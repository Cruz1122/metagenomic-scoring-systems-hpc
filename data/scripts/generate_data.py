#!/usr/bin/env python3
"""
generate_data.py — Generador de dataset metagenómico sintético-compatible.

Genera el dataset cMD_CRC100_balanced (100 muestras × 500 items) para el
modelo de scoring HPC:

    P_i = W1 * T_i + W2 * S_i + W3 * F_i
    Score_muestra = A[muestra, :] @ P
    AUC = auc(labels, Score)

Matemática del dataset:
  - A  : matriz de abundancias relativas (100 × 500), filas suman ~1.0.
  - y  : vector binario de etiquetas (0 = healthy, 1 = CRC).
  - T_i: magnitud diferencial del item i entre CRC y healthy, en [0, 1].
  - S_i: asociación del item i con metadatos poblacionales, en [0, 1].
  - F_i: perfil funcional del item i, media de 5 marcadores, en [0, 1].

Salida:
    data/csv/       → 6 archivos CSV
    data/npy/       → 3 archivos NumPy .npy
    data/dataset_manifest.json

Uso:
    python data/scripts/generate_data.py
    python data/scripts/generate_data.py --seed 42 --signal 2.0
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constantes del dataset
# ---------------------------------------------------------------------------

# Proporción de items enriquecidos en cada grupo
CRC_ENRICHED_FRACTION = 0.10     # 10 % → CRC_enriched
HEALTHY_ENRICHED_FRACTION = 0.20 # 20 % → healthy_enriched

# Nombre del estudio (constante para todo el dataset)
STUDY_NAME = "synthetic_CRC_study"

# Columnas de marcadores funcionales en la matriz funcional
FUNCTIONAL_MARKER_COLS = [
    "resistance_marker",
    "virulence_marker",
    "inflammation_marker",
    "metabolic_marker",
    "beneficial_marker",
]


# ---------------------------------------------------------------------------
# Funciones auxiliares de asociación estadística (sin scipy)
# ---------------------------------------------------------------------------

def _abs_pearson(x: np.ndarray, y: np.ndarray) -> float:
    """Coeficiente de correlación de Pearson en valor absoluto.

    Retorna 0.0 si alguna variable no tiene varianza o si hay menos de
    3 puntos válidos.
    """
    # Eliminar pares con NaN
    mask = ~(np.isnan(x) | np.isnan(y))
    x_clean, y_clean = x[mask], y[mask]

    if len(x_clean) < 3:
        return 0.0
    if np.std(x_clean) == 0.0 or np.std(y_clean) == 0.0:
        return 0.0

    r = np.corrcoef(x_clean, y_clean)[0, 1]
    return abs(r) if not np.isnan(r) else 0.0


def _correlation_ratio(x: np.ndarray, categories: np.ndarray) -> float:
    """Eta (raíz cuadrada del correlation ratio) entre abundancia y categorías.

    Mide la fracción de la varianza total explicada por la pertenencia a
    cada categoría. Retorna 0.0 si hay menos de 2 categorías o varianza
    total nula.
    """
    unique_cats = np.unique(categories)
    if len(unique_cats) < 2:
        return 0.0

    grand_mean = float(np.mean(x))
    ss_between = 0.0
    for cat in unique_cats:
        mask = categories == cat
        n_cat = int(mask.sum())
        if n_cat == 0:
            continue
        cat_mean = float(np.mean(x[mask]))
        ss_between += n_cat * (cat_mean - grand_mean) ** 2

    ss_total = float(np.sum((x - grand_mean) ** 2))
    if ss_total == 0.0:
        return 0.0

    return np.sqrt(ss_between / ss_total)


# ---------------------------------------------------------------------------
# Bloques de construcción del dataset
# ---------------------------------------------------------------------------

def build_samples(n_healthy: int, n_crc: int) -> pd.DataFrame:
    """Construye la tabla de muestras con etiquetas.

    Args:
        n_healthy: Número de muestras sanas (label = 0).
        n_crc: Número de muestras con CRC (label = 1).

    Returns:
        DataFrame con columnas sample_id, label, group.
    """
    rows = []
    for i in range(n_healthy):
        rows.append({
            "sample_id": f"CTRL_{i + 1:03d}",
            "label": 0,
            "group": "healthy",
        })
    for i in range(n_crc):
        rows.append({
            "sample_id": f"CRC_{i + 1:03d}",
            "label": 1,
            "group": "CRC",
        })
    return pd.DataFrame(rows)


def build_metadata(samples: pd.DataFrame,
                   rng: np.random.Generator) -> pd.DataFrame:
    """Genera metadatos poblacionales por muestra.

    Simula variables clínicas y demográficas con distribución diferenciada
    entre grupos (edad y BMI más altos en CRC).

    Args:
        samples: DataFrame con la columna sample_id y label.
        rng: Generador de números aleatorios NumPy.

    Returns:
        DataFrame con sample_id, age, sex, bmi, country, study_name, disease.
    """
    n = len(samples)
    labels = samples["label"].to_numpy()
    countries = ["France", "Germany", "Spain", "Italy", "Denmark"]

    # Edad: CRC tiende a ser mayor
    age = rng.normal(loc=np.where(labels == 1, 63.0, 52.0), scale=7.0)
    age = np.clip(np.round(age), 30, 85).astype(int)

    # BMI: ligeramente más alto en CRC
    bmi = rng.normal(loc=np.where(labels == 1, 26.5, 24.5), scale=2.5)
    bmi = np.clip(bmi, 18.0, 35.0).round(1)

    metadata = samples[["sample_id"]].copy()
    metadata["age"] = age
    metadata["sex"] = rng.choice(["male", "female"], size=n)
    metadata["bmi"] = bmi
    metadata["country"] = rng.choice(countries, size=n)
    metadata["study_name"] = STUDY_NAME
    metadata["disease"] = np.where(labels == 1, "CRC", "healthy")

    return metadata


def build_abundance_matrix(n_samples: int,
                           n_items: int,
                           labels: np.ndarray,
                           rng: np.random.Generator) -> np.ndarray:
    """Genera la matriz de abundancias relativas A.

    Los primeros items se enriquecen en CRC o healthy según las fracciones
    definidas en las constantes globales. El resto son items neutros con
    ruido. Cada fila se normaliza para sumar ~1.0.

    Args:
        n_samples: Número de filas (muestras).
        n_items: Número de columnas (items/taxones).
        labels: Vector binario 0/1.
        rng: Generador de números aleatorios NumPy.

    Returns:
        Matriz float32 de forma (n_samples, n_items).
    """
    n_crc_enriched = max(1, int(round(CRC_ENRICHED_FRACTION * n_items)))
    n_healthy_enriched = max(1, int(round(HEALTHY_ENRICHED_FRACTION * n_items)))

    is_healthy = (labels == 0)
    is_crc = (labels == 1)

    # Abundancia base uniforme con algo de dispersión
    abundances = rng.uniform(0.05, 1.0, size=(n_samples, n_items)).astype(np.float64)

    # Inyectar señal diferencial
    for i in range(n_crc_enriched):
        abundances[is_healthy, i] *= 0.3   # menos en sanos
        abundances[is_crc, i] *= 1.7       # más en CRC

    for i in range(n_crc_enriched, n_crc_enriched + n_healthy_enriched):
        abundances[is_healthy, i] *= 1.7   # más en sanos
        abundances[is_crc, i] *= 0.3       # menos en CRC

    # Ruido multiplicativo para evitar muestras idénticas dentro del mismo grupo
    abundances *= rng.uniform(0.8, 1.2, size=abundances.shape)

    # Normalización composicional
    abundances = np.maximum(abundances, 0.0)
    row_sums = abundances.sum(axis=1, keepdims=True)
    abundances /= row_sums

    return abundances.astype(np.float32)


def build_functional_matrix(n_items: int,
                            n_crc_enriched: int,
                            n_healthy_enriched: int,
                            rng: np.random.Generator) -> pd.DataFrame:
    """Genera la matriz de marcadores funcionales por item.

    Las probabilidades de cada marcador dependen del tipo de item:
    - CRC_enriched:     alta probabilidad de virulence e inflammation.
    - Healthy_enriched: alta probabilidad de metabolic y beneficial.
    - Neutro:           probabilidades bajas o moderadas.

    Args:
        n_items: Número total de items.
        n_crc_enriched: Items enriquecidos en CRC.
        n_healthy_enriched: Items enriquecidos en healthy.
        rng: Generador de números aleatorios NumPy.

    Returns:
        DataFrame con columnas item_id y los 5 marcadores (0/1).
    """
    # Definir probabilidades por tipo de item
    # Cada tupla: (resistance, virulence, inflammation, metabolic, beneficial)
    probability_map = []
    for i in range(n_items):
        if i < n_crc_enriched:
            probability_map.append((0.40, 0.80, 0.90, 0.35, 0.05))
        elif i < n_crc_enriched + n_healthy_enriched:
            probability_map.append((0.05, 0.05, 0.05, 0.70, 0.65))
        else:
            probability_map.append((0.05, 0.08, 0.08, 0.35, 0.15))

    # Construir fila por fila
    rows = []
    for i, probs in enumerate(probability_map):
        p_res, p_vir, p_inf, p_met, p_ben = probs
        rows.append({
            "item_id": f"item_{i:03d}",
            "resistance_marker": int(rng.random() < p_res),
            "virulence_marker": int(rng.random() < p_vir),
            "inflammation_marker": int(rng.random() < p_inf),
            "metabolic_marker": int(rng.random() < p_met),
            "beneficial_marker": int(rng.random() < p_ben),
        })

    return pd.DataFrame(rows)


def build_item_mapping(n_items: int) -> pd.DataFrame:
    """Crea el mapeo item_id → taxon_name para todos los items.

    Args:
        n_items: Número total de items.

    Returns:
        DataFrame con columnas item_id, taxon_name, original_feature_id, rank.
    """
    rows = []
    for i in range(n_items):
        rows.append({
            "item_id": f"item_{i:03d}",
            "taxon_name": f"Synthetic taxon {i:03d}",
            "original_feature_id": f"synthetic_species_{i:03d}",
            "rank": "species",
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Cálculo de los perfiles T, S, F
# ---------------------------------------------------------------------------

def compute_profiles(
    A: np.ndarray,
    labels: np.ndarray,
    metadata: pd.DataFrame,
    functional_matrix: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Calcula los tres perfiles por item: T, S, F.

    T (taxonómico):
        Magnitud de la diferencia de abundancia entre grupos, normalizada
        a [0, 1]. Dirección en columna separada `taxon_direction`.

    S (ecológico/poblacional):
        Asociación entre la abundancia del item y cada variable de metadata
        con variabilidad real. Se usa Pearson para numéricas (age, bmi) y
        correlation ratio (eta) para categóricas (sex, country).
        study_name es constante, se excluye.

    F (funcional):
        Media simple de los 5 marcadores funcionales (0/1), ya en [0, 1].

    Args:
        A: Matriz de abundancias (n_samples, n_items).
        labels: Vector de etiquetas (n_samples,).
        metadata: DataFrame con variables poblacionales.
        functional_matrix: DataFrame con marcadores funcionales.

    Returns:
        item_profiles: DataFrame con item_id, taxon_name, T, taxon_direction,
                       S, F.
        profiles_TSF: array float32 (n_items, 3) con columnas [T, S, F].
    """
    n_items = A.shape[1]

    # ------------------------------------------------------------------
    #  T: magnitud diferencial
    # ------------------------------------------------------------------
    mean_crc = A[labels == 1].mean(axis=0)
    mean_healthy = A[labels == 0].mean(axis=0)

    raw_T = np.abs(mean_crc - mean_healthy)
    max_raw_T = float(raw_T.max())
    T_profile = raw_T / max_raw_T if max_raw_T > 0 else np.zeros(n_items)

    # Dirección biológica (separada de T)
    taxon_direction = np.where(
        mean_crc > mean_healthy,
        "CRC_enriched",
        np.where(mean_crc < mean_healthy, "healthy_enriched", "neutral"),
    )

    # ------------------------------------------------------------------
    #  S: asociación con metadatos
    # ------------------------------------------------------------------
    # Seleccionar solo variables con variabilidad real
    usable_variables = []  # Lista de tuplas (tipo, nombre, valores)

    for var_name in ["age", "bmi"]:
        values = metadata[var_name].to_numpy(dtype=float)
        if np.std(values) > 1e-12:
            usable_variables.append(("numeric", var_name, values))

    for var_name in ["sex", "country"]:
        values = metadata[var_name].to_numpy()
        if len(np.unique(values)) >= 2:
            usable_variables.append(("categorical", var_name, values))

    # study_name es constante → se excluye automáticamente

    S_raw = np.zeros(n_items)

    if usable_variables:
        for i in range(n_items):
            item_abundance = A[:, i]
            associations = []

            for var_type, _var_name, var_values in usable_variables:
                if var_type == "numeric":
                    assoc = _abs_pearson(item_abundance, var_values.astype(float))
                else:
                    assoc = _correlation_ratio(item_abundance, var_values)
                associations.append(assoc)

            S_raw[i] = float(np.mean(associations))

    max_S = float(S_raw.max())
    S_profile = S_raw / max_S if max_S > 0 else np.zeros(n_items)

    # ------------------------------------------------------------------
    #  F: perfil funcional (media de marcadores)
    # ------------------------------------------------------------------
    marker_data = functional_matrix[FUNCTIONAL_MARKER_COLS].to_numpy(dtype=float)
    F_profile = marker_data.mean(axis=1)  # ya en [0, 1]

    # ------------------------------------------------------------------
    #  Ensamblar DataFrames de salida
    # ------------------------------------------------------------------
    item_ids = [f"item_{i:03d}" for i in range(n_items)]
    taxon_names = [f"Synthetic taxon {i:03d}" for i in range(n_items)]

    item_profiles = pd.DataFrame({
        "item_id": item_ids,
        "taxon_name": taxon_names,
        "T": T_profile,
        "taxon_direction": taxon_direction,
        "S": S_profile,
        "F": F_profile,
    })

    profiles_TSF = np.column_stack([T_profile, S_profile, F_profile]).astype(np.float32)

    return item_profiles, profiles_TSF


# ---------------------------------------------------------------------------
# Punto de entrada principal
# ---------------------------------------------------------------------------

def main() -> None:
    """Genera el dataset cMD_CRC100_balanced y lo escribe en disco."""
    parser = argparse.ArgumentParser(
        description="Genera dataset metagenómico sintético-compatible",
    )
    parser.add_argument("--seed", type=int, default=42,
                        help="Semilla para reproducibilidad")
    parser.add_argument("--signal", type=float, default=2.0,
                        help="Controla separabilidad (mantenido por compatibilidad)")
    parser.add_argument("--out-dir", type=Path, default=Path("data"),
                        help="Directorio raíz de salida")
    args = parser.parse_args()

    # Directorios de salida
    output_root = args.out_dir.resolve()
    csv_dir = output_root / "csv"
    npy_dir = output_root / "npy"

    csv_dir.mkdir(parents=True, exist_ok=True)
    npy_dir.mkdir(parents=True, exist_ok=True)

    # Generador reproducible
    rng = np.random.default_rng(args.seed)

    # Parámetros del dataset
    N_SAMPLES = 100
    N_ITEMS = 500
    N_HEALTHY = N_SAMPLES // 2
    N_CRC = N_SAMPLES // 2

    print("Generando dataset cMD_CRC100_balanced "
          f"({N_SAMPLES} muestras × {N_ITEMS} items) ...")

    # Construir componentes
    samples = build_samples(N_HEALTHY, N_CRC)
    labels = samples["label"].to_numpy(dtype=np.int32)

    metadata = build_metadata(samples, rng)
    abundance_matrix = build_abundance_matrix(N_SAMPLES, N_ITEMS, labels, rng)

    n_crc_enriched = max(1, int(round(CRC_ENRICHED_FRACTION * N_ITEMS)))
    n_healthy_enriched = max(1, int(round(HEALTHY_ENRICHED_FRACTION * N_ITEMS)))

    functional_matrix = build_functional_matrix(
        N_ITEMS, n_crc_enriched, n_healthy_enriched, rng,
    )
    item_mapping = build_item_mapping(N_ITEMS)

    # Calcular perfiles T, S, F
    item_profiles, profiles_TSF = compute_profiles(
        abundance_matrix, labels, metadata, functional_matrix,
    )

    # ------------------------------------------------------------------
    #  Escribir archivos CSV
    # ------------------------------------------------------------------
    samples.to_csv(csv_dir / "samples.csv", index=False)

    matrix_A_csv = pd.DataFrame(
        abundance_matrix,
        columns=[f"item_{i:03d}" for i in range(N_ITEMS)],
    )
    matrix_A_csv.insert(0, "sample_id", samples["sample_id"].to_numpy())
    matrix_A_csv.to_csv(csv_dir / "matrix_A.csv",
                        index=False, float_format="%.9g")

    metadata.to_csv(csv_dir / "metadata.csv", index=False)
    functional_matrix.to_csv(csv_dir / "functional_matrix.csv", index=False)
    item_profiles.to_csv(csv_dir / "item_profiles.csv",
                         index=False, float_format="%.9g")
    item_mapping.to_csv(csv_dir / "item_mapping.csv", index=False)

    # ------------------------------------------------------------------
    #  Escribir archivos NumPy
    # ------------------------------------------------------------------
    np.save(npy_dir / "matrix_A.npy", abundance_matrix.astype(np.float32))
    np.save(npy_dir / "labels.npy", labels.astype(np.int32))
    np.save(npy_dir / "profiles_TSF.npy", profiles_TSF.astype(np.float32))

    # ------------------------------------------------------------------
    #  Manifest JSON
    # ------------------------------------------------------------------
    manifest = {
        "dataset_name": "cMD_CRC100_balanced",
        "dataset_type": "synthetic-compatible",
        "source_reference": "curatedMetagenomicData / Bioconductor compatible",
        "disease_task": "colorectal_cancer_vs_healthy",
        "positive_class": "CRC",
        "negative_class": "healthy/control",
        "seed": args.seed,
        "n_samples": N_SAMPLES,
        "n_healthy": N_HEALTHY,
        "n_crc": N_CRC,
        "n_items": N_ITEMS,
        "abundance_type": "relative_abundance",
        "T_formula": "abs(mean_crc_i - mean_healthy_i) / max_raw_T",
        "S_formula": "mean normalized association between item abundance "
                     "and usable metadata variables",
        "F_formula": "mean of 5 functional markers",
        "note": "Synthetic-compatible dataset for debugging and HPC benchmarks. "
                "Final presentation dataset should be reconstructed from "
                "curatedMetagenomicData.",
    }
    with open(output_root / "dataset_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------
    #  Reporte final
    # ------------------------------------------------------------------
    print("OK")
    print(f"    CSV -> {csv_dir}/")
    print(f"    NPY -> {npy_dir}/")
    print(f"    A: {abundance_matrix.shape}  dtype={abundance_matrix.dtype}")
    print(f"    labels: {labels.shape}  dtype={labels.dtype}  "
          f"healthy={N_HEALTHY}  CRC={N_CRC}")
    print(f"    profiles_TSF: {profiles_TSF.shape}  dtype={profiles_TSF.dtype}")


if __name__ == "__main__":
    main()
