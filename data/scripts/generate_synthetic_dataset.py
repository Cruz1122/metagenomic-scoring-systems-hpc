#!/usr/bin/env python3
"""
generate_synthetic_dataset.py

Generador reproducible de dataset metagenómico compatible con el modelo HPC
de scoring binario healthy/control vs CRC.

Este script NO descarga datos reales. Produce un dataset sintético-compatible
para desarrollo, debugging, pruebas de I/O y benchmarks preliminares.

Salida principal:
    data/csv/samples.csv
    data/csv/matrix_A.csv
    data/csv/metadata.csv
    data/csv/functional_matrix.csv
    data/csv/item_profiles.csv
    data/csv/item_mapping.csv
    data/npy/matrix_A.npy
    data/npy/labels.npy
    data/npy/profiles_TSF.npy
    data/dataset_manifest.json

Modelo:
    P_i = W1*T_i + W2*S_i + W3*F_i
    Score = A @ P

Uso:
    python data/scripts/generate_synthetic_dataset.py \
        --out-dir data \
        --n-samples 100 \
        --n-items 500 \
        --seed 42 \
        --signal 2.2

Autor: proyecto scoring_metagenomico
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


REALISTIC_TAXA = [
    # CRC-associated / oral-like / opportunistic taxa frequently discussed in CRC microbiome literature
    "Fusobacterium nucleatum",
    "Peptostreptococcus stomatis",
    "Parvimonas micra",
    "Solobacterium moorei",
    "Porphyromonas asaccharolytica",
    "Prevotella intermedia",
    "Campylobacter showae",
    "Gemella morbillorum",
    "Leptotrichia buccalis",
    "Streptococcus anginosus",
    "Bacteroides fragilis",
    "Escherichia coli",
    "Klebsiella pneumoniae",
    "Citrobacter freundii",
    "Enterococcus faecalis",
    "Enterococcus faecium",
    "Clostridium symbiosum",
    "Clostridium hathewayi",
    "Alistipes finegoldii",
    "Akkermansia muciniphila",

    # gut commensals / health-associated taxa
    "Faecalibacterium prausnitzii",
    "Roseburia intestinalis",
    "Roseburia hominis",
    "Eubacterium rectale",
    "Eubacterium hallii",
    "Eubacterium eligens",
    "Bifidobacterium adolescentis",
    "Bifidobacterium longum",
    "Bifidobacterium bifidum",
    "Blautia obeum",
    "Blautia wexlerae",
    "Subdoligranulum variabile",
    "Ruminococcus bromii",
    "Ruminococcus callidus",
    "Coprococcus comes",
    "Coprococcus eutactus",
    "Anaerostipes hadrus",
    "Anaerobutyricum hallii",
    "Christensenella minuta",
    "Butyricicoccus pullicaecorum",

    # abundant gut taxa / mixed behavior
    "Bacteroides vulgatus",
    "Bacteroides uniformis",
    "Bacteroides ovatus",
    "Bacteroides thetaiotaomicron",
    "Bacteroides stercoris",
    "Bacteroides dorei",
    "Prevotella copri",
    "Parabacteroides distasonis",
    "Alistipes putredinis",
    "Alistipes shahii",
    "Collinsella aerofaciens",
    "Dorea longicatena",
    "Dorea formicigenerans",
    "Lachnospira pectinoschiza",
    "Oscillibacter valericigenes",
    "Flavonifractor plautii",
    "Eggerthella lenta",
    "Bilophila wadsworthia",
    "Desulfovibrio piger",
    "Methanobrevibacter smithii",
]


CRC_LIKE_TAXA = {
    "Fusobacterium nucleatum",
    "Peptostreptococcus stomatis",
    "Parvimonas micra",
    "Solobacterium moorei",
    "Porphyromonas asaccharolytica",
    "Prevotella intermedia",
    "Campylobacter showae",
    "Gemella morbillorum",
    "Leptotrichia buccalis",
    "Streptococcus anginosus",
    "Bacteroides fragilis",
    "Escherichia coli",
    "Klebsiella pneumoniae",
    "Citrobacter freundii",
    "Enterococcus faecalis",
    "Clostridium symbiosum",
    "Clostridium hathewayi",
}


HEALTHY_LIKE_TAXA = {
    "Faecalibacterium prausnitzii",
    "Roseburia intestinalis",
    "Roseburia hominis",
    "Eubacterium rectale",
    "Eubacterium hallii",
    "Eubacterium eligens",
    "Bifidobacterium adolescentis",
    "Bifidobacterium longum",
    "Bifidobacterium bifidum",
    "Blautia obeum",
    "Blautia wexlerae",
    "Subdoligranulum variabile",
    "Ruminococcus bromii",
    "Coprococcus comes",
    "Anaerostipes hadrus",
    "Christensenella minuta",
    "Butyricicoccus pullicaecorum",
}


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Sigmoide estable para transformar efectos en factores positivos."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def _minmax(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Normaliza un vector al rango [0, 1]."""
    lo = float(np.min(x))
    hi = float(np.max(x))
    return (x - lo) / (hi - lo + eps)


def _safe_row_normalize(A: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Normaliza filas para que cada muestra sume 1."""
    row_sum = A.sum(axis=1, keepdims=True)
    return A / np.maximum(row_sum, eps)


def build_taxon_names(n_items: int) -> list[str]:
    """Construye nombres taxonómicos plausibles para n_items.

    Los primeros nombres son taxones reales frecuentes en literatura intestinal/CRC.
    Si n_items excede la lista base, completa con nombres proxy.
    """
    names = list(REALISTIC_TAXA[:n_items])
    while len(names) < n_items:
        idx = len(names)
        genus = [
            "Bacteroides", "Clostridium", "Eubacterium", "Ruminococcus",
            "Lachnospiraceae", "Oscillibacter", "Alistipes", "Blautia",
            "Prevotella", "Anaerostipes", "Coprococcus", "Roseburia",
        ][idx % 12]
        names.append(f"{genus} synthetic_species_{idx:03d}")
    return names


def build_samples(n_healthy: int, n_disease: int) -> pd.DataFrame:
    """Crea samples.csv con etiquetas 0/1 y grupos healthy/CRC."""
    rows = []
    for i in range(n_healthy):
        rows.append({"sample_id": f"CTRL_{i + 1:03d}", "label": 0, "group": "healthy"})
    for i in range(n_disease):
        rows.append({"sample_id": f"CRC_{i + 1:03d}", "label": 1, "group": "CRC"})
    return pd.DataFrame(rows)


def build_metadata(samples: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Genera metadata poblacional/ecológica plausible por muestra.

    Las distribuciones están diseñadas para parecer cohortes humanas heterogéneas:
    edad, sexo, BMI, dieta, antibióticos, país, estudio y ambiente.
    """
    n = len(samples)
    y = samples["label"].to_numpy()

    countries = np.array(["France", "Germany", "Spain", "Italy", "Denmark"])
    studies = np.array(["ZellerG_2014", "FengQ_2015", "WirbelJ_2019"])
    diets = np.array(["western", "mediterranean", "omnivore", "high_fiber"])
    sexes = np.array(["female", "male"])

    # CRC tiende a edad algo mayor y BMI ligeramente mayor en esta simulación.
    age = rng.normal(loc=np.where(y == 1, 63.0, 53.0), scale=8.5)
    age = np.clip(np.rint(age), 35, 82).astype(int)

    bmi = rng.normal(loc=np.where(y == 1, 26.6, 24.2), scale=3.1)
    bmi = np.clip(bmi, 18.0, 34.5).round(2)

    sex = rng.choice(sexes, size=n, p=[0.52, 0.48])

    diet = []
    for label in y:
        if label == 1:
            diet.append(rng.choice(diets, p=[0.38, 0.17, 0.35, 0.10]))
        else:
            diet.append(rng.choice(diets, p=[0.18, 0.25, 0.34, 0.23]))

    antibiotic_use = rng.choice(["no", "yes"], size=n, p=[0.90, 0.10])
    environment = rng.choice(["urban", "suburban", "rural"], size=n, p=[0.70, 0.20, 0.10])
    country = rng.choice(countries, size=n, p=[0.22, 0.18, 0.22, 0.20, 0.18])
    study_name = rng.choice(studies, size=n, p=[0.45, 0.35, 0.20])

    meta = samples[["sample_id"]].copy()
    meta["age"] = age
    meta["sex"] = sex
    meta["bmi"] = bmi
    meta["diet"] = diet
    meta["antibiotic_use"] = antibiotic_use
    meta["environment"] = environment
    meta["location"] = country
    meta["country"] = country
    meta["study_name"] = study_name
    meta["disease"] = np.where(y == 1, "CRC", "healthy")

    return meta


def build_item_effects(
    taxon_names: list[str],
    rng: np.random.Generator,
    crc_fraction: float = 0.16,
    healthy_fraction: float = 0.24,
) -> tuple[np.ndarray, np.ndarray]:
    """Genera efectos taxonómicos latentes.

    Returns:
        effect: valor continuo. Positivo = enriquecido en CRC; negativo = healthy.
        item_class: array de strings {crc_enriched, healthy_enriched, neutral}
    """
    n_items = len(taxon_names)
    effect = rng.normal(0.0, 0.18, size=n_items)
    item_class = np.array(["neutral"] * n_items, dtype=object)

    # Primero fuerza taxones conocidos a clases biológicamente plausibles.
    for i, name in enumerate(taxon_names):
        if name in CRC_LIKE_TAXA:
            effect[i] += rng.normal(1.15, 0.25)
            item_class[i] = "crc_enriched"
        elif name in HEALTHY_LIKE_TAXA:
            effect[i] -= rng.normal(1.00, 0.25)
            item_class[i] = "healthy_enriched"

    # Luego completa proporciones si hacen falta.
    target_crc = max(1, int(round(crc_fraction * n_items)))
    target_healthy = max(1, int(round(healthy_fraction * n_items)))

    current_crc = np.where(item_class == "crc_enriched")[0]
    current_healthy = np.where(item_class == "healthy_enriched")[0]
    neutral_idx = np.where(item_class == "neutral")[0]

    if len(current_crc) < target_crc and len(neutral_idx) > 0:
        add = rng.choice(neutral_idx, size=min(target_crc - len(current_crc), len(neutral_idx)), replace=False)
        effect[add] += rng.normal(0.85, 0.25, size=len(add))
        item_class[add] = "crc_enriched"

    neutral_idx = np.where(item_class == "neutral")[0]
    if len(current_healthy) < target_healthy and len(neutral_idx) > 0:
        add = rng.choice(neutral_idx, size=min(target_healthy - len(current_healthy), len(neutral_idx)), replace=False)
        effect[add] -= rng.normal(0.80, 0.25, size=len(add))
        item_class[add] = "healthy_enriched"

    effect = np.clip(effect, -2.0, 2.0)
    return effect.astype(np.float64), item_class


def build_functional_matrix(
    item_ids: list[str],
    taxon_names: list[str],
    effect: np.ndarray,
    item_class: np.ndarray,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Genera matriz funcional proxy por item.

    Las probabilidades dependen del tipo de item:
    - CRC-enriched: más virulence/inflammation/resistance.
    - Healthy-enriched: más beneficial/metabolic.
    - Neutral: principalmente metabolic con probabilidades moderadas.
    """
    rows = []
    for item_id, taxon, eff, cls in zip(item_ids, taxon_names, effect, item_class):
        if cls == "crc_enriched":
            p_res = 0.14 + 0.10 * _sigmoid(np.array([eff]))[0]
            p_vir = 0.35 + 0.20 * _sigmoid(np.array([eff]))[0]
            p_inf = 0.45 + 0.25 * _sigmoid(np.array([eff]))[0]
            p_met = 0.58
            p_ben = 0.05
        elif cls == "healthy_enriched":
            p_res = 0.03
            p_vir = 0.04
            p_inf = 0.07
            p_met = 0.72
            p_ben = 0.48
        else:
            p_res = 0.05
            p_vir = 0.08
            p_inf = 0.12
            p_met = 0.55
            p_ben = 0.18

        rows.append(
            {
                "item_id": item_id,
                "resistance_marker": int(rng.random() < p_res),
                "virulence_marker": int(rng.random() < p_vir),
                "inflammation_marker": int(rng.random() < p_inf),
                "metabolic_marker": int(rng.random() < p_met),
                "beneficial_marker": int(rng.random() < p_ben),
            }
        )

    return pd.DataFrame(rows)


def build_profiles(
    effect: np.ndarray,
    metadata: pd.DataFrame,
    functional_matrix: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Construye perfiles T, S, F por item.

    T:
        efecto taxonómico escalado. Positivo = CRC, negativo = healthy.

    S:
        asociación simulada con covariables poblacionales/ecológicas.
        Usa una combinación de ruido estructurado + intensidad de confusión.

    F:
        score funcional normalizado derivado de marcadores proxy.
    """
    n_items = len(effect)

    # T conserva signo. Se escala a rango aprox [-1, 1].
    T = effect / (np.max(np.abs(effect)) + 1e-12)

    # S refleja asociación con metadatos. No debe ser constante.
    # Se modela como intensidad no negativa dependiente de variables simuladas.
    age_sd = float(metadata["age"].std())
    bmi_sd = float(metadata["bmi"].std())
    country_entropy_proxy = metadata["country"].nunique() / max(1, len(metadata))
    study_entropy_proxy = metadata["study_name"].nunique() / max(1, len(metadata))

    metadata_strength = min(1.0, 0.08 * age_sd + 0.06 * bmi_sd + 4.0 * country_entropy_proxy + 3.0 * study_entropy_proxy)
    S_raw = (
        0.35 * np.abs(effect)
        + 0.45 * rng.beta(1.4, 3.0, size=n_items)
        + 0.20 * metadata_strength * rng.random(n_items)
    )
    S = _minmax(S_raw)

    fm = functional_matrix[
        [
            "resistance_marker",
            "virulence_marker",
            "inflammation_marker",
            "metabolic_marker",
            "beneficial_marker",
        ]
    ].to_numpy(dtype=np.float64)

    # Marcadores pro-enfermedad suman positivo; beneficial también entra en F
    # como densidad funcional, no necesariamente como riesgo.
    F_raw = (
        0.20 * fm[:, 0]  # resistance
        + 0.30 * fm[:, 1]  # virulence
        + 0.30 * fm[:, 2]  # inflammation
        + 0.15 * fm[:, 3]  # metabolic
        + 0.25 * fm[:, 4]  # beneficial
    )
    F = _minmax(F_raw + 0.05 * rng.random(n_items))

    return pd.DataFrame({"T": T, "S": S, "F": F})


def build_abundance_matrix(
    labels: np.ndarray,
    effect: np.ndarray,
    rng: np.random.Generator,
    signal: float = 2.2,
    concentration: float = 240.0,
    dropout_rate: float = 0.30,
) -> np.ndarray:
    """Genera matriz composicional de abundancias relativas.

    La matriz se genera con un esquema Dirichlet/log-normal:
    - base_abundance: abundancia media pesada/heterogénea.
    - effect: desplaza abundancias según label healthy/CRC.
    - Dirichlet: agrega variabilidad muestral y respeta composicionalidad.
    - dropout: introduce ceros estructurales de baja abundancia.

    Args:
        labels: vector 0/1.
        effect: efecto por item. Positivo CRC, negativo healthy.
        rng: generador numpy.
        signal: fuerza de separación entre grupos.
        concentration: concentración Dirichlet. Menor = más ruido.
        dropout_rate: tasa base de ceros para items muy raros.

    Returns:
        A: matriz (n_samples, n_items), filas normalizadas.
    """
    n_samples = len(labels)
    n_items = len(effect)

    # Abundancia base heavy-tailed: pocos taxones abundantes, muchos raros.
    base = rng.lognormal(mean=-3.2, sigma=1.15, size=n_items)
    base = base / base.sum()

    A = np.zeros((n_samples, n_items), dtype=np.float64)

    for j, label in enumerate(labels):
        direction = 1.0 if label == 1 else -1.0

        # Ruido individual para que muestras del mismo grupo no sean clones.
        individual_noise = rng.normal(0.0, 0.20, size=n_items)

        log_shift = signal * direction * effect + individual_noise
        mu = base * np.exp(log_shift)
        mu = mu / mu.sum()

        # Dirichlet composicional.
        alpha = np.maximum(mu * concentration, 1e-5)
        row = rng.dirichlet(alpha)

        # Dropout dependiente de rareza: taxones de baja abundancia caen más.
        rarity = 1.0 - _minmax(mu)
        p_dropout = np.clip(dropout_rate * rarity, 0.0, 0.85)
        mask = rng.random(n_items) < p_dropout
        row[mask] = 0.0

        if row.sum() <= 0:
            row = mu.copy()

        A[j, :] = row / row.sum()

    return A.astype(np.float32)


def build_dataset(
    n_samples: int = 100,
    n_items: int = 500,
    seed: int = 42,
    signal: float = 2.2,
    concentration: float = 240.0,
    dropout_rate: float = 0.30,
) -> dict[str, object]:
    """Construye todas las tablas del dataset sintético-compatible."""
    if n_samples < 10:
        raise ValueError("n_samples debe ser >= 10.")
    if n_samples % 2 != 0:
        raise ValueError("n_samples debe ser par para mantener balance 50/50.")
    if n_items < 10:
        raise ValueError("n_items debe ser >= 10.")
    if not (0.0 <= dropout_rate < 1.0):
        raise ValueError("dropout_rate debe estar en [0, 1).")

    rng = np.random.default_rng(seed)
    n_healthy = n_samples // 2
    n_disease = n_samples // 2

    samples = build_samples(n_healthy, n_disease)
    labels = samples["label"].to_numpy(dtype=np.int32)

    metadata = build_metadata(samples, rng)

    item_ids = [f"item_{i:03d}" for i in range(n_items)]
    taxon_names = build_taxon_names(n_items)

    effect, item_class = build_item_effects(taxon_names, rng)
    functional_matrix = build_functional_matrix(item_ids, taxon_names, effect, item_class, rng)
    profiles = build_profiles(effect, metadata, functional_matrix, rng)

    A = build_abundance_matrix(
        labels=labels,
        effect=effect,
        rng=rng,
        signal=signal,
        concentration=concentration,
        dropout_rate=dropout_rate,
    )

    matrix_A = pd.DataFrame(A, columns=item_ids)
    matrix_A.insert(0, "sample_id", samples["sample_id"].to_numpy())

    item_profiles = pd.DataFrame(
        {
            "item_id": item_ids,
            "taxon_name": taxon_names,
            "T": profiles["T"].to_numpy(),
            "S": profiles["S"].to_numpy(),
            "F": profiles["F"].to_numpy(),
        }
    )

    item_mapping = pd.DataFrame(
        {
            "item_id": item_ids,
            "taxon_name": taxon_names,
            "feature_type": "species",
            "source_feature_id": [f"synthetic_cMD_species_proxy_{i:03d}" for i in range(n_items)],
            "item_class": item_class,
            "crc_effect": effect,
        }
    )

    profiles_TSF = item_profiles[["T", "S", "F"]].to_numpy(dtype=np.float32)

    return {
        "samples": samples,
        "matrix_A": matrix_A,
        "metadata": metadata,
        "functional_matrix": functional_matrix,
        "item_profiles": item_profiles,
        "item_mapping": item_mapping,
        "A": A,
        "labels": labels,
        "profiles_TSF": profiles_TSF,
        "effect": effect,
        "item_class": item_class,
    }


def validate_dataset(dataset: dict[str, object], atol: float = 1e-5) -> dict[str, object]:
    """Valida integridad básica del dataset generado."""
    samples: pd.DataFrame = dataset["samples"]  # type: ignore[assignment]
    metadata: pd.DataFrame = dataset["metadata"]  # type: ignore[assignment]
    matrix_A: pd.DataFrame = dataset["matrix_A"]  # type: ignore[assignment]
    functional_matrix: pd.DataFrame = dataset["functional_matrix"]  # type: ignore[assignment]
    item_profiles: pd.DataFrame = dataset["item_profiles"]  # type: ignore[assignment]
    A: np.ndarray = dataset["A"]  # type: ignore[assignment]
    labels: np.ndarray = dataset["labels"]  # type: ignore[assignment]
    profiles_TSF: np.ndarray = dataset["profiles_TSF"]  # type: ignore[assignment]

    errors: list[str] = []

    if A.ndim != 2:
        errors.append(f"A debe ser 2D, recibido {A.ndim}D.")
    if labels.ndim != 1:
        errors.append(f"labels debe ser 1D, recibido {labels.ndim}D.")
    if A.shape[0] != labels.shape[0]:
        errors.append(f"A filas != labels: {A.shape[0]} != {labels.shape[0]}.")
    if profiles_TSF.shape != (A.shape[1], 3):
        errors.append(f"profiles_TSF shape inválido: {profiles_TSF.shape}, esperado {(A.shape[1], 3)}.")
    if len(samples) != A.shape[0]:
        errors.append("samples.csv no coincide con filas de A.")
    if len(metadata) != A.shape[0]:
        errors.append("metadata.csv no coincide con filas de A.")
    if len(functional_matrix) != A.shape[1]:
        errors.append("functional_matrix.csv no coincide con columnas de A.")
    if len(item_profiles) != A.shape[1]:
        errors.append("item_profiles.csv no coincide con columnas de A.")
    if not np.all(A >= 0):
        errors.append("A contiene valores negativos.")
    if not np.allclose(A.sum(axis=1), 1.0, atol=atol):
        errors.append("No todas las filas de A suman 1.")
    if set(np.unique(labels).tolist()) != {0, 1}:
        errors.append("labels debe contener exactamente clases {0,1}.")

    n_healthy = int((labels == 0).sum())
    n_crc = int((labels == 1).sum())
    if n_healthy != n_crc:
        errors.append(f"Dataset no balanceado: healthy={n_healthy}, CRC={n_crc}.")

    report = {
        "ok": len(errors) == 0,
        "errors": errors,
        "n_samples": int(A.shape[0]),
        "n_items": int(A.shape[1]),
        "healthy": n_healthy,
        "CRC": n_crc,
        "A_shape": list(A.shape),
        "labels_shape": list(labels.shape),
        "profiles_TSF_shape": list(profiles_TSF.shape),
        "row_sum_min": float(A.sum(axis=1).min()),
        "row_sum_max": float(A.sum(axis=1).max()),
        "zero_fraction": float((A == 0).mean()),
        "T_min": float(profiles_TSF[:, 0].min()),
        "T_max": float(profiles_TSF[:, 0].max()),
        "S_min": float(profiles_TSF[:, 1].min()),
        "S_max": float(profiles_TSF[:, 1].max()),
        "F_min": float(profiles_TSF[:, 2].min()),
        "F_max": float(profiles_TSF[:, 2].max()),
    }
    return report


def write_dataset(
    dataset: dict[str, object],
    out_dir: Path,
    seed: int,
    signal: float,
    concentration: float,
    dropout_rate: float,
    dataset_name: str,
) -> None:
    """Escribe CSV en csv/, NPY en npy/ y manifest al directorio de salida."""
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_dir = out_dir / "csv"
    npy_dir = out_dir / "npy"
    csv_dir.mkdir(parents=True, exist_ok=True)
    npy_dir.mkdir(parents=True, exist_ok=True)

    samples: pd.DataFrame = dataset["samples"]  # type: ignore[assignment]
    matrix_A: pd.DataFrame = dataset["matrix_A"]  # type: ignore[assignment]
    metadata: pd.DataFrame = dataset["metadata"]  # type: ignore[assignment]
    functional_matrix: pd.DataFrame = dataset["functional_matrix"]  # type: ignore[assignment]
    item_profiles: pd.DataFrame = dataset["item_profiles"]  # type: ignore[assignment]
    item_mapping: pd.DataFrame = dataset["item_mapping"]  # type: ignore[assignment]
    A: np.ndarray = dataset["A"]  # type: ignore[assignment]
    labels: np.ndarray = dataset["labels"]  # type: ignore[assignment]
    profiles_TSF: np.ndarray = dataset["profiles_TSF"]  # type: ignore[assignment]

    samples.to_csv(csv_dir / "samples.csv", index=False)
    matrix_A.to_csv(csv_dir / "matrix_A.csv", index=False, float_format="%.9g")
    metadata.to_csv(csv_dir / "metadata.csv", index=False)
    functional_matrix.to_csv(csv_dir / "functional_matrix.csv", index=False)
    item_profiles.to_csv(csv_dir / "item_profiles.csv", index=False, float_format="%.9g")
    item_mapping.to_csv(csv_dir / "item_mapping.csv", index=False, float_format="%.9g")

    np.save(npy_dir / "matrix_A.npy", A.astype(np.float32))
    np.save(npy_dir / "labels.npy", labels.astype(np.int32))
    np.save(npy_dir / "profiles_TSF.npy", profiles_TSF.astype(np.float32))

    validation = validate_dataset(dataset)

    manifest = {
        "dataset_name": dataset_name,
        "dataset_type": "synthetic-compatible",
        "purpose": "debugging, development, I/O tests, preliminary HPC benchmarks",
        "seed": int(seed),
        "signal": float(signal),
        "concentration": float(concentration),
        "dropout_rate": float(dropout_rate),
        "n_samples": validation["n_samples"],
        "n_items": validation["n_items"],
        "healthy": validation["healthy"],
        "CRC": validation["CRC"],
        "outputs": [
            "csv/samples.csv",
            "csv/matrix_A.csv",
            "csv/metadata.csv",
            "csv/functional_matrix.csv",
            "csv/item_profiles.csv",
            "csv/item_mapping.csv",
            "npy/matrix_A.npy",
            "npy/labels.npy",
            "npy/profiles_TSF.npy",
        ],
        "validation": validation,
        "notes": [
            "Dataset sintético compatible con el esquema del proyecto.",
            "No corresponde a mediciones clínicas reales.",
            "Para presentación final se recomienda reconstruir un dataset real desde curatedMetagenomicData.",
        ],
    }

    with (out_dir / "dataset_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    if not validation["ok"]:
        raise RuntimeError("Dataset inválido:\n" + "\n".join(validation["errors"]))


def parse_args() -> argparse.Namespace:
    """Argumentos de línea de comandos."""
    ap = argparse.ArgumentParser(
        description="Genera dataset metagenómico sintético-compatible para scoring HPC."
    )
    ap.add_argument("--out-dir", type=Path, default=Path("data"), help="Directorio de salida.")
    ap.add_argument("--dataset-name", type=str, default="synthetic_CRC100_balanced")
    ap.add_argument("--n-samples", type=int, default=100, help="Número total de muestras, debe ser par.")
    ap.add_argument("--n-items", type=int, default=500, help="Número de items/taxones.")
    ap.add_argument("--seed", type=int, default=42, help="Semilla RNG reproducible.")
    ap.add_argument("--signal", type=float, default=2.2, help="Fuerza de señal healthy vs CRC.")
    ap.add_argument(
        "--concentration",
        type=float,
        default=240.0,
        help="Concentración Dirichlet. Menor = más ruido muestral.",
    )
    ap.add_argument(
        "--dropout-rate",
        type=float,
        default=0.30,
        help="Tasa base de dropout/ceros para taxones raros.",
    )
    return ap.parse_args()


def main() -> None:
    """Entry point."""
    args = parse_args()

    dataset = build_dataset(
        n_samples=args.n_samples,
        n_items=args.n_items,
        seed=args.seed,
        signal=args.signal,
        concentration=args.concentration,
        dropout_rate=args.dropout_rate,
    )

    write_dataset(
        dataset=dataset,
        out_dir=args.out_dir,
        seed=args.seed,
        signal=args.signal,
        concentration=args.concentration,
        dropout_rate=args.dropout_rate,
        dataset_name=args.dataset_name,
    )

    validation = validate_dataset(dataset)

    print("[OK] Dataset sintético-compatible generado")
    print(f"     out_dir:        {args.out_dir}")
    print(f"     dataset_name:   {args.dataset_name}")
    print(f"     seed:           {args.seed}")
    print(f"     A shape:        {tuple(validation['A_shape'])}")
    print(f"     labels shape:   {tuple(validation['labels_shape'])}")
    print(f"     profiles shape: {tuple(validation['profiles_TSF_shape'])}")
    print(f"     healthy / CRC:  {validation['healthy']} / {validation['CRC']}")
    print(f"     zero_fraction:  {validation['zero_fraction']:.4f}")
    print(f"     row_sum range:  {validation['row_sum_min']:.6f} .. {validation['row_sum_max']:.6f}")


if __name__ == "__main__":
    main()
