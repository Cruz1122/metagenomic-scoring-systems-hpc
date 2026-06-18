#!/usr/bin/env python3
"""
generate_dataset.py — synthetic-compatible metagenomic scoring dataset for HPC benchmarking.

Default dataset:
  EVAL: 2000 samples = 1000 healthy + 1000 CRC
  REF:  1000 samples = 500 healthy + 500 CRC
  Items: 10000
  Seed: 42

Contract:
  P = profiles_TSF @ W
  Score = matrix_A @ P
  AUC = auc(labels, Score)

Why REF/EVAL:
  T and S are estimated from REF.
  matrix_A.npy and labels.npy are exported from EVAL.
  This prevents computing profiles from the same cohort used for benchmark AUC.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_N_EVAL = 2000
DEFAULT_N_REF = 1000
DEFAULT_N_ITEMS = 10000
DEFAULT_SEED = 42

DEFAULT_SIGNAL = 0.35
DEFAULT_T_STRENGTH = 0.80
DEFAULT_METADATA_STRENGTH = 0.07
DEFAULT_FUNCTIONAL_OVERLAP = 0.22
DEFAULT_METADATA_FRACTION = 0.16
DEFAULT_ZERO_INFLATION = 0.12
DEFAULT_NOISE_SIGMA = 0.50

CRC_ENRICHED_FRACTION = 0.030
HEALTHY_ENRICHED_FRACTION = 0.030

STUDY_NAME = "synthetic_CRC_study_scaled"

FUNCTIONAL_MARKER_COLS = [
    "resistance_marker",
    "virulence_marker",
    "inflammation_marker",
    "metabolic_marker",
    "beneficial_marker",
]


def require_even_positive(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} debe ser positivo. Recibido: {value}")
    if value % 2 != 0:
        raise ValueError(f"{name} debe ser par para balancear clases. Recibido: {value}")


def zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    sd = float(np.std(x))
    if sd <= 1e-12:
        return np.zeros_like(x, dtype=np.float64)
    return (x - float(np.mean(x))) / sd


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def normalize_rows(A: np.ndarray) -> np.ndarray:
    A = np.asarray(A, dtype=np.float64)
    A[~np.isfinite(A)] = 0.0
    A[A < 0.0] = 0.0

    row_sums = A.sum(axis=1)
    bad = row_sums <= 0.0
    if np.any(bad):
        # Fallback defensivo: filas completamente vacías reciben abundancia uniforme.
        A[bad, :] = 1.0
        row_sums = A.sum(axis=1)

    A = A / row_sums[:, None]
    return A.astype(np.float32)


def auc_rank(y_true: np.ndarray, scores: np.ndarray) -> float:
    """ROC AUC por rangos con promedio de empates; sin dependencia sklearn."""
    y = np.asarray(y_true, dtype=np.int32)
    s = np.asarray(scores, dtype=np.float64)

    pos = y == 1
    n_pos = int(pos.sum())
    n_neg = int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(s, kind="mergesort")
    sorted_s = s[order]

    ranks = np.empty(len(s), dtype=np.float64)
    i = 0
    while i < len(s):
        j = i + 1
        while j < len(s) and sorted_s[j] == sorted_s[i]:
            j += 1
        avg_rank = 0.5 * (i + 1 + j)
        ranks[order[i:j]] = avg_rank
        i = j

    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def generate_labels(n: int, cohort: str, rng: np.random.Generator) -> pd.DataFrame:
    require_even_positive(f"n_{cohort}", n)

    n_h = n // 2
    rows = []
    for i in range(n_h):
        rows.append(
            {
                "sample_id": f"{cohort}_CTRL_{i + 1:05d}",
                "label": 0,
                "group": "healthy",
            }
        )
    for i in range(n_h):
        rows.append(
            {
                "sample_id": f"{cohort}_CRC_{i + 1:05d}",
                "label": 1,
                "group": "CRC",
            }
        )

    df = pd.DataFrame(rows)
    perm = rng.permutation(len(df))
    return df.iloc[perm].reset_index(drop=True)


def generate_metadata(samples: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """
    Metadata no separa directamente CRC vs healthy.

    age y bmi se generan con la misma distribución marginal para ambas clases.
    La variable disease solo se conserva como etiqueta descriptiva.
    """
    n = len(samples)

    age = rng.normal(loc=58.0, scale=8.0, size=n)
    age = np.clip(np.rint(age), 30, 85).astype(np.int32)

    bmi = rng.normal(loc=25.2, scale=2.8, size=n)
    bmi = np.clip(bmi, 18.0, 36.0).round(1)

    sex = rng.choice(["male", "female"], size=n, p=[0.50, 0.50])
    country = rng.choice(
        ["France", "Germany", "Spain", "Italy", "Denmark", "Netherlands"],
        size=n,
        p=[0.18, 0.18, 0.16, 0.16, 0.16, 0.16],
    )

    meta = pd.DataFrame(
        {
            "sample_id": samples["sample_id"].to_numpy(),
            "age": age,
            "sex": sex,
            "bmi": bmi,
            "country": country,
            "study_name": STUDY_NAME,
            "disease": samples["group"].to_numpy(),
        }
    )
    return meta


def metadata_risk_from_metadata(meta: pd.DataFrame) -> np.ndarray:
    age_z = zscore(meta["age"].to_numpy(dtype=np.float64))
    bmi_z = zscore(meta["bmi"].to_numpy(dtype=np.float64))
    risk = 0.65 * age_z + 0.35 * bmi_z
    return zscore(risk)


def generate_item_groups(n_items: int, rng: np.random.Generator) -> np.ndarray:
    n_crc = max(1, int(round(CRC_ENRICHED_FRACTION * n_items)))
    n_healthy = max(1, int(round(HEALTHY_ENRICHED_FRACTION * n_items)))

    if n_crc + n_healthy >= n_items:
        raise ValueError("Las fracciones de items diferenciales exceden n_items.")

    groups = np.array(["neutral"] * n_items, dtype=object)
    groups[:n_crc] = "CRC_enriched"
    groups[n_crc : n_crc + n_healthy] = "healthy_enriched"
    rng.shuffle(groups)
    return groups


def generate_taxon_names(item_groups: np.ndarray, rng: np.random.Generator) -> list[str]:
    crc_genera = [
        "Fusobacterium",
        "Parvimonas",
        "Peptostreptococcus",
        "Porphyromonas",
        "Gemella",
        "Solobacterium",
        "Campylobacter",
        "Morganella",
    ]
    healthy_genera = [
        "Faecalibacterium",
        "Roseburia",
        "Eubacterium",
        "Anaerostipes",
        "Agathobacter",
        "Coprococcus",
        "Bifidobacterium",
        "Akkermansia",
    ]
    neutral_genera = [
        "Bacteroides",
        "Prevotella",
        "Alistipes",
        "Ruminococcus",
        "Dorea",
        "Collinsella",
        "Blautia",
        "Clostridium",
        "Oscillibacter",
        "Subdoligranulum",
    ]

    names = []
    counters: dict[str, int] = {}
    for g in item_groups:
        if g == "CRC_enriched":
            genus = str(rng.choice(crc_genera))
        elif g == "healthy_enriched":
            genus = str(rng.choice(healthy_genera))
        else:
            genus = str(rng.choice(neutral_genera))

        counters[genus] = counters.get(genus, 0) + 1
        names.append(f"{genus} synthetic_species_{counters[genus]:03d}")

    return names


def build_metadata_effects(
    item_groups: np.ndarray,
    rng: np.random.Generator,
    metadata_fraction: float,
) -> np.ndarray:
    """
    Selecciona items sensibles a metadata.

    El efecto no se calcula con labels, pero se fuerza un solapamiento moderado
    con los grupos diferenciales para que S tenga señal útil sin convertirse en
    fuga directa de etiqueta.
    """
    n_items = len(item_groups)
    n_active = max(1, int(round(metadata_fraction * n_items)))

    effect = np.zeros(n_items, dtype=np.float64)
    crc_idx = np.where(item_groups == "CRC_enriched")[0]
    healthy_idx = np.where(item_groups == "healthy_enriched")[0]
    neutral_idx = np.where(item_groups == "neutral")[0]

    # 40% de activos diferenciales, 60% neutrales.
    n_diff_active = min(len(crc_idx) + len(healthy_idx), int(round(0.40 * n_active)))
    n_neutral_active = max(0, n_active - n_diff_active)

    diff_pool = np.concatenate([crc_idx, healthy_idx])
    if len(diff_pool) > 0 and n_diff_active > 0:
        diff_active = rng.choice(diff_pool, size=n_diff_active, replace=False)
        for j in diff_active:
            if item_groups[j] == "CRC_enriched":
                effect[j] = rng.uniform(0.55, 1.00)
            elif item_groups[j] == "healthy_enriched":
                effect[j] = -rng.uniform(0.55, 1.00)

    if len(neutral_idx) > 0 and n_neutral_active > 0:
        neutral_active = rng.choice(neutral_idx, size=min(n_neutral_active, len(neutral_idx)), replace=False)
        effect[neutral_active] = rng.choice([-1.0, 1.0], size=len(neutral_active)) * rng.uniform(
            0.30, 0.75, size=len(neutral_active)
        )

    return effect


def generate_abundance(
    labels: np.ndarray,
    item_groups: np.ndarray,
    metadata_risk: np.ndarray,
    metadata_effect: np.ndarray,
    rng: np.random.Generator,
    signal: float,
    metadata_strength: float,
    zero_inflation: float,
    noise_sigma: float,
) -> np.ndarray:
    n_samples = len(labels)
    n_items = len(item_groups)

    # Base composicional: gamma normalizada crea abundancias relativas heterogéneas.
    base = rng.gamma(shape=0.45, scale=1.0, size=n_items)
    base = base / base.sum()

    raw = np.empty((n_samples, n_items), dtype=np.float64)

    class_effect = np.ones(n_items, dtype=np.float64)
    crc_mask = item_groups == "CRC_enriched"
    healthy_mask = item_groups == "healthy_enriched"

    # Señal moderada; con 2000 muestras, señales muy altas vuelven el AUC trivial.
    up = math.exp(0.24 * signal)
    down = math.exp(-0.14 * signal)

    for i in range(n_samples):
        factors = class_effect.copy()

        if labels[i] == 1:
            factors[crc_mask] *= up
            factors[healthy_mask] *= down
        else:
            factors[crc_mask] *= down
            factors[healthy_mask] *= up

        if metadata_strength > 0:
            factors *= np.exp(metadata_strength * metadata_risk[i] * metadata_effect)

        noise = rng.lognormal(mean=0.0, sigma=noise_sigma, size=n_items)
        row = base * factors * noise

        if zero_inflation > 0:
            prevalence = sigmoid(-1.15 + 6.0 * np.sqrt(base / base.max()))
            keep_prob = np.clip((1.0 - zero_inflation) * prevalence + 0.10, 0.05, 1.0)
            keep = rng.random(n_items) < keep_prob
            if keep.sum() < max(10, int(0.03 * n_items)):
                keep[rng.choice(n_items, size=max(10, int(0.03 * n_items)), replace=False)] = True
            row *= keep

        raw[i, :] = row

    return normalize_rows(raw)


def compute_T(A_ref: np.ndarray, y_ref: np.ndarray, t_strength: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    eps = 1e-9
    mean_crc = A_ref[y_ref == 1].mean(axis=0)
    mean_h = A_ref[y_ref == 0].mean(axis=0)

    log2fc = np.log2((mean_crc + eps) / (mean_h + eps))
    T0 = 0.5 + 0.5 * np.tanh(log2fc / 2.0)
    T = 0.5 + t_strength * (T0 - 0.5)
    T = np.clip(T, 0.0, 1.0).astype(np.float32)

    delta = mean_crc - mean_h
    threshold = float(np.quantile(np.abs(delta), 0.70))
    direction = np.where(
        delta > threshold,
        "CRC_enriched",
        np.where(delta < -threshold, "healthy_enriched", "neutral"),
    )
    return T, direction, log2fc.astype(np.float32)


def compute_S(A_ref: np.ndarray, metadata_risk_ref: np.ndarray) -> np.ndarray:
    S = np.empty(A_ref.shape[1], dtype=np.float32)
    r = np.asarray(metadata_risk_ref, dtype=np.float64)

    for j in range(A_ref.shape[1]):
        x = A_ref[:, j].astype(np.float64)
        if np.std(x) <= 1e-12 or np.std(r) <= 1e-12:
            S[j] = 0.5
        else:
            c = np.corrcoef(x, r)[0, 1]
            if not np.isfinite(c):
                c = 0.0
            S[j] = 0.5 + 0.5 * c

    return np.clip(S, 0.0, 1.0).astype(np.float32)


def compute_F(
    item_groups: np.ndarray,
    taxon_names: list[str],
    rng: np.random.Generator,
    functional_overlap: float,
) -> tuple[np.ndarray, pd.DataFrame]:
    n = len(item_groups)
    raw = np.full(n, 0.5, dtype=np.float64)

    crc_idx = np.where(item_groups == "CRC_enriched")[0]
    healthy_idx = np.where(item_groups == "healthy_enriched")[0]

    n_crc_active = int(round(functional_overlap * len(crc_idx)))
    n_healthy_active = int(round(functional_overlap * len(healthy_idx)))

    crc_active = set(rng.choice(crc_idx, size=n_crc_active, replace=False).tolist()) if n_crc_active > 0 else set()
    healthy_active = set(rng.choice(healthy_idx, size=n_healthy_active, replace=False).tolist()) if n_healthy_active > 0 else set()

    resistance = np.zeros(n, dtype=np.int32)
    virulence = np.zeros(n, dtype=np.int32)
    inflammation = np.zeros(n, dtype=np.int32)
    metabolic = np.zeros(n, dtype=np.int32)
    beneficial = np.zeros(n, dtype=np.int32)

    for i, group in enumerate(item_groups):
        if group == "CRC_enriched":
            if i in crc_active:
                raw[i] = rng.normal(0.76, 0.06)
                virulence[i] = 1
                inflammation[i] = 1
                resistance[i] = int(rng.random() < 0.35)
            else:
                raw[i] = rng.normal(0.58, 0.06)
                virulence[i] = int(rng.random() < 0.25)
        elif group == "healthy_enriched":
            if i in healthy_active:
                raw[i] = rng.normal(0.25, 0.06)
                beneficial[i] = 1
                metabolic[i] = 1
            else:
                raw[i] = rng.normal(0.42, 0.06)
                beneficial[i] = int(rng.random() < 0.30)
        else:
            raw[i] = rng.normal(0.50, 0.08)
            metabolic[i] = int(rng.random() < 0.25)
            beneficial[i] = int(rng.random() < 0.15)
            virulence[i] = int(rng.random() < 0.08)
            inflammation[i] = int(rng.random() < 0.10)
            resistance[i] = int(rng.random() < 0.05)

    F = np.clip(raw, 0.0, 1.0).astype(np.float32)

    markers = pd.DataFrame(
        {
            "resistance_marker": resistance,
            "virulence_marker": virulence,
            "inflammation_marker": inflammation,
            "metabolic_marker": metabolic,
            "beneficial_marker": beneficial,
        }
    )
    return F, markers


def write_dataset(
    output_dir: Path,
    samples_eval: pd.DataFrame,
    metadata_eval: pd.DataFrame,
    A_eval: np.ndarray,
    profiles_TSF: np.ndarray,
    item_profiles: pd.DataFrame,
    item_mapping: pd.DataFrame,
    functional_matrix: pd.DataFrame,
    manifest: dict[str, Any],
    write_matrix_csv: bool = True,
) -> None:
    csv_dir = output_dir / "csv"
    npy_dir = output_dir / "npy"
    csv_dir.mkdir(parents=True, exist_ok=True)
    npy_dir.mkdir(parents=True, exist_ok=True)

    item_ids = item_mapping["item_id"].to_list()

    samples_eval.to_csv(csv_dir / "samples.csv", index=False)

    if write_matrix_csv:
        matrix_df = pd.DataFrame(A_eval, columns=item_ids)
        matrix_df.insert(0, "sample_id", samples_eval["sample_id"].to_numpy())
        matrix_df.to_csv(csv_dir / "matrix_A.csv", index=False, float_format="%.9g")
    else:
        # Para datasets muy grandes, matrix_A.npy es la fuente eficiente.
        # Se deja un README explícito en lugar de crear un CSV de cientos de MB.
        (csv_dir / "matrix_A.README.txt").write_text(
            "matrix_A.csv omitido por --no-matrix-csv. Usa npy/matrix_A.npy como matriz principal.\n",
            encoding="utf-8",
        )

    metadata_eval.to_csv(csv_dir / "metadata.csv", index=False)
    functional_matrix.to_csv(csv_dir / "functional_matrix.csv", index=False)
    item_profiles.to_csv(csv_dir / "item_profiles.csv", index=False, float_format="%.9g")
    item_mapping.to_csv(csv_dir / "item_mapping.csv", index=False)

    np.save(npy_dir / "matrix_A.npy", A_eval.astype(np.float32))
    np.save(npy_dir / "labels.npy", samples_eval["label"].to_numpy(np.int32))
    np.save(npy_dir / "profiles_TSF.npy", profiles_TSF.astype(np.float32))

    with open(output_dir / "dataset_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


def validate_dataset(A: np.ndarray, y: np.ndarray, profiles: np.ndarray, min_eval: int, n_items: int) -> dict[str, Any]:
    row_sums = A.sum(axis=1)
    validation = {
        "A_shape": list(A.shape),
        "labels_shape": list(y.shape),
        "profiles_TSF_shape": list(profiles.shape),
        "n_healthy": int((y == 0).sum()),
        "n_crc": int((y == 1).sum()),
        "row_sum_min": float(row_sums.min()),
        "row_sum_max": float(row_sums.max()),
        "profiles_min": float(profiles.min()),
        "profiles_max": float(profiles.max()),
        "A_nonnegative": bool(np.all(A >= 0)),
        "profiles_in_unit_interval": bool(np.all((profiles >= 0) & (profiles <= 1))),
        "rows_sum_to_one": bool(np.allclose(row_sums, 1.0, atol=1e-5)),
        "balanced_labels": bool((y == 0).sum() == (y == 1).sum()),
        "min_eval_ok": bool(A.shape[0] >= min_eval),
        "n_items_ok": bool(A.shape[1] == n_items),
    }

    failed = [k for k, v in validation.items() if isinstance(v, bool) and not v]
    if failed:
        raise RuntimeError(f"Validación falló: {failed}. Detalle: {validation}")

    return validation


def quick_eval(A: np.ndarray, y: np.ndarray, profiles: np.ndarray, rng: np.random.Generator, k: int) -> dict[str, Any]:
    """
    Sanity check rápido para datasets grandes.

    Evita recalcular A @ (profiles @ W) para cada W. Por asociatividad:
        A @ (profiles @ W) == (A @ profiles) @ W
    Primero calcula Z = A @ profiles, de tamaño n_samples x 3. Luego cada
    evaluación de W cuesta O(n_samples * 3), no O(n_samples * n_items).
    Esto es clave para n_items=10000.
    """
    Z = (A.astype(np.float32, copy=False) @ profiles.astype(np.float32, copy=False)).astype(np.float32)

    def eval_w(w: np.ndarray) -> float:
        scores = Z @ w.astype(np.float32, copy=False)
        return auc_rank(y, scores)

    result: dict[str, Any] = {
        "T_only_auc": eval_w(np.array([1.0, 0.0, 0.0], dtype=np.float32)),
        "S_only_auc": eval_w(np.array([0.0, 1.0, 0.0], dtype=np.float32)),
        "F_only_auc": eval_w(np.array([0.0, 0.0, 1.0], dtype=np.float32)),
        "equal_auc": eval_w(np.array([1 / 3, 1 / 3, 1 / 3], dtype=np.float32)),
    }

    if k <= 0:
        result["best_auc"] = None
        result["best_w"] = None
        return result

    best_auc = -1.0
    best_w = None

    # Dirichlet sobre el simplex: W_i > 0 y sum(W)=1.
    weights = rng.dirichlet(np.ones(3, dtype=np.float64), size=k).astype(np.float32)
    for w in weights:
        auc = eval_w(w)
        if auc > best_auc:
            best_auc = auc
            best_w = w.copy()

    result["best_auc"] = float(best_auc)
    result["best_w"] = [float(x) for x in best_w]
    return result


def build_dataset(args: argparse.Namespace) -> Path:
    require_even_positive("n_eval", args.n_eval)
    require_even_positive("n_ref", args.n_ref)

    if args.n_eval < 2000 and not args.allow_small:
        raise ValueError(
            f"Este generador está configurado para mínimo 2000 muestras EVAL. "
            f"Recibido --n-eval {args.n_eval}. Usa --allow-small solo para pruebas."
        )

    rng = np.random.default_rng(args.seed)

    samples_ref = generate_labels(args.n_ref, "REF", rng)
    samples_eval = generate_labels(args.n_eval, "EVAL", rng)

    y_ref = samples_ref["label"].to_numpy(np.int32)
    y_eval = samples_eval["label"].to_numpy(np.int32)

    metadata_ref = generate_metadata(samples_ref, rng)
    metadata_eval = generate_metadata(samples_eval, rng)

    metadata_risk_ref = metadata_risk_from_metadata(metadata_ref)
    metadata_risk_eval = metadata_risk_from_metadata(metadata_eval)

    item_groups = generate_item_groups(args.n_items, rng)
    taxon_names = generate_taxon_names(item_groups, rng)
    metadata_effect = build_metadata_effects(item_groups, rng, args.metadata_fraction)

    A_ref = generate_abundance(
        labels=y_ref,
        item_groups=item_groups,
        metadata_risk=metadata_risk_ref,
        metadata_effect=metadata_effect,
        rng=rng,
        signal=args.signal,
        metadata_strength=args.metadata_strength,
        zero_inflation=args.zero_inflation,
        noise_sigma=args.noise_sigma,
    )

    A_eval = generate_abundance(
        labels=y_eval,
        item_groups=item_groups,
        metadata_risk=metadata_risk_eval,
        metadata_effect=metadata_effect,
        rng=rng,
        signal=args.signal,
        metadata_strength=args.metadata_strength,
        zero_inflation=args.zero_inflation,
        noise_sigma=args.noise_sigma,
    )

    T, taxon_direction_estimated, log2fc_ref = compute_T(A_ref, y_ref, args.t_strength)
    S = compute_S(A_ref, metadata_risk_ref)
    F, marker_values = compute_F(item_groups, taxon_names, rng, args.functional_overlap)

    profiles_TSF = np.vstack([T, S, F]).T.astype(np.float32)

    item_ids = [f"item_{i:03d}" for i in range(args.n_items)]

    samples_eval_out = samples_eval.copy()
    metadata_eval_out = metadata_eval.copy()

    item_mapping = pd.DataFrame(
        {
            "item_id": item_ids,
            "taxon_name": taxon_names,
            "original_feature_id": [f"synthetic_feature_{i:03d}" for i in range(args.n_items)],
            "rank": "species",
            "true_group": item_groups,
            "metadata_effect": metadata_effect.astype(np.float32),
            "log2fc_ref": log2fc_ref,
        }
    )

    functional_matrix = pd.concat(
        [pd.DataFrame({"item_id": item_ids}), marker_values],
        axis=1,
    )

    item_profiles = pd.DataFrame(
        {
            "item_id": item_ids,
            "taxon_name": taxon_names,
            "T": profiles_TSF[:, 0],
            "taxon_direction": taxon_direction_estimated,
            "true_group": item_groups,
            "S": profiles_TSF[:, 1],
            "F": profiles_TSF[:, 2],
        }
    )

    validation = validate_dataset(A_eval, y_eval, profiles_TSF, min_eval=2000, n_items=args.n_items)
    qe = quick_eval(A_eval, y_eval, profiles_TSF, rng, args.quick_k)

    dataset_name = args.name
    if dataset_name == "auto":
        dataset_name = f"synthetic_CRC{args.n_eval}x{args.n_items}_balanced"

    output_dir = args.out_dir / dataset_name

    manifest: dict[str, Any] = {
        "dataset_name": dataset_name,
        "dataset_type": "synthetic-compatible-scaled",
        "disease_task": "colorectal_cancer_vs_healthy",
        "positive_class": "CRC",
        "negative_class": "healthy/control",
        "seed": args.seed,
        "n_eval_samples": int(args.n_eval),
        "n_eval_healthy": int((y_eval == 0).sum()),
        "n_eval_crc": int((y_eval == 1).sum()),
        "n_ref_samples": int(args.n_ref),
        "n_ref_healthy": int((y_ref == 0).sum()),
        "n_ref_crc": int((y_ref == 1).sum()),
        "n_items": int(args.n_items),
        "n_crc_enriched_true": int((item_groups == "CRC_enriched").sum()),
        "n_healthy_enriched_true": int((item_groups == "healthy_enriched").sum()),
        "n_neutral_true": int((item_groups == "neutral").sum()),
        "signal": float(args.signal),
        "t_strength": float(args.t_strength),
        "metadata_strength": float(args.metadata_strength),
        "metadata_fraction": float(args.metadata_fraction),
        "functional_overlap": float(args.functional_overlap),
        "zero_inflation": float(args.zero_inflation),
        "noise_sigma": float(args.noise_sigma),
        "abundance_type": "relative_abundance_synthetic",
        "abundance_model": "gamma base abundance + lognormal sample noise + class factor + metadata factor + row normalization",
        "T_formula": "T = clip(0.5 + t_strength * ((0.5 + 0.5*tanh(log2fc_ref/2)) - 0.5), 0, 1)",
        "T_source": "estimated from independent REF cohort only",
        "S_formula": "S = 0.5 + 0.5*corr(A_ref[:, i], metadata_risk_ref)",
        "S_source": "estimated from REF metadata only; disease label not used",
        "F_formula": "controlled functional proxy from item group with functional_overlap",
        "F_source": "synthetic functional markers with controlled overlap",
        "reference_split_note": "T and S are computed from REF. EVAL is exported for benchmark to avoid label leakage.",
        "matrix_A_csv_written": bool(not args.no_matrix_csv),
        "validation": validation,
        "quick_eval": qe,
    }

    write_dataset(
        output_dir=output_dir,
        samples_eval=samples_eval_out,
        metadata_eval=metadata_eval_out,
        A_eval=A_eval,
        profiles_TSF=profiles_TSF,
        item_profiles=item_profiles,
        item_mapping=item_mapping,
        functional_matrix=functional_matrix,
        manifest=manifest,
        write_matrix_csv=not args.no_matrix_csv,
    )

    print(f"Dataset generado: {output_dir}")
    print(f"A_eval: {A_eval.shape} {A_eval.dtype}")
    print(
        f"labels: {y_eval.shape} healthy={int((y_eval == 0).sum())} "
        f"CRC={int((y_eval == 1).sum())}"
    )
    print(f"profiles_TSF: {profiles_TSF.shape} {profiles_TSF.dtype}")
    print(f"row_sum_min={validation['row_sum_min']:.8f} row_sum_max={validation['row_sum_max']:.8f}")
    print(
        "AUC sanity: "
        f"T={qe['T_only_auc']:.4f} "
        f"S={qe['S_only_auc']:.4f} "
        f"F={qe['F_only_auc']:.4f} "
        f"equal={qe['equal_auc']:.4f} "
        f"best={qe['best_auc'] if qe['best_auc'] is not None else 'NA'}"
    )
    if qe["best_w"] is not None:
        print("best_w:", " ".join(f"{x:.4f}" for x in qe["best_w"]))

    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Genera dataset sintético-compatible escalado para scoring metagenómico HPC."
    )
    parser.add_argument("--name", type=str, default="auto")
    parser.add_argument("--n-eval", type=int, default=DEFAULT_N_EVAL)
    parser.add_argument("--n-ref", type=int, default=DEFAULT_N_REF)
    parser.add_argument("--n-items", type=int, default=DEFAULT_N_ITEMS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--signal", type=float, default=DEFAULT_SIGNAL)
    parser.add_argument("--t-strength", type=float, default=DEFAULT_T_STRENGTH)
    parser.add_argument("--metadata-strength", type=float, default=DEFAULT_METADATA_STRENGTH)
    parser.add_argument("--metadata-fraction", type=float, default=DEFAULT_METADATA_FRACTION)
    parser.add_argument("--functional-overlap", type=float, default=DEFAULT_FUNCTIONAL_OVERLAP)
    parser.add_argument("--zero-inflation", type=float, default=DEFAULT_ZERO_INFLATION)
    parser.add_argument("--noise-sigma", type=float, default=DEFAULT_NOISE_SIGMA)
    parser.add_argument("--out-dir", type=Path, default=Path("data") / "processed")
    parser.add_argument("--quick-k", type=int, default=500, help="K de random search para sanity check; 0 lo desactiva.")
    parser.add_argument("--allow-small", action="store_true", help="Permite n-eval < 2000 solo para pruebas.")
    parser.add_argument("--no-matrix-csv", action="store_true", help="Omite csv/matrix_A.csv; conserva npy/matrix_A.npy. Recomendado para 2000x10000 si el repo no debe inflarse.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_dataset(args)


if __name__ == "__main__":
    main()
