#!/usr/bin/env python3
"""
generate_dataset.py — Generador con perfiles T, S, F calibrados.

Parámetros de calibración:
  --t-strength        Compresión de T hacia 0.5 (0.70-1.00)
  --metadata-strength Efecto de metadata_risk sobre abundancia (0.00-0.30)
  --functional-overlap % de items enriquecidos con perfil funcional alineado (0.30-0.50)

Arquitectura:
  REF (200) → T, S
  EVAL (100) → benchmark
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
DEFAULT_N_EVAL = 100
DEFAULT_N_REF = 200
DEFAULT_N_ITEMS = 500
DEFAULT_SEED = 42
DEFAULT_SIGNAL = 1.0
DEFAULT_T_STRENGTH = 0.85
DEFAULT_METADATA_STRENGTH = 0.20
DEFAULT_FUNCTIONAL_OVERLAP = 0.40

CRC_ENRICHED_FRACTION = 0.05
HEALTHY_ENRICHED_FRACTION = 0.10

STUDY_NAME = "synthetic_CRC_study"

FUNCTIONAL_MARKER_COLS = [
    "resistance_marker", "virulence_marker", "inflammation_marker",
    "metabolic_marker", "beneficial_marker",
]

# Probabilidades funcionales (se mantienen fijas, el overlap controla cuántos
# items las reciben)
PROBS_CRC = (0.10, 0.25, 0.30, 0.45, 0.10)
PROBS_HEALTHY = (0.04, 0.06, 0.08, 0.55, 0.35)
PROBS_NEUTRAL = (0.05, 0.08, 0.10, 0.45, 0.18)


# ---------------------------------------------------------------------------
# Auxiliares estadísticos
# ---------------------------------------------------------------------------

def _abs_pearson(x: np.ndarray, y: np.ndarray) -> float:
    mask = ~(np.isnan(x) | np.isnan(y))
    x_c, y_c = x[mask], y[mask]
    if len(x_c) < 3 or np.std(x_c) == 0 or np.std(y_c) == 0:
        return 0.0
    r = np.corrcoef(x_c, y_c)[0, 1]
    return abs(r) if not np.isnan(r) else 0.0


def _correlation_ratio(x: np.ndarray, categories: np.ndarray) -> float:
    cats = np.unique(categories)
    if len(cats) < 2:
        return 0.0
    gm = float(np.mean(x))
    ssb = sum(int((categories == c).sum()) * (float(np.mean(x[categories == c])) - gm) ** 2 for c in cats)
    sst = float(np.sum((x - gm) ** 2))
    return np.sqrt(ssb / sst) if sst > 0 else 0.0


# ---------------------------------------------------------------------------
# 1. Componentes base
# ---------------------------------------------------------------------------

def generate_labels(n: int, rng: np.random.Generator) -> pd.DataFrame:
    n_h = n // 2
    rows = [{"sample_id": f"CTRL_{i+1:03d}", "label": 0, "group": "healthy"} for i in range(n_h)]
    rows += [{"sample_id": f"CRC_{i+1:03d}", "label": 1, "group": "CRC"} for i in range(n - n_h)]
    return pd.DataFrame(rows)


def generate_metadata(labels: np.ndarray, rng: np.random.Generator) -> pd.DataFrame:
    """Metadata IDÉNTICA entre grupos (sin correlación mr-label).
    
    La señal de S viene exclusivamente del metadata_effect en abundancia,
    no de diferencias entre grupos. Esto evita que S sea un proxy de T.
    """
    n = len(labels)
    countries = ["France", "Germany", "Spain", "Italy", "Denmark"]
    age = rng.normal(loc=58.0, scale=7.0, size=n)
    age = np.clip(np.round(age), 30, 85).astype(int)
    bmi = rng.normal(loc=25.3, scale=2.5, size=n)
    bmi = np.clip(bmi, 18.0, 35.0).round(1)
    samples = pd.DataFrame({"sample_id": [f"S{i+1:03d}" for i in range(n)]})
    meta = samples.copy()
    meta["age"] = age
    meta["sex"] = rng.choice(["male", "female"], size=n)
    meta["bmi"] = bmi
    meta["country"] = rng.choice(countries, size=n)
    meta["study_name"] = STUDY_NAME
    meta["disease"] = np.where(labels == 1, "CRC", "healthy")
    return meta


def _signal_factors(signal: float):
    up = 1.0 + 0.10 * signal
    down = 1.0 - 0.05 * signal
    return up, down


def generate_item_groups(n_items: int, rng: np.random.Generator) -> np.ndarray:
    n_crc = max(1, int(round(CRC_ENRICHED_FRACTION * n_items)))
    n_healthy = max(1, int(round(HEALTHY_ENRICHED_FRACTION * n_items)))
    groups = np.array(["neutral"] * n_items, dtype=object)
    groups[:n_crc] = "CRC_enriched"
    groups[n_crc:n_crc + n_healthy] = "healthy_enriched"
    return groups


# ---------------------------------------------------------------------------
# 2. Abundancia con señal de clase + metadata
# ---------------------------------------------------------------------------

def generate_abundance(
    labels: np.ndarray,
    item_groups: np.ndarray,
    rng: np.random.Generator,
    signal: float = 1.0,
    metadata_risk: np.ndarray | None = None,
    item_metadata_effect: np.ndarray | None = None,
    metadata_strength: float = 0.0,
) -> np.ndarray:
    """Genera abundancia con efecto de clase y efecto opcional de metadata.
    
    Args:
        labels: Vector 0/1 (n,).
        item_groups: Grupo por item (n_items,).
        rng: RNG.
        signal: Nivel de señal de clase.
        metadata_risk: Score de riesgo por muestra (n,), opcional.
        item_metadata_effect: Efecto de metadata por item (n_items,), opcional.
        metadata_strength: Peso del efecto metadata.
    """
    n = len(labels)
    n_items = len(item_groups)
    is_h = (labels == 0)
    is_c = (labels == 1)

    abundances = rng.uniform(0.05, 1.0, size=(n, n_items)).astype(np.float64)
    up, down = _signal_factors(signal)

    for i in range(n_items):
        g = item_groups[i]
        if g == "CRC_enriched":
            abundances[is_h, i] *= down
            abundances[is_c, i] *= up
        elif g == "healthy_enriched":
            abundances[is_h, i] *= up
            abundances[is_c, i] *= down

    # Efecto metadata (independiente de clase)
    if metadata_risk is not None and item_metadata_effect is not None and metadata_strength > 0:
        mr = (metadata_risk - metadata_risk.mean()) / (metadata_risk.std() + 1e-12)
        for i in range(n_items):
            if item_metadata_effect[i] != 0:
                abundances[:, i] *= np.exp(metadata_strength * mr * item_metadata_effect[i])

    # Ruido
    noise = rng.lognormal(mean=0.0, sigma=0.35, size=abundances.shape)
    abundances *= noise
    abundances = np.maximum(abundances, 0.0)
    abundances /= abundances.sum(axis=1, keepdims=True)
    return abundances.astype(np.float32)


# ---------------------------------------------------------------------------
# 3. Matriz funcional (Bernoulli, decorativa) + F directo
# ---------------------------------------------------------------------------

def generate_functional_matrix(
    item_groups: np.ndarray,
    rng: np.random.Generator,
    functional_overlap: float = 0.40,
) -> pd.DataFrame:
    """Genera marcadores Bernoulli para CSV (decorativo).
    
    functional_overlap: proporción de items enriquecidos que reciben
    perfil funcional alineado. El resto recibe neutral.
    Los neutrales tienen 10% de recibir perfil alineado por azar.
    """
    n_items = len(item_groups)
    rows = []
    for i in range(n_items):
        g = item_groups[i]
        if g == "CRC_enriched":
            probs = PROBS_CRC if rng.random() < functional_overlap else PROBS_NEUTRAL
        elif g == "healthy_enriched":
            probs = PROBS_HEALTHY if rng.random() < functional_overlap else PROBS_NEUTRAL
        else:
            rv = rng.random()
            if rv < 0.10:
                probs = PROBS_CRC
            elif rv < 0.20:
                probs = PROBS_HEALTHY
            else:
                probs = PROBS_NEUTRAL
        pR, pV, pI, pM, pB = probs
        rows.append({
            "item_id": f"item_{i:03d}",
            "resistance_marker": int(rng.random() < pR),
            "virulence_marker": int(rng.random() < pV),
            "inflammation_marker": int(rng.random() < pI),
            "metabolic_marker": int(rng.random() < pM),
            "beneficial_marker": int(rng.random() < pB),
        })
    return pd.DataFrame(rows)


def compute_F(
    item_groups: np.ndarray,
    rng: np.random.Generator,
    functional_overlap: float = 0.40,
) -> np.ndarray:
    """Calcula F directo desde grupo del item con overlap controlado.
    
    En lugar de Bernoulli, asigna valores fijos + ruido.
    functional_overlap controla cuántos items enriquecidos tienen F alineado.
    """
    n_items = len(item_groups)
    F_CRC = 0.85
    F_HEALTHY = 0.15
    F_NEUTRAL = 0.50

    Fp = np.full(n_items, F_NEUTRAL, dtype=float)
    for i in range(n_items):
        g = item_groups[i]
        if g == "CRC_enriched":
            if rng.random() < functional_overlap:
                Fp[i] = F_CRC
        elif g == "healthy_enriched":
            if rng.random() < functional_overlap:
                Fp[i] = F_HEALTHY
        else:
            rv = rng.random()
            if rv < 0.10:
                Fp[i] = F_CRC
            elif rv < 0.20:
                Fp[i] = F_HEALTHY

    noise = rng.normal(0, 0.04, size=n_items)
    Fp = np.clip(Fp + noise, 0.0, 1.0)
    return Fp


# ---------------------------------------------------------------------------
# 4. Perfiles T, S, F
# ---------------------------------------------------------------------------

def compute_T(A_ref: np.ndarray, y_ref: np.ndarray,
              t_strength: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """T con compresión opcional hacia 0.5.
    
    T = 0.5 + t_strength * (T_original - 0.5)
    """
    eps = 1e-9
    mc = A_ref[y_ref == 1].mean(axis=0)
    mh = A_ref[y_ref == 0].mean(axis=0)
    delta = mc - mh
    log2fc = np.log2((mc + eps) / (mh + eps))
    T_orig = 0.5 + 0.5 * np.tanh(log2fc / 2.0)
    T = 0.5 + t_strength * (T_orig - 0.5)
    return T, delta


def compute_taxon_direction(delta: np.ndarray) -> np.ndarray:
    threshold = np.percentile(np.abs(delta), 70)
    direction = np.array(["neutral"] * len(delta), dtype=object)
    direction[delta > threshold] = "CRC_enriched"
    direction[delta < -threshold] = "healthy_enriched"
    return direction


def compute_S(A_ref: np.ndarray, metadata: pd.DataFrame,
              y_ref: np.ndarray | None = None,
              t_strength: float = 1.0) -> np.ndarray:
    """S orientado a riesgo poblacional, con residualización parcial.
    
    Elimina el 50% de la señal entre-grupos (T-proxy) de la abundancia
    antes de correlacionar con metadata_risk. Así S retiene señal
    independiente sin volverse proxy de T.
    
    metadata_risk = 0.65 * zscore(age) + 0.35 * zscore(bmi)
    """
    age = metadata["age"].to_numpy(float)
    bmi = metadata["bmi"].to_numpy(float)
    age_z = (age - age.mean()) / (age.std() + 1e-12)
    bmi_z = (bmi - bmi.mean()) / (bmi.std() + 1e-12)
    metadata_risk = 0.65 * age_z + 0.35 * bmi_z

    # Residualización parcial: elimina 50% de la señal entre-grupos
    A_partial = A_ref.copy()
    if y_ref is not None:
        for grp_val in np.unique(y_ref):
            mask = y_ref == grp_val
            if mask.sum() > 0:
                grp_mean = A_ref[mask].mean(axis=0)
                A_partial[mask] -= 0.50 * grp_mean  # solo 50%

    n_items = A_ref.shape[1]
    S_raw = np.zeros(n_items)
    for i in range(n_items):
        corr_i = np.corrcoef(A_partial[:, i], metadata_risk)[0, 1]
        S_raw[i] = 0.5 + 0.5 * (corr_i if not np.isnan(corr_i) else 0.0)

    if t_strength < 1.0:
        S_raw = 0.5 + t_strength * (S_raw - 0.5)

    return np.clip(S_raw, 0.0, 1.0)


def build_item_mapping(n_items: int) -> pd.DataFrame:
    rows = [{
        "item_id": f"item_{i:03d}",
        "taxon_name": f"Synthetic taxon {i:03d}",
        "original_feature_id": f"synthetic_species_{i:03d}",
        "rank": "species",
    } for i in range(n_items)]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 5. Construcción del dataset
# ---------------------------------------------------------------------------

def write_dataset(output_dir: Path, samples, metadata, A_eval,
                  functional_matrix, item_profiles, item_mapping,
                  profiles_TSF, manifest):
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_dir = output_dir / "csv"
    npy_dir = output_dir / "npy"
    csv_dir.mkdir(exist_ok=True)
    npy_dir.mkdir(exist_ok=True)

    samples.to_csv(csv_dir / "samples.csv", index=False)
    mA = pd.DataFrame(A_eval, columns=[f"item_{i:03d}" for i in range(A_eval.shape[1])])
    mA.insert(0, "sample_id", samples["sample_id"].to_numpy())
    mA.to_csv(csv_dir / "matrix_A.csv", index=False, float_format="%.9g")
    metadata.to_csv(csv_dir / "metadata.csv", index=False)
    functional_matrix.to_csv(csv_dir / "functional_matrix.csv", index=False)
    item_profiles.to_csv(csv_dir / "item_profiles.csv", index=False, float_format="%.9g")
    item_mapping.to_csv(csv_dir / "item_mapping.csv", index=False)

    np.save(npy_dir / "matrix_A.npy", A_eval.astype(np.float32))
    np.save(npy_dir / "labels.npy", samples["label"].to_numpy(np.int32))
    np.save(npy_dir / "profiles_TSF.npy", profiles_TSF.astype(np.float32))

    with open(output_dir / "dataset_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


def build_one_dataset(
    name: str,
    n_eval_samples: int = DEFAULT_N_EVAL,
    n_ref_samples: int = DEFAULT_N_REF,
    n_items: int = DEFAULT_N_ITEMS,
    seed: int = DEFAULT_SEED,
    signal: float = DEFAULT_SIGNAL,
    t_strength: float = DEFAULT_T_STRENGTH,
    metadata_strength: float = DEFAULT_METADATA_STRENGTH,
    functional_overlap: float = DEFAULT_FUNCTIONAL_OVERLAP,
    output_root: Path = Path("data") / "processed",
    verbose: bool = True,
) -> Path:
    if verbose:
        print(f"\nGenerando '{name}' (eval={n_eval_samples} ref={n_ref_samples} "
              f"items={n_items} signal={signal} t={t_strength} "
              f"meta={metadata_strength} overlap={functional_overlap})")

    rng = np.random.default_rng(seed)

    # Generar ambas cohortes
    samples_ref = generate_labels(n_ref_samples, rng)
    labels_ref = samples_ref["label"].to_numpy(np.int32)
    metadata_ref = generate_metadata(labels_ref, rng)

    samples_eval = generate_labels(n_eval_samples, rng)
    labels_eval = samples_eval["label"].to_numpy(np.int32)
    metadata_eval = generate_metadata(labels_eval, rng)

    item_groups = generate_item_groups(n_items, rng)

    # Metadata risk score
    age_all = np.concatenate([metadata_ref["age"].to_numpy(float),
                              metadata_eval["age"].to_numpy(float)])
    bmi_all = np.concatenate([metadata_ref["bmi"].to_numpy(float),
                              metadata_eval["bmi"].to_numpy(float)])
    age_z_all = (age_all - age_all.mean()) / (age_all.std() + 1e-12)
    bmi_z_all = (bmi_all - bmi_all.mean()) / (bmi_all.std() + 1e-12)
    metadata_risk_all = 0.65 * age_z_all + 0.35 * bmi_z_all
    metadata_risk_ref = metadata_risk_all[:n_ref_samples]
    metadata_risk_eval = metadata_risk_all[n_ref_samples:]

    # Efecto de metadata sobre TODOS los items (para que S comparta
    # items con T y no sea puramente independiente)
    item_metadata_effect = np.zeros(n_items, dtype=float)
    n_meta_items = max(1, int(0.10 * n_items))  # 10% de todos los items
    all_idx = np.arange(n_items)
    meta_idx = rng.choice(all_idx, n_meta_items, replace=False)
    half = n_meta_items // 2
    item_metadata_effect[meta_idx[:half]] = 1.0
    item_metadata_effect[meta_idx[half:2*half]] = -1.0

    # Abundancia para T (SIN metadata_effect, para que T no capture señal de S)
    A_ref_T = generate_abundance(labels_ref, item_groups, rng, signal)
    # Abundancia para S (CON metadata_effect)
    A_ref_S = generate_abundance(labels_ref, item_groups, rng, signal,
                                  metadata_risk_ref, item_metadata_effect,
                                  metadata_strength)
    # Abundancia para EVAL (CON metadata_effect, para benchmark)
    A_eval = generate_abundance(labels_eval, item_groups, rng, signal,
                                 metadata_risk_eval, item_metadata_effect,
                                 metadata_strength)

    # Perfiles
    T_profile, delta = compute_T(A_ref_T, labels_ref, t_strength)
    S_profile = compute_S(A_ref_S, metadata_ref, y_ref=labels_ref, t_strength=t_strength)
    functional_matrix = generate_functional_matrix(item_groups, rng, functional_overlap)
    F_profile = compute_F(item_groups, rng, functional_overlap)

    taxon_direction = compute_taxon_direction(delta)
    n_neutral = int((taxon_direction == "neutral").sum())

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

    assert np.all(profiles_TSF >= 0) and np.all(profiles_TSF <= 1)

    if verbose:
        print(f"    taxon_direction: CRC={(taxon_direction=='CRC_enriched').sum()}, "
              f"healthy={(taxon_direction=='healthy_enriched').sum()}, "
              f"neutral={n_neutral}")

    item_mapping = build_item_mapping(n_items)
    n_crc_enc = max(1, int(round(CRC_ENRICHED_FRACTION * n_items)))
    n_health_enc = max(1, int(round(HEALTHY_ENRICHED_FRACTION * n_items)))
    up_f, down_f = _signal_factors(signal)

    manifest = {
        "dataset_name": name,
        "dataset_type": "synthetic-compatible",
        "disease_task": "colorectal_cancer_vs_healthy",
        "positive_class": "CRC",
        "negative_class": "healthy/control",
        "seed": seed,
        "signal": signal,
        "t_strength": t_strength,
        "metadata_strength": metadata_strength,
        "functional_overlap": functional_overlap,
        "n_eval_samples": n_eval_samples,
        "n_eval_healthy": n_eval_samples // 2,
        "n_eval_crc": n_eval_samples - n_eval_samples // 2,
        "n_ref_samples": n_ref_samples,
        "n_ref_healthy": n_ref_samples // 2,
        "n_ref_crc": n_ref_samples - n_ref_samples // 2,
        "n_items": n_items,
        "n_crc_enriched": n_crc_enc,
        "n_healthy_enriched": n_health_enc,
        "abundance_type": "relative_abundance",
        "abundance_signal": f"up={up_f:.4f}, down={down_f:.4f}",
        "abundance_noise": "lognormal(mean=0.0, sigma=0.35)",
        "T_formula": f"0.5 + {t_strength} * (T_original - 0.5)",
        "T_source": "estimated from independent reference cohort (A_ref, y_ref)",
        "S_formula": "0.5 + 0.5 * corr(abundance, 0.65*age_z + 0.35*bmi_z)",
        "S_source": "estimated from reference cohort, metadata_risk-oriented (not abs)",
        "F_formula": "direct from item group + noise (not from Bernoulli markers)",
        "F_source": "computed from item group with controlled overlap",
        "taxon_direction_method": "percentile 70 of |delta|",
        "reference_split_note": "T and S from independent ref cohort. "
                                 "Prevents label leakage.",
    }

    write_dataset(output_root / name, samples_eval, metadata_eval, A_eval,
                  functional_matrix, item_profiles, item_mapping,
                  profiles_TSF, manifest)

    if verbose:
        print(f"  OK -> {output_root / name}/")
    return output_root / name


# ---------------------------------------------------------------------------
# 6. Evaluación rápida (para grid)
# ---------------------------------------------------------------------------

def quick_eval(data_dir: Path, k: int = 5000, seed: int = 42) -> dict:
    """Evalúa dataset y retorna métricas.
    
    No modifica archivos. Usa k=5000 para diagnóstico rápido.
    """
    sys.path.insert(0, str(Path("python").resolve()))
    from common import load_data, evaluate, random_search
    from scipy.stats import pearsonr

    A, y, profiles = load_data(data_dir)
    T, S, F = profiles[:, 0], profiles[:, 1], profiles[:, 2]

    res = {}
    for name, w in [("T_only", [1,0,0]), ("S_only", [0,1,0]),
                     ("F_only", [0,0,1]), ("equal", [1/3,1/3,1/3])]:
        auc_v, cons_v = evaluate(A, y, profiles, np.array(w))
        res[name] = (auc_v, cons_v)

    best_auc, best_cons, best_w = random_search(A, y, profiles, k, seed)
    res["best"] = (best_auc, best_cons)
    res["best_w"] = best_w

    res["corr_TF"] = pearsonr(T, F)[0]
    res["corr_TS"] = pearsonr(T, S)[0]
    res["corr_SF"] = pearsonr(S, F)[0]

    # signal=0 test
    rng0 = np.random.default_rng(seed)
    lab0 = generate_labels(300, rng0)["label"].to_numpy(np.int32)
    grp0 = generate_item_groups(500, rng0)
    A0 = generate_abundance(lab0, grp0, rng0, 0.0)
    idx_p = rng0.permutation(300)
    T0, _ = compute_T(A0[idx_p[:200]], lab0[idx_p[:200]], 1.0)
    F0 = compute_F(grp0, rng0, 0.40)
    meta0 = generate_metadata(lab0[idx_p[:200]], rng0)
    S0 = compute_S(A0[idx_p[:200]], meta0, y_ref=lab0[idx_p[:200]], t_strength=1.0)
    p0 = np.column_stack([T0, S0, F0]).astype(np.float32)
    sig0_t, _ = evaluate(A0[idx_p[200:]], lab0[idx_p[200:]], p0, np.array([1,0,0]))
    res["signal0_T"] = sig0_t

    # neutral count
    import pandas as pd
    ip = pd.read_csv(data_dir / "csv" / "item_profiles.csv")
    res["neutral_pct"] = 100 * (ip["taxon_direction"] == "neutral").sum() / len(ip)

    return res


# ---------------------------------------------------------------------------
# Grid de hiperparámetros
# ---------------------------------------------------------------------------

def run_grid(output_root: Path = Path("data") / "grid",
             k: int = 5000, seed: int = 42) -> list[dict]:
    """Barrido pequeño de t_strength × metadata_strength × functional_overlap."""
    t_strengths = [0.80, 0.90, 1.00]
    meta_strengths = [0.00, 0.15, 0.30]
    overlaps = [0.30, 0.40, 0.50]

    results = []
    total = len(t_strengths) * len(meta_strengths) * len(overlaps)
    idx = 0

    for ts in t_strengths:
        for ms in meta_strengths:
            for ov in overlaps:
                idx += 1
                name = f"ts{ts:.2f}_ms{ms:.2f}_ov{ov:.2f}"
                print(f"\n[{idx}/{total}] {name}")
                try:
                    d = build_one_dataset(name, n_eval_samples=100, n_ref_samples=200,
                                          n_items=500, seed=seed, signal=1.0,
                                          t_strength=ts, metadata_strength=ms,
                                          functional_overlap=ov,
                                          output_root=output_root, verbose=False)
                    r = quick_eval(d, k=k, seed=seed)
                    r["t_strength"] = ts
                    r["metadata_strength"] = ms
                    r["functional_overlap"] = ov
                    results.append(r)

                    print(f"  T={r['T_only'][0]:.4f} S={r['S_only'][0]:.4f} "
                          f"F={r['F_only'][0]:.4f} equal={r['equal'][0]:.4f} "
                          f"best={r['best'][0]:.4f} w=[{r['best_w'][0]:.4f} "
                          f"{r['best_w'][1]:.4f} {r['best_w'][2]:.4f}] "
                          f"corrTF={r['corr_TF']:.4f} sig0={r['signal0_T']:.4f} "
                          f"neut={r['neutral_pct']:.0f}%")
                except Exception as e:
                    print(f"  ERROR: {e}")

    return results


# ---------------------------------------------------------------------------
# Punto de entrada
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Genera dataset con perfiles T,S,F calibrados")
    parser.add_argument("--name", type=str, default="benchmark")
    parser.add_argument("--n-eval", type=int, default=DEFAULT_N_EVAL)
    parser.add_argument("--n-ref", type=int, default=DEFAULT_N_REF)
    parser.add_argument("--n-items", type=int, default=DEFAULT_N_ITEMS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--signal", type=float, default=DEFAULT_SIGNAL)
    parser.add_argument("--t-strength", type=float, default=DEFAULT_T_STRENGTH)
    parser.add_argument("--metadata-strength", type=float, default=DEFAULT_METADATA_STRENGTH)
    parser.add_argument("--functional-overlap", type=float, default=DEFAULT_FUNCTIONAL_OVERLAP)
    parser.add_argument("--out-dir", type=Path, default=Path("data") / "processed")
    parser.add_argument("--grid", action="store_true",
                        help="Ejecutar grid de hiperparámetros")
    parser.add_argument("--k", type=int, default=5000,
                        help="K para random search en grid")
    args = parser.parse_args()

    if args.grid:
        results = run_grid(output_root=args.out_dir, k=args.k, seed=args.seed)
        print("\n\n=== GRID RESULTS ===")
        print("ts,ms,ov,T_only,S_only,F_only,equal,best,wT,wS,wF,corrTF,sig0T,neut%")
        for r in results:
            print(f"{r['t_strength']:.2f},{r['metadata_strength']:.2f},"
                  f"{r['functional_overlap']:.2f},"
                  f"{r['T_only'][0]:.4f},{r['S_only'][0]:.4f},"
                  f"{r['F_only'][0]:.4f},{r['equal'][0]:.4f},"
                  f"{r['best'][0]:.4f},{r['best_w'][0]:.4f},"
                  f"{r['best_w'][1]:.4f},{r['best_w'][2]:.4f},"
                  f"{r['corr_TF']:.4f},{r['signal0_T']:.4f},"
                  f"{r['neutral_pct']:.0f}")
    else:
        d = build_one_dataset(args.name,
                               n_eval_samples=args.n_eval,
                               n_ref_samples=args.n_ref,
                               n_items=args.n_items,
                               seed=args.seed,
                               signal=args.signal,
                               t_strength=args.t_strength,
                               metadata_strength=args.metadata_strength,
                               functional_overlap=args.functional_overlap,
                               output_root=args.out_dir)
        r = quick_eval(d, k=5000, seed=args.seed)
        print(f"\n  T_only={r['T_only'][0]:.4f}  S_only={r['S_only'][0]:.4f}  "
              f"F_only={r['F_only'][0]:.4f}")
        print(f"  equal={r['equal'][0]:.4f}  best={r['best'][0]:.4f}  "
              f"w=[{r['best_w'][0]:.4f} {r['best_w'][1]:.4f} {r['best_w'][2]:.4f}]")
        print(f"  corr(T,F)={r['corr_TF']:.4f}  signal=0 T={r['signal0_T']:.4f}  "
              f"neutral={r['neutral_pct']:.0f}%")


if __name__ == "__main__":
    main()
