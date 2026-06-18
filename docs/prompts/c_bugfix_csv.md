# C_sequential y OpenMP vs Python en 2000 muestras

**Session ID:** ses_12c6d87d0ffeg1sCtqQRgjg0zC
**Created:** 6/16/2026, 10:14:12 PM
**Updated:** 6/16/2026, 10:27:17 PM

---

## User

c_sequential y openmp no me estan dando igual que python con el dataset de 2000 muestras. revisa bien que puede estar pasando, pero no te quedes en "puede ser la seed". quiero que mires carga de datos, orden de muestras, precision, auc, consistencia, pesos generados y si C esta leyendo lo mismo que python. necesito la causa real y como comprobarla.

---

## Assistant (Quality · DeepSeek V4 Flash · 9.7s)

# Verdict

**Root cause found: column index bug in `load_csv_profiles()`.** C reads wrong columns for S and F profile data, producing corrupted profiles → wrong AUC.

---
# Findings by severity

## 🔴 Critical — Wrong column indices in CSV profile parsing

**File:** `C_OpenMP_MPI/shared/common.c:233-235`

**Evidence:**
```
CSV actual columns (7 total):
  0:item_id  1:taxon_name  2:T  3:taxon_direction  4:true_group  5:S  6:F

C code read:
  col 2 → T  ✓
  col 4 → S  ✗ (column 4 = "true_group" = STRING like "neutral")
  col 5 → F  ✗ (column 5 = S, numeric but wrong column)

strtod("neutral") → 0.0, so S column becomes all zeros.
F column receives S values instead of F values.
```

**Impact:**
- Before fix: `c_sequential` AUC = **0.796571** (wrong data)
- Python: AUC = **0.814503** (correct data)  
- After fix: `c_sequential` AUC = **0.814503** (matches Python exactly)
- Delta: **~0.018 AUC (2.2% relative error)**

**Root cause:** The `item_profiles.csv` format was expanded from 6 to 7 columns (added `true_group` after `taxon_direction`), but C parser wasn't updated.

**Fix applied:** `col 4→5` for S, `col 5→6` for F. Comment also updated.

---
# Simplification plan

No architectural changes needed. This is a data format drift bug.

---
# Dead code candidates

None found.

---
# Complexity/perf risks

**Not directly related** but noted: `evaluate()` in common.c allocates/frees `P_vec` and `scores` on every call (2× malloc + 2× free per iteration, ~10K allocations in a run). This is a perf drag for the inner loop. Could be hoisted to caller with scratch buffers.

---
# Missing tests

No regression tests exist for CSV loading correctness. Add:
1. CSV profile column integrity test — load CSV, verify T/S/F columns match NPY reference
2. AUC numerical regression test — fixed w = [0.3, 0.3, 0.4] should produce known AUC

---
# Architecture recommendation

**Add a schema assertion:** validate CSV column header on load. If `item_profiles.csv` header doesn't match expected pattern, abort with clear message. This would have caught the format drift immediately.

---
# Verification gate

| Check | Before fix | After fix |
|-------|-----------|-----------|
| C sequential AUC (k=5000, seed=42) | 0.796571 | **0.814503** ✓ |
| Python sequential AUC (same params) | 0.814503 | 0.814503 ✓ |
| C OpenMP AUC (same params) | 0.796569 | 0.814473 ✓ |
| Best weights match Python | No | **Yes** ✓ |

---

## User

modulariza los dos datasets, el de 100 muestras y el de 2000

---

## Assistant (Quality · DeepSeek V4 Flash · 6.1s)

# Verdict

Datasets modularizados. Estructura limpia con ambos datasets en `data/processed/`.

---

## Nueva estructura

```
data/
├── npy/ → processed/synthetic_CRC2000x500_balanced/npy/     (symlink)
├── csv/ → processed/synthetic_CRC2000x500_balanced/csv/     (symlink)
├── dataset_manifest.json → .../dataset_manifest.json         (symlink)
├── processed/
│   ├── synthetic_CRC100x500_balanced/    ← 100 muestras
│   │   ├── npy/   (matrix_A.npy, labels.npy, profiles_TSF.npy)
│   │   ├── csv/   (matrix_A.csv, samples.csv, item_profiles.csv, ...)
│   │   └── dataset_manifest.json
│   └── synthetic_CRC2000x500_balanced/   ← 2000 muestras (default)
│       ├── npy/
│       ├── csv/
│       └── dataset_manifest.json
└── scripts/
    └── generate_dataset.py
```

## Cambios realizados

| Archivo | Cambio |
|---------|--------|
| `data/processed/synthetic_CRC100x500_balanced/` | **Nuevo** — dataset 100 muestras (50+50), 500 items |
| `data/npy/`, `data/csv/`, `data/dataset_manifest.json` | Ahora symlinks a `processed/synthetic_CRC2000x500_balanced/` |
| `data/scripts/generate_dataset.py:664` | `min_eval=2000` → `min_eval=args.n_eval` (permite generar ambos tamaños) |
| `DATASET.md` | Reescribir — documenta ambos datasets, estructura, uso |
| `README.md` | Actualizar ejemplos con `DATASET_SIZE` y el generador correcto |
| `run_all.sh` | Soporte `DATASET_SIZE=100`/`2000`, fix `generate_data.py` → `generate_dataset.py` |
| `Makefile` | Soporte `DATASET_SIZE`, fix target `data:` al script correcto, clean ya no elimina symlinks |

## Uso

```bash
# 2000 muestras (default)
make python-seq K=10000
DATASET_SIZE=2000 ./run_all.sh

# 100 muestras (rápido para desarrollo)
make python-seq DATASET_SIZE=100 K=10000
DATASET_SIZE=100 ./run_all.sh

# Explícito
python python/sequential.py --k 10000 --data-dir data/processed/synthetic_CRC100x500_balanced
```

---
