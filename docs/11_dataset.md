# 11. Dataset del proyecto

## 1. Dataset

```text
Nombre:          cMD_CRC100_balanced
Muestras:        100 (50 healthy, 50 CRC)
Items/taxones:   500
Seed:            42
Abundancia:      relativa (filas de A suman ~1)
```

Modelo:

```text
P_i = W1*T_i + W2*S_i + W3*F_i
Score = A @ P
AUC = auc(labels, Score)
```

## 2. Estructura

```
data/
├── csv/          → samples.csv, matrix_A.csv, metadata.csv,
│                   functional_matrix.csv, item_profiles.csv, item_mapping.csv
├── npy/          → matrix_A.npy, labels.npy, profiles_TSF.npy
└── dataset_manifest.json
```

## 3. Perfiles

### T (taxonómico) — `[0, 1]`

```python
T_i = abs(mean_crc_i - mean_healthy_i) / max_raw_T
```

Magnitud diferencial. La dirección va en `taxon_direction` (`CRC_enriched`,
`healthy_enriched`, `neutral`).

### S (ecológico/poblacional) — `[0, 1]`

Asociación entre `A[:, i]` y metadata usable:

- `age`, `bmi`: `abs(Pearson)`
- `sex`, `country`: `eta = sqrt(SS_between / SS_total)`

`study_name` es constante, excluido.

### F (funcional) — `[0, 1]`

```python
F_i = mean(resistance, virulence, inflammation, metabolic, beneficial)
```

## 4. Generación

```bash
python data/scripts/generate_data.py
```

Dataset **synthetic-compatible**. Para presentación final se reconstruirá
desde `curatedMetagenomicData` (ver `data/scripts/build_final_dataset.R`).
