# Datos del proyecto

Dataset metagenómico sintético-compatible `cMD_CRC100_balanced`
(100 muestras × 500 items) para el modelo de scoring HPC.

## Generador

```text
data/scripts/generate_data.py
```

```bash
python data/scripts/generate_data.py
python data/scripts/generate_data.py --seed 42 --signal 2.0
```

## Estructura

```text
data/
├── csv/
│   ├── samples.csv
│   ├── matrix_A.csv
│   ├── metadata.csv
│   ├── functional_matrix.csv
│   ├── item_profiles.csv
│   └── item_mapping.csv
├── npy/
│   ├── matrix_A.npy
│   ├── labels.npy
│   └── profiles_TSF.npy
├── dataset_manifest.json
├── scripts/
│   ├── generate_data.py       → generador principal
│   ├── build_final_dataset.R  → reconstrucción desde curatedMetagenomicData
│   └── validate_dataset.py    → validador
└── README.md
```

## Validación

```bash
python data/scripts/validate_dataset.py
```

## Nota

Dataset **synthetic-compatible**. No corresponde a mediciones clínicas reales.
Para la presentación final se reconstruirá desde `curatedMetagenomicData`.
