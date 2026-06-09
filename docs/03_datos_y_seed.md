# Datos y seed

## Dataset

`cMD_CRC100_balanced`: 100 muestras (50 healthy, 50 CRC), 500 items.

## Generación

```bash
python data/scripts/generate_data.py --seed 42
```

Parámetros: `--seed` (default 42), `--signal` (default 2.0).

## Archivos generados

```
data/csv/samples.csv, matrix_A.csv, metadata.csv, functional_matrix.csv,
     item_profiles.csv, item_mapping.csv
data/npy/matrix_A.npy, labels.npy, profiles_TSF.npy
data/dataset_manifest.json
```

## Seed

Controla reproducibilidad de los datos. Cada implementación (Python, C, CUDA)
genera sus candidatos W con su propio RNG sobre el mismo dataset.

## Dataset real

Para presentación final: `data/scripts/build_final_dataset.R` reconstruye
desde curatedMetagenomicData.
