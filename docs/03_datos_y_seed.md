# Datos y seed

Comando (generación sintética para desarrollo):

```bash
python data/scripts/generate_synthetic_dataset.py \
    --out-dir data \
    --n-samples 100 \
    --n-items 500 \
    --seed 42 \
    --signal 2.2
```

Parámetros:

| Flag | Default | Descripción |
|---|---|---|
| `--n-samples` | 100 | Número total de muestras (debe ser par) |
| `--n-items` | 500 | Número de items (columnas de `A` y filas de `profiles`) |
| `--seed` | 42 | Semilla para reproducibilidad |
| `--signal` | 2.2 | Controla separabilidad: más alto = más separación entre sanos y enfermos |
| `--concentration` | 240.0 | Concentración Dirichlet: menor = más ruido muestral |
| `--dropout-rate` | 0.30 | Tasa base de dropout/ceros para taxones raros |
| `--out-dir` | data | Directorio de salida |

Archivos generados (bajo `{out-dir}/`):

```text
csv/samples.csv           → metadatos de muestras (sample_id, label, group)
csv/matrix_A.csv          → matriz A (100 x 500) en CSV
npy/matrix_A.npy          → matriz A (100 x 500) en NumPy
npy/labels.npy            → vector y (100 enteros: 50×0, 50×1)
npy/profiles_TSF.npy      → matriz de perfiles (500 x 3): columnas T, S, F
csv/metadata.csv          → variables poblacionales/ecológicas
csv/functional_matrix.csv → marcadores funcionales proxy por item
csv/item_profiles.csv     → perfiles T, S, F con nombres de taxones
csv/item_mapping.csv      → mapeo item_id → taxon_name
dataset_manifest.json     → metadatos del dataset generado
```

> También existe `data/scripts/build_final_dataset.R` para reconstruir el dataset real desde `curatedMetagenomicData` (Bioconductor). Ese script es la ruta prevista para la presentación final del proyecto.

`metadata.json` incluye los pesos sintéticos reales usados para inducir la señal:

```json
{
  "true_w_synthetic": [0.45, 0.35, 0.20],
  "signal": 6.0
}
```

## Modelo de generación

Cada fila de `A` se genera como `Dirichlet(α)` donde `α = 0.5 + signal · risk` para enfermos y `α = 0.5 + signal · (1 - risk)` para sanos. El vector `risk` se calcula como `profiles @ true_w`. Esto crea separabilidad controlada entre grupos.

## Supuesto explícito

El contrato define `T_i`, `S_i`, `F_i`, pero el script mínimo solo guardaba `A` e `y`. Sin perfiles no se puede calcular `P_i`. Por eso el scaffold genera `profiles.npy` y `profiles.csv` con columnas `T,S,F`.

## Seed

El seed controla los datos y los candidatos por implementación. No se exige que Python, C y CUDA generen exactamente los mismos candidatos, pero sí que cada implementación sea reproducible y use el mismo dataset.

En Python y PyCUDA los candidatos se generan con `np.random.default_rng(seed).dirichlet(...)`. En C (OpenMP y MPI) se usa Xorshift más transformación exponencial. En CUDA C se usa `std::mt19937_64` más exponencial.
