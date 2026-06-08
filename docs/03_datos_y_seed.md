# Datos y seed

Comando:

```bash
python data/generate_data.py --n-items 500 --seed 42 --signal 6.0 --out-dir data
```

Parámetros:

| Flag | Default | Descripción |
|---|---|---|
| `--n-items` | 50 | Número de items (columnas de `A` y filas de `profiles`) |
| `--seed` | 42 | Semilla para reproducibilidad |
| `--signal` | 6.0 | Controla separabilidad: más alto = más separación entre sanos y enfermos |
| `--out-dir` | data | Directorio de salida |

Archivos generados:

```text
matrix_A.npy / matrix_A.csv     → matriz A (10 x N)
labels.npy / labels.csv          → vector y (10 enteros: 5×0, 5×1)
profiles.npy / profiles.csv      → matriz de perfiles (N x 3): columnas T, S, F
metadata.json                    → seed, n_items, signal, true_w_synthetic
```

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
