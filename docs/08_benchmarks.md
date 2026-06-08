# Benchmarks

Ejecutar:

```bash
./run_all.sh
```

Configurable:

```bash
N_ITEMS=500 K=100000 SEED=42 THREADS_LIST="1 2 4 8" WORKERS_LIST="2 4" MPI_RANKS_LIST="2 4" ./run_all.sh
```

## Pipeline de salida

1. `run_all.sh` genera datos, ejecuta cada implementación y escribe en `results/benchmark_raw.csv`.
2. `scripts/postprocess_benchmark.py` lee `benchmark_raw.csv`, localiza el baseline `python_sequential`, calcula `speedup` y `efficiency`, y escribe `results/benchmark.csv`.

Archivos:

```text
results/benchmark_raw.csv   → datos crudos por implementación
results/benchmark.csv        → incluye speedup y eficiencia
```

Columnas de `benchmark.csv`:

```text
implementation, parallel_units, n_items, k, time_sec, auc, consistency, w1, w2, w3, seed, speedup, efficiency
```

Donde:
- `speedup = T_python_sequential / T_impl`
- `efficiency = speedup / parallel_units`

## Gráficas

```bash
python scripts/plot_benchmark.py --input results/benchmark.csv --out-dir results/plots
```

Genera `results/plots/` con barras para `time_sec`, `speedup`, `efficiency` y `auc`.

## Repeticiones

Para informe final, haz varias repeticiones y reporta promedio/desviación. Una sola corrida sirve para depurar, no para concluir.
