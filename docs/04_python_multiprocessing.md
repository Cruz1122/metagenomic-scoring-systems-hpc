# Python y multiprocessing

Ambos scripts comparten el módulo `python/common.py` que contiene:

- `random_search(A, y, profiles, k, seed)`: random search vectorizado por batches de 8192 candidatos usando `rng.dirichlet` y multiplicación matricial.
- `auc_matrix(scores, y)`: cálculo vectorizado de AUC para múltiples candidatos a la vez.
- `load_data(data_dir)`: carga `matrix_A.npy`, `labels.npy` y `profiles_TSF.npy`.

## Secuencial

`python/sequential.py` es el baseline. Un solo proceso ejecuta `random_search` con `timed_search`.

```bash
python python/sequential.py --k 10000 --seed 42 --data-dir data
```

Flags: `--csv` para salida en formato CSV (una línea), sin `--csv` imprime JSON.

## Multiprocessing

`python/multicore.py` divide `K` entre procesos usando `multiprocessing.Pool`. Estrategia:

1. Divide `K` en chunks equitativos (`divmod`).
2. Cada worker recibe una semilla derivada: `seed + 100003 * i`.
3. Cada worker carga datos independientemente (copia propia de `A`, `y`, `profiles`).
4. Se recogen resultados y se retorna el mejor AUC.

```bash
python python/multicore.py --k 10000 --workers 4 --seed 42 --data-dir data
```

## Advertencia

No uses `K` pequeño para juzgar multiprocessing. Crear procesos, cargar datos y recolectar resultados tiene overhead. Con `K ≥ 10000` el speedup empieza a ser medible.
