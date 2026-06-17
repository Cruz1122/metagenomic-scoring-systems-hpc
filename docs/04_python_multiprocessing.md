# Python y multiprocessing

Ambos scripts (`sequential.py` y `multicore.py`) comparten el módulo `python/common.py`
que contiene:

- `load_data(data_dir)`: carga `matrix_A.npy`, `labels.npy` y `profiles_TSF.npy`.
- `evaluate(A, y, profiles, w)`: calcula `P = profiles @ w`, `scores = A @ P`,
  `auc = roc_auc_score(y, scores)` y `consistency = mejor balanced accuracy`.
- `SearchResult`: dataclass con métricas de una corrida; método `csv_row()`.

## Secuencial

`python/sequential.py` es el baseline. Un solo proceso ejecuta `random_search`
generando un peso W por iteración con `rng.dirichlet(np.ones(3))` y evaluándolo
al instante.

```bash
python python/sequential.py --k 10000 --seed 42 --data-dir data
```

Tres estrategias de búsqueda (`--search`):
- `random`  — Dirichlet(1,1,1) puro.
- `grid`    — barrido sistemático del simplex con step=0.02 (~1326 puntos).
- `hybrid`  — grid + random + refinamiento local con Dirichlet concentrada.

## Multiprocessing

`python/multicore.py` reparte la **evaluación** de candidatos entre procesos
usando `multiprocessing.Pool.map`. La generación de pesos es centralizada
para garantizar determinismo frente a sequential.py.

### Diseño

1. El proceso principal genera los K pesos de una sola vez:
   `rng.dirichlet(np.ones(3), size=K)`.
2. `np.array_split(weights, workers)` divide en chunks balanceados.
3. Cada chunk se etiqueta con un `worker_id` (0, 1, …, workers−1).
4. `Pool.map(partial(_eval_chunk, A, y, profiles), chunks_with_ids)` evalúa
   cada chunk en un proceso distinto.
5. Cada worker retorna su mejor local `(best_auc, best_consistency, w, local_idx, worker_id)`.
6. El proceso principal recolecta los resultados, reporta el mejor local de
   cada worker y selecciona el mejor global (AUC primario, consistencia desempate).

No hay variables compartidas, locks, Manager, Queue ni Value. La comunicación
se reduce al retorno de `pool.map`.

### Tres estrategias de búsqueda

- `random`  — genera K pesos Dirichlet, paraleliza toda la evaluación.
- `grid`    — genera ~1326 puntos del simplex, paraleliza evaluación.
- `hybrid`  — fase 1 (grid) + fase 2 (random batch) + fase 3 (local batch
              concentrada con concentration=300 y 1000). Cada fase genera
              sus pesos secuencialmente pero evalúa en paralelo.

### CLI

```bash
python python/multicore.py --k 10000 --seed 42 --workers 4

python python/multicore.py --k 10000 --seed 42 --workers 4 --search grid

python python/multicore.py --k 5000 --seed 42 --workers 4 --search hybrid --theta 0.5
```

### Argumentos

| Flag | Default | Descripción |
|---|---|---|
| `--k` | 10000 | Número de candidatos |
| `--seed` | 42 | Semilla RNG (genera mismos pesos que sequential) |
| `--workers` | cpu_count() | Número de procesos workers |
| `--search` | random | Estrategia: random, grid, hybrid |
| `--theta` | — | Umbral fijo para consistencia (post-hoc); si se omite, se usa median(scores) |
| `--data-dir` | data | Directorio con los .npy |
| `--csv` | false | Salida CSV una línea (sin colores ni etiquetas) |

### Salida (modo normal)

```text
  [W0] AUC 0.655200  consist=0.6800  w=[0.3374 0.3279 0.3347]  (2500 cand.)
  [W1] AUC 0.721300  consist=0.7000  w=[0.5168 0.1155 0.3677]  (2500 cand.)
  [W2] AUC 0.758800  consist=0.7200  w=[0.5671 0.1334 0.2995]  (2500 cand.)  ★
  [W3] AUC 0.654400  consist=0.6800  w=[0.2076 0.4275 0.3649]  (2500 cand.)

  ➜  [W2] AUC 0.758800  (initial)  iter 1,728/10,000  consist=0.7200  w=[0.5671 0.1334 0.2995]
  ...
```

- `[WN]` identifica qué worker encontró cada mejora global.
- `★` marca el mejor global entre los workers.
- La consistencia final se calcula post-hoc con `--theta` o `median(scores)`.

### Salida (CSV)

```text
implementation,parallel_units,n_items,k,time_sec,auc,consistency,w1,w2,w3,seed,search_mode,iterations_until_best
```

## Determinismo

Con la misma `seed` y mismo `K`, `multicore.py` genera exactamente los mismos
pesos que `sequential.py`. El `best_auc` debe coincidir con tolerancia `1e-12`.
