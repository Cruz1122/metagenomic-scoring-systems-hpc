# PyCUDA

CUDA asigna un candidato `W` por hilo GPU. Esto encaja con random search porque los candidatos son independientes.

Archivo:

```text
CUDA/scoring_pycuda.py   → wrapper Python con kernel embebido
```

## PyCUDA (`scoring_pycuda.py`)

```bash
make python-pycuda K=100000 SEED=42 SEARCH=random
make python-pycuda-fast K=10000 SEED=42 SEARCH=random   # benchmark sin logging en vivo
# o
python CUDA/scoring_pycuda.py --k 100000 --seed 42 --data-dir data --search random
python CUDA/scoring_pycuda.py --k 10000 --seed 42 --search random --fast
```

- Lee datos desde `.npy` (comparte formato con Python).
- Genera candidatos con `np.random.default_rng(seed).dirichlet(...)`.
- El kernel CUDA se embebe como string y se compila en runtime con PyCUDA.
- AUC y consistencia se calculan en **GPU** (float32).
- AUC: Mann-Whitney por rangos; **scores acumulados en double** (igual que `common.c` / sklearn).
- Logger con jerarquía CUDA: **grilla** (kernel launch) · **bloque** · **thread**.

## Paralelismo

```cuda
// BLOCK_SIZE = 256, GRID = ceil(K / 256)
// Ejemplo K=10000 → 40 bloques, 256 hilos/bloque
evaluate_full<<<grid, block, n_samples * sizeof(float)>>>(
    dA, dP, dLabels, dW, dAuc, dCons, n_samples, n_items, K, index_offset);
reduce_best_stage1<<<grid, block>>>(dAuc, dCons, dPartial, K);
reduce_best_stage2<<<1, block>>>(dPartial, dBestAuc, dBestCons, dBestIdx, grid);
```

- **1 hilo = 1 candidato** `W_k`.
- Hilos inactivos del último bloque participan en `__syncthreads()` durante la carga cooperativa.

## Memoria compartida

En modo `full`, cada bloque cachea la columna `A[:, i]` (filas por muestra) en `__shared__` antes de que los 256 hilos acumulen `scores[s] += A[s,i] * p_i(w)`.

- Tamaño dinámico por launch: `n_samples × sizeof(float)` (≈ 8 KB con 2000 muestras).
- `profiles` y `labels` permanecen en memoria global (solo lectura).

## Transferencias Host↔Device

| Buffer | Cuándo |
|--------|--------|
| `A`, `profiles`, `y` (labels) | Una vez al inicio |
| `B = A @ profiles` (modo precompute) | Una vez al inicio |
| `weights` (K×3) | Una vez (hybrid: por fase, mismo buffer pre-asignado) |
| `auc`, `consistency` | Modo `--fast`: reduction en GPU; D2H solo del mejor |
| | Modo live: D2H parcial cada 32 candidatos (logging) |

## Modos de ejecución

| Flag | Comportamiento |
|------|----------------|
| *(default)* | Un launch CUDA por bloque (256 cand.); solo log cuando hay nuevo mejor global |
| `--fast` | Un launch sobre K + reduction GPU; ideal para benchmark |

## Reglas de memoria

- Copiar `A`, `profiles`, `y` una sola vez al device. Si copias por candidato, destruyes el rendimiento.
- Cada launch evalúa un batch de candidatos en paralelo; un thread = un candidato.
- El mejor global se obtiene con **reduction kernel** estándar en dos fases (sin race multi-bloque).
