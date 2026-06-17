# OpenMP

OpenMP se usa para memoria compartida en CPU. Todos los hilos leen `A`, `profiles` e `y` (compartido, solo lectura), y cada hilo evalúa candidatos `W` independientes con RNG propio.

Archivo: `C_OpenMP_MPI/scoring_openmp.c`

## Compilar y ejecutar

```bash
make -C C_OpenMP_MPI scoring_openmp
OMP_NUM_THREADS=4 ./C_OpenMP_MPI/scoring_openmp --k 100000 --seed 42 --data-dir data --search random
```

CLI:

| Flag | Default | Descripción |
|---|---|---|
| `--k` | 10000 | Candidatos (random/hybrid) |
| `--seed` | 42 | Semilla base RNG |
| `--data-dir` | "data" | Directorio con CSV |
| `--search` | "random" | `random` \| `grid` \| `hybrid` |
| `--step` | 0.02 | Paso del grid (grid/hybrid) |

Hilos se controlan vía variable de entorno `OMP_NUM_THREADS` (estándar OpenMP).

## 3 estrategias de búsqueda

### Random search
- `simplex(pcg, w)` genera Dirichlet(1,1,1)
- `K` iteraciones paralelas con `schedule(static)`
- Cada hilo: RNG `pcg64_seed(pcg, seed + tid)` — stream único

### Grid search
- Barre w1, w2 con paso `step` en el simplex 2D
- Precomputa array plano de puntos, `#pragma omp for schedule(static)`
- Sin RNG (grid determinista)

### Hybrid search
3 fases secuenciales, cada una paralela internamente:
1. **Grid** step=0.02 (~1326 puntos)
2. **Random** Dirichlet(1,1,1) (~50% del resto)
3. **Local** Dirichlet concentrado alrededor de best_w con concentration=300 y 1000 (~50% resto)

La fase local usa `dirichlet_general(alpha, pcg, w)` que implementa muestreo Gamma(alpha_i,1) vía Marsaglia-Tsang + normalización.

## Estrategia de RNG

Cada hilo usa PCG64 (réplica de numpy) inicializado con:

```c
uint64_t pcg[4];
pcg64_seed(pcg, seed + omp_get_thread_num());
```

Streams independientes por hilo. Con `OMP_NUM_THREADS=1` los resultados son **idénticos** al secuencial (misma semilla).

## Patrón de paralelización

```c
#pragma omp parallel
{
    // RNG local + Best local (privado)
    Best local = { .auc = -1.0 };
    #pragma omp for schedule(static)
    for (long i = 0; i < K; i++) {
        simplex(pcg, w);
        evaluate(..., &auc, &cons);
        if (auc > local.auc) local = ...;
    }
    // Merge post-loop (única sincronización)
    #pragma omp critical
    if (local.auc > global.auc) global = local;
}
```

- **Sin acceso a variables compartidas durante el loop.** Solo memoria privada.
- `#pragma omp critical` solo una vez por hilo al final, no por iteración.
- Worker report post-loop: `log_worker_report(tid, ...)` con tags `[W0]`..`[W3]`.
- No se usa `reduction` porque `Best` no es tipo nativo C.

## Sincronización compartida

| Variable | Acceso |
|---|---|
| `ds->A`, `ds->profiles`, `ds->y` | Solo lectura (compartido, seguro) |
| `pcg[4]`, `w[3]`, `local_best` | Privado por hilo |
| `global_best` | Shared, solo en `#pragma omp critical` post-loop |

## Métricas

Tiempo medido con `clock_gettime(CLOCK_MONOTONIC)`. La salida incluye `c_openmp`, `search_mode`, `workers` (via `omp_get_max_threads()`).
