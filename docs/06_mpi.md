# MPI

MPI se usa para memoria distribuida. Cada rank evalúa un bloque de candidatos.

Archivo: `C_OpenMP_MPI/scoring_mpi.c`

**Estado: scaffold.** El archivo incluye `shared/common.h`, `shared/rng.h`, `shared/logger.h` (reemplazando los stubs locales anteriores). Falta implementar la lógica MPI real.

## Compilar

```bash
make -C C_OpenMP_MPI scoring_mpi
mpirun -np 4 ./C_OpenMP_MPI/scoring_mpi --k 10000 --seed 42 --data-dir data
```

Requiere MPICH, OpenMPI o equivalente.

## Pipeline sugerido

1. **Rank 0** carga datos con `load_data(data_dir, &ds)` (shared/common.c).
2. **Broadcast** dimensiones (`n_samples`, `n_items`) y matrices (`A`, `profiles`, `y`) a todos los ranks.
3. Cada rank: RNG propio `pcg64_seed(pcg, seed + rank)`, itera su chunk de `K/size` candidatos.
4. `MPI_Gather` o `MPI_Reduce` de mejores locales → rank 0.
5. Rank 0: `log_complete()`, salida parseable.

## Recursos compartidos disponibles

| Función | Archivo | Propósito |
|---|---|---|
| `load_data()` | `shared/common.h` | Carga CSV → Dataset |
| `evaluate()` | `shared/common.h` | scores = A @ (profiles @ w), AUC, consistency |
| `simplex()` | `shared/rng.h` | Dirichlet(1,1,1) |
| `pcg64_seed()` | `shared/rng.h` | Inicializar RNG con semilla |
| `log_header/completer()` | `shared/logger.h` | Logger colorido |
| `free_dataset()` | `shared/common.h` | Liberar Dataset |

## Advertencia

MPI puede no acelerar con `K` pequeño o en una sola máquina. MPI brilla en clústeres con alta latencia de red pero muchos nodos, donde `K` es suficientemente grande para que el cómputo domine sobre la comunicación.
