# Índice de documentación

Este directorio concentra la documentación operativa del proyecto. El archivo contractual completo está en [`../PROJECT.md`](../PROJECT.md). Si una decisión técnica contradice `PROJECT.md`, la decisión está mal salvo que esté marcada como supuesto explícito.

> 🏗️ **Scaffold parcial:** `C_OpenMP_MPI/scoring_openmp.c` está implementado (3 estrategias OpenMP). `C_OpenMP_MPI/scoring_mpi.c`, `CUDA/` y `scripts/` son esqueleto con TODO.

## Guías

1. [Resumen técnico](00_resumen_tecnico.md)
2. [Convenciones](01_convenciones.md)
3. [Modelo de scoring](02_modelo_scoring.md)
4. [Datos y seed](03_datos_y_seed.md)
5. [Python y multiprocessing](04_python_multiprocessing.md)
6. [OpenMP](05_openmp.md) — implementado: PCG64, 3 estrategias, worker-report
7. [MPI](06_mpi.md) — scaffold: reusa shared/, pipeline sugerido
8. [CUDA y PyCUDA](07_cuda.md)
9. [Benchmarks](08_benchmarks.md)
10. [Amdahl y Gustafson](09_amdahl_gustafson.md)
11. [Entregables](10_entregables.md)
