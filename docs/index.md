# Indice de documentacion

Este directorio contiene la documentacion tecnica detallada del proyecto. Cada archivo desarrolla en profundidad los temas presentados en el README principal.

## Problema y contexto

| Archivo | Contenido |
|---|---|
| [01_problema.md](01_problema.md) | Planteamiento del problema: scoring metagenomico, objetivo, relevancia computacional |
| [02_dataset.md](02_dataset.md) | Dataset sintetico: origen, dimensiones, generacion, separacion REF/EVAL |
| [03_modelo_matematico.md](03_modelo_matematico.md) | Modelo matematico: perfiles, scores, funcion objetivo, simplex, Dirichlet |

## Implementaciones

| Archivo | Contenido |
|---|---|
| [04_python_secuencial.md](04_python_secuencial.md) | Python secuencial: baseline, evaluacion por candidato, algoritmo |
| [05_python_multicore.md](05_python_multicore.md) | Python multicore: multiprocessing, imap_unordered, memoria independiente |
| [06_c_secuencial.md](06_c_secuencial.md) | C secuencial: implementacion nativa, RNG PCG64, optimizaciones |
| [07_c_openmp.md](07_c_openmp.md) | C OpenMP: memoria compartida, fork/join, merge serial post-loop |
| [08_c_mpi.md](08_c_mpi.md) | C MPI: memoria distribuida, Bcast, Scatterv, Gather |
| [09_pycuda.md](09_pycuda.md) | PyCUDA: GPU, kernel CUDA, reduccion, modos full y precompute |

## Estrategias y evaluacion

| Archivo | Contenido |
|---|---|
| [10_estrategias_busqueda.md](10_estrategias_busqueda.md) | Estrategias de busqueda: random, grid, hybrid |
| [11_benchmark.md](11_benchmark.md) | Benchmark: metricas, resultados K=2M, graficas en results/plots/ |

## Referencias externas

- [`PROJECT.md`](../PROJECT.md): especificacion contractual completa del proyecto
- [`fuente_real_dataset_sintetico_crc.md`](../fuente_real_dataset_sintetico_crc.md): origen biologico y referencia del dataset sintetico
