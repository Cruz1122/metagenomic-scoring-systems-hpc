# Entregables

Mínimo:

```text
python/common.py             → lógica compartida (load_data, random_search, AUC, consistencia)
python/sequential.py         → baseline secuencial
python/multicore.py          → multiprocessing
C_OpenMP_MPI/scoring_openmp.c → OpenMP memoria compartida
C_OpenMP_MPI/scoring_mpi.c   → MPI memoria distribuida
CUDA/scoring_kernel.cu       → CUDA C
CUDA/scoring_pycuda.py       → PyCUDA
scripts/postprocess_benchmark.py → agrega speedup y eficiencia
scripts/plot_benchmark.py    → genera gráficas
run_all.sh                   → benchmark completo
results/benchmark.csv        → resultados consolidados
docs/                        → documentación técnica (12 archivos)
report/informe_tecnico.pdf   → informe final
```

El informe debe incluir modelo, datos/seed, arquitectura, sincronización, memoria, benchmarks, speedup, eficiencia, Amdahl y discusión de AUC/consistencia. No llenes el PDF con teoría genérica: defiende resultados medidos.
