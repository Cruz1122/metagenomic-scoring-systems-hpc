#!/usr/bin/env bash
# run_all.sh — Pipeline completa: genera datos, ejecuta todos los implementaciones, post-procesa.
set -euo pipefail
cd "$(dirname "$0")"

N_ITEMS="${N_ITEMS:-500}"
K="${K:-10000}"
SEED="${SEED:-42}"
WORKERS_LIST="${WORKERS_LIST:-2 4}"
THREADS_LIST="${THREADS_LIST:-1 2 4}"
MPI_RANKS_LIST="${MPI_RANKS_LIST:-2 4}"
DATA_DIR="${DATA_DIR:-data}"
RAW="results/benchmark_raw.csv"
OUT="results/benchmark.csv"
mkdir -p results results/plots

# 1. Generar datos sintéticos
python data/generate_data.py --n-items "$N_ITEMS" --seed "$SEED" --out-dir "$DATA_DIR"

# 2. Benchmark header
echo "implementation,parallel_units,n_items,k,time_sec,auc,consistency,w1,w2,w3,seed" > "$RAW"

# 3. Python secuencial
python python/sequential.py --k "$K" --seed "$SEED" --data-dir "$DATA_DIR" --csv >> "$RAW"

# 4. Python multi-core
for W in $WORKERS_LIST; do
  python python/multicore.py --k "$K" --seed "$SEED" --workers "$W" --data-dir "$DATA_DIR" --csv >> "$RAW"
done

# 5. C OpenMP
if command -v gcc >/dev/null 2>&1 && make -C C_OpenMP_MPI scoring_openmp >/dev/null 2>&1; then
  for T in $THREADS_LIST; do
    ./C_OpenMP_MPI/scoring_openmp --k "$K" --seed "$SEED" --threads "$T" --data-dir "$DATA_DIR" >> "$RAW"
  done
else
  echo "[WARN] OpenMP no disponible; omitido." >&2
fi

# 6. C MPI
if command -v mpicc >/dev/null 2>&1 && command -v mpirun >/dev/null 2>&1 && make -C C_OpenMP_MPI scoring_mpi >/dev/null 2>&1; then
  for R in $MPI_RANKS_LIST; do
    mpirun --allow-run-as-root -np "$R" ./C_OpenMP_MPI/scoring_mpi --k "$K" --seed "$SEED" --data-dir "$DATA_DIR" >> "$RAW" || true
  done
else
  echo "[WARN] MPI no disponible; omitido." >&2
fi

# 7. CUDA C
if command -v nvcc >/dev/null 2>&1 && make -C CUDA scoring_cuda >/dev/null 2>&1; then
  ./CUDA/scoring_cuda --k "$K" --seed "$SEED" --data-dir "$DATA_DIR" >> "$RAW" || true
else
  echo "[WARN] CUDA C no disponible; omitido." >&2
fi

# 8. PyCUDA
if python -c 'import pycuda.autoinit' >/dev/null 2>&1; then
  python CUDA/scoring_pycuda.py --k "$K" --seed "$SEED" --data-dir "$DATA_DIR" --csv >> "$RAW" || true
else
  echo "[WARN] PyCUDA no disponible; omitido." >&2
fi

# 9. Post-process (speedup, efficiency)
python scripts/postprocess_benchmark.py --input "$RAW" --output "$OUT"
echo "Benchmark consolidado: $OUT"
