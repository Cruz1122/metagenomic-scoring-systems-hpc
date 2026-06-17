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
# Dataset: 100 | 2000 (default)
DATASET_SIZE="${DATASET_SIZE:-2000}"
if [ "$DATASET_SIZE" = "100" ]; then
    DATA_DIR="${DATA_DIR:-data/processed/synthetic_CRC100x500_balanced}"
elif [ "$DATASET_SIZE" = "2000" ]; then
    DATA_DIR="${DATA_DIR:-data}"
else
    echo "ERROR: DATASET_SIZE must be 100 or 2000" >&2
    exit 1
fi
RAW="results/benchmark_raw.csv"
OUT="results/benchmark.csv"
mkdir -p results results/plots

# 1. Generar datos sintéticos (si no existen)
if [ "$DATASET_SIZE" = "100" ] && [ ! -f "$DATA_DIR/dataset_manifest.json" ]; then
    python data/scripts/generate_dataset.py --name "synthetic_CRC100x500_balanced" \
        --n-eval 100 --n-ref 200 --seed "$SEED" --allow-small
elif [ "$DATASET_SIZE" = "2000" ] && [ ! -f "data/processed/synthetic_CRC2000x500_balanced/dataset_manifest.json" ]; then
    python data/scripts/generate_dataset.py --name "synthetic_CRC2000x500_balanced" \
        --n-eval 2000 --n-ref 1000 --seed "$SEED"
fi

# 2. Benchmark header
echo "implementation,parallel_units,n_items,k,time_sec,auc,consistency,w1,w2,w3,seed,search_mode,iterations_until_best" > "$RAW"

# 3. Python secuencial — random baseline
python python/sequential.py --k "$K" --seed "$SEED" --data-dir "$DATA_DIR" --search random --csv >> "$RAW"

# 3b. Python secuencial — grid search
python python/sequential.py --k "$K" --seed "$SEED" --data-dir "$DATA_DIR" --search grid --csv >> "$RAW"

# 3c. Python secuencial — hybrid search
python python/sequential.py --k "$K" --seed "$SEED" --data-dir "$DATA_DIR" --search hybrid --csv >> "$RAW"

# 4. Python multi-core — random search
for W in $WORKERS_LIST; do
  python python/multicore.py --k "$K" --seed "$SEED" --workers "$W" --data-dir "$DATA_DIR" --csv >> "$RAW"
done

# 4b. Python multi-core — grid search
for W in $WORKERS_LIST; do
  python python/multicore.py --k "$K" --seed "$SEED" --workers "$W" --data-dir "$DATA_DIR" --search grid --csv >> "$RAW"
done

# 4c. Python multi-core — hybrid search
for W in $WORKERS_LIST; do
  python python/multicore.py --k "$K" --seed "$SEED" --workers "$W" --data-dir "$DATA_DIR" --search hybrid --csv >> "$RAW"
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
