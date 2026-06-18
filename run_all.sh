#!/usr/bin/env bash
# run_all.sh — Pipeline completa: genera datos, ejecuta implementaciones, CSV crudo.
set -euo pipefail
cd "$(dirname "$0")"

K="${K:-10000}"
K_LIST="${K_LIST:-$K}"
SEED="${SEED:-42}"
STEP="${STEP:-0.02}"
SEARCH_LIST="${SEARCH_LIST:-random}"
NPROC="${NPROC:-$(nproc)}"
DATASET_SIZE="${DATASET_SIZE:-2000}"
DATA_DIR="${DATA_DIR:-}"
PYTHON="${PYTHON:-.venv/bin/python}"
RAW="results/benchmark_raw.csv"
HEADER="implementation,parallel_units,n_items,k,time_sec,auc,consistency,w1,w2,w3,seed,search_mode,iterations_until_best"
CSV_RE='^(python_sequential|python_multicore|c_sequential|c_openmp|c_mpi|pycuda|cuda_c),'

if [ -z "$DATA_DIR" ]; then
  if [ "$DATASET_SIZE" = "100" ]; then
    DATA_DIR="data/processed/synthetic_CRC100x500_balanced"
  else
    DATA_DIR="data/processed/synthetic_CRC2000x10000_balanced"
  fi
fi
DATA_DIR="${DATA_DIR%/}"

mkdir -p results

append_benchmark() {
  local out err
  out=$(mktemp)
  err=$(mktemp)
  set +e
  "$@" >"$out" 2>"$err"
  local ec=$?
  set -e
  if grep -qE "$CSV_RE" "$out"; then
    grep -E "$CSV_RE" "$out" >>"$RAW"
  else
    echo "[WARN] Sin línea CSV (--benchmark?): $*" >&2
    [ -s "$out" ] && tail -5 "$out" >&2
    [ -s "$err" ] && tail -5 "$err" >&2
  fi
  rm -f "$out" "$err"
  return "$ec"
}

if [ ! -f "$DATA_DIR/dataset_manifest.json" ] && [ ! -f "$DATA_DIR/npy/matrix_A.npy" ]; then
  echo ">> Generando dataset en $DATA_DIR..."
  "$PYTHON" data/scripts/generate_dataset.py \
    --name synthetic_CRC2000x10000_balanced \
    --n-eval 2000 --n-ref 1000 --n-items 10000 \
    --seed "$SEED" --quick-k 500
fi

if command -v gcc >/dev/null 2>&1; then
  make -C C_OpenMP_MPI all
fi

echo "$HEADER" >"$RAW"
echo ">> Paralelismo máximo: NPROC=$NPROC" >&2

for K in $K_LIST; do
  echo ">> K=$K search=$SEARCH_LIST" >&2
  COMMON=(--k "$K" --seed "$SEED" --data-dir "$DATA_DIR" --step "$STEP")

  for mode in $SEARCH_LIST; do
    append_benchmark "$PYTHON" python/sequential.py "${COMMON[@]}" --search "$mode" --benchmark || true

    append_benchmark "$PYTHON" python/multicore.py "${COMMON[@]}" --workers "$NPROC" --search "$mode" --benchmark || true

    if [ "$mode" = "random" ] && [ -x C_OpenMP_MPI/scoring_sequential ]; then
      append_benchmark ./C_OpenMP_MPI/scoring_sequential --k "$K" --seed "$SEED" --data-dir "$DATA_DIR" --benchmark || true
    fi

    if [ -x C_OpenMP_MPI/scoring_openmp ]; then
      append_benchmark ./C_OpenMP_MPI/scoring_openmp "${COMMON[@]}" --threads "$NPROC" --search "$mode" --benchmark || true
    fi

    if [ -x C_OpenMP_MPI/scoring_mpi ] && command -v mpirun >/dev/null 2>&1; then
      append_benchmark mpirun --allow-run-as-root --oversubscribe -np "$NPROC" ./C_OpenMP_MPI/scoring_mpi \
        "${COMMON[@]}" --search "$mode" --benchmark || true
    fi

    if "$PYTHON" -c 'import pycuda.autoinit' >/dev/null 2>&1; then
      append_benchmark "$PYTHON" CUDA/scoring_pycuda.py "${COMMON[@]}" --search "$mode" --benchmark || true
    fi
  done
done

if ! "$PYTHON" -c 'import pycuda.autoinit' >/dev/null 2>&1; then
  echo "[WARN] PyCUDA omitido." >&2
fi

echo "Benchmark crudo: $RAW"
"$PYTHON" scripts/validate_benchmark_csv.py --input "$RAW"
