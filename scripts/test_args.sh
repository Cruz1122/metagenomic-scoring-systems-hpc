#!/usr/bin/env bash
# Verifica que todos los args CLI (K, seed, search, step, data-dir,
# threads/workers/ranks) funcionen en cada implementación.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

K=50
SEED=99
STEP=0.05
DATA_DIR="data/processed/synthetic_CRC100x500_balanced"
SEARCH_MODES=(random grid hybrid)
PASS=0
FAIL=0

# Activar venv (fish) si existe; fallback a .venv/bin/python
if [[ -f .venv/bin/activate.fish ]]; then
  PYTHON="$(fish -c "source .venv/bin/activate.fish; command -v python" 2>/dev/null || true)"
fi
PYTHON="${PYTHON:-.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON="$(command -v python3 || command -v python)"
fi

pass() { echo "  OK  $*"; PASS=$((PASS + 1)); }
fail() { echo "  FAIL $*" >&2; FAIL=$((FAIL + 1)); }

# Extrae clave=valor de la salida (última ocurrencia)
parse_out() {
  local key="$1" out="$2"
  echo "$out" | grep -E "^${key}=" | tail -1 | cut -d= -f2-
}

check_kv() {
  local label="$1" out="$2" key="$3" expected="$4"
  local got
  got="$(parse_out "$key" "$out")"
  if [[ "$got" == "$expected" ]]; then
    pass "$label: $key=$got"
  else
    fail "$label: esperado $key=$expected, obtuvo '$got'"
  fi
}

check_run() {
  local label="$1"
  shift
  local out
  if out="$("$@" 2>&1)"; then
    echo "$out" | grep -q "best_auc=" || { fail "$label: sin best_auc en salida"; return; }
    echo "$out"
  else
    fail "$label: exit code != 0"
    echo "$out" >&2
    return 1
  fi
}

echo "== test-args (K=$K SEED=$SEED STEP=$STEP python=$PYTHON) =="
echo ""

if [[ ! -f "$DATA_DIR/dataset_manifest.json" ]]; then
  echo "Generando dataset pequeño..."
  "$PYTHON" data/scripts/generate_dataset.py \
    --name synthetic_CRC100x500_balanced \
    --n-eval 100 --n-ref 200 --seed "$SEED" --allow-small
fi

if [[ ! -x C_OpenMP_MPI/scoring_sequential ]]; then
  echo "Compilando C..."
  make c
fi

ARGS=(--k "$K" --seed "$SEED" --data-dir "$DATA_DIR" --step "$STEP")

# ── Python secuencial ───────────────────────────────────────────────
for mode in "${SEARCH_MODES[@]}"; do
  label="python-sequential search=$mode"
  out="$(check_run "$label" "$PYTHON" python/sequential.py "${ARGS[@]}" --search "$mode")" || continue
  check_kv "$label" "$out" "implementation" "python_sequential"
  if [[ "$mode" == "grid" ]]; then
    local_k="$(parse_out K "$out")"
    [[ "$local_k" -gt 0 ]] && pass "$label: K=$local_k (grid points)" \
      || fail "$label: K grid inválido '$local_k'"
  else
    check_kv "$label" "$out" "K" "$K"
  fi
done

# ── Python multicore (workers) ──────────────────────────────────────
for w in 2 3; do
  for mode in "${SEARCH_MODES[@]}"; do
    label="python-multicore workers=$w search=$mode"
    out="$(check_run "$label" "$PYTHON" python/multicore.py "${ARGS[@]}" \
      --search "$mode" --workers "$w")" || continue
    check_kv "$label" "$out" "workers" "$w"
    if [[ "$mode" == "grid" ]]; then
      local_k="$(parse_out K "$out")"
      [[ "$local_k" -gt 0 ]] && pass "$label: K=$local_k (grid points)" \
        || fail "$label: K grid inválido '$local_k'"
    else
      check_kv "$label" "$out" "K" "$K"
    fi
  done
done

# ── C secuencial ────────────────────────────────────────────────────
label="c-sequential"
out="$(check_run "$label" ./C_OpenMP_MPI/scoring_sequential \
  --k "$K" --seed "$SEED" --data-dir "$DATA_DIR")" || true
if [[ -n "${out:-}" ]]; then
  check_kv "$label" "$out" "K" "$K"
  check_kv "$label" "$out" "implementation" "c_sequential"
fi

# ── C OpenMP (threads + search) ─────────────────────────────────────
for t in 1 2; do
  for mode in "${SEARCH_MODES[@]}"; do
    label="c-openmp threads=$t search=$mode"
    out="$(check_run "$label" ./C_OpenMP_MPI/scoring_openmp "${ARGS[@]}" \
      --search "$mode" --threads "$t")" || continue
    check_kv "$label" "$out" "workers" "$t"
    check_kv "$label" "$out" "search_mode" "$mode"
    if [[ "$mode" == "random" ]]; then
      check_kv "$label" "$out" "K" "$K"
    fi
  done
done

# ── C MPI (WORKERS via MPI_RANKS) ───────────────────────────────────
if command -v mpirun >/dev/null 2>&1; then
  for r in 2 3; do
    for mode in "${SEARCH_MODES[@]}"; do
      label="c-mpi ranks=$r search=$mode"
      out="$(check_run "$label" mpirun --allow-run-as-root -np "$r" \
        ./C_OpenMP_MPI/scoring_mpi "${ARGS[@]}" --search "$mode")" || continue
      check_kv "$label" "$out" "workers" "$r"
      check_kv "$label" "$out" "search_mode" "$mode"
    done
  done

  # Verificar alias Makefile: WORKERS=3 → ranks=3
  label="make c-mpi WORKERS=3"
  out="$(check_run "$label" make -s c-mpi K="$K" SEED="$SEED" SEARCH=random \
    STEP="$STEP" DATASET_SIZE=100 WORKERS=3 2>&1)" || true
  if [[ -n "${out:-}" ]]; then
    check_kv "$label" "$out" "workers" "3"
  fi
else
  echo "[SKIP] mpirun no disponible"
fi

echo ""
echo "== resumen: $PASS OK, $FAIL FAIL =="
[[ "$FAIL" -eq 0 ]]
