SHELL := /bin/bash

# ── Parámetros (override: make c-openmp K=500 THREADS=2 SEARCH=random) ──
K            ?= 10000
SEED         ?= 42
SEARCH       ?= random
STEP         ?= 0.02
WORKERS      ?= 4
THREADS      ?= 4
MPI_RANKS    ?= $(WORKERS)
PYTHON       ?= .venv/bin/python
DATA_DIR     ?= data/processed/synthetic_CRC2000x10000_balanced
DATASET_SIZE ?= 2000
BENCHMARK_OUT ?= results/benchmark.csv

ifeq ($(DATA_DIR),)
  ifeq ($(DATASET_SIZE),100)
    DATA_DIR := data/processed/synthetic_CRC100x500_balanced
  else
    DATA_DIR := data/processed/synthetic_CRC2000x500_balanced
  endif
endif
# Normalizar slash final (también con override CLI: make DATA_DIR=foo/)
NORM_DATA_DIR = $(shell echo "$(DATA_DIR)" | sed 's|/*$$||')

# Args compartidos
RUN_ARGS = --k $(K) --seed $(SEED) --data-dir $(NORM_DATA_DIR) --search $(SEARCH) --step $(STEP)

# ── Flags de compilación (solo targets build) ──
export CC          ?= gcc
export MPICC       ?= mpicc
export CFLAGS      ?= -O3 -std=c11 -Wall -Wextra
export LDFLAGS     ?= -lm
export OPENMPFLAGS ?= -fopenmp

# CUDA toolkit (PyCUDA JIT)
CUDA_HOME ?= $(shell \
  if command -v nvcc >/dev/null 2>&1; then dirname "$$(dirname "$$(command -v nvcc)")"; \
  elif [ -x /opt/cuda/bin/nvcc ]; then echo /opt/cuda; \
  elif [ -x /usr/local/cuda/bin/nvcc ]; then echo /usr/local/cuda; \
  else echo ""; fi)
CUDA_LIB_DIR = $(CUDA_HOME)/targets/x86_64-linux/lib
CUDA_ENV = CUDA_HOME="$(CUDA_HOME)" PATH="$(CUDA_HOME)/bin:$$PATH" LD_LIBRARY_PATH="$(CUDA_LIB_DIR):$(CUDA_HOME)/lib64:$${LD_LIBRARY_PATH:-}"

.PHONY: help data data-100 data-2000 \
        python-sequential python-multicore python-pycuda python-pycuda-fast \
        python-sequential-benchmark python-multicore-benchmark python-pycuda-benchmark \
        c-sequential c-openmp c-mpi \
        c-sequential-benchmark c-openmp-benchmark c-mpi-benchmark \
        c benchmark plots test-args clean

help:
	@echo "Build:"
	@echo "  make c          -> compilar C secuencial + OpenMP + MPI"
	@echo ""
	@echo "Run (no compila; usa 'make c' antes si hace falta):"
	@echo "  make python-sequential   Python secuencial"
	@echo "  make python-multicore    Python multicore  (WORKERS=$(WORKERS))"
	@echo "  make python-pycuda       PyCUDA"
	@echo "  make python-pycuda-fast  PyCUDA (--fast, sin logging en vivo)"
	@echo "  make c-sequential        C secuencial"
	@echo "  make c-openmp            C OpenMP          (THREADS=$(THREADS))"
	@echo "  make c-mpi               C MPI             (MPI_RANKS=$(MPI_RANKS))"
	@echo ""
	@echo "Benchmark (sin logging, salida CSV):"
	@echo "  make python-sequential-benchmark | python-multicore-benchmark | python-pycuda-benchmark"
	@echo "  make c-sequential-benchmark | c-openmp-benchmark | c-mpi-benchmark"
	@echo "  make benchmark  -> una corrida por implementación (SEARCH=$(SEARCH), K=$(K))"
	@echo ""
	@echo "Vars: K, SEED, SEARCH, STEP, DATA_DIR, THREADS, MPI_RANKS, WORKERS"
	@echo "  BENCHMARK_OUT (solo make benchmark)"
	@echo "  make benchmark K=\"5000 10000 20000\" SEARCH=random"
	@echo "  make c-mpi WORKERS=3   (MPI_RANKS hereda WORKERS si no se pasa)"
	@echo ""
	@echo "  make data | make test-args | make benchmark | make benchmark-all | make plots | make clean"

data:
	$(PYTHON) data/scripts/generate_dataset.py \
	  --name synthetic_CRC2000x10000_balanced \
	  --n-eval 2000 \
	  --n-ref 1000 \
	  --n-items 10000 \
	  --seed $(SEED) \
	  --quick-k 500

# ── Python ──────────────────────────────────────────────────────────

python-sequential:
	@echo ">> python-sequential: K=$(K) search=$(SEARCH) data=$(NORM_DATA_DIR)"
	$(PYTHON) python/sequential.py $(RUN_ARGS)

python-multicore:
	@echo ">> python-multicore: K=$(K) workers=$(WORKERS) search=$(SEARCH) data=$(NORM_DATA_DIR)"
	$(PYTHON) python/multicore.py $(RUN_ARGS) --workers $(WORKERS)

python-pycuda:
	@test -n "$(CUDA_HOME)" || { echo "ERROR: nvcc no encontrado — instala CUDA toolkit (/opt/cuda)"; exit 1; }
	@echo ">> python-pycuda: K=$(K) search=$(SEARCH) data=$(NORM_DATA_DIR)"
	$(CUDA_ENV) $(PYTHON) CUDA/scoring_pycuda.py $(RUN_ARGS)

python-pycuda-fast:
	@test -n "$(CUDA_HOME)" || { echo "ERROR: nvcc no encontrado — instala CUDA toolkit (/opt/cuda)"; exit 1; }
	@echo ">> python-pycuda-fast: K=$(K) search=$(SEARCH) data=$(NORM_DATA_DIR)"
	$(CUDA_ENV) $(PYTHON) CUDA/scoring_pycuda.py $(RUN_ARGS) --fast

python-sequential-benchmark:
	@echo ">> python-sequential-benchmark: K=$(K) search=$(SEARCH) data=$(NORM_DATA_DIR)"
	$(PYTHON) python/sequential.py $(RUN_ARGS) --benchmark

python-multicore-benchmark:
	@echo ">> python-multicore-benchmark: K=$(K) workers=$(WORKERS) search=$(SEARCH) data=$(NORM_DATA_DIR)"
	$(PYTHON) python/multicore.py $(RUN_ARGS) --workers $(WORKERS) --benchmark

python-pycuda-benchmark:
	@test -n "$(CUDA_HOME)" || { echo "ERROR: nvcc no encontrado — instala CUDA toolkit (/opt/cuda)"; exit 1; }
	@echo ">> python-pycuda-benchmark: K=$(K) search=$(SEARCH) data=$(NORM_DATA_DIR)"
	$(CUDA_ENV) $(PYTHON) CUDA/scoring_pycuda.py $(RUN_ARGS) --benchmark

# ── C ───────────────────────────────────────────────────────────────

c-sequential:
	@test -x C_OpenMP_MPI/scoring_sequential || { echo "ERROR: binario no encontrado — ejecuta 'make c'"; exit 1; }
	@echo ">> c-sequential: K=$(K) seed=$(SEED) data=$(NORM_DATA_DIR)"
	./C_OpenMP_MPI/scoring_sequential --k $(K) --seed $(SEED) --data-dir $(NORM_DATA_DIR)

c-openmp:
	@test -x C_OpenMP_MPI/scoring_openmp || { echo "ERROR: binario no encontrado — ejecuta 'make c'"; exit 1; }
	@echo ">> c-openmp: K=$(K) threads=$(THREADS) search=$(SEARCH) data=$(NORM_DATA_DIR)"
	./C_OpenMP_MPI/scoring_openmp $(RUN_ARGS) --threads $(THREADS)

c-mpi:
	@test -x C_OpenMP_MPI/scoring_mpi || { echo "ERROR: binario no encontrado — ejecuta 'make c'"; exit 1; }
	@command -v mpirun >/dev/null || { echo "ERROR: mpirun no encontrado"; exit 1; }
	@echo ">> c-mpi: K=$(K) ranks=$(MPI_RANKS) search=$(SEARCH) data=$(NORM_DATA_DIR)"
	mpirun --allow-run-as-root -np $(MPI_RANKS) ./C_OpenMP_MPI/scoring_mpi $(RUN_ARGS)

c-sequential-benchmark:
	@test -x C_OpenMP_MPI/scoring_sequential || { echo "ERROR: binario no encontrado — ejecuta 'make c'"; exit 1; }
	@echo ">> c-sequential-benchmark: K=$(K) seed=$(SEED) data=$(NORM_DATA_DIR)"
	./C_OpenMP_MPI/scoring_sequential --k $(K) --seed $(SEED) --data-dir $(NORM_DATA_DIR) --benchmark

c-openmp-benchmark:
	@test -x C_OpenMP_MPI/scoring_openmp || { echo "ERROR: binario no encontrado — ejecuta 'make c'"; exit 1; }
	@echo ">> c-openmp-benchmark: K=$(K) threads=$(THREADS) search=$(SEARCH) data=$(NORM_DATA_DIR)"
	./C_OpenMP_MPI/scoring_openmp $(RUN_ARGS) --threads $(THREADS) --benchmark

c-mpi-benchmark:
	@test -x C_OpenMP_MPI/scoring_mpi || { echo "ERROR: binario no encontrado — ejecuta 'make c'"; exit 1; }
	@command -v mpirun >/dev/null || { echo "ERROR: mpirun no encontrado"; exit 1; }
	@echo ">> c-mpi-benchmark: K=$(K) ranks=$(MPI_RANKS) search=$(SEARCH) data=$(NORM_DATA_DIR)"
	mpirun --allow-run-as-root -np $(MPI_RANKS) ./C_OpenMP_MPI/scoring_mpi $(RUN_ARGS) --benchmark

# ── Build ───────────────────────────────────────────────────────────

c:
	$(MAKE) -C C_OpenMP_MPI all

# ── Benchmark completo (scripts/benchmark_pipeline.py) ───────────────

benchmark:
	@set -euo pipefail; \
	mkdir -p results; \
	if [ ! -f "$(NORM_DATA_DIR)/dataset_manifest.json" ] && \
	   [ ! -f "$(NORM_DATA_DIR)/npy/matrix_A.npy" ]; then \
	  echo ">> Generando dataset en $(NORM_DATA_DIR)..."; \
	  $(MAKE) data SEED=$(SEED); \
	fi; \
	if command -v gcc >/dev/null 2>&1; then $(MAKE) c; fi; \
	if [ -n "$(CUDA_HOME)" ]; then \
	  export CUDA_HOME="$(CUDA_HOME)"; \
	  export PATH="$(CUDA_HOME)/bin:$$PATH"; \
	  export LD_LIBRARY_PATH="$(CUDA_LIB_DIR):$(CUDA_HOME)/lib64:$${LD_LIBRARY_PATH:-}"; \
	fi; \
	$(PYTHON) scripts/benchmark_pipeline.py \
	  --all-strategies --search $(SEARCH) \
	  --k $(K) --seed $(SEED) --data-dir $(NORM_DATA_DIR) --step $(STEP) \
	  --output $(BENCHMARK_OUT); \
	echo "Benchmark consolidado: $(BENCHMARK_OUT)"

benchmark-all:
	$(PYTHON) scripts/benchmark_all.py --k-list $(K_LIST) --workers $(WORKERS) --data-dir $(NORM_DATA_DIR)

plots:
	@echo "ERROR: scripts/plot_benchmark.py no existe; genera gráficas manualmente desde $(BENCHMARK_OUT)" >&2; \
	exit 1

test-args:
	@bash scripts/test_args.sh

clean:
	$(MAKE) -C C_OpenMP_MPI clean || true
	rm -f results/benchmark.csv results/benchmark_raw.csv \
	      results/benchmark_pipeline.csv results/benchmark_smoke.csv \
	      results/benchmark_run.log results/plots/*.png
