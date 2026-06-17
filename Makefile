SHELL := /bin/bash

# ── Parámetros (override: make c-openmp K=500 THREADS=2 SEARCH=random) ──
K            ?= 10000
SEED         ?= 42
SEARCH       ?= random
STEP         ?= 0.02
THREADS      ?= 4
MPI_RANKS    ?= 4
WORKERS      ?= 4
DATA_DIR     ?=
DATASET_SIZE ?= 2000
WORKERS_LIST ?= 2 4
THREADS_LIST ?= 1 2 4
MPI_RANKS_LIST ?= 2 4

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

.PHONY: help data data-100 data-2000 \
        python-sequential python-multicore python-pycuda \
        c-sequential c-openmp c-mpi c-cuda \
        c cuda benchmark plots clean

help:
	@echo "Build:"
	@echo "  make c          -> compilar C secuencial + OpenMP + MPI"
	@echo "  make cuda       -> compilar CUDA C"
	@echo ""
	@echo "Run (no compila; usa 'make c' antes si hace falta):"
	@echo "  make python-sequential   Python secuencial"
	@echo "  make python-multicore    Python multicore  (WORKERS=$(WORKERS))"
	@echo "  make python-pycuda       PyCUDA"
	@echo "  make c-sequential        C secuencial"
	@echo "  make c-openmp            C OpenMP          (THREADS=$(THREADS))"
	@echo "  make c-mpi               C MPI             (MPI_RANKS=$(MPI_RANKS))"
	@echo "  make c-cuda              CUDA C"
	@echo ""
	@echo "Vars: K, SEED, SEARCH, STEP, DATA_DIR, THREADS, MPI_RANKS, WORKERS"
	@echo "  make c-openmp K=500 THREADS=2 SEARCH=random"
	@echo ""
	@echo "  make data | make benchmark | make plots | make clean"

data-100:
	python data/scripts/generate_dataset.py --name synthetic_CRC100x500_balanced \
		--n-eval 100 --n-ref 200 --seed $(SEED) --allow-small

data-2000: data

data:
	python data/scripts/generate_dataset.py --seed $(SEED)

# ── Python ──────────────────────────────────────────────────────────

python-sequential:
	@echo ">> python-sequential: K=$(K) search=$(SEARCH) data=$(NORM_DATA_DIR)"
	python python/sequential.py $(RUN_ARGS)

python-multicore:
	@echo ">> python-multicore: K=$(K) workers=$(WORKERS) search=$(SEARCH) data=$(NORM_DATA_DIR)"
	python python/multicore.py $(RUN_ARGS) --workers $(WORKERS)

python-pycuda:
	@echo ">> python-pycuda: K=$(K) data=$(NORM_DATA_DIR)"
	python CUDA/scoring_pycuda.py --k $(K) --seed $(SEED) --data-dir $(NORM_DATA_DIR)

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

c-cuda:
	@test -x CUDA/scoring_cuda || { echo "ERROR: binario no encontrado — ejecuta 'make cuda'"; exit 1; }
	@echo ">> c-cuda: K=$(K) data=$(NORM_DATA_DIR)"
	./CUDA/scoring_cuda --k $(K) --seed $(SEED) --data-dir $(NORM_DATA_DIR)

# ── Build ───────────────────────────────────────────────────────────

c:
	$(MAKE) -C C_OpenMP_MPI all

cuda:
	$(MAKE) -C CUDA all

# ── Benchmark completo ──────────────────────────────────────────────

benchmark:
	@set -euo pipefail; \
	RAW=results/benchmark_raw.csv; \
	OUT=results/benchmark.csv; \
	mkdir -p results results/plots; \
	if [ "$(DATASET_SIZE)" = "100" ] && [ ! -f "$(NORM_DATA_DIR)/dataset_manifest.json" ]; then \
	  $(MAKE) data-100 SEED=$(SEED); \
	elif [ "$(DATASET_SIZE)" = "2000" ] && [ ! -f "$(NORM_DATA_DIR)/dataset_manifest.json" ]; then \
	  $(MAKE) data SEED=$(SEED); \
	fi; \
	if command -v gcc >/dev/null 2>&1; then $(MAKE) c; fi; \
	echo "implementation,parallel_units,n_items,k,time_sec,auc,consistency,w1,w2,w3,seed,search_mode,iterations_until_best" > "$$RAW"; \
	ARGS="--k $(K) --seed $(SEED) --data-dir $(NORM_DATA_DIR) --step $(STEP)"; \
	for mode in random grid hybrid; do \
	  python python/sequential.py $$ARGS --search $$mode --csv >> "$$RAW"; \
	done; \
	for w in $(WORKERS_LIST); do \
	  for mode in random grid hybrid; do \
	    python python/multicore.py $$ARGS --search $$mode --workers $$w --csv >> "$$RAW"; \
	  done; \
	done; \
	if [ -x C_OpenMP_MPI/scoring_openmp ]; then \
	  for t in $(THREADS_LIST); do \
	    for mode in random grid hybrid; do \
	      ./C_OpenMP_MPI/scoring_openmp $$ARGS --search $$mode --threads $$t >> "$$RAW"; \
	    done; \
	  done; \
	else echo "[WARN] OpenMP omitido." >&2; fi; \
	if [ -x C_OpenMP_MPI/scoring_mpi ] && command -v mpirun >/dev/null 2>&1; then \
	  for r in $(MPI_RANKS_LIST); do \
	    for mode in random grid hybrid; do \
	      mpirun --allow-run-as-root -np $$r ./C_OpenMP_MPI/scoring_mpi $$ARGS --search $$mode >> "$$RAW" || true; \
	    done; \
	  done; \
	else echo "[WARN] MPI omitido." >&2; fi; \
	if command -v nvcc >/dev/null 2>&1 && $(MAKE) -C CUDA scoring_cuda >/dev/null 2>&1; then \
	  ./CUDA/scoring_cuda --k $(K) --seed $(SEED) --data-dir $(NORM_DATA_DIR) >> "$$RAW" || true; \
	else echo "[WARN] CUDA C omitido." >&2; fi; \
	if python -c 'import pycuda.autoinit' >/dev/null 2>&1; then \
	  python CUDA/scoring_pycuda.py --k $(K) --seed $(SEED) --data-dir $(NORM_DATA_DIR) --csv >> "$$RAW" || true; \
	else echo "[WARN] PyCUDA omitido." >&2; fi; \
	python scripts/postprocess_benchmark.py --input "$$RAW" --output "$$OUT"; \
	echo "Benchmark consolidado: $$OUT"

plots:
	python scripts/plot_benchmark.py --input results/benchmark.csv --out-dir results/plots

clean:
	$(MAKE) -C C_OpenMP_MPI clean || true
	$(MAKE) -C CUDA clean || true
	rm -f results/benchmark.csv results/benchmark_raw.csv results/plots/*.png
