SHELL := /bin/bash
N_ITEMS ?= 500
SEED ?= 42
K ?= 20000
WORKERS ?= 4
THETA ?=
THREADS ?= 4
MPI_RANKS ?= 4
DATASET_SIZE ?= 2000
DATA_DIR ?= $(shell if [ "$(DATASET_SIZE)" = "100" ]; then echo "data/processed/synthetic_CRC100x500_balanced"; else echo "data"; fi)

.PHONY: help data python-seq python-mp python-mp-grid python-mp-hybrid c openmp-run mpi-run cuda cuda-run pycuda-run benchmark plots clean

help:
	@echo "make data | make data-100   -> generate datasets (2000 | 100 samples)"
	@echo "make python-seq K=10000     -> sequential random (default)"
	@echo "make python-seq-grid        -> sequential grid search"
	@echo "make python-seq-hybrid      -> sequential hybrid search"
	@echo "make python-mp K=10000 WORKERS=4   -> multicore random"
	@echo "make python-mp-grid         -> multicore grid search"
	@echo "make python-mp-hybrid       -> multicore hybrid search"
	@echo "make c                      -> compile all C binaries"
	@echo "make seq-run                -> C sequential baseline"
	@echo "make openmp-run             -> C/OpenMP (set OMP_NUM_THREADS)"
	@echo "make mpi-run                -> C/MPI (scaffold)"
	@echo "make cuda | make cuda-run   -> CUDA C (scaffold)"
	@echo "make pycuda-run             -> PyCUDA (scaffold)"
	@echo "make benchmark | make plots | make clean"
	@echo ""
	@echo "Dataset selection: DATASET_SIZE=100 (100 samples) or 2000 (default)"

data-100:
	python data/scripts/generate_dataset.py --name synthetic_CRC100x500_balanced \
		--n-eval 100 --n-ref 200 --seed $(SEED) --allow-small

data-2000: data

data:
	python data/scripts/generate_dataset.py --seed $(SEED)

python-seq:
	python python/sequential.py --k $(K) --seed $(SEED) --data-dir $(DATA_DIR) --search random

python-seq-grid:
	python python/sequential.py --k $(K) --seed $(SEED) --data-dir $(DATA_DIR) --search grid

python-seq-hybrid:
	python python/sequential.py --k $(K) --seed $(SEED) --data-dir $(DATA_DIR) --search hybrid

python-mp:
	python python/multicore.py --k $(K) --seed $(SEED) --workers $(WORKERS) --data-dir $(DATA_DIR) $(if $(THETA),--theta $(THETA))

python-mp-grid:
	python python/multicore.py --k $(K) --seed $(SEED) --workers $(WORKERS) --data-dir $(DATA_DIR) --search grid $(if $(THETA),--theta $(THETA))

python-mp-hybrid:
	python python/multicore.py --k $(K) --seed $(SEED) --workers $(WORKERS) --data-dir $(DATA_DIR) --search hybrid $(if $(THETA),--theta $(THETA))

c:
	$(MAKE) -C C_OpenMP_MPI all

seq-run: c
	./C_OpenMP_MPI/scoring_sequential --k $(K) --seed $(SEED) --data-dir $(DATA_DIR)

openmp-run: c
	./C_OpenMP_MPI/scoring_openmp --k $(K) --seed $(SEED) --data-dir $(DATA_DIR)

mpi-run: c
	mpirun -np $(MPI_RANKS) ./C_OpenMP_MPI/scoring_mpi --k $(K) --seed $(SEED) --data-dir $(DATA_DIR)

cuda:
	$(MAKE) -C CUDA all

cuda-run: cuda
	./CUDA/scoring_cuda --k $(K) --seed $(SEED) --data-dir $(DATA_DIR)

pycuda-run:
	python CUDA/scoring_pycuda.py --k $(K) --seed $(SEED) --data-dir $(DATA_DIR)

benchmark:
	./run_all.sh

plots:
	python scripts/plot_benchmark.py --input results/benchmark.csv --out-dir results/plots

clean:
	$(MAKE) -C C_OpenMP_MPI clean || true
	$(MAKE) -C CUDA clean || true
	rm -f results/benchmark.csv results/benchmark_raw.csv results/plots/*.png
	# Nota: data/npy/, data/csv/, data/dataset_manifest.json son symlinks,
	# no se borran. Para regenerar datasets usa 'make data'.
