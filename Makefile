SHELL := /bin/bash
N_ITEMS ?= 500
SEED ?= 42
K ?= 10000
WORKERS ?= 4
THETA ?=
THREADS ?= 4
MPI_RANKS ?= 4
DATA_DIR ?= data

.PHONY: help data python-seq python-mp python-mp-grid python-mp-hybrid c openmp-run mpi-run cuda cuda-run pycuda-run benchmark plots clean

help:
	@echo "make data                -> python data/scripts/generate_data.py"
	@echo "make python-seq K=10000"
	@echo "make python-mp K=10000 WORKERS=4"
	@echo "make python-mp-grid K=10000 WORKERS=4"
	@echo "make python-mp-hybrid K=5000 WORKERS=4"
	@echo "make c | make openmp-run | make mpi-run"
	@echo "make cuda | make cuda-run | make pycuda-run"
	@echo "make benchmark | make plots | make clean"

data:
	python data/scripts/generate_data.py --seed $(SEED)

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
	rm -rf data/npy/ data/csv/ data/dataset_manifest.json
