# C_OpenMP_MPI — builds sequential, OpenMP, and MPI scoring binaries.
# Flags can be overridden from the root Makefile:
#   make -C C_OpenMP_MPI all CFLAGS="-O3 -g" OPENMPFLAGS="-fopenmp"

CC          ?= gcc
MPICC       ?= mpicc
CFLAGS      ?= -O3 -std=c11 -Wall -Wextra
LDFLAGS     ?= -lm
OPENMPFLAGS ?= -fopenmp

.PHONY: all clean mpi sequential openmp

all: scoring_sequential scoring_openmp scoring_mpi

SHARED_SRC = shared/common.c shared/ziggurat.c shared/logger.c shared/rng.c
SHARED_HDR = shared/common.h shared/ziggurat.h shared/logger.h shared/rng.h

scoring_sequential: scoring_sequential.c $(SHARED_SRC) $(SHARED_HDR)
	$(CC) $(CFLAGS) scoring_sequential.c $(SHARED_SRC) -o scoring_sequential $(LDFLAGS)

scoring_openmp: scoring_openmp.c $(SHARED_SRC) $(SHARED_HDR)
	$(CC) $(CFLAGS) $(OPENMPFLAGS) scoring_openmp.c $(SHARED_SRC) -o scoring_openmp $(LDFLAGS)

scoring_mpi: scoring_mpi.c $(SHARED_SRC) $(SHARED_HDR)
	@echo "  [MPI] Compilando scoring_mpi..."
	$(MPICC) $(CFLAGS) scoring_mpi.c $(SHARED_SRC) -o scoring_mpi $(LDFLAGS) || \
		(echo "  [SKIP] mpicc no disponible" && false)

sequential: scoring_sequential
openmp: scoring_openmp
mpi: scoring_mpi

clean:
	rm -f scoring_openmp scoring_sequential scoring_mpi *.o
