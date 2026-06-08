CC ?= gcc
MPICC ?= mpicc
CFLAGS ?= -O3 -std=c11 -Wall -Wextra
OMPFLAGS ?= -fopenmp
LDFLAGS ?= -lm
.PHONY: all clean
all: scoring_openmp scoring_mpi
scoring_openmp: scoring_openmp.c
	$(CC) $(CFLAGS) $(OMPFLAGS) $< -o $@ $(LDFLAGS)
scoring_mpi: scoring_mpi.c
	$(MPICC) $(CFLAGS) $< -o $@ $(LDFLAGS)
clean:
	rm -f scoring_openmp scoring_mpi *.o
