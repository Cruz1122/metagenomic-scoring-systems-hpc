/**
 * scoring_mpi.c — Búsqueda aleatoria con MPI.
 *
 * Reutiliza shared/common.h, shared/rng.h, shared/logger.h
 * en vez de implementar sus propios stubs.
 *
 * Compilar:
 *   make -C C_OpenMP_MPI scoring_mpi
 *
 * Ejecutar:
 *   mpirun -np 4 ./C_OpenMP_MPI/scoring_mpi --k 10000 --seed 42 --data-dir data
 */
#define _GNU_SOURCE
#include "shared/common.h"
#include "shared/rng.h"
#include "shared/logger.h"
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static const char* arg(int argc, char **argv, const char *name, const char *fallback) {
    for (int i = 1; i < argc - 1; i++)
        if (strcmp(argv[i], name) == 0)
            return argv[i + 1];
    return fallback;
}

/* ================================================================== */
/*  Main MPI                                                           */
/* ================================================================== */

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    long K   = atol(arg(argc, argv, "--k",    "10000"));
    int  seed = atoi(arg(argc, argv, "--seed", "42"));
    const char *data_dir = arg(argc, argv, "--data-dir", "data");

    /* TODO: implementar búsqueda MPI usando shared/common.h, shared/rng.h
     *
     * Pipeline sugerido:
     *   1. Rank 0: load_data(data_dir, &ds)
     *   2. Broadcast dimensiones + matrices (A, profiles, y) a todos los ranks
     *   3. Cada rank: RNG propio pcg64_seed(pcg, seed + rank)
     *      for i in chunk_local: simplex(pcg, w) → evaluate(...)
     *   4. MPI_Gather o MPI_Reduce de mejores locales → rank 0
     *   5. Rank 0: log_complete, salida parseable
     *   6. free_dataset en rank 0, free() en los demás
     */

    if (rank == 0) {
        Dataset ds;
        if (load_data(data_dir, &ds) != 0) {
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        log_header("c_mpi", ds.n_items, K);
        fprintf(stderr, "  [MPI scaffold] rank=%d/%d  K=%ld  seed=%d\n",
                rank, size, K, seed);
        log_complete("c_mpi", 0.0, 0.0, (double[3]){0,0,0}, 0.0);

        printf("implementation=c_mpi\n");
        printf("ranks=%d\n", size);
        printf("N=%d\n", ds.n_items);
        printf("K=%ld\n", K);
        printf("best_auc=%.6f\n", 0.0);
        printf("best_w=[0.0, 0.0, 0.0]\n");
        printf("best_w_sum=0.0\n");
        printf("time_sec=0.0\n");

        free_dataset(&ds);
    }

    MPI_Finalize();
    return 0;
}
