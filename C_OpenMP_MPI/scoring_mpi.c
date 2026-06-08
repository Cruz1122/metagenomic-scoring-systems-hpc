/**
 * scoring_mpi.c — Búsqueda aleatoria con MPI.
 *
 * Compilar:
 *   make -C C_OpenMP_MPI scoring_mpi
 *
 * Ejecutar:
 *   mpirun -np 4 ./C_OpenMP_MPI/scoring_mpi --k 10000 --seed 42 --data-dir data
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <mpi.h>

/** Mejor resultado encontrado (AUC, consistencia, pesos). */
typedef struct { double auc, cons, w[3]; } Best;

/* ------------------------------------------------------------------ */
/*  Utilidades básicas (RNG, argumentos, CSV, evaluación)             */
/* ------------------------------------------------------------------ */

static uint64_t xs(uint64_t *s) {
    /* TODO: xorshift RNG */
    return *s;
}

static double u01(uint64_t *s) {
    /* TODO: uniforme [0,1) */
    return 0.0;
}

static void simplex(uint64_t *s, double w[3]) {
    /* TODO: muestra Dirichlet(1,1,1) */
}

static const char* arg(int argc, char **argv, const char *n, const char *d) {
    /* TODO: extraer argumento --name valor */
    return d;
}

static void path(char *out, size_t cap, const char *d, const char *f) {
    /* TODO: construir ruta d/f */
}

static int lines(const char *p) {
    /* TODO: contar líneas de archivo CSV */
    return -1;
}

static int cols(const char *p) {
    /* TODO: contar columnas de archivo CSV */
    return -1;
}

static double* matrix(const char *p, int *r, int *c) {
    /* TODO: leer matriz CSV a double* */
    return NULL;
}

static int* labels(const char *p, int *n) {
    /* TODO: leer etiquetas CSV a int* */
    return NULL;
}

static double auc10(const double s[10], const int y[10]) {
    /* TODO: AUC para 10 muestras */
    return 0.0;
}

static double cons10(const double s[10], const int y[10]) {
    /* TODO: consistencia para 10 muestras */
    return 0.0;
}

static void eval(const double *A, const double *P, const int *y,
                 int N, const double w[3],
                 double *out_auc, double *out_cons) {
    /* TODO: calcular scores = A @ (P @ w) y evaluar */
}

/* ------------------------------------------------------------------ */
/*  Main MPI                                                           */
/* ------------------------------------------------------------------ */

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    long K = atol(arg(argc, argv, "--k", "10000"));
    int seed = atoi(arg(argc, argv, "--seed", "42"));
    const char *d = arg(argc, argv, "--data-dir", "data");

    /* TODO: rank 0 carga datos, scatter candidatos, reduce mejor, print */
    fprintf(stderr, "MPI scaffold — rank %d/%d, implementar búsqueda\n", rank, size);

    MPI_Finalize();
    return 1;
}
