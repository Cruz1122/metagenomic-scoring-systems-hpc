/**
 * @file scoring_sequential.c
 * @brief Búsqueda aleatoria secuencial de pesos Dirichlet.
 *
 * Réplica de python/sequential.py.
 * SIN paralelización — implementación secuencial pura.
 *
 * ## Pipeline (idéntico a sequential.py::random_search)
 *   1. Carga A, profiles, y (delega en shared/common.c::load_data)
 *   2. Itera K veces:
 *      - w ~ Dirichlet(1,1,1) via shared/rng.c::simplex
 *      - scores = A @ (profiles @ w) via shared/common.c::evaluate
 *      - AUC y consistencia
 *      - Guarda el mejor
 *   3. Imprime resultado (logger ANSI + salida parseable)
 *
 * ## Dependencias
 *   shared/common.h/c  → load_data, evaluate, auc, consistency
 *   shared/rng.h/c     → pcg64_seed, xs, simplex (SeedSequence + PCG64 + Dirichlet)
 *   shared/ziggurat.h/c → standard_exponential
 *   shared/logger.h/c  → log_header, log_improvement, log_complete
 *
 * Compilar:
 *   make -C C_OpenMP_MPI scoring_sequential
 *
 * Uso:
 *   ./C_OpenMP_MPI/scoring_sequential --k 10000 --seed 42 --data-dir data
 */
#define _GNU_SOURCE
#include "shared/common.h"
#include "shared/rng.h"
#include "shared/logger.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

/* ================================================================== */
/*  Argumentos CLI — --nombre valor                                    */
/* ================================================================== */

/**
 * @brief Extrae el valor de un argumento tipo `--nombre valor`.
 *
 * @param argc  Cantidad de argumentos.
 * @param argv  Arreglo de argumentos.
 * @param name  Nombre del flag (ej. "--k").
 * @param fallback Valor por defecto.
 * @return      Valor asociado al flag, o fallback.
 */
static const char* arg(int argc, char **argv, const char *name, const char *fallback) {
    for (int i = 1; i < argc - 1; i++)
        if (strcmp(argv[i], name) == 0)
            return argv[i + 1];
    return fallback;
}

/* ================================================================== */
/*  Búsqueda aleatoria secuencial                                      */
/*  Réplica de python/sequential.py::random_search()                  */
/* ================================================================== */

/**
 * @brief Búsqueda aleatoria de pesos Dirichlet — versión secuencial.
 *
 * Itera K veces:
 *   - w ~ Dirichlet(1,1,1)
 *   - P = profiles @ w
 *   - scores = A @ P
 *   - AUC y consistencia
 *   - guarda el mejor (AUC primario)
 *
 * @param ds    Dataset cargado (A, profiles, y).
 * @param k     Número de candidatos a evaluar.
 * @param seed  Semilla RNG.
 * @return      Mejor resultado encontrado.
 */
static Best random_search(const Dataset *ds, long k, uint64_t seed, int verbose) {
    /* Inicializar RNG (SeedSequence + PCG64) */
    uint64_t pcg[4];
    pcg64_seed(pcg, seed);

    Best best = { .auc = -1.0, .cons = 0.0, .w = {0,0,0}, .iter = -1 };

    for (long i = 0; i < k; i++) {
        /* w ~ Dirichlet(1,1,1) */
        double w[3];
        simplex(pcg, w);

        /* Evaluar */
        double auc_val, cons_val;
        evaluate(ds->A, ds->n_samples, ds->n_items,
                 ds->profiles, ds->y, w,
                 &auc_val, &cons_val);

        /* Guardar si mejora AUC */
        if (auc_val > best.auc) {
            double prev = best.auc;
            best.auc  = auc_val;
            best.cons = cons_val;
            best.w[0] = w[0];
            best.w[1] = w[1];
            best.w[2] = w[2];
            best.iter = i;
            if (verbose)
                log_improvement(i, k, auc_val, prev, cons_val, w, -1);
        }
    }

    return best;
}

/* ================================================================== */
/*  grid_search — secuencial                                           */
/* ================================================================== */

static Best grid_search(const Dataset *ds, double step, int verbose, long *out_total)
{
    long total = 0;
    for (double w1 = 0.0; w1 <= 1.0 + 1e-12; w1 += step)
        for (double w2 = 0.0; w2 <= 1.0 - w1 + 1e-12; w2 += step)
            total++;
    *out_total = total;

    Best best = { .auc = -1.0, .cons = 0.0, .w = {0,0,0}, .iter = -1 };

    long idx = 0;
    for (double w1 = 0.0; w1 <= 1.0 + 1e-12; w1 += step) {
        for (double w2 = 0.0; w2 <= 1.0 - w1 + 1e-12; w2 += step) {
            double w[3] = { w1, w2, 1.0 - w1 - w2 };
            double auc_val, cons_val;
            evaluate(ds->A, ds->n_samples, ds->n_items,
                     ds->profiles, ds->y, w, &auc_val, &cons_val);

            if (auc_val > best.auc) {
                double prev = best.auc;
                best.auc  = auc_val;
                best.cons = cons_val;
                best.w[0] = w[0];
                best.w[1] = w[1];
                best.w[2] = w[2];
                best.iter = idx;
                if (verbose)
                    log_improvement(idx, total, auc_val, prev, cons_val, w, -1);
            }
            idx++;
        }
    }

    return best;
}

/* ================================================================== */
/*  hybrid_search — secuencial                                         */
/* ================================================================== */

static Best hybrid_search(const Dataset *ds, long k, uint64_t seed,
                          double step, int verbose)
{
    long grid_total;
    Best best = grid_search(ds, step, verbose, &grid_total);

    long remaining = k - grid_total;
    if (remaining <= 0)
        return best;

    long random_n = remaining / 2;
    long local_n  = remaining - random_n;

    uint64_t pcg[4];
    pcg64_seed(pcg, seed);

    for (long i = 0; i < random_n; i++) {
        double w[3], auc_val, cons_val;
        simplex(pcg, w);
        evaluate(ds->A, ds->n_samples, ds->n_items,
                 ds->profiles, ds->y, w, &auc_val, &cons_val);

        if (auc_val > best.auc) {
            double prev = best.auc;
            best.auc  = auc_val;
            best.cons = cons_val;
            best.w[0] = w[0];
            best.w[1] = w[1];
            best.w[2] = w[2];
            best.iter = grid_total + i;
            if (verbose)
                log_improvement(grid_total + i, k, auc_val, prev, cons_val, w, -1);
        }
    }

    if (local_n > 0 && best.auc >= 0) {
        int    splits[2] = { (int)(local_n / 2), (int)(local_n - local_n / 2) };
        double concs[2]  = { 300.0, 1000.0 };
        long   base      = grid_total + random_n;

        for (int ph = 0; ph < 2; ph++) {
            int count = splits[ph];
            if (count <= 0)
                continue;

            double alpha[3];
            for (int j = 0; j < 3; j++)
                alpha[j] = fmax(best.w[j] * concs[ph], 1e-3);

            for (int i = 0; i < count; i++) {
                double w[3], auc_val, cons_val;
                dirichlet_general(alpha, pcg, w);
                evaluate(ds->A, ds->n_samples, ds->n_items,
                         ds->profiles, ds->y, w, &auc_val, &cons_val);

                if (auc_val > best.auc) {
                    double prev = best.auc;
                    best.auc  = auc_val;
                    best.cons = cons_val;
                    best.w[0] = w[0];
                    best.w[1] = w[1];
                    best.w[2] = w[2];
                    best.iter = base + i;
                    if (verbose)
                        log_improvement(base + i, k, auc_val, prev, cons_val, w, -1);
                }
            }
            base += count;
        }
    }

    return best;
}

static void log_search_mode(const char *mode, double step) {
    if (strcmp(mode, "grid") == 0 || strcmp(mode, "hybrid") == 0)
        fprintf(stderr, "  search=%s  step=%.4f\n", mode, step);
    else
        fprintf(stderr, "  search=%s\n", mode);
}

/* ================================================================== */
/*  Programa principal                                                 */
/* ================================================================== */

/**
 * @brief Punto de entrada.
 *
 * CLI:
 *   --k N        Número de candidatos (default 10000)
 *   --seed N     Semilla RNG (default 42)
 *   --data-dir   Directorio con CSV (default "data")
 *
 * Flujo: cargar datos → cronometrar búsqueda → imprimir resultados.
 */
int main(int argc, char **argv) {
    long K         = atol(arg(argc, argv, "--k",         "10000"));
    int  seed      = atoi(arg(argc, argv, "--seed",      "42"));
    const char *data_dir = parse_data_dir(argc, argv);
    const char *search   = arg(argc, argv, "--search",   "random");
    double step          = atof(arg(argc, argv, "--step", "0.02"));
    int  benchmark = cli_flag(argc, argv, "--benchmark");

    /* Cargar datos (solo CSV) */
    Dataset ds;
    if (load_data(data_dir, &ds, benchmark) != 0)
        return 1;

    if (!benchmark) {
        log_header("c_sequential", ds.n_items, K);
        log_search_mode(search, step);
    }

    /* Cronometrar búsqueda */
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    Best best;
    long actual_k = K;

    if (strcmp(search, "grid") == 0) {
        best = grid_search(&ds, step, !benchmark, &actual_k);
    } else if (strcmp(search, "hybrid") == 0) {
        best = hybrid_search(&ds, K, (uint64_t)seed, step, !benchmark);
    } else {
        best = random_search(&ds, K, (uint64_t)seed, !benchmark);
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double elapsed = (t1.tv_sec - t0.tv_sec)
                   + (t1.tv_nsec - t0.tv_nsec) * 1e-9;

    if (benchmark) {
        print_csv_row("c_sequential", 1, ds.n_items, actual_k, elapsed,
                      best.auc, best.cons, best.w, seed, search, best.iter);
    } else {
        log_complete("c_sequential", best.auc, best.cons, best.w, elapsed);

        printf("implementation=c_sequential\n");
        printf("search_mode=%s\n", search);
        printf("N=%d\n", ds.n_items);
        printf("K=%ld\n", actual_k);
        printf("best_auc=%.6f\n", best.auc);
        printf("best_w=[%.8f, %.8f, %.8f]\n",
               best.w[0], best.w[1], best.w[2]);
        printf("best_w_sum=%.8f\n",
               best.w[0] + best.w[1] + best.w[2]);
        printf("time_sec=%.6f\n", elapsed);
    }

    free_dataset(&ds);
    return 0;
}
