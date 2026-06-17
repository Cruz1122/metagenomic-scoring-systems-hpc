/**
 * @file scoring_openmp.c
 * @brief Búsqueda paralela (OpenMP) de pesos Dirichlet — 3 estrategias.
 *
 * Estrategias:
 *   - random:  Dirichlet(1,1,1) aleatorio, K iteraciones
 *   - grid:    Barrido sistemático sobre el simplex 2D con paso fijo
 *   - hybrid:  Grid + Random global + Refinamiento local Dirichlet
 *
 * Pipeline por iteración (idéntico al secuencial):
 *   1. w ~ Dirichlet (según estrategia)
 *   2. scores = A @ (profiles @ w)
 *   3. AUC y consistencia
 *   4. Guarda el mejor local de cada hilo, merge post-loop
 *
 * Compilar:
 *   make -C C_OpenMP_MPI scoring_openmp
 *
 * Uso:
 *   OMP_NUM_THREADS=4 ./C_OpenMP_MPI/scoring_openmp --k 10000 --seed 42 \
 *       --data-dir data --search random
 *
 * Dependencias:
 *   shared/common.h/c  → load_data, evaluate, auc, consistency
 *   shared/rng.h/c     → pcg64_seed, xs, simplex, gamma_sample, dirichlet_general
 *   shared/ziggurat.h/c → standard_exponential
 *   shared/logger.h/c  → log_header, log_improvement, log_complete, log_worker_report
 */
#define _GNU_SOURCE
#include "shared/common.h"
#include "shared/rng.h"
#include "shared/logger.h"
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

/* ================================================================== */
/*  Argumentos CLI — --nombre valor                                    */
/* ================================================================== */

static const char* arg(int argc, char **argv, const char *name, const char *fallback) {
    for (int i = 1; i < argc - 1; i++)
        if (strcmp(argv[i], name) == 0)
            return argv[i + 1];
    return fallback;
}

/* ================================================================== */
/*  Logger: search mode                                                */
/* ================================================================== */

static void log_search_mode(const char *mode, double step) {
    if (strcmp(mode, "grid") == 0 || strcmp(mode, "hybrid") == 0)
        fprintf(stderr, "  search=%s  step=%.4f\n", mode, step);
    else
        fprintf(stderr, "  search=%s\n", mode);
}

/* ================================================================== */
/*  Resultado por hilo (para worker_report)                            */
/* ================================================================== */

typedef struct {
    Best best;
    int  chunk_size;
} ThreadResult;

/* ================================================================== */
/*  random_search — OpenMP                                             */
/* ================================================================== */

/**
 * @brief Búsqueda aleatoria de pesos Dirichlet — versión OpenMP.
 *
 * Cada hilo:
 *   - RNG propio: pcg64_seed(pcg, seed + tid)
 *   - Acumula solo mejor local (sin compartir nada durante el loop)
 *   - Merge local→global vía #pragma omp critical post-loop
 *   - Reporta su mejor local al final
 *
 * @param ds      Dataset (A, profiles, y).
 * @param k       Número de candidatos a evaluar.
 * @param seed    Semilla base RNG (cada hilo usa seed + tid).
 * @param verbose 1 para logging en vivo.
 * @return        Mejor resultado global.
 */
static Best random_search_openmp(const Dataset *ds, long k, uint64_t seed,
                                  int verbose)
{
    Best global_best = { .auc = -1.0, .cons = 0.0, .w = {0,0,0}, .iter = -1 };
    ThreadResult *threads = NULL;
    int n_threads = 0;

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        #pragma omp single
        {
            n_threads = omp_get_num_threads();
            threads = (ThreadResult*)calloc((size_t)n_threads, sizeof(ThreadResult));
        }

        /* Cada hilo: su propio RNG, stream único */
        uint64_t pcg[4];
        pcg64_seed(pcg, seed + (uint64_t)tid);

        Best local_best = { .auc = -1.0, .cons = 0.0, .w = {0,0,0}, .iter = -1 };

        #pragma omp for schedule(static)
        for (long i = 0; i < k; i++) {
            double w[3], auc_val, cons_val;
            simplex(pcg, w);
            evaluate(ds->A, ds->n_samples, ds->n_items,
                     ds->profiles, ds->y, w, &auc_val, &cons_val);

            /* Solo mejor local (sin acceso shared) */
            if (auc_val > local_best.auc) {
                local_best.auc  = auc_val;
                local_best.cons = cons_val;
                local_best.w[0] = w[0];
                local_best.w[1] = w[1];
                local_best.w[2] = w[2];
                local_best.iter = i;
            }
        }

        /* Guardar resultado local y chunk_size */
        threads[tid].best = local_best;
        long chunk_base = k / n_threads;
        int  extra      = (int)(k % n_threads);
        threads[tid].chunk_size = (int)chunk_base + (tid < extra ? 1 : 0);

        #pragma omp barrier

        /* Merge local → global (por si nunca ganó el critical race) */
        #pragma omp critical
        if (local_best.auc > global_best.auc)
            global_best = local_best;
    }

    /* Reportar worker results (hilo master) */
    if (verbose && threads) {
        printf("\n  --- best by worker (random) ---\n");
        for (int t = 0; t < n_threads; t++) {
            int is_global = (threads[t].best.auc == global_best.auc &&
                             threads[t].best.cons == global_best.cons);
            log_worker_report(t, threads[t].best.auc, threads[t].best.cons,
                              threads[t].best.w, threads[t].chunk_size, is_global);
        }
        printf("\n");
    }
    free(threads);
    return global_best;
}

/* ================================================================== */
/*  grid_search — OpenMP                                               */
/* ================================================================== */

/**
 * @brief Búsqueda sistemática sobre el simplex 2D — versión OpenMP.
 *
 * Genera todos los puntos del grid (w1, w2, w3) en un array plano,
 * luego evalúa en paralelo con schedule(static).
 * No necesita RNG — el grid es determinista.
 *
 * @param ds       Dataset.
 * @param step     Paso del grid (default 0.02).
 * @param verbose  1 para logging.
 * @param out_total Cantidad total de puntos evaluados.
 * @return         Mejor resultado.
 */
static Best grid_search_openmp(const Dataset *ds, double step,
                                int verbose, long *out_total)
{
    /* Fase 1: contar puntos del grid */
    long total = 0;
    for (double w1 = 0.0; w1 <= 1.0 + 1e-12; w1 += step)
        for (double w2 = 0.0; w2 <= 1.0 - w1 + 1e-12; w2 += step)
            total++;
    *out_total = total;

    /* Fase 2: generar grid (serial, O(N) vs evaluate O(N²)) */
    double (*grid)[3] = (double(*)[3])malloc((size_t)total * sizeof(double[3]));
    if (!grid) return (Best){ .auc = -1.0 };

    long idx = 0;
    for (double w1 = 0.0; w1 <= 1.0 + 1e-12; w1 += step) {
        for (double w2 = 0.0; w2 <= 1.0 - w1 + 1e-12; w2 += step) {
            double w3 = 1.0 - w1 - w2;
            grid[idx][0] = w1; grid[idx][1] = w2; grid[idx][2] = w3;
            idx++;
        }
    }

    /* Fase 3: evaluar en paralelo (mismo patrón que random) */
    Best global_best = { .auc = -1.0, .cons = 0.0, .w = {0,0,0}, .iter = -1 };
    ThreadResult *threads = NULL;
    int n_threads = 0;

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        #pragma omp single
        {
            n_threads = omp_get_num_threads();
            threads = (ThreadResult*)calloc((size_t)n_threads, sizeof(ThreadResult));
        }

        Best local_best = { .auc = -1.0, .cons = 0.0, .w = {0,0,0}, .iter = -1 };

        #pragma omp for schedule(static)
        for (long i = 0; i < total; i++) {
            double auc_val, cons_val;
            evaluate(ds->A, ds->n_samples, ds->n_items,
                     ds->profiles, ds->y, grid[i], &auc_val, &cons_val);

            if (auc_val > local_best.auc) {
                local_best.auc  = auc_val;
                local_best.cons = cons_val;
                local_best.w[0] = grid[i][0];
                local_best.w[1] = grid[i][1];
                local_best.w[2] = grid[i][2];
                local_best.iter = i;
            }
        }

        threads[tid].best = local_best;
        long chunk_base = total / n_threads;
        int  extra      = (int)(total % n_threads);
        threads[tid].chunk_size = (int)chunk_base + (tid < extra ? 1 : 0);

        #pragma omp barrier

        #pragma omp critical
        if (local_best.auc > global_best.auc)
            global_best = local_best;
    }

    /* Reportar workers */
    if (verbose && threads) {
        printf("\n  --- best by worker (grid) ---\n");
        for (int t = 0; t < n_threads; t++) {
            int is_global = (threads[t].best.auc == global_best.auc &&
                             threads[t].best.cons == global_best.cons);
            log_worker_report(t, threads[t].best.auc, threads[t].best.cons,
                              threads[t].best.w, threads[t].chunk_size, is_global);
        }
        printf("\n");
    }
    free(threads);
    free(grid);
    return global_best;
}

/* ================================================================== */
/*  hybrid_search — OpenMP                                             */
/* ================================================================== */

/**
 * @brief Búsqueda híbrida en tres fases — versión OpenMP.
 *
 * Fase 1 — Grid step=0.02 (~1326 puntos, si step=0.02)
 * Fase 2 — Random Dirichlet(1,1,1) global  (~50% del resto)
 * Fase 3 — Local Dirichlet concentrada alrededor del mejor W (~50% del resto)
 *          dividida entre concentration=300 y concentration=1000.
 *
 * Cada fase es internamente paralela con OpenMP;
 * las fases se ejecutan secuencialmente.
 *
 * @param ds     Dataset.
 * @param k      Presupuesto total de candidatos.
 * @param seed   Semilla base.
 * @param step   Paso del grid.
 * @param verbose 1 para logging.
 * @return       Mejor resultado.
 */
static Best hybrid_search_openmp(const Dataset *ds, long k, uint64_t seed,
                                  double step, int verbose)
{
    /* ── Fase 1: Grid grueso ──────────────────────────────────────── */
    long grid_total;
    Best global_best = grid_search_openmp(ds, step, verbose, &grid_total);

    long remaining = k - grid_total;
    if (remaining <= 0) return global_best;

    long random_n = remaining / 2;
    long local_n  = remaining - random_n;

    /* ── Fase 2: Random global ────────────────────────────────────── */
    if (random_n > 0) {
        ThreadResult *rnd_threads = NULL;
        int rnd_n_threads = 0;

        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            #pragma omp single
            {
                rnd_n_threads = omp_get_num_threads();
                rnd_threads = (ThreadResult*)calloc((size_t)rnd_n_threads,
                                                     sizeof(ThreadResult));
            }

            uint64_t pcg[4];
            pcg64_seed(pcg, seed + (uint64_t)tid);
            Best local_best = global_best; /* partir del mejor actual */

            #pragma omp for schedule(static)
            for (long i = 0; i < random_n; i++) {
                double w[3], auc_val, cons_val;
                simplex(pcg, w);
                evaluate(ds->A, ds->n_samples, ds->n_items,
                         ds->profiles, ds->y, w, &auc_val, &cons_val);

                if (auc_val > local_best.auc) {
                    local_best.auc  = auc_val;
                    local_best.cons = cons_val;
                    local_best.w[0] = w[0];
                    local_best.w[1] = w[1];
                    local_best.w[2] = w[2];
                    local_best.iter = grid_total + i;
                }
            }

            rnd_threads[tid].best = local_best;
            long chunk_base = random_n / rnd_n_threads;
            int  extra      = (int)(random_n % rnd_n_threads);
            rnd_threads[tid].chunk_size = (int)chunk_base + (tid < extra ? 1 : 0);

            #pragma omp barrier

            #pragma omp critical
            if (local_best.auc > global_best.auc)
                global_best = local_best;
        }

        if (verbose && rnd_threads) {
            printf("\n  --- best by worker (random phase) ---\n");
            for (int t = 0; t < rnd_n_threads; t++) {
                int is_global = (rnd_threads[t].best.auc == global_best.auc &&
                                 rnd_threads[t].best.cons == global_best.cons);
                log_worker_report(t, rnd_threads[t].best.auc,
                                  rnd_threads[t].best.cons,
                                  rnd_threads[t].best.w,
                                  rnd_threads[t].chunk_size, is_global);
            }
            printf("\n");
        }
        free(rnd_threads);
    }

    /* ── Fase 3: Refinamiento local Dirichlet ─────────────────────── */
    if (local_n > 0 && global_best.auc >= 0) {
        int    splits[2] = { (int)(local_n / 2), (int)(local_n - local_n / 2) };
        double concs[2]  = { 300.0, 1000.0 };
        long   base      = grid_total + random_n;

        for (int ph = 0; ph < 2; ph++) {
            int count = splits[ph];
            if (count <= 0) continue;

            /* alpha_i = best_w_i * concentration, floor at 1e-3 */
            double alpha[3];
            for (int j = 0; j < 3; j++)
                alpha[j] = fmax(global_best.w[j] * concs[ph], 1e-3);

            ThreadResult *loc_threads = NULL;
            int loc_n_threads = 0;

            #pragma omp parallel
            {
                int tid = omp_get_thread_num();
                #pragma omp single
                {
                    loc_n_threads = omp_get_num_threads();
                    loc_threads = (ThreadResult*)calloc((size_t)loc_n_threads,
                                                         sizeof(ThreadResult));
                }

                uint64_t pcg[4];
                pcg64_seed(pcg, seed + (uint64_t)tid);
                Best local_best = global_best;

                #pragma omp for schedule(static)
                for (int i = 0; i < count; i++) {
                    double w[3], auc_val, cons_val;
                    dirichlet_general(alpha, pcg, w);
                    evaluate(ds->A, ds->n_samples, ds->n_items,
                             ds->profiles, ds->y, w, &auc_val, &cons_val);

                    if (auc_val > local_best.auc) {
                        local_best.auc  = auc_val;
                        local_best.cons = cons_val;
                        local_best.w[0] = w[0];
                        local_best.w[1] = w[1];
                        local_best.w[2] = w[2];
                        local_best.iter = base + i;
                    }
                }

                loc_threads[tid].best = local_best;
                long chunk_base = count / loc_n_threads;
                int  extra      = (int)(count % loc_n_threads);
                loc_threads[tid].chunk_size = (int)chunk_base + (tid < extra ? 1 : 0);

                #pragma omp barrier

                #pragma omp critical
                if (local_best.auc > global_best.auc)
                    global_best = local_best;
            }

            if (verbose && loc_threads) {
                printf("\n  --- best by worker (local conc=%.0f) ---\n", concs[ph]);
                for (int t = 0; t < loc_n_threads; t++) {
                    int is_global = (loc_threads[t].best.auc == global_best.auc &&
                                     loc_threads[t].best.cons == global_best.cons);
                    log_worker_report(t, loc_threads[t].best.auc,
                                      loc_threads[t].best.cons,
                                      loc_threads[t].best.w,
                                      loc_threads[t].chunk_size, is_global);
                }
                printf("\n");
            }
            free(loc_threads);
        }
    }

    return global_best;
}

/* ================================================================== */
/*  Programa principal                                                 */
/* ================================================================== */

int main(int argc, char **argv) {
    long K         = atol(arg(argc, argv, "--k",         "10000"));
    int  seed      = atoi(arg(argc, argv, "--seed",      "42"));
    const char *data_dir = arg(argc, argv, "--data-dir", "data");
    const char *search   = arg(argc, argv, "--search",   "random");
    double step          = atof(arg(argc, argv, "--step", "0.02"));

    /* Cargar datos (serial) */
    Dataset ds;
    if (load_data(data_dir, &ds) != 0)
        return 1;

    /* Logger: cabecera */
    log_header("c_openmp", ds.n_items, K);
    log_search_mode(search, step);

    /* Cronometrar búsqueda */
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    Best best;
    long actual_k = K;

    if (strcmp(search, "grid") == 0) {
        best = grid_search_openmp(&ds, step, 1, &actual_k);
    } else if (strcmp(search, "hybrid") == 0) {
        best = hybrid_search_openmp(&ds, K, (uint64_t)seed, step, 1);
    } else {
        best = random_search_openmp(&ds, K, (uint64_t)seed, 1);
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double elapsed = (t1.tv_sec - t0.tv_sec)
                   + (t1.tv_nsec - t0.tv_nsec) * 1e-9;

    /* Logger: resumen */
    log_complete("c_openmp", best.auc, best.cons, best.w, elapsed);

    /* Salida parseable (formato compatible con sequential + extras) */
    printf("implementation=c_openmp\n");
    printf("search_mode=%s\n", search);
    printf("workers=%d\n", omp_get_max_threads());
    printf("N=%d\n", ds.n_items);
    printf("K=%ld\n", actual_k);
    printf("best_auc=%.6f\n", best.auc);
    printf("best_w=[%.8f, %.8f, %.8f]\n",
           best.w[0], best.w[1], best.w[2]);
    printf("best_w_sum=%.8f\n",
           best.w[0] + best.w[1] + best.w[2]);
    if (strcmp(search, "grid") == 0)
        printf("step=%.4f\n", step);
    printf("time_sec=%.6f\n", elapsed);

    free_dataset(&ds);
    return 0;
}
