/**
 * scoring_mpi.c — Búsqueda paralela de pesos con MPI.
 *
 * Estrategias: random, grid, hybrid
 * Paralelización por candidatos W (cada rank tiene copia completa de datos).
 * Logger idéntico a OpenMP: rank 0 imprime todo; mejoras en vivo vía MPI.
 *
 * Compilar:
 *   make -C C_OpenMP_MPI mpi
 *
 * Ejecutar:
 *   mpirun -np 4 C_OpenMP_MPI/scoring_mpi --search random --k 10000 --seed 42
 */
#define _GNU_SOURCE
#include "shared/common.h"
#include "shared/rng.h"
#include "shared/logger.h"
#include <mpi.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ================================================================== */
/*  BestResult + live logging                                          */
/* ================================================================== */

typedef struct {
    double auc;
    double consistency;
    long   candidate_idx;
    double w1, w2, w3;
} BestResult;

#define TAG_IMPROVE 100

typedef struct {
    long   iter;
    double auc;
    double cons;
    double w[3];
    int    worker_id;
} ImproveMsg;

static double live_best_auc = -1.0;

static void live_log_reset(void) {
    live_best_auc = -1.0;
}

static void try_log_global_improvement(long k_total, const ImproveMsg *m) {
    if (m->auc > live_best_auc) {
        double prev = live_best_auc;
        double w[3] = { m->w[0], m->w[1], m->w[2] };
        log_improvement(m->iter, k_total, m->auc, prev, m->cons, w, m->worker_id);
        live_best_auc = m->auc;
    }
}

static void drain_improvements(long k_total) {
    int flag;
    MPI_Status status;
    ImproveMsg msg;

    while (1) {
        MPI_Iprobe(MPI_ANY_SOURCE, TAG_IMPROVE, MPI_COMM_WORLD, &flag, &status);
        if (!flag)
            break;
        MPI_Recv(&msg, (int)sizeof(msg), MPI_BYTE, status.MPI_SOURCE,
                 TAG_IMPROVE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        msg.worker_id = status.MPI_SOURCE;
        try_log_global_improvement(k_total, &msg);
    }
}

static MPI_Request pending_improve_req = MPI_REQUEST_NULL;
static ImproveMsg  pending_improve_msg;

static void flush_pending_improvement(int rank) {
    if (rank > 0 && pending_improve_req != MPI_REQUEST_NULL) {
        MPI_Wait(&pending_improve_req, MPI_STATUS_IGNORE);
        pending_improve_req = MPI_REQUEST_NULL;
    }
}

static void notify_local_improvement(int rank, long k_total, long gidx,
                                     double auc, double cons, const double w[3]) {
    ImproveMsg msg = { .iter = gidx, .auc = auc, .cons = cons,
                       .w = { w[0], w[1], w[2] }, .worker_id = rank };

    if (rank == 0) {
        try_log_global_improvement(k_total, &msg);
        return;
    }

    if (pending_improve_req != MPI_REQUEST_NULL)
        MPI_Wait(&pending_improve_req, MPI_STATUS_IGNORE);

    pending_improve_msg = msg;
    MPI_Isend(&pending_improve_msg, (int)sizeof(pending_improve_msg), MPI_BYTE,
              0, TAG_IMPROVE, MPI_COMM_WORLD, &pending_improve_req);
}

/* ================================================================== */
/*  CLI                                                                */
/* ================================================================== */

static const char* arg(int argc, char **argv, const char *name, const char *fallback) {
    for (int i = 1; i < argc - 1; i++)
        if (strcmp(argv[i], name) == 0) return argv[i + 1];
    return fallback;
}
static int arg_int(int argc, char **argv, const char *name, int fallback) {
    const char *v = arg(argc, argv, name, NULL); return v ? atoi(v) : fallback;
}
static long arg_long(int argc, char **argv, const char *name, long fallback) {
    const char *v = arg(argc, argv, name, NULL); return v ? atol(v) : fallback;
}

static void log_search_mode(const char *mode, double step) {
    if (strcmp(mode, "grid") == 0 || strcmp(mode, "hybrid") == 0)
        fprintf(stderr, "  search=%s  step=%.4f\n", mode, step);
    else
        fprintf(stderr, "  search=%s\n", mode);
}

/* ================================================================== */
/*  Broadcast del problema                                             */
/* ================================================================== */

static void broadcast_problem(Dataset *ds, int rank) {
    int dims[2] = { ds->n_samples, ds->n_items };
    MPI_Bcast(dims, 2, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank != 0) { ds->n_samples = dims[0]; ds->n_items = dims[1]; }

    int64_t nA = (int64_t)ds->n_samples * ds->n_items;
    int64_t nP = (int64_t)ds->n_items * 3;
    int64_t nY = ds->n_samples;

    if (rank != 0) {
        ds->A = (double*)malloc((size_t)nA * sizeof(double));
        ds->profiles = (double*)malloc((size_t)nP * sizeof(double));
        ds->y = (int*)malloc((size_t)nY * sizeof(int));
        if (!ds->A || !ds->profiles || !ds->y) MPI_Abort(MPI_COMM_WORLD, 2);
    }
    MPI_Bcast(ds->A, (int)nA, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(ds->profiles, (int)nP, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(ds->y, (int)nY, MPI_INT, 0, MPI_COMM_WORLD);
}

/* ================================================================== */
/*  Generación de candidatos                                           */
/* ================================================================== */

static double* generate_candidates_random(long k, int seed, long *actual_k) {
    *actual_k = k;
    double *cand = (double*)malloc((size_t)k * 3 * sizeof(double));
    if (!cand) return NULL;
    uint64_t pcg[4];
    pcg64_seed(pcg, (uint64_t)seed);
    for (long i = 0; i < k; i++) {
        double w[3]; simplex(pcg, w);
        cand[i*3+0] = w[0]; cand[i*3+1] = w[1]; cand[i*3+2] = w[2];
    }
    return cand;
}

static double* generate_candidates_grid_step(double step, long *actual_k) {
    long total = 0;
    for (double w1 = 0.0; w1 <= 1.0 + 1e-12; w1 += step)
        for (double w2 = 0.0; w2 <= 1.0 - w1 + 1e-12; w2 += step)
            total++;
    *actual_k = total;
    double *cand = (double*)malloc((size_t)total * 3 * sizeof(double));
    if (!cand) return NULL;
    long idx = 0;
    for (double w1 = 0.0; w1 <= 1.0 + 1e-12; w1 += step) {
        for (double w2 = 0.0; w2 <= 1.0 - w1 + 1e-12; w2 += step) {
            double w3 = 1.0 - w1 - w2;
            cand[idx*3+0] = w1; cand[idx*3+1] = w2; cand[idx*3+2] = w3;
            idx++;
        }
    }
    return cand;
}

static double* generate_candidates_grid(int grid_steps, long *actual_k) {
    long total = 0;
    for (int i = 0; i <= grid_steps; i++)
        for (int j = 0; j <= grid_steps - i; j++) total++;
    *actual_k = total;
    double *cand = (double*)malloc((size_t)total * 3 * sizeof(double));
    if (!cand) return NULL;
    long idx = 0;
    for (int i = 0; i <= grid_steps; i++)
        for (int j = 0; j <= grid_steps - i; j++) {
            int kk = grid_steps - i - j;
            cand[idx*3+0] = (double)i/grid_steps;
            cand[idx*3+1] = (double)j/grid_steps;
            cand[idx*3+2] = (double)kk/grid_steps;
            idx++;
        }
    return cand;
}

static double* generate_candidates_refinement(const double best_w[3],
                                               long count, int seed) {
    if (count <= 0) return NULL;
    double *cand = (double*)malloc((size_t)count * 3 * sizeof(double));
    if (!cand) return NULL;
    uint64_t pcg[4];
    pcg64_seed(pcg, (uint64_t)seed + 9999);
    double alpha[3];
    for (int j = 0; j < 3; j++) alpha[j] = fmax(best_w[j] * 300.0, 1e-3);
    for (long i = 0; i < count; i++) {
        double w[3]; dirichlet_general(alpha, pcg, w);
        cand[i*3+0] = w[0]; cand[i*3+1] = w[1]; cand[i*3+2] = w[2];
    }
    return cand;
}

/* ================================================================== */
/*  Scatter de candidatos                                              */
/* ================================================================== */

static int chunk_size_for_rank(int r, long total_k, int size) {
    long base = total_k / size;
    int  rem  = (int)(total_k % size);
    return (int)base + (r < rem ? 1 : 0);
}

static long scatter_candidates(const double *all_cand, long total_k,
                                double **local_cand, int rank, int size) {
    long base = total_k / size;
    int  rem  = (int)(total_k % size);

    int *sendcounts = NULL, *displs = NULL;
    if (rank == 0) {
        sendcounts = (int*)malloc((size_t)size * sizeof(int));
        displs     = (int*)malloc((size_t)size * sizeof(int));
        int offset = 0;
        for (int r = 0; r < size; r++) {
            long cnt = base + (r < rem ? 1 : 0);
            sendcounts[r] = (int)(cnt * 3);
            displs[r]     = offset;
            offset += (int)(cnt * 3);
        }
    }

    long local_count = base + (rank < rem ? 1 : 0);
    *local_cand = (double*)malloc((size_t)local_count * 3 * sizeof(double));
    MPI_Scatterv(all_cand, sendcounts, displs, MPI_DOUBLE,
                 *local_cand, (int)(local_count * 3), MPI_DOUBLE,
                 0, MPI_COMM_WORLD);
    free(sendcounts); free(displs);
    return local_count;
}

/* ================================================================== */
/*  Evaluación local                                                   */
/* ================================================================== */

static BestResult evaluate_local(const Dataset *ds,
                                  const double *candidates, long count,
                                  long global_offset,
                                  int rank, long k_total, int live_log) {
    BestResult best = { -1.0, 0.0, -1, 0, 0, 0 };
    for (long i = 0; i < count; i++) {
        double w[3] = { candidates[i*3+0], candidates[i*3+1], candidates[i*3+2] };
        double auc_val, cons_val;
        evaluate(ds->A, ds->n_samples, ds->n_items,
                 ds->profiles, ds->y, w, &auc_val, &cons_val);
        long gidx = global_offset + i;
        int better = 0;
        if (auc_val > best.auc) better = 1;
        else if (auc_val == best.auc && cons_val > best.consistency) better = 1;
        else if (auc_val == best.auc && cons_val == best.consistency && gidx < best.candidate_idx) better = 1;
        if (better) {
            best.auc = auc_val; best.consistency = cons_val;
            best.candidate_idx = gidx;
            best.w1 = w[0]; best.w2 = w[1]; best.w3 = w[2];
            if (live_log)
                notify_local_improvement(rank, k_total, gidx, auc_val, cons_val, w);
        }
        if (rank == 0 && live_log && (i & 31) == 0)
            drain_improvements(k_total);
    }
    if (live_log) {
        flush_pending_improvement(rank);
        if (rank == 0)
            drain_improvements(k_total);
    }
    return best;
}

/* ================================================================== */
/*  Gather de mejores locales → rank 0 + Bcast                        */
/* ================================================================== */

static void log_worker_reports_from_gather(const double *all, int size,
                                            long total_k, const BestResult *global,
                                            const char *phase_label) {
    printf("\n  --- best by worker (%s) ---\n", phase_label);
    for (int r = 0; r < size; r++) {
        const double *p = all + r * 6;
        double w[3] = { p[3], p[4], p[5] };
        int is_global = (p[0] == global->auc && p[1] == global->consistency);
        log_worker_report(r, p[0], p[1], w,
                          chunk_size_for_rank(r, total_k, size), is_global);
    }
    printf("\n");
}

static BestResult merge_from_gather(const double *all, int size, BestResult seed) {
    BestResult global = seed;
    for (int r = 0; r < size; r++) {
        const double *p = all + r * 6;
        int better = 0;
        if (p[0] > global.auc) better = 1;
        else if (p[0] == global.auc && p[1] > global.consistency) better = 1;
        else if (p[0] == global.auc && p[1] == global.consistency
                 && (long)p[2] < global.candidate_idx) better = 1;
        if (better) {
            global.auc = p[0]; global.consistency = p[1];
            global.candidate_idx = (long)p[2];
            global.w1 = p[3]; global.w2 = p[4]; global.w3 = p[5];
        }
    }
    return global;
}

static BestResult gather_best(BestResult local, int rank, int size,
                               long total_k, const char *phase_label) {
    double pack[6] = { local.auc, local.consistency, (double)local.candidate_idx,
                       local.w1, local.w2, local.w3 };
    double *all = NULL;
    if (rank == 0) all = (double*)malloc((size_t)size * 6 * sizeof(double));
    MPI_Gather(pack, 6, MPI_DOUBLE, all, 6, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    BestResult global = local;
    if (rank == 0 && all) {
        global = merge_from_gather(all, size, local);
        if (phase_label)
            log_worker_reports_from_gather(all, size, total_k, &global, phase_label);
        free(all);
    }

    double bp[6] = { global.auc, global.consistency, (double)global.candidate_idx,
                     global.w1, global.w2, global.w3 };
    MPI_Bcast(bp, 6, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (rank != 0) {
        global.auc = bp[0]; global.consistency = bp[1];
        global.candidate_idx = (long)bp[2];
        global.w1 = bp[3]; global.w2 = bp[4]; global.w3 = bp[5];
    }
    return global;
}

/* ================================================================== */
/*  Estrategias: devuelven BestResult (no imprimen)                   */
/* ================================================================== */

static BestResult run_random(const Dataset *ds, long k, int seed,
                              int rank, int size, long *out_actual_k,
                              int live_log) {
    double *all = NULL; long actual_k = 0;
    if (rank == 0) { all = generate_candidates_random(k, seed, &actual_k); }
    MPI_Bcast(&actual_k, 1, MPI_LONG, 0, MPI_COMM_WORLD);
    double *local = NULL;
    long nlocal = scatter_candidates(all, actual_k, &local, rank, size);
    long base = actual_k / size, rem = actual_k % size;
    long off = rank * base + (rank < rem ? rank : rem);
    BestResult lb = evaluate_local(ds, local, nlocal, off, rank, actual_k, live_log);
    free(local); if (rank == 0) free(all);
    *out_actual_k = actual_k;
    return gather_best(lb, rank, size, actual_k, live_log ? "random" : NULL);
}

static BestResult run_grid_step(const Dataset *ds, double step,
                                 int rank, int size, long *out_actual_k,
                                 int live_log) {
    double *all = NULL; long actual_k = 0;
    if (rank == 0) { all = generate_candidates_grid_step(step, &actual_k); }
    MPI_Bcast(&actual_k, 1, MPI_LONG, 0, MPI_COMM_WORLD);
    double *local = NULL;
    long nlocal = scatter_candidates(all, actual_k, &local, rank, size);
    long base = actual_k / size, rem = actual_k % size;
    long off = rank * base + (rank < rem ? rank : rem);
    BestResult lb = evaluate_local(ds, local, nlocal, off, rank, actual_k, live_log);
    free(local); if (rank == 0) free(all);
    *out_actual_k = actual_k;
    return gather_best(lb, rank, size, actual_k, live_log ? "grid" : NULL);
}

static BestResult run_grid(const Dataset *ds, int grid_steps,
                            int rank, int size, long *out_actual_k,
                            int live_log) {
    double *all = NULL; long actual_k = 0;
    if (rank == 0) { all = generate_candidates_grid(grid_steps, &actual_k); }
    MPI_Bcast(&actual_k, 1, MPI_LONG, 0, MPI_COMM_WORLD);
    double *local = NULL;
    long nlocal = scatter_candidates(all, actual_k, &local, rank, size);
    long base = actual_k / size, rem = actual_k % size;
    long off = rank * base + (rank < rem ? rank : rem);
    BestResult lb = evaluate_local(ds, local, nlocal, off, rank, actual_k, live_log);
    free(local); if (rank == 0) free(all);
    *out_actual_k = actual_k;
    return gather_best(lb, rank, size, actual_k, live_log ? "grid" : NULL);
}

static BestResult run_hybrid(const Dataset *ds, long k, int seed,
                              int refine_steps,
                              int rank, int size, long *out_actual_k,
                              int live_log) {
    int ref_local = (refine_steps > 0) ? refine_steps : (int)(k * 0.2);
    if (ref_local < 1) ref_local = 1;

    /* Fase 1: Random */
    double *rand_all = NULL; long rand_k = 0;
    if (rank == 0) { rand_all = generate_candidates_random(k, seed, &rand_k); }
    MPI_Bcast(&rand_k, 1, MPI_LONG, 0, MPI_COMM_WORLD);
    double *local_r = NULL;
    long nlr = scatter_candidates(rand_all, rand_k, &local_r, rank, size);
    long base_r = rand_k / size, rem_r = rand_k % size;
    long off_r = rank * base_r + (rank < rem_r ? rank : rem_r);
    BestResult lb_r = evaluate_local(ds, local_r, nlr, off_r, rank, rand_k, live_log);
    free(local_r); if (rank == 0) free(rand_all);
    BestResult p1 = gather_best(lb_r, rank, size, rand_k,
                                live_log ? "random phase" : NULL);

    if (live_log && rank == 0)
        drain_improvements(rand_k);
    MPI_Barrier(MPI_COMM_WORLD);

    /* Fase 2: Refinamiento */
    double best_w[3] = { p1.w1, p1.w2, p1.w3 };
    double *ref_all = NULL; long ref_k = 0;
    if (rank == 0) {
        ref_all = generate_candidates_refinement(best_w, ref_local, seed + 1);
        ref_k = ref_local;
    }
    MPI_Bcast(&ref_k, 1, MPI_LONG, 0, MPI_COMM_WORLD);
    long k_total = rand_k + ref_k;
    double *local_f = NULL; long nlf = 0;
    if (ref_k > 0) nlf = scatter_candidates(ref_all, ref_k, &local_f, rank, size);
    long base_f = (ref_k > 0) ? ref_k / size : 0;
    int  rem_f  = (ref_k > 0) ? (int)(ref_k % size) : 0;
    long off_f = (ref_k > 0) ? rank * base_f + (rank < rem_f ? rank : rem_f) : 0;
    off_f += rand_k;
    BestResult lb_f = { -1.0, 0.0, -1, 0, 0, 0 };
    if (ref_k > 0 && nlf > 0) {
        lb_f = evaluate_local(ds, local_f, nlf, off_f, rank, k_total, live_log);
        free(local_f);
    }
    if (rank == 0) free(ref_all);
    BestResult p2 = gather_best(lb_f, rank, size, ref_k,
                                live_log ? "refinement phase" : NULL);

    BestResult best = p1;
    if (p2.auc > p1.auc) best = p2;
    else if (p2.auc == p1.auc && p2.consistency > p1.consistency) best = p2;
    else if (p2.auc == p1.auc && p2.consistency == p1.consistency
             && p2.candidate_idx < p1.candidate_idx) best = p2;

    *out_actual_k = rand_k + ref_k;
    double bp[6] = { best.auc, best.consistency, (double)best.candidate_idx,
                     best.w1, best.w2, best.w3 };
    MPI_Bcast(bp, 6, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (rank != 0) {
        best.auc = bp[0]; best.consistency = bp[1];
        best.candidate_idx = (long)bp[2];
        best.w1 = bp[3]; best.w2 = bp[4]; best.w3 = bp[5];
    }
    return best;
}

/* ================================================================== */
/*  Main                                                               */
/* ================================================================== */

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const char *search_arg = arg(argc, argv, "--search", NULL);
    const char *strategy   = search_arg ? search_arg
                                        : arg(argc, argv, "--strategy", "random");
    long  k                = arg_long(argc, argv, "--k",          10000);
    int   seed             = arg_int(argc, argv, "--seed",        42);
    int   refine_steps     = arg_int(argc, argv, "--refine-steps", 0);
    const char *data_dir   = parse_data_dir(argc, argv);
    double step            = atof(arg(argc, argv, "--step", "0.02"));
    int   grid_steps       = arg_int(argc, argv, "--grid-steps", 0);
    int   benchmark        = cli_flag(argc, argv, "--benchmark");
    int   live_log         = !benchmark;

    Dataset ds;
    if (rank == 0) {
        if (load_data(data_dir, &ds, benchmark) != 0) MPI_Abort(MPI_COMM_WORLD, 1);
        if (!benchmark) {
            log_header("c_mpi", ds.n_items, k);
            log_search_mode(strategy, step);
            live_log_reset();
        }
    }
    broadcast_problem(&ds, rank);

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    long actual_k = 0;
    BestResult best = { -1.0, 0.0, -1, 0, 0, 0 };

    if (strcmp(strategy, "grid") == 0) {
        if (grid_steps > 0)
            best = run_grid(&ds, grid_steps, rank, size, &actual_k, live_log);
        else
            best = run_grid_step(&ds, step, rank, size, &actual_k, live_log);
    } else if (strcmp(strategy, "hybrid") == 0) {
        best = run_hybrid(&ds, k, seed, refine_steps, rank, size, &actual_k, live_log);
    } else {
        best = run_random(&ds, k, seed, rank, size, &actual_k, live_log);
    }

    double t1 = MPI_Wtime();
    double local_elapsed = t1 - t0;
    double global_elapsed = 0.0;
    MPI_Reduce(&local_elapsed, &global_elapsed, 1, MPI_DOUBLE,
               MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        double w[3] = { best.w1, best.w2, best.w3 };

        if (benchmark) {
            print_csv_row("c_mpi", size, ds.n_items, actual_k, global_elapsed,
                          best.auc, best.consistency, w, seed, strategy,
                          best.candidate_idx);
        } else {
            log_complete("c_mpi", best.auc, best.consistency, w, global_elapsed);

            printf("implementation=c_mpi\n");
            printf("search_mode=%s\n", strategy);
            printf("workers=%d\n", size);
            printf("N=%d\n", ds.n_items);
            printf("K=%ld\n", actual_k);
            printf("best_auc=%.6f\n", best.auc);
            printf("best_w=[%.8f, %.8f, %.8f]\n", best.w1, best.w2, best.w3);
            printf("best_w_sum=%.8f\n", best.w1 + best.w2 + best.w3);
            if (strcmp(strategy, "grid") == 0)
                printf("step=%.4f\n", step);
            printf("time_sec=%.6f\n", global_elapsed);
        }
    }

    if (rank == 0) free_dataset(&ds);
    else { free(ds.A); free(ds.profiles); free(ds.y); }

    MPI_Finalize();
    return 0;
}
