/**
 * @file logger.c
 * @brief Logger ANSI colorido — réplica de python/logger.py.
 *
 * Dependencias: logger.h, common.h, <stdio.h>, <string.h>.
 *
 * Uso (mismo que en logger.py):
 *   log_header("c_sequential", n_items, K);
 *   // ... en cada mejora:
 *   log_improvement(iter, K, auc, prev_auc, cons, w);
 *   // ... al final:
 *   log_complete("c_sequential", best.auc, best.cons, best.w, elapsed);
 */
#include "logger.h"
#include <stdio.h>
#include <string.h>

/* ================================================================== */
/*  Colores ANSI — réplica de logger.py::_Style                       */
/* ================================================================== */

#define ANSI_RESET   "\033[0m"
#define ANSI_BOLD    "\033[1m"
#define ANSI_DIM     "\033[2m"
#define ANSI_GREEN   "\033[32m"
#define ANSI_CYAN    "\033[36m"
#define ANSI_MAGENTA "\033[35m"
#define ANSI_GOLD    "\033[1;33m"   /* bold yellow */

/* ================================================================== */
/*  Estado interno del logger                                          */
/* ================================================================== */

static int    _improvement_count = 0;
static double _best_auc          = -1.0;

/* ================================================================== */
/*  Implementación                                                     */
/* ================================================================== */

void log_header(const char *impl, int n_items, long k) {
    /*
     * Formato (idéntico a logger.py::_header):
     *   ╭─ c_sequential ──────────────────────────────────────╮
     *   │  BÚSQUEDA DE PESOS ÓPTIMOS
     *   │  items (N) ... 500
     *   │  candidatos   10000
     *   ╰────────────────────────────────────────────────────────╯
     */
    printf("\n");
    printf(ANSI_BOLD ANSI_CYAN "  ╭─ %s ", impl);
    int pad = 50 - (int)strlen(impl);
    if (pad < 2) pad = 2;
    for (int i = 0; i < pad; i++) printf("─");
    printf("╮\n" ANSI_RESET);
    printf(ANSI_CYAN "  │  " ANSI_BOLD "BÚSQUEDA DE PESOS ÓPTIMOS\n" ANSI_RESET);
    printf(ANSI_CYAN "  │  items (N) ... %d\n", n_items);
    printf(ANSI_CYAN "  │  candidatos   %ld\n", k);
    printf(ANSI_CYAN "  ╰");
    for (int i = 0; i < 56; i++) printf("─");
    printf("╯\n" ANSI_RESET);
    printf("\n");

    _improvement_count = 0;
    _best_auc          = -1.0;
}

void log_improvement(long iteration, long k,
                     double auc, double prev_auc,
                     double cons, const double w[3])
{
    /*
     * Formato (idéntico a logger.py::_improvement_line):
     *   -> AUC 0.755600  (+0.000400)  iter 64/10000  consist=0.7300  w=[0.5954 0.1198 0.2848]
     *   -> AUC 0.721200  (initial)    iter 0/10000   consist=0.6800  w=[0.3374 0.3279 0.3347]
     */
    _improvement_count++;
    _best_auc = auc;

    const char *delta_str;
    char delta_buf[32];
    if (_improvement_count == 1) {
        delta_str = "initial";
    } else {
        snprintf(delta_buf, sizeof(delta_buf), "+%.6f", auc - prev_auc);
        delta_str = delta_buf;
    }

    printf("  " ANSI_GOLD "->" ANSI_RESET "  "
           ANSI_BOLD "AUC %.6f" ANSI_RESET "  "
           ANSI_GREEN "(%s)" ANSI_RESET "  "
           "iter %ld/%ld  "
           "consist=%.4f  "
           "w=[%.4f %.4f %.4f]\n",
           auc, delta_str, iteration, k, cons,
           w[0], w[1], w[2]);
}

void log_complete(const char *impl, double auc, double cons,
                  const double w[3], double time_sec)
{
    /*
     * Formato (idéntico a logger.py::_summary):
     *   ╭─ MEJOR RESULTADO ────────────────────────────────────────╮
     *   │  implementacion ... c_sequential
     *   │  mejoras        ... 4
     *   │  AUC            ... 0.755600000
     *   │  consistencia   ... 0.7300
     *   │  pesos W        ... [0.595409066, 0.119754974, 0.284835960]
     *   │  suma W         ... 1.000000000
     *   │  tiempo         ... 0.006782 s
     *   ╰────────────────────────────────────────────────────────╯
     */
    printf("\n");
    printf(ANSI_BOLD ANSI_MAGENTA "  ╭─ MEJOR RESULTADO ");
    for (int i = 0; i < 40; i++) printf("─");
    printf("╮\n" ANSI_RESET);
    printf(ANSI_BOLD ANSI_MAGENTA "  │  implementacion ... %s\n" ANSI_RESET, impl);
    printf(ANSI_MAGENTA "  │  mejoras        ... %d\n", _improvement_count);
    printf(ANSI_MAGENTA "  │  AUC            ... " ANSI_GOLD "%.9f" ANSI_RESET ANSI_MAGENTA "\n", auc);
    printf(ANSI_MAGENTA "  │  consistencia   ... %.4f\n", cons);
    printf(ANSI_MAGENTA "  │  pesos W        ... [%.9f, %.9f, %.9f]\n",
           w[0], w[1], w[2]);
    printf(ANSI_MAGENTA "  │  suma W         ... %.9f\n", w[0] + w[1] + w[2]);
    printf(ANSI_MAGENTA "  │  tiempo         ... " ANSI_CYAN "%.6f s" ANSI_RESET ANSI_MAGENTA "\n", time_sec);
    printf(ANSI_MAGENTA "  ╰");
    for (int i = 0; i < 56; i++) printf("─");
    printf("╯\n" ANSI_RESET);
    printf("\n");
}

void log_reset(void) {
    _improvement_count = 0;
    _best_auc          = -1.0;
}

void log_worker_report(int worker_id, double auc, double cons,
                       const double w[3], int chunk_size, int is_best)
{
    /*
     * Formato (idéntico a logger.py::worker_report):
     *   [W2]  AUC 0.755600  consist=0.7300  w=[0.5954 0.1198 0.2848]  (500 cand.) ★
     */
    printf("  " ANSI_DIM ANSI_CYAN "[W%d]" ANSI_RESET "  "
           ANSI_BOLD "AUC %.6f" ANSI_RESET "  "
           "consist=%.4f  "
           "w=[%.4f %.4f %.4f]  "
           "(%d cand.)%s\n",
           worker_id, auc, cons,
           w[0], w[1], w[2],
           chunk_size,
           is_best ? " " ANSI_GOLD "\xe2\x98\x85" ANSI_RESET : "");
}
