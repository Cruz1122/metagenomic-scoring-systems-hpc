/**
 * @file common.c
 * @brief Carga de datos (CSV), AUC, consistencia, evaluación.
 *
 * Réplica de python/common.py.
 *
 ## Funciones (réplica de python/common.py)
 *   load_data()      → common.py::load_data()
 *   auc()            → common.py::auc_vector()
 *   consistency()    → common.py::consistency()
 *   evaluate()       → common.py::evaluate()
 *
 * Dependencias: common.h, <math.h>, <stdio.h>, <stdlib.h>, <string.h>.
 */
#define _GNU_SOURCE
#include "common.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ================================================================== */
/*  Internas: parseo de CSV                                           */
/*  (compartido por load_csv_matrix, load_csv_profiles, load_csv_labels) */
/* ================================================================== */

static void strip_eol(char *line) {
    size_t n = strlen(line);
    while (n > 0 && (line[n - 1] == '\n' || line[n - 1] == '\r'))
        line[--n] = '\0';
}

static int line_nonempty(const char *line) {
    for (const char *p = line; *p; p++)
        if (*p != ' ' && *p != '\t') return 1;
    return 0;
}

/**
 * @brief Lee una línea CSV con buffer dinámico (sin límite fijo de 64 KiB).
 * @return 0 en éxito, -1 en EOF o error.
 */
static int read_csv_line(FILE *fh, char **line, size_t *cap) {
    ssize_t n = getline(line, cap, fh);
    if (n < 0) return -1;
    strip_eol(*line);
    return 0;
}

/**
 * @brief Cuenta campos separados por comas en una línea.
 */
static int csv_fields(const char *line) {
    int n = 1;
    for (const char *p = line; *p; p++)
        if (*p == ',') n++;
    return n;
}

/**
 * @brief Avanza `n` campos desde `p` y devuelve el inicio del siguiente.
 */
static const char *csv_skip_fields(const char *p, int n) {
    for (int i = 0; i < n; i++) {
        while (*p && *p != ',') p++;
        if (*p == ',') p++;
        else if (i < n - 1) return NULL;
    }
    return p;
}

/**
 * @brief Parsea `data_cols` doubles tras omitir el primer campo (sample_id).
 * Un solo recorrido O(data_cols) por fila.
 */
static int parse_matrix_row(char *line, int data_cols, double *row) {
    char *p = line;
    while (*p && *p != ',') p++;
    if (*p != ',') return -1;
    p++;

    for (int c = 0; c < data_cols; c++) {
        char *end = NULL;
        row[c] = strtod(p, &end);
        if (end == p) return -1;
        p = end;
        if (c < data_cols - 1) {
            if (*p != ',') return -1;
            p++;
        }
    }
    return 0;
}

/* ================================================================== */
/*  Carga de CSVs individuales                                        */
/* ================================================================== */

/**
 * @brief Carga matrix_A.csv → double* row-major.
 *
 * Formato:
 *   sample_id,item_000,item_001,...   (header)
 *   CTRL_001,0.000816,0.000620,...    (100 filas, 501 columnas)
 *
 * Se omite la columna 0 (sample_id).
 *
 * @return  Puntero a malloc, o NULL.  Escribe rows, cols.
 */
static double* load_csv_matrix(const char *path, int *rows, int *cols) {
    FILE *fh = fopen(path, "rb");
    if (!fh) { perror(path); return NULL; }

    char *line = NULL;
    size_t cap = 0;

    if (read_csv_line(fh, &line, &cap) != 0) {
        fprintf(stderr, "ERROR: CSV vacío: %s\n", path);
        free(line); fclose(fh); return NULL;
    }

    int total_cols = csv_fields(line);
    int data_cols  = total_cols - 1;
    if (data_cols < 1) {
        free(line); fclose(fh); return NULL;
    }

    int nrows = 0;
    while (read_csv_line(fh, &line, &cap) == 0) {
        if (line_nonempty(line)) nrows++;
    }
    if (nrows == 0) {
        free(line); fclose(fh); return NULL;
    }

    double *out = malloc((size_t)nrows * (size_t)data_cols * sizeof(double));
    if (!out) { free(line); fclose(fh); return NULL; }

    rewind(fh);
    if (read_csv_line(fh, &line, &cap) != 0) {
        free(out); free(line); fclose(fh); return NULL;
    }

    for (int r = 0; r < nrows; r++) {
        if (read_csv_line(fh, &line, &cap) != 0 ||
            parse_matrix_row(line, data_cols, out + (size_t)r * data_cols) != 0) {
            free(out); free(line); fclose(fh); return NULL;
        }
    }

    free(line);
    fclose(fh);
    *rows = nrows;
    *cols = data_cols;
    return out;
}

/**
 * @brief Carga samples.csv → int* de etiquetas.
 *
 * Formato:
 *   sample_id,label,group   (header)
 *   CTRL_001,0,healthy       (100 filas)
 *
 * Extrae columna 1 (label).
 */
static int* load_csv_labels(const char *path, int *n) {
    FILE *fh = fopen(path, "rb");
    if (!fh) { perror(path); return NULL; }

    char *line = NULL;
    size_t cap = 0;

    if (read_csv_line(fh, &line, &cap) != 0) {
        free(line); fclose(fh); return NULL;
    }

    int nrows = 0;
    while (read_csv_line(fh, &line, &cap) == 0) {
        if (line_nonempty(line)) nrows++;
    }
    if (nrows == 0) {
        free(line); fclose(fh); return NULL;
    }

    int *out = malloc((size_t)nrows * sizeof(int));
    if (!out) { free(line); fclose(fh); return NULL; }

    rewind(fh);
    if (read_csv_line(fh, &line, &cap) != 0) {
        free(out); free(line); fclose(fh); return NULL;
    }

    for (int r = 0; r < nrows; r++) {
        const char *p;
        if (read_csv_line(fh, &line, &cap) != 0 ||
            (p = csv_skip_fields(line, 1)) == NULL) {
            free(out); free(line); fclose(fh); return NULL;
        }
        out[r] = (int)strtol(p, NULL, 10);
    }

    free(line);
    fclose(fh);
    *n = nrows;
    return out;
}

/**
 * @brief Carga item_profiles.csv → double* row-major (T, S, F).
 *
 * Formato:
 *   item_id,taxon_name,T,taxon_direction,true_group,S,F   (header)
 *   item_000,Synthetic taxon 000,0.581010,neutral,neutral,0.443418,0.475849
 *
 * Extrae columnas 2 (T), 5 (S), 6 (F).
 * Asume que ningún campo contiene comas incrustadas.
 */
static double* load_csv_profiles(const char *path, int *rows, int *cols) {
    FILE *fh = fopen(path, "rb");
    if (!fh) { perror(path); return NULL; }

    char *line = NULL;
    size_t cap = 0;

    if (read_csv_line(fh, &line, &cap) != 0) {
        free(line); fclose(fh); return NULL;
    }

    int nrows = 0;
    while (read_csv_line(fh, &line, &cap) == 0) {
        if (line_nonempty(line)) nrows++;
    }
    if (nrows == 0) {
        free(line); fclose(fh); return NULL;
    }

    double *out = malloc((size_t)nrows * 3 * sizeof(double));
    if (!out) { free(line); fclose(fh); return NULL; }

    rewind(fh);
    if (read_csv_line(fh, &line, &cap) != 0) {
        free(out); free(line); fclose(fh); return NULL;
    }

    for (int r = 0; r < nrows; r++) {
        const char *pT, *pS, *pF;
        if (read_csv_line(fh, &line, &cap) != 0 ||
            (pT = csv_skip_fields(line, 2)) == NULL ||
            (pS = csv_skip_fields(line, 5)) == NULL ||
            (pF = csv_skip_fields(line, 6)) == NULL) {
            free(out); free(line); fclose(fh); return NULL;
        }
        out[r * 3 + 0] = strtod(pT, NULL);
        out[r * 3 + 1] = strtod(pS, NULL);
        out[r * 3 + 2] = strtod(pF, NULL);
    }

    free(line);
    fclose(fh);
    *rows = nrows;
    *cols = 3;
    return out;
}

/* ================================================================== */
/*  API pública                                                        */
/* ================================================================== */

/**
 * @brief Carga matrices A, etiquetas y perfiles desde archivos CSV.
 *
 * Busca en data_dir/csv/:
 *   - matrix_A.csv   → A (n_samples × n_items)
 *   - samples.csv    → y (n_samples, columna 1 = label)
 *   - item_profiles.csv → profiles (n_items × 3, columnas: 2=T, 4=S, 5=F)
 *
 * Valida consistencia de dimensiones y contenido de etiquetas {0, 1}.
 *
 * ## Formato matrix_A.csv
 *   sample_id,item_000,item_001,...   (header con 501 columnas)
 *   CTRL_001,0.000816107495,0.000619835919,...   (100 filas de datos)
 *
 * ## Formato samples.csv
 *   sample_id,label,group   (header)
 *   CTRL_001,0,healthy      (100 filas)
 *
 * ## Formato item_profiles.csv
 *   item_id,taxon_name,T,taxon_direction,S,F   (header)
 *   item_000,Synthetic taxon 000,0.581010222,CRC_enriched,0.443418457,0.475849094
 *
 * Réplica de python/common.py::load_data().
 *
 * @param data_dir  Directorio raíz de datos.
 * @param[out] ds   Dataset recién allocado (llamar free_dataset al terminar).
 * @return          0 en éxito, -1 en error con mensaje a stderr.
 */
int load_data(const char *data_dir, Dataset *ds) {
    char pA[512], pY[512], pP[512];
    fprintf(stderr, "  cargando CSV desde %s/csv/ ...\n", data_dir);
    fflush(stderr);
    snprintf(pA, sizeof(pA), "%s/csv/matrix_A.csv", data_dir);
    snprintf(pY, sizeof(pY), "%s/csv/samples.csv", data_dir);
    snprintf(pP, sizeof(pP), "%s/csv/item_profiles.csv", data_dir);

    int rA = 0, cA = 0, rP = 0, cP = 0, nY = 0;
    double *A = load_csv_matrix(pA, &rA, &cA);
    int    *y = load_csv_labels(pY, &nY);
    double *P = load_csv_profiles(pP, &rP, &cP);

    if (!A || !y || !P) {
        free(A); free(y); free(P);
        fprintf(stderr, "ERROR: no se pudieron cargar los CSV en %s/csv/\n",
                data_dir);
        return -1;
    }

    /* Validar */
    if (cP != 3) {
        fprintf(stderr, "ERROR: profiles debe tener 3 columnas\n");
        free(A); free(y); free(P); return -1;
    }
    if (cA != rP) {
        fprintf(stderr, "ERROR: columnas A (%d) != filas profiles (%d)\n",
                cA, rP);
        free(A); free(y); free(P); return -1;
    }
    if (rA != nY) {
        fprintf(stderr, "ERROR: filas A (%d) != largo y (%d)\n", rA, nY);
        free(A); free(y); free(P); return -1;
    }
    for (int i = 0; i < nY; i++)
        if (y[i] != 0 && y[i] != 1) {
            fprintf(stderr, "ERROR: y[%d] = %d (debe ser 0 o 1)\n", i, y[i]);
            free(A); free(y); free(P); return -1;
        }

    ds->A = A; ds->n_samples = rA; ds->n_items = cA;
    ds->profiles = P; ds->y = y;
    return 0;
}

static const char *cli_arg(int argc, char **argv,
                           const char *name, const char *fallback) {
    for (int i = 1; i < argc - 1; i++)
        if (strcmp(argv[i], name) == 0)
            return argv[i + 1];
    return fallback;
}

const char *parse_data_dir(int argc, char **argv) {
    const char *v = cli_arg(argc, argv, "--data-dir", NULL);
    if (!v) v = cli_arg(argc, argv, "--data", NULL);
    if (!v) return "data/processed/synthetic_CRC2000x500_balanced";
    /* quitar slash final para evitar data/foo//csv/ */
    static char buf[512];
    size_t n = strlen(v);
    if (n >= sizeof(buf)) n = sizeof(buf) - 1;
    memcpy(buf, v, n);
    buf[n] = '\0';
    while (n > 1 && buf[n - 1] == '/') {
        buf[n - 1] = '\0';
        n--;
    }
    return buf;
}

void free_dataset(Dataset *ds) {
    free(ds->A);
    free(ds->profiles);
    free(ds->y);
}

/* ================================================================== */
/*  Matriz × vector                                                    */
/* ================================================================== */

static void matvec(const double *M, int rows, int cols,
                   const double *v, double *out)
{
    for (int i = 0; i < rows; i++) {
        double sum = 0.0;
        for (int j = 0; j < cols; j++)
            sum += M[i * cols + j] * v[j];
        out[i] = sum;
    }
}

/* ================================================================== */
/*  AUC — Métrica área bajo la curva ROC                              */
/*  Implementación: estadístico U de Mann-Whitney                      */
/*  Réplica de python/common.py::auc_vector()                         */
/*                                                                     */
/*  Algoritmo:                                                         */
/*    1. Ordenar scores asc. con labels                                */
/*    2. Asignar rangos (1-indexed, promedio para ties)                */
/*    3. Sumar rangos de los positivos                                 */
/*    4. AUC = (sum_ranks_pos - n_pos*(n_pos+1)/2) / (n_pos * n_neg)  */
/* ================================================================== */

static int cmp_scored(const void *a, const void *b) {
    double sa = ((const ScoredSample *)a)->score;
    double sb = ((const ScoredSample *)b)->score;
    if (sa < sb) return -1;
    if (sa > sb) return  1;
    return 0;
}

double auc(const double *scores, const int *y, int n) {
    int n_pos = 0, n_neg = 0;
    for (int i = 0; i < n; i++) {
        if (y[i] == 1) n_pos++;
        else           n_neg++;
    }
    if (n_pos == 0 || n_neg == 0) return 0.5;

    ScoredSample *pairs = malloc((size_t)n * sizeof(ScoredSample));
    if (!pairs) return 0.5;

    for (int i = 0; i < n; i++) {
        pairs[i].score = scores[i];
        pairs[i].label = y[i];
    }
    qsort(pairs, (size_t)n, sizeof(ScoredSample), cmp_scored);

    double sum_ranks_pos = 0.0;
    int i = 0;
    while (i < n) {
        int j = i;
        while (j < n && pairs[j].score == pairs[i].score) j++;
        double avg_rank = (double)(i + 1 + j) / 2.0;
        for (int k = i; k < j; k++)
            if (pairs[k].label == 1) sum_ranks_pos += avg_rank;
        i = j;
    }
    free(pairs);

    double val = (sum_ranks_pos - (double)n_pos * (n_pos + 1) / 2.0)
                 / ((double)n_pos * n_neg);
    if (val < 0.0) val = 0.0;
    if (val > 1.0) val = 1.0;
    return val;
}

/* ================================================================== */
/*  Consistencia — máximo balanced accuracy barriendo thresholds      */
/*  Réplica de python/common.py::consistency()                        */
/*                                                                     */
/*  Algoritmo:                                                         */
/*    1. Ordenar scores asc. con labels                                */
/*    2. Barre todos los thresholds entre scores consecutivos          */
/*    3. En cada paso: (TPR + TNR) / 2                                 */
/*    4. Retorna el máximo                                             */
/*                                                                     */
/*  Inicialmente todos los puntos están "sobre" el umbral.             */
/*  Al mover el umbral hacia arriba, las muestras pasan al lado        */
/*  "bajo el umbral" una por una.                                      */
/* ================================================================== */

double consistency(const double *scores, const int *y, int n) {
    int n_pos = 0, n_neg = 0;
    for (int i = 0; i < n; i++) {
        if (y[i] == 1) n_pos++;
        else           n_neg++;
    }
    if (n_pos == 0 || n_neg == 0) return 1.0;

    ScoredSample *pairs = malloc((size_t)n * sizeof(ScoredSample));
    if (!pairs) return 0.0;

    for (int i = 0; i < n; i++) {
        pairs[i].score = scores[i];
        pairs[i].label = y[i];
    }
    qsort(pairs, (size_t)n, sizeof(ScoredSample), cmp_scored);

    /* Inicialmente todos sobre el umbral */
    int tp = 0;
    for (int i = 0; i < n; i++)
        if (pairs[i].label == 1) tp++;
    int tn = 0;

    double best = 0.0;
    for (int i = 0; i < n; i++) {
        double tpr = (n_pos > 0) ? (double)tp / n_pos : 1.0;
        double tnr = (n_neg > 0) ? (double)tn / n_neg : 1.0;
        double bal = (tpr + tnr) / 2.0;
        if (bal > best) best = bal;
        if (pairs[i].label == 1) tp--;
        else                     tn++;
    }

    free(pairs);
    return best;
}

/* ================================================================== */
/*  Evaluación completa                                                */
/*  Réplica de python/common.py::evaluate()                            */
/*                                                                     */
/*  Pipeline:                                                          */
/*    1. P_vec = profiles @ w       (n_items × 3 · 3 → n_items)        */
/*    2. scores = A @ P_vec         (n_samples × n_items → n_samples)  */
/*    3. AUC     = auc(scores, y)                                       */
/*    4. consist = consistency(scores, y)                               */
/* ================================================================== */

void evaluate(const double *A, int n_samples, int n_items,
              const double *profiles, const int *y,
              const double w[3],
              double *out_auc, double *out_cons)
{
    double *P_vec  = malloc((size_t)n_items  * sizeof(double));
    double *scores = malloc((size_t)n_samples * sizeof(double));

    if (!P_vec || !scores) {
        free(P_vec); free(scores);
        *out_auc = *out_cons = 0.0;
        return;
    }

    /* P_vec = profiles @ w (n_items × 3 · 3 → n_items) */
    matvec(profiles, n_items, 3, w, P_vec);
    /* scores = A @ P_vec (n_samples × n_items · n_items → n_samples) */
    matvec(A, n_samples, n_items, P_vec, scores);

    *out_auc  = auc(scores, y, n_samples);
    *out_cons = consistency(scores, y, n_samples);

    free(P_vec);
    free(scores);
}
