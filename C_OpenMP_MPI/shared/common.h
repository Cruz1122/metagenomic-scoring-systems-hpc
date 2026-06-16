/**
 * @file common.h
 * @brief Tipos y funciones compartidas — réplica de python/common.py.
 *
 * Arquitectura:
 *   - SearchResult:  resultado de una corrida (implementation, N, K, AUC, pesos, tiempo).
 *   - ScoredSample:  par (score, label) para ordenar en métricas.
 *   - Dataset:       matrices cargadas desde CSV (A, profiles, y).
 *   - Funciones:     load_data, free_dataset, auc, consistency, evaluate.
 *
 * Dependencias: <stdint.h>
 */
#pragma once
#include <stdint.h>

/* ================================================================== */
/*  Tipos de datos  (réplica de python/common.py::SearchResult)       */
/* ================================================================== */

/**
 * @struct Best
 * @brief Resultado de una corrida de búsqueda de pesos.
 *
 * Réplica de python/common.py::SearchResult (versión simplificada).
 */
typedef struct {
    double auc;       /**< AUC del mejor hallazgo */
    double cons;      /**< Consistencia (balanced accuracy) asociada */
    double w[3];       /**< Pesos W = [w1, w2, w3] en el simplex */
    long   iter;      /**< Iteración 0-indexada donde se encontró */
} Best;

/**
 * @struct ScoredSample
 * @brief Par (score, label) para ordenar en AUC y consistencia.
 */
typedef struct {
    double score;     /**< Score producido por A @ (profiles @ w) */
    int    label;     /**< Etiqueta real {0, 1} */
} ScoredSample;

/**
 * @struct Dataset
 * @brief Contiene las tres matrices cargadas desde archivos CSV.
 *
 * Equivalente a la tupla (A, y, profiles) retornada por common.py::load_data().
 */
typedef struct {
    double *A;         /**< Matriz de abundancia  (n_samples × n_items) */
    int    n_samples;  /**< Cantidad de muestras (filas de A) */
    int    n_items;    /**< Cantidad de items/features (columnas de A) */
    double *profiles;  /**< Perfiles funcionales (n_items × 3) */
    int    *y;         /**< Etiquetas binarias (n_samples,)  {0, 1} */
} Dataset;

/* ================================================================== */
/*  Funciones  (réplica de python/common.py)                          */
/* ================================================================== */

/**
 * @brief Carga matrices A, etiquetas y perfiles desde archivos CSV.
 *
 * Busca en data_dir/csv/:
 *   - matrix_A.csv   → A (n_samples × n_items)
 *   - samples.csv    → y (n_samples)
 *   - item_profiles.csv → profiles (n_items × 3), columnas T, S, F
 *
 * Réplica de python/common.py::load_data().
 *
 * @param data_dir  Directorio raíz de datos.
 * @param[out] ds   Dataset (llamar free_dataset al terminar).
 * @return          0 en éxito, -1 en error.
 */
int    load_data(const char *data_dir, Dataset *ds);

/**
 * @brief Libera la memoria asociada a un Dataset.
 * @param ds  Dataset previamente cargado con load_data().
 */
void   free_dataset(Dataset *ds);

/**
 * @brief Calcula el Área Bajo la Curva ROC (AUC).
 *
 * Implementación vía estadístico U de Mann-Whitney.
 * Réplica de python/common.py::auc_vector().
 *
 * @param scores  Arreglo de scores (double) de largo n.
 * @param y       Arreglo de etiquetas {0, 1} de largo n.
 * @param n       Cantidad de muestras.
 * @return        AUC en [0.0, 1.0]; 0.5 si n_pos == 0 o n_neg == 0.
 */
double auc(const double *scores, const int *y, int n);

/**
 * @brief Calcula la consistencia como el máximo balanced accuracy
 *        sobre todos los umbrales posibles.
 *
 * Barre todos los puntos de corte entre scores consecutivos y
 * retorna el máximo (TPR + TNR) / 2.
 * Réplica de python/common.py::consistency().
 *
 * @param scores  Arreglo de scores (double) de largo n.
 * @param y       Arreglo de etiquetas {0, 1} de largo n.
 * @param n       Cantidad de muestras.
 * @return        Máximo balanced accuracy en [0.0, 1.0].
 */
double consistency(const double *scores, const int *y, int n);

/**
 * @brief Evalúa AUC y consistencia para un vector de pesos w.
 *
 * Pipeline:
 *   1. P = profiles @ w              (n_items × 3 · 3 → n_items)
 *   2. scores = A @ P                (n_samples × n_items → n_samples)
 *   3. auc    = auc(scores, y)
 *   4. cons   = consistency(scores, y)
 *
 * Réplica de python/common.py::evaluate().
 *
 * @param A         Matriz de abundancia (n_samples × n_items).
 * @param n_samples  Filas de A.
 * @param n_items    Columnas de A (y filas de profiles).
 * @param profiles  Perfiles funcionales (n_items × 3).
 * @param y         Etiquetas (n_samples).
 * @param w         Pesos [w1, w2, w3] en el simplex.
 * @param[out] out_auc   AUC calculada.
 * @param[out] out_cons  Consistencia calculada.
 */
void   evaluate(const double *A, int n_samples, int n_items,
                const double *profiles, const int *y,
                const double w[3],
                double *out_auc, double *out_cons);
