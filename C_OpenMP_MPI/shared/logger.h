/**
 * @file logger.h
 * @brief Logger colorido para búsqueda de scoring — réplica de python/logger.py.
 *
 * Arquitectura (idéntica a logger.py):
 *   - `Log` imprime cabecera con implementación, N, K.
 *   - `log_improvement()` imprime cada mejora de AUC con flecha dorada.
 *   - `log_complete()` imprime resumen con recuadros.
 *   - Usa ANSI escape codes; degrada gracefulmente si la terminal no soporta color.
 *
 * Dependencias: stdio.h, common.h (Best)
 */
#pragma once
#include "common.h"

/**
 * @brief Imprime cabecera de inicio: implementación, N, K.
 *
 * @param impl     Nombre de la implementación (ej. "c_sequential").
 * @param n_items  Cantidad de items (N).
 * @param k        Total de candidatos a evaluar.
 */
void log_header(const char *impl, int n_items, long k);

/**
 * @brief Imprime línea de mejora cuando se supera el mejor AUC.
 *
 * @param iteration  Iteración 0-indexada.
 * @param k          Total de candidatos.
 * @param auc        Nuevo mejor AUC.
 * @param prev_auc   AUC anterior (para el delta).
 * @param cons       Consistencia asociada.
 * @param w          Pesos que produjeron la mejora.
 */
void log_improvement(long iteration, long k,
                     double auc, double prev_auc,
                     double cons, const double w[3]);

/**
 * @brief Imprime resumen final con recuadro.
 *
 * @param impl     Nombre de la implementación.
 * @param auc      Mejor AUC encontrado.
 * @param cons     Consistencia asociada.
 * @param w        Pesos del mejor hallazgo.
 * @param time_sec Tiempo de ejecución en segundos.
 */
void log_complete(const char *impl, double auc, double cons,
                  const double w[3], double time_sec);

/**
 * @brief Reinicia el contador interno del logger.
 *
 * Útil para re-ejecuciones sin reinicializar todo el logger.
 */
void log_reset(void);
