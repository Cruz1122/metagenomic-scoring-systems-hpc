/**
 * @file ziggurat.h
 * @brief Generador Exponential(1) vía ziggurat — réplica de numpy.
 *
 * El algoritmo y tablas son idénticos a
 * numpy/random/src/distributions/random_standard_exponential.c
 * y ziggurat_constants.h.
 *
 * Dependencias: stdint.h
 */
#pragma once
#include <stdint.h>

/**
 * @brief Genera un variate ~ Exponential(1) usando ziggurat.
 *
 * Réplica exacta de numpy's random_standard_exponential().
 * Consume 1 × next_uint64 del PCG64 en el caso común (~98.9%).
 *
 * @param pcg_state  Estado PCG64: arreglo[4] {state_hi,state_lo,inc_hi,inc_lo}
 * @return           double con distribución Exponential(1).
 */
double standard_exponential(uint64_t *pcg_state);
