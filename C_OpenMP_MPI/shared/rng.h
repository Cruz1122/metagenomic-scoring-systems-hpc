/**
 * @file rng.h
 * @brief RNG compatible con numpy — SeedSequence + PCG64 + Dirichlet.
 *
 * Provee todo lo necesario para generar secuencias de números aleatorios
 * idénticas a numpy (PCG64 XSH-RR con SeedSequence), incluyendo
 * muestreo Dirichlet(1,1,1).
 *
 * Dependencias: <stdint.h>
 */
#pragma once
#include <stdint.h>
#include <math.h>

/**
 * @brief Inicializa estado PCG64[4] desde una semilla usando SeedSequence.
 *
 * s[0] = state_hi, s[1] = state_lo (128-bit state)
 * s[2] = inc_hi,   s[3] = inc_lo   (128-bit increment)
 *
 * Réplica de numpy::PCG64(seed).
 *
 * @param s     Arreglo[4] uint64 (se llena).
 * @param seed  Semilla (ej. 42).
 */
void pcg64_seed(uint64_t *s, uint64_t seed);

/**
 * @brief Genera uint64 pseudo-aleatorio via PCG64 XSH-RR.
 *
 * Paso LCG:  state = state * MULT + inc
 * Output:    rotr64(state_hi ^ state_lo, state_hi >> 58)
 *
 * Réplica de numpy::PCG64.random_raw().
 *
 * @param s  Estado PCG64[4] (se muta).
 * @return   uint64 pseudo-aleatorio.
 */
uint64_t xs(uint64_t *s);

/**
 * @brief Genera double uniforme en [0, 1) con 53 bits.
 *
 * @param s  Estado PCG64[4] (se muta).
 * @return   double en [0.0, 1.0).
 */
double u01(uint64_t *s);

/**
 * @brief Muestra Dirichlet(1,1,1) — réplica exacta de numpy.
 *
 * Genera 3 × Exponential(1) via ziggurat, luego normaliza.
 *
 * @param s      Estado PCG64[4].
 * @param[out] w  Pesos [w1, w2, w3] con suma ≈ 1.
 */
void simplex(uint64_t *s, double w[3]);

/**
 * @brief Muestra Gamma(alpha, 1) para alpha > 0.
 *
 * Marsaglia-Tsang (alpha >= 1) + small-alpha rejection (0 < alpha < 1).
 * Box-Muller para normal variate interna.
 *
 * @param alpha  Shape parameter (must be > 0).
 * @param pcg    Estado PCG64 (se muta).
 * @return       Muestra ~ Gamma(alpha, 1).
 */
double gamma_sample(double alpha, uint64_t *pcg);

/**
 * @brief Muestra Dirichlet(alpha[0], alpha[1], alpha[2]).
 *
 * w_i = Gamma(alpha_i, 1) / sum_j Gamma(alpha_j, 1)
 *
 * @param alpha  Parámetros de concentración [3] (cada uno > 0).
 * @param pcg    Estado PCG64 (se muta).
 * @param[out] w  Pesos [w1, w2, w3] con suma ≈ 1.
 */
void dirichlet_general(const double alpha[3], uint64_t *pcg, double w[3]);
