/**
 * @file scoring_openmp.c
 * @brief Alias de scoring_sequential (compatibilidad histórica).
 *
 * Compilar:
 *   make -C C_OpenMP_MPI scoring_openmp
 *
 * Equivalente a:
 *   make -C C_OpenMP_MPI scoring_sequential
 */
#define _GNU_SOURCE
#include "scoring_sequential.c"
