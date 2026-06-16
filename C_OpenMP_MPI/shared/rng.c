/**
 * @file rng.c
 * @brief RNG compatible con numpy — SeedSequence + PCG64 + Dirichlet.
 *
 * Contiene:
 *   - SeedSequence (ss_hashmix, ss_mix, ss_init, ss_generate)
 *   - PCG64 XSH-RR (pcg64_seed, xs, u01)
 *   - Dirichlet(1,1,1) (simplex)
 *
 * Dependencias: rng.h, ziggurat.h, <math.h>, <stdint.h>.
 *
 * ## SeedSequence
 *   Réplica de numpy.random.SeedSequence.
 *   Fuente: numpy/random/bit_generator.pyx  (clase SeedSequence)
 *   Algoritmo: mezcla uint32 con hashmix/mix (sin SHA, puro aritmético).
 *
 * ## PCG64
 *   Réplica de numpy.random.PCG64 (PCG XSL RR 128/64).
 *   Fuente: numpy/random/src/pcg64/pcg64.h
 *
 * ## Dirichlet
 *   Réplica de numpy: w_i = Gamma(1,1)_i / sum(Gamma(1,1)_j)
 *   donde Gamma(1,1) = Exponential(1) via ziggurat.
 */
#include "rng.h"
#include "ziggurat.h"
#include <math.h>

/* ================================================================== */
/*  SeedSequence                                                       */
/* ================================================================== */

#define SS_INIT_A   0x43b0d7e5U
#define SS_MULT_A   0x931e8875U
#define SS_INIT_B   0x8b51f9ddU
#define SS_MULT_B   0x58f38dedU
#define SS_MIX_L    0xca01f9ddU
#define SS_MIX_R    0x4973f715U
#define SS_XSHIFT   16U
#define SS_MASK32   0xFFFFFFFFU
#define SS_POOL_SZ  4U

static uint32_t ss_hashmix(uint32_t value, uint32_t *hc) {
    value ^= *hc;
    *hc = (*hc * SS_MULT_A) & SS_MASK32;
    value = (value * *hc) & SS_MASK32;
    value ^= (value >> SS_XSHIFT);
    return value;
}

static uint32_t ss_mix(uint32_t x, uint32_t y) {
    uint32_t r = (SS_MIX_L * x - SS_MIX_R * y) & SS_MASK32;
    r ^= (r >> SS_XSHIFT);
    return r;
}

static void ss_init(uint32_t pool[SS_POOL_SZ], uint64_t seed) {
    uint32_t entropy[2];
    int n_entropy;
    if (seed <= SS_MASK32) {
        entropy[0] = (uint32_t)seed;
        n_entropy = 1;
    } else {
        entropy[0] = (uint32_t)(seed & SS_MASK32);
        entropy[1] = (uint32_t)(seed >> 32);
        n_entropy = 2;
    }
    for (unsigned i = 0; i < SS_POOL_SZ; i++)
        pool[i] = 0;
    uint32_t hc = SS_INIT_A;
    for (unsigned i = 0; i < SS_POOL_SZ; i++) {
        uint32_t val = (i < (unsigned)n_entropy) ? entropy[i] : 0U;
        pool[i] = ss_hashmix(val, &hc);
    }
    for (unsigned src = 0; src < SS_POOL_SZ; src++) {
        for (unsigned dst = 0; dst < SS_POOL_SZ; dst++) {
            if (src != dst)
                pool[dst] = ss_mix(pool[dst], ss_hashmix(pool[src], &hc));
        }
    }
}

static void ss_generate(uint32_t pool[SS_POOL_SZ], uint64_t *out,
                        unsigned n_words)
{
    unsigned n_u32 = n_words * 2;
    uint32_t buf[64];
    if (n_u32 > 64) n_u32 = 64;
    uint32_t hc = SS_INIT_B;
    for (unsigned i = 0; i < n_u32; i++) {
        uint32_t dv = pool[i % SS_POOL_SZ];
        dv ^= hc;
        hc = (hc * SS_MULT_B) & SS_MASK32;
        dv = (dv * hc) & SS_MASK32;
        dv ^= (dv >> SS_XSHIFT);
        buf[i] = dv;
    }
    for (unsigned i = 0; i < n_words; i++)
        out[i] = (uint64_t)buf[2 * i] | ((uint64_t)buf[2 * i + 1] << 32);
}

/* ================================================================== */
/*  PCG64 — Permutation Congruential Generator (XSH-RR)               */
/* ================================================================== */

#define PCG_MULT_HI 2549297995355413924ULL
#define PCG_MULT_LO 4865540595714422341ULL

static inline uint64_t rotr64(uint64_t x, unsigned r) {
    return (x >> r) | (x << ((-r) & 63));
}

void pcg64_seed(uint64_t *s, uint64_t seed) {
    uint32_t pool[SS_POOL_SZ];
    uint64_t ss_out[4];
    ss_init(pool, seed);
    ss_generate(pool, ss_out, 4);
    __uint128_t initstate = ((__uint128_t)ss_out[0] << 64) | ss_out[1];
    __uint128_t initseq   = ((__uint128_t)ss_out[2] << 64) | ss_out[3];
    __uint128_t inc = (initseq << 1) | 1;
    __uint128_t mult = ((__uint128_t)PCG_MULT_HI << 64) | PCG_MULT_LO;
    __uint128_t state = 0;
    state = state * mult + inc;
    state += initstate;
    state = state * mult + inc;
    s[0] = (uint64_t)(state >> 64);
    s[1] = (uint64_t)state;
    s[2] = (uint64_t)(inc >> 64);
    s[3] = (uint64_t)inc;
}

uint64_t xs(uint64_t *s) {
    __uint128_t st  = ((__uint128_t)s[0] << 64) | s[1];
    __uint128_t inc = ((__uint128_t)s[2] << 64) | s[3];
    __uint128_t mult = ((__uint128_t)PCG_MULT_HI << 64) | PCG_MULT_LO;
    st = st * mult + inc;
    s[0] = (uint64_t)(st >> 64);
    s[1] = (uint64_t)st;
    uint64_t hi = s[0], lo = s[1];
    unsigned rot = (unsigned)(hi >> 58);
    return rotr64(hi ^ lo, rot);
}

double u01(uint64_t *s) {
    return (xs(s) >> 11) * (1.0 / 9007199254740992.0);
}

/* ================================================================== */
/*  Dirichlet(1,1,1)                                                   */
/*  Gamma(1,1) = Exponential(1) via ziggurat (idéntico a numpy)       */
/* ================================================================== */

void simplex(uint64_t *s, double w[3]) {
    double a = standard_exponential(s);
    double b = standard_exponential(s);
    double c = standard_exponential(s);
    double sum = a + b + c;
    w[0] = a / sum;
    w[1] = b / sum;
    w[2] = c / sum;
}
