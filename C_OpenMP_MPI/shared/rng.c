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

/* M_PI no es estándar C11; definir por si acaso */
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

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

/* ================================================================== */
/*  Box-Muller: genera ~ N(0,1) interno                                */
/* ================================================================== */

static double box_muller(uint64_t *pcg) {
    double u1 = u01(pcg);
    /* evitar log(0) */
    while (u1 == 0.0) u1 = u01(pcg);
    double u2 = u01(pcg);
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

/* ================================================================== */
/*  Gamma(alpha, 1) — Marsaglia-Tsang (alpha >= 1)                    */
/*  Fuente: https://dl.acm.org/doi/10.1145/358407.358414              */
/* ================================================================== */

static double gamma_mt(double alpha, uint64_t *pcg) {
    double d = alpha - (1.0 / 3.0);
    double c = 1.0 / sqrt(9.0 * d);
    for (;;) {
        double x = box_muller(pcg);
        double v = 1.0 + c * x;
        v = v * v * v;   /* (1 + c*x)^3 */
        if (v <= 0.0) continue;
        double u = u01(pcg);
        /* Rango de aceptación rápido (~99% de los casos) */
        if (u < 1.0 - 0.0331 * (x * x) * (x * x))
            return d * v;
        if (log(u) < 0.5 * x * x + d * (1.0 - v + log(v)))
            return d * v;
    }
}

/* ================================================================== */
/*  Gamma(alpha, 1) — Best (1983) para 0 < alpha < 1                  */
/*  Fuente: https://dl.acm.org/doi/10.2307/2347345                    */
/* ================================================================== */

static double gamma_small(double alpha, uint64_t *pcg) {
    double e = 2.71828182845904523536; /* exp(1) */
    double threshold = e / (e + alpha);
    for (;;) {
        double u = u01(pcg);
        double v = u01(pcg);
        if (u <= threshold) {
            /* Region 1: z = (v * alpha)^(1/alpha), z in (0, 1] */
            double z = pow(v * alpha, 1.0 / alpha);
            if (z <= 1.0) return z;
        } else {
            /* Region 2: z = 1 - log(v), z > 1 */
            double z = 1.0 - log(v);
            if (u <= pow(z, alpha - 1.0)) return z;
        }
    }
}

/* ================================================================== */
/*  Gamma(alpha, 1) — API pública                                      */
/* ================================================================== */

double gamma_sample(double alpha, uint64_t *pcg) {
    if (alpha <= 0.0) return 0.0;
    if (alpha == 1.0) return standard_exponential(pcg);
    if (alpha < 1.0)  return gamma_small(alpha, pcg);
    return gamma_mt(alpha, pcg);
}

/* ================================================================== */
/*  Dirichlet(alpha[0], alpha[1], alpha[2])                            */
/* ================================================================== */

void dirichlet_general(const double alpha[3], uint64_t *pcg, double w[3]) {
    double g[3], sum = 0.0;
    for (int i = 0; i < 3; i++) {
        g[i] = gamma_sample(alpha[i], pcg);
        sum += g[i];
    }
    for (int i = 0; i < 3; i++)
        w[i] = g[i] / sum;
}
