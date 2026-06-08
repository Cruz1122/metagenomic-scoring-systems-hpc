/**
 * scoring_kernel.cu — Búsqueda aleatoria en GPU con CUDA C.
 *
 * Compilar:
 *   make -C CUDA scoring_cuda
 *
 * Ejecutar:
 *   ./CUDA/scoring_cuda --k 10000 --seed 42 --data-dir data
 */
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#define CK(x) do { \
    cudaError_t e = (x); \
    if (e != cudaSuccess) { \
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(e)); \
        return 2; \
    } \
} while (0)

/* ------------------------------------------------------------------ */
/*  Utilidades host                                                    */
/* ------------------------------------------------------------------ */

static const char* arg(int argc, char **argv, const char *n, const char *d) {
    /* TODO: extraer argumento --name valor */
    return d;
}

static std::vector<double> mat(const std::string &p, int &r, int &c) {
    /* TODO: leer matriz CSV a vector */
    return {};
}

static std::vector<int> lab(const std::string &p) {
    /* TODO: leer etiquetas CSV a vector */
    return {};
}

/* ------------------------------------------------------------------ */
/*  Kernel CUDA                                                        */
/* ------------------------------------------------------------------ */

__device__ double aucd(double s[10], const int *y) {
    /* TODO: AUC para 10 muestras (device) */
    return 0.0;
}

__global__ void kernel(const double *A, const double *P, const int *y,
                        const double *W, double *auc,
                        int N, int K) {
    /* TODO: cada hilo evalúa un candidato W[k] */
}

/* ------------------------------------------------------------------ */
/*  Main                                                               */
/* ------------------------------------------------------------------ */

int main(int argc, char **argv) {
    int K = atoi(arg(argc, argv, "--k", "10000"));
    int seed = atoi(arg(argc, argv, "--seed", "42"));
    std::string d = arg(argc, argv, "--data-dir", "data");

    /* TODO: cargar datos, generar candidatos, copiar a GPU, lanzar kernel,
       recoger mejor resultado, imprimir CSV */
    fprintf(stderr, "CUDA scaffold — implementar búsqueda\n");
    return 1;
}
