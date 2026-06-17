#!/usr/bin/env python3
"""Implementación PyCUDA del scoring metagenómico.

Tres estrategias: random, grid, hybrid
Dos modos: full (P=profiles@W, scores=A@P), precompute (scores=B@W, B=A@profiles)
El AUC y la consistencia se calculan en GPU.
"""
from __future__ import annotations
import argparse
import math
import sys
import time
from pathlib import Path
import numpy as np

_python_dir = str(Path(__file__).resolve().parent.parent / 'python')
if _python_dir not in sys.path:
    sys.path.insert(0, _python_dir)

from common import load_data
from logger import Log

# Module-level kernel references (set by _compile_kernels)
evaluate_full = None
evaluate_precompute = None

# ── CUDA kernel source ────────────────────────────────────────────────
KERNEL_CODE = """
#define MAX_SAMPLES 4096

// ── AUC por conteo de pares positivo-negativo ──
__device__ float auc_device(const float* scores, const int* labels, int n) {
    int n_pos = 0, n_neg = 0;
    for (int i = 0; i < n; i++) {
        if (labels[i] == 1) n_pos++;
        else n_neg++;
    }
    if (n_pos == 0 || n_neg == 0) return 0.5f;

    float correct = 0.0f;
    float ties = 0.0f;
    float total = (float)(n_pos * n_neg);

    for (int i = 0; i < n; i++) {
        if (labels[i] != 1) continue;
        float si = scores[i];
        for (int j = 0; j < n; j++) {
            if (labels[j] != 0) continue;
            float sj = scores[j];
            if (si > sj) correct += 1.0f;
            else if (si == sj) ties += 1.0f;
        }
    }
    return (correct + 0.5f * ties) / total;
}

// ── Consistencia (mejor balanced accuracy sobre todos los umbrales) ──
__device__ float consistency_device(const float* scores, const int* labels, int n) {
    int idx[MAX_SAMPLES];
    for (int i = 0; i < n; i++) idx[i] = i;

    for (int i = 1; i < n; i++) {
        float key_score = scores[idx[i]];
        int key_idx = idx[i];
        int j = i - 1;
        while (j >= 0 && scores[idx[j]] > key_score) {
            idx[j + 1] = idx[j];
            j--;
        }
        idx[j + 1] = key_idx;
    }

    int n_pos = 0, n_neg = 0;
    for (int i = 0; i < n; i++) {
        if (labels[i] == 1) n_pos++;
        else n_neg++;
    }

    int tp = n_pos, tn = 0;
    float best = 0.0f;
    for (int i = 0; i < n; i++) {
        float tpr = (n_pos > 0) ? (float)tp / n_pos : 1.0f;
        float tnr = (n_neg > 0) ? (float)tn / n_neg : 1.0f;
        float bal_acc = (tpr + tnr) * 0.5f;
        if (bal_acc > best) best = bal_acc;
        if (labels[idx[i]] == 1) tp--;
        else tn++;
    }
    return best;
}

// ── Kernel modo full: P = profiles @ W, scores = A @ P ──
__global__ void evaluate_full(
    const float* A, const float* profiles, const int* labels,
    const float* weights, float* out_auc, float* out_consistency,
    int n_samples, int n_items, int K)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= K) return;

    float w0 = weights[idx * 3 + 0];
    float w1 = weights[idx * 3 + 1];
    float w2 = weights[idx * 3 + 2];

    float scores[MAX_SAMPLES];
    for (int s = 0; s < n_samples; s++) scores[s] = 0.0f;

    for (int i = 0; i < n_items; i++) {
        float p = profiles[i * 3 + 0] * w0 +
                  profiles[i * 3 + 1] * w1 +
                  profiles[i * 3 + 2] * w2;
        for (int s = 0; s < n_samples; s++) {
            scores[s] += A[s * n_items + i] * p;
        }
    }

    out_auc[idx] = auc_device(scores, labels, n_samples);
    out_consistency[idx] = consistency_device(scores, labels, n_samples);
}

// ── Kernel modo precompute: scores = B @ W ──
__global__ void evaluate_precompute(
    const float* B, const int* labels,
    const float* weights, float* out_auc, float* out_consistency,
    int n_samples, int K)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= K) return;

    float w0 = weights[idx * 3 + 0];
    float w1 = weights[idx * 3 + 1];
    float w2 = weights[idx * 3 + 2];

    float scores[MAX_SAMPLES];
    for (int s = 0; s < n_samples; s++) {
        scores[s] = B[s * 3 + 0] * w0 +
                    B[s * 3 + 1] * w1 +
                    B[s * 3 + 2] * w2;
    }

    out_auc[idx] = auc_device(scores, labels, n_samples);
    out_consistency[idx] = consistency_device(scores, labels, n_samples);
}
"""


def _compile_kernels():
    """Compila los kernels CUDA con PyCUDA."""
    import pycuda.driver as cuda
    from pycuda.compiler import SourceModule
    mod = SourceModule(KERNEL_CODE, options=['-O3', '--std=c++17',
                       f'-arch=sm_{cuda.Device(0).compute_capability() // 10}{cuda.Device(0).compute_capability() % 10}'])
    evaluate_full = mod.get_function('evaluate_full')
    evaluate_precompute = mod.get_function('evaluate_precompute')
    return evaluate_full, evaluate_precompute


def _validate(A, y, profiles):
    """Valida dimensiones y etiquetas."""
    assert A.ndim == 2, f"A debe tener 2 dimensiones, tiene {A.ndim}"
    assert y.ndim == 1, f"y debe tener 1 dimensión, tiene {y.ndim}"
    assert profiles.ndim == 2, f"profiles debe tener 2 dimensiones, tiene {profiles.ndim}"
    assert profiles.shape[1] == 3, f"profiles debe tener 3 columnas, tiene {profiles.shape[1]}"
    assert A.shape[1] == profiles.shape[0], \
        f"columnas de A ({A.shape[1]}) != filas de profiles ({profiles.shape[0]})"
    assert A.shape[0] == y.shape[0], \
        f"filas de A ({A.shape[0]}) != largo de y ({y.shape[0]})"
    classes = set(np.unique(y))
    assert classes == {0, 1}, f"y debe contener clases 0 y 1, contiene {classes}"


def _evaluate_batch(gpu_data, weights_batch, block_size, stream=None):
    """Evalúa un batch de pesos en GPU, retorna (aucs, consistencies)."""
    import pycuda.driver as cuda
    import pycuda.gpuarray as gpuarray

    A_gpu, profiles_gpu, B_gpu, labels_gpu, n_samples, n_items, mode_full = gpu_data
    K_batch = weights_batch.shape[0]

    w_gpu = gpuarray.to_gpu(weights_batch.astype(np.float32).ravel())
    auc_gpu = gpuarray.empty((K_batch,), dtype=np.float32)
    cons_gpu = gpuarray.empty((K_batch,), dtype=np.float32)

    grid = (int(math.ceil(K_batch / block_size)), 1, 1)
    block = (block_size, 1, 1)

    if mode_full:
        evaluate_full(
            A_gpu, profiles_gpu, labels_gpu,
            w_gpu, auc_gpu, cons_gpu,
            np.int32(n_samples), np.int32(n_items), np.int32(K_batch),
            block=block, grid=grid, stream=stream
        )
    else:
        evaluate_precompute(
            B_gpu, labels_gpu,
            w_gpu, auc_gpu, cons_gpu,
            np.int32(n_samples), np.int32(K_batch),
            block=block, grid=grid, stream=stream
        )

    return auc_gpu.get(), cons_gpu.get()


def _find_best(aucs, consistencies, indices):
    """Encuentra el mejor candidato. Desempate: consistencia, luego índice."""
    best_i = 0
    for i in range(1, len(aucs)):
        better = False
        if aucs[i] > aucs[best_i]:
            better = True
        elif aucs[i] == aucs[best_i] and consistencies[i] > consistencies[best_i]:
            better = True
        elif (aucs[i] == aucs[best_i] and consistencies[i] == consistencies[best_i]
              and indices[i] < indices[best_i]):
            better = True
        if better:
            best_i = i
    return best_i


# ── Random search ──────────────────────────────────────────────────────
def random_search(gpu_data, K, seed, block_size, batch_size):
    """Búsqueda aleatoria: genera Dirichlet(1,1,1), evalúa en GPU."""
    rng = np.random.default_rng(seed)
    weights = rng.dirichlet(np.ones(3), size=K).astype(np.float32)

    best_auc = -np.inf
    best_cons = 0.0
    best_w = None
    best_idx = -1

    for off in range(0, K, batch_size):
        batch = weights[off:off + batch_size]
        aucs, cons = _evaluate_batch(gpu_data, batch, block_size)
        offset_indices = np.arange(off, off + len(batch))
        bi = _find_best(aucs, cons, offset_indices)
        gi = off + bi
        better = False
        if aucs[bi] > best_auc:
            better = True
        elif aucs[bi] == best_auc and cons[bi] > best_cons:
            better = True
        elif aucs[bi] == best_auc and cons[bi] == best_cons and gi < best_idx:
            better = True
        if better:
            best_auc = aucs[bi]
            best_cons = cons[bi]
            best_w = batch[bi].copy()
            best_idx = gi

    return best_auc, best_cons, tuple(best_w), best_idx


# ── Grid search ────────────────────────────────────────────────────────
def _generate_grid(resolution):
    """Genera todos los puntos del simplex 2D. Retorna (weights, actual_K)."""
    weights = []
    for i in range(resolution + 1):
        for j in range(resolution + 1 - i):
            k = resolution - i - j
            weights.append([i / resolution, j / resolution, k / resolution])
    return np.array(weights, dtype=np.float32), len(weights)


def _resolution_for_K(K):
    return max(int(math.sqrt(2.0 * K)), 1)


def grid_search(gpu_data, K_hint, block_size, batch_size, grid_resolution=0):
    """Búsqueda sistemática sobre el simplex. Reporta actual_K real."""
    resolution = grid_resolution if grid_resolution > 0 else _resolution_for_K(K_hint)
    weights, actual_K = _generate_grid(resolution)

    best_auc = -np.inf
    best_cons = 0.0
    best_w = None
    best_idx = -1

    for off in range(0, actual_K, batch_size):
        batch = weights[off:off + batch_size]
        aucs, cons = _evaluate_batch(gpu_data, batch, block_size)
        offset_indices = np.arange(off, off + len(batch))
        bi = _find_best(aucs, cons, offset_indices)
        gi = off + bi
        better = False
        if aucs[bi] > best_auc:
            better = True
        elif aucs[bi] == best_auc and cons[bi] > best_cons:
            better = True
        elif aucs[bi] == best_auc and cons[bi] == best_cons and gi < best_idx:
            better = True
        if better:
            best_auc = aucs[bi]
            best_cons = cons[bi]
            best_w = batch[bi].copy()
            best_idx = gi

    return best_auc, best_cons, tuple(best_w), best_idx, actual_K


# ── Hybrid search ──────────────────────────────────────────────────────
def hybrid_search(gpu_data, K, seed, block_size, batch_size):
    """Tres fases: grid (20%) + random (60%) + local (20%)."""
    rng = np.random.default_rng(seed)

    K_grid = min(int(K * 0.2), 2000)
    resolution = _resolution_for_K(K_grid)
    grid_w, actual_grid = _generate_grid(resolution)
    K_grid = actual_grid

    K_random = int(K * 0.6)
    K_local = K - K_grid - K_random
    if K_local < 0:
        K_random = K - K_grid
        K_local = 0

    best_auc = -np.inf
    best_cons = 0.0
    best_w = None
    best_idx = -1

    def _eval_and_track(batch, offset):
        nonlocal best_auc, best_cons, best_w, best_idx
        if len(batch) == 0:
            return
        aucs, cons = _evaluate_batch(gpu_data, batch, block_size)
        offset_indices = np.arange(offset, offset + len(batch))
        bi = _find_best(aucs, cons, offset_indices)
        gi = offset + bi
        better = False
        if aucs[bi] > best_auc:
            better = True
        elif aucs[bi] == best_auc and cons[bi] > best_cons:
            better = True
        elif aucs[bi] == best_auc and cons[bi] == best_cons and gi < best_idx:
            better = True
        if better:
            best_auc = aucs[bi]
            best_cons = cons[bi]
            best_w = batch[bi].copy()
            best_idx = gi

    # Fase 1: Grid
    for off in range(0, K_grid, batch_size):
        _eval_and_track(grid_w[off:off + batch_size], off)

    # Fase 2: Random
    random_w = rng.dirichlet(np.ones(3), size=K_random).astype(np.float32)
    for off in range(0, K_random, batch_size):
        _eval_and_track(random_w[off:off + batch_size], K_grid + off)

    # Fase 3: Local
    if K_local > 0 and best_w is not None:
        alpha = np.maximum(np.array(best_w, dtype=np.float32) * 100.0, 1e-3)
        local_w = rng.dirichlet(alpha, size=K_local).astype(np.float32)
        for off in range(0, K_local, batch_size):
            _eval_and_track(local_w[off:off + batch_size], K_grid + K_random + off)

    return best_auc, best_cons, best_w, best_idx


# ── Timed search ───────────────────────────────────────────────────────
def timed_search(name, parallel_units, A, y, profiles, K, seed,
                 log=None, search_mode='random', block_size=256, batch_size=1000000,
                 mode='full', grid_resolution=0):
    """Ejecuta búsqueda con medición de tiempo. El AUC se calcula en GPU."""
    import pycuda.driver as cuda
    import pycuda.gpuarray as gpuarray

    n_samples, n_items = A.shape
    A_cont = np.ascontiguousarray(A.astype(np.float32))
    P_cont = np.ascontiguousarray(profiles.astype(np.float32))
    y_int = np.ascontiguousarray(y.astype(np.int32))

    A_gpu = gpuarray.to_gpu(A_cont)
    profiles_gpu = gpuarray.to_gpu(P_cont)
    labels_gpu = gpuarray.to_gpu(y_int)

    mode_full = (mode == 'full')

    if not mode_full:
        B = (A_cont @ P_cont).astype(np.float32)
        B_gpu = gpuarray.to_gpu(B)
    else:
        B_gpu = None

    gpu_data = (A_gpu, profiles_gpu, B_gpu, labels_gpu, n_samples, n_items, mode_full)

    start = time.perf_counter()

    if search_mode == 'grid':
        best_auc, best_consistency, best_weights, best_iter, actual_k = \
            grid_search(gpu_data, K, block_size, batch_size, grid_resolution)
        K = actual_k
    elif search_mode == 'hybrid':
        best_auc, best_consistency, best_weights, best_iter = \
            hybrid_search(gpu_data, K, seed, block_size, batch_size)
    else:
        best_auc, best_consistency, best_weights, best_iter = \
            random_search(gpu_data, K, seed, block_size, batch_size)

    elapsed = time.perf_counter() - start

    # Ensure best_w is a tuple
    if best_weights is None:
        best_weights = (1.0 / 3, 1.0 / 3, 1.0 / 3)
    elif isinstance(best_weights, np.ndarray):
        best_weights = tuple(best_weights.tolist())

    out = {
        'implementation': name,
        'search': search_mode,
        'mode': mode,
        'K': K,
        'actual_k': K if search_mode != 'grid' else actual_k,
        'N': n_items,
        'best_auc': best_auc,
        'best_consistency': best_consistency,
        'best_w0': best_weights[0],
        'best_w1': best_weights[1],
        'best_w2': best_weights[2],
        'time_sec': elapsed,
        'seed': seed,
        'block_size': block_size,
    }

    if log is not None:
        best_iter_display = best_iter if best_iter is not None else -1
        log.improvement(best_iter_display, best_auc, best_consistency, best_weights)

    return out


# ── Main ───────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description='Scoring metagenómico — PyCUDA')
    ap.add_argument('--k', type=int, default=10000, help='Número de candidatos')
    ap.add_argument('--seed', type=int, default=42, help='Semilla RNG')
    ap.add_argument('--search', choices=['random', 'grid', 'hybrid'],
                    default='random', help='Estrategia de búsqueda')
    ap.add_argument('--data-dir', type=Path, default=Path('data/npy'),
                    help='Directorio de datos')
    ap.add_argument('--block-size', type=int, default=256, help='Tamaño de bloque CUDA')
    ap.add_argument('--batch-size', type=int, default=1000000,
                    help='Candidatos por batch')
    ap.add_argument('--mode', choices=['full', 'precompute'], default='full',
                    help='Modo de evaluación')
    ap.add_argument('--grid-resolution', type=int, default=0,
                    help='Resolución del grid (0 = automática)')
    ap.add_argument('--csv', action='store_true', help='Salida en CSV')
    args = ap.parse_args()

    try:
        import pycuda.autoinit  # noqa
        import pycuda.driver as cuda
    except Exception as e:
        print(f'PyCUDA no disponible: {e}', file=sys.stderr)
        raise SystemExit(2)

    global evaluate_full, evaluate_precompute
    evaluate_full, evaluate_precompute = _compile_kernels()

    A, y, profiles = load_data(args.data_dir)
    _validate(A, y, profiles)

    n_samples, n_items = A.shape
    log = None if args.csv else Log('pycuda', n_items, args.k)

    gpu_cores = 1  # placeholder for parallel_units in SearchResult
    result = timed_search('pycuda', gpu_cores, A, y, profiles,
                          args.k, args.seed, log=log,
                          search_mode=args.search,
                          block_size=args.block_size,
                          batch_size=args.batch_size,
                          mode=args.mode,
                          grid_resolution=args.grid_resolution)

    # ── Salida CSV obligatoria ──
    # implementation,search,mode,K,actual_k,N,best_auc,best_consistency,
    # best_w0,best_w1,best_w2,time_sec,seed,block_size
    print(f"{result['implementation']},{result['search']},{result['mode']},"
          f"{result['K']},{result['actual_k']},{result['N']},"
          f"{result['best_auc']:.9f},{result['best_consistency']:.9f},"
          f"{result['best_w0']:.9f},{result['best_w1']:.9f},{result['best_w2']:.9f},"
          f"{result['time_sec']:.9f},{result['seed']},{result['block_size']}")


if __name__ == '__main__':
    main()
