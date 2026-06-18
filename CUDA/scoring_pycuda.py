#!/usr/bin/env python3
"""Implementación PyCUDA del scoring metagenómico.

Tres estrategias: random, grid, hybrid
Dos modos: full (P=profiles@W, scores=A@P), precompute (scores=B@W, B=A@profiles)
El AUC y la consistencia se calculan en GPU.
"""
from __future__ import annotations
import argparse
import math
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
import numpy as np

_python_dir = str(Path(__file__).resolve().parent.parent / 'python')
if _python_dir not in sys.path:
    sys.path.insert(0, _python_dir)

from common import SearchResult, load_data
from logger import Log

# Module-level kernel references (set by _compile_kernels)
evaluate_full = None
evaluate_precompute = None
reduce_best_stage1 = None
reduce_best_stage1_from = None
reduce_best_stage2 = None

DEFAULT_BLOCK_SIZE = 256
LOG_INTERVAL = 32

# ── CUDA kernel source ────────────────────────────────────────────────
KERNEL_CODE = """
#define MAX_SAMPLES 4096
#define BLOCK_SIZE 256

struct BestVal {
    float auc;
    float consistency;
    int index;
};

__device__ bool is_better_dev(const BestVal& a, const BestVal& b) {
    if (a.auc > b.auc) return true;
    if (a.auc < b.auc) return false;
    if (a.consistency > b.consistency) return true;
    if (a.consistency < b.consistency) return false;
    return a.index < b.index;
}

// ── AUC Mann-Whitney (réplica de common.c / sklearn); scores en double ──
__device__ float auc_device(const double* scores, const int* labels, int n) {
    int n_pos = 0, n_neg = 0;
    for (int i = 0; i < n; i++) {
        if (labels[i] == 1) n_pos++;
        else n_neg++;
    }
    if (n_pos == 0 || n_neg == 0) return 0.5f;

    int idx[MAX_SAMPLES];
    for (int i = 0; i < n; i++) idx[i] = i;

    for (int i = 1; i < n; i++) {
        int key = idx[i];
        double key_score = scores[key];
        int j = i - 1;
        while (j >= 0 && scores[idx[j]] > key_score) {
            idx[j + 1] = idx[j];
            j--;
        }
        idx[j + 1] = key;
    }

    double sum_ranks_pos = 0.0;
    int i = 0;
    while (i < n) {
        int j = i;
        double s = scores[idx[i]];
        while (j < n && scores[idx[j]] == s) j++;
        double avg_rank = ((double)(i + 1) + (double)j) / 2.0;
        for (int k = i; k < j; k++) {
            if (labels[idx[k]] == 1) sum_ranks_pos += avg_rank;
        }
        i = j;
    }

    double val = (sum_ranks_pos - (double)n_pos * (n_pos + 1) / 2.0)
                 / ((double)n_pos * n_neg);
    if (val < 0.0) val = 0.0;
    if (val > 1.0) val = 1.0;
    return (float)val;
}

// ── Consistencia (mejor balanced accuracy sobre todos los umbrales) ──
__device__ float consistency_device(const double* scores, const int* labels, int n) {
    int idx[MAX_SAMPLES];
    for (int i = 0; i < n; i++) idx[i] = i;

    for (int i = 1; i < n; i++) {
        int key = idx[i];
        double key_score = scores[key];
        int j = i - 1;
        while (j >= 0 && scores[idx[j]] > key_score) {
            idx[j + 1] = idx[j];
            j--;
        }
        idx[j + 1] = key;
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
// Memoria compartida cachea la columna A[:, i] (filas por muestra) por bloque.
__global__ void evaluate_full(
    const float* __restrict__ A,
    const float* __restrict__ profiles,
    const int* __restrict__ labels,
    const float* __restrict__ weights,
    float* __restrict__ out_auc,
    float* __restrict__ out_consistency,
    int n_samples, int n_items, int K, int index_offset)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int global_idx = index_offset + idx;
    bool active = (idx < K);
    extern __shared__ float sh_col[];

    double w0 = 0.0, w1 = 0.0, w2 = 0.0;
    if (active) {
        w0 = (double)weights[global_idx * 3 + 0];
        w1 = (double)weights[global_idx * 3 + 1];
        w2 = (double)weights[global_idx * 3 + 2];
    }

    double scores[MAX_SAMPLES];
    if (active) {
        for (int s = 0; s < n_samples; s++) scores[s] = 0.0;
    }

    for (int i = 0; i < n_items; i++) {
        for (int s = threadIdx.x; s < n_samples; s += blockDim.x)
            sh_col[s] = A[s * n_items + i];
        __syncthreads();

        if (active) {
            double p = (double)profiles[i * 3 + 0] * w0 +
                       (double)profiles[i * 3 + 1] * w1 +
                       (double)profiles[i * 3 + 2] * w2;
            for (int s = 0; s < n_samples; s++)
                scores[s] += (double)sh_col[s] * p;
        }
        __syncthreads();
    }

    if (active) {
        out_auc[global_idx] = auc_device(scores, labels, n_samples);
        out_consistency[global_idx] = consistency_device(scores, labels, n_samples);
    }
}

// ── Kernel modo precompute: scores = B @ W ──
__global__ void evaluate_precompute(
    const float* __restrict__ B,
    const int* __restrict__ labels,
    const float* __restrict__ weights,
    float* __restrict__ out_auc,
    float* __restrict__ out_consistency,
    int n_samples, int K, int index_offset)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= K) return;

    int global_idx = index_offset + idx;
    double w0 = (double)weights[global_idx * 3 + 0];
    double w1 = (double)weights[global_idx * 3 + 1];
    double w2 = (double)weights[global_idx * 3 + 2];

    double scores[MAX_SAMPLES];
    for (int s = 0; s < n_samples; s++) {
        scores[s] = (double)B[s * 3 + 0] * w0 +
                    (double)B[s * 3 + 1] * w1 +
                    (double)B[s * 3 + 2] * w2;
    }

    out_auc[global_idx] = auc_device(scores, labels, n_samples);
    out_consistency[global_idx] = consistency_device(scores, labels, n_samples);
}

// ── Reduction stage 1: auc/cons arrays → partial best por bloque CUDA ──
__global__ void reduce_best_stage1(
    const float* __restrict__ aucs,
    const float* __restrict__ consistencies,
    BestVal* __restrict__ partial,
    int K)
{
    extern __shared__ BestVal sdata[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    if (idx < K) {
        sdata[tid].auc = aucs[idx];
        sdata[tid].consistency = consistencies[idx];
        sdata[tid].index = idx;
    } else {
        sdata[tid].auc = -1.0f;
        sdata[tid].consistency = -1.0f;
        sdata[tid].index = K;
    }
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (is_better_dev(sdata[tid + s], sdata[tid]))
                sdata[tid] = sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0)
        partial[blockIdx.x] = sdata[0];
}

// ── Reduction stage 1 (BestVal → partial) para passes iterativos ──
__global__ void reduce_best_stage1_from(
    const BestVal* __restrict__ input,
    BestVal* __restrict__ partial,
    int K)
{
    extern __shared__ BestVal sdata[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    if (idx < K)
        sdata[tid] = input[idx];
    else {
        sdata[tid].auc = -1.0f;
        sdata[tid].consistency = -1.0f;
        sdata[tid].index = K;
    }
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (is_better_dev(sdata[tid + s], sdata[tid]))
                sdata[tid] = sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0)
        partial[blockIdx.x] = sdata[0];
}

// ── Reduction stage 2: partial[] → escalar global ──
__global__ void reduce_best_stage2(
    const BestVal* __restrict__ partial,
    float* __restrict__ out_best_auc,
    float* __restrict__ out_best_consistency,
    int* __restrict__ out_best_index,
    int n_blocks)
{
    extern __shared__ BestVal sdata[];
    int tid = threadIdx.x;

    if (tid < n_blocks)
        sdata[tid] = partial[tid];
    else {
        sdata[tid].auc = -1.0f;
        sdata[tid].consistency = -1.0f;
        sdata[tid].index = n_blocks;
    }
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (is_better_dev(sdata[tid + s], sdata[tid]))
                sdata[tid] = sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        *out_best_auc = sdata[0].auc;
        *out_best_consistency = sdata[0].consistency;
        *out_best_index = sdata[0].index;
    }
}
"""


def _setup_cuda_toolkit() -> None:
    """Añade nvcc y libs de CUDA al entorno si el toolkit no está en PATH."""
    if shutil.which('nvcc'):
        return

    candidates: list[str] = []
    if os.environ.get('CUDA_HOME'):
        candidates.append(os.environ['CUDA_HOME'])
    candidates.extend(['/opt/cuda', '/usr/local/cuda'])

    for root in candidates:
        nvcc = os.path.join(root, 'bin', 'nvcc')
        if not os.path.isfile(nvcc):
            continue
        os.environ['CUDA_HOME'] = root
        os.environ['PATH'] = os.path.join(root, 'bin') + os.pathsep + os.environ.get('PATH', '')
        for lib_dir in (
            os.path.join(root, 'targets', 'x86_64-linux', 'lib'),
            os.path.join(root, 'lib64'),
            os.path.join(root, 'lib'),
        ):
            if os.path.isdir(lib_dir):
                prev = os.environ.get('LD_LIBRARY_PATH', '')
                os.environ['LD_LIBRARY_PATH'] = lib_dir + (os.pathsep + prev if prev else '')
        return

    print('ERROR: nvcc no encontrado. Instala CUDA toolkit o exporta CUDA_HOME.',
          file=sys.stderr)
    raise SystemExit(2)


def _compile_kernels():
    """Compila los kernels CUDA con PyCUDA."""
    import pycuda.driver as cuda
    from pycuda.compiler import SourceModule
    cc = cuda.Device(0).compute_capability()
    if isinstance(cc, tuple):
        arch = f'sm_{cc[0]}{cc[1]}'
    else:
        arch = f'sm_{cc // 10}{cc % 10}'
    mod = SourceModule(KERNEL_CODE, options=['-O3', '--std=c++17', f'-arch={arch}'])
    return (
        mod.get_function('evaluate_full'),
        mod.get_function('evaluate_precompute'),
        mod.get_function('reduce_best_stage1'),
        mod.get_function('reduce_best_stage1_from'),
        mod.get_function('reduce_best_stage2'),
    )


def _grid_size(k: int, block_size: int) -> int:
    return max(int(math.ceil(k / block_size)), 1)


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


@dataclass
class GPUContext:
    """Buffers GPU persistentes: H2D estático una vez, evaluación por offset."""
    A_gpu: object
    profiles_gpu: object
    B_gpu: object | None
    labels_gpu: object
    weights_gpu: object
    auc_gpu: object
    cons_gpu: object
    partial_a: object
    partial_b: object
    best_auc_gpu: object
    best_cons_gpu: object
    best_idx_gpu: object
    n_samples: int
    n_items: int
    mode_full: bool
    block_size: int
    shared_eval_bytes: int
    shared_reduce_bytes: int

    @classmethod
    def create(cls, A, y, profiles, k_max: int, mode: str = 'full',
               block_size: int = DEFAULT_BLOCK_SIZE):
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

        weights_gpu = gpuarray.empty((k_max * 3,), dtype=np.float32)
        auc_gpu = gpuarray.empty((k_max,), dtype=np.float32)
        cons_gpu = gpuarray.empty((k_max,), dtype=np.float32)

        grid_max = _grid_size(k_max, block_size)
        bestval_dtype = np.dtype([('auc', 'f4'), ('consistency', 'f4'), ('index', 'i4')])
        partial_a = gpuarray.empty((grid_max,), dtype=bestval_dtype)
        partial_b = gpuarray.empty((grid_max,), dtype=bestval_dtype)

        best_auc_gpu = gpuarray.zeros((1,), dtype=np.float32)
        best_cons_gpu = gpuarray.zeros((1,), dtype=np.float32)
        best_idx_gpu = gpuarray.zeros((1,), dtype=np.int32)

        shared_eval_bytes = n_samples * 4
        shared_reduce_bytes = block_size * bestval_dtype.itemsize

        return cls(
            A_gpu=A_gpu,
            profiles_gpu=profiles_gpu,
            B_gpu=B_gpu,
            labels_gpu=labels_gpu,
            weights_gpu=weights_gpu,
            auc_gpu=auc_gpu,
            cons_gpu=cons_gpu,
            partial_a=partial_a,
            partial_b=partial_b,
            best_auc_gpu=best_auc_gpu,
            best_cons_gpu=best_cons_gpu,
            best_idx_gpu=best_idx_gpu,
            n_samples=n_samples,
            n_items=n_items,
            mode_full=mode_full,
            block_size=block_size,
            shared_eval_bytes=shared_eval_bytes,
            shared_reduce_bytes=shared_reduce_bytes,
        )

    def upload_weights(self, weights: np.ndarray, offset: int = 0) -> None:
        """Sube pesos al buffer device (offset en candidatos, no en floats)."""
        flat = weights.astype(np.float32).ravel()
        if offset == 0 and flat.size == self.weights_gpu.size:
            self.weights_gpu.set(flat)
        else:
            self.weights_gpu[offset * 3:offset * 3 + flat.size].set(flat)

    def evaluate(self, k_chunk: int, index_offset: int = 0, stream=None) -> None:
        """Lanza evaluate_* sobre k_chunk candidatos en index_offset."""
        grid = (_grid_size(k_chunk, self.block_size), 1, 1)
        block = (self.block_size, 1, 1)
        kwargs = dict(block=block, grid=grid, stream=stream)

        if self.mode_full:
            evaluate_full(
                self.A_gpu, self.profiles_gpu, self.labels_gpu,
                self.weights_gpu, self.auc_gpu, self.cons_gpu,
                np.int32(self.n_samples), np.int32(self.n_items),
                np.int32(k_chunk), np.int32(index_offset),
                shared=self.shared_eval_bytes,
                **kwargs,
            )
        else:
            evaluate_precompute(
                self.B_gpu, self.labels_gpu,
                self.weights_gpu, self.auc_gpu, self.cons_gpu,
                np.int32(self.n_samples), np.int32(k_chunk), np.int32(index_offset),
                **kwargs,
            )

    def fetch_chunk_results(self, k_chunk: int, index_offset: int = 0):
        """D2H parcial de auc/cons para un chunk (modo logging)."""
        sl = slice(index_offset, index_offset + k_chunk)
        return self.auc_gpu[sl].get(), self.cons_gpu[sl].get()

    def reduce_best(self, k_total: int) -> tuple[float, float, int]:
        """Reduction en GPU del mejor global entre k_total candidatos."""
        import pycuda.driver as cuda

        n = k_total
        src_auc, src_cons = self.auc_gpu, self.cons_gpu
        input_vals = None
        out_partial = self.partial_a
        alt_partial = self.partial_b

        block = (self.block_size, 1, 1)

        while True:
            grid_n = _grid_size(n, self.block_size)
            if input_vals is None:
                reduce_best_stage1(
                    src_auc, src_cons, out_partial,
                    np.int32(n),
                    block=block,
                    grid=(grid_n, 1, 1),
                    shared=self.shared_reduce_bytes,
                )
            else:
                reduce_best_stage1_from(
                    input_vals, out_partial,
                    np.int32(n),
                    block=block,
                    grid=(grid_n, 1, 1),
                    shared=self.shared_reduce_bytes,
                )

            if grid_n == 1:
                best = out_partial.get()[0]
                return float(best['auc']), float(best['consistency']), int(best['index'])

            if grid_n <= self.block_size:
                reduce_best_stage2(
                    out_partial,
                    self.best_auc_gpu, self.best_cons_gpu, self.best_idx_gpu,
                    np.int32(grid_n),
                    block=(self.block_size, 1, 1),
                    grid=(1, 1, 1),
                    shared=self.shared_reduce_bytes,
                )
                cuda.Context.synchronize()
                return (
                    float(self.best_auc_gpu.get()[0]),
                    float(self.best_cons_gpu.get()[0]),
                    int(self.best_idx_gpu.get()[0]),
                )

            input_vals = out_partial
            out_partial, alt_partial = alt_partial, out_partial
            n = grid_n


def _is_better(auc: float, cons: float, idx: int,
               best_auc: float, best_cons: float, best_idx: int) -> bool:
    if auc > best_auc:
        return True
    if auc == best_auc and cons > best_cons:
        return True
    if auc == best_auc and cons == best_cons and idx < best_idx:
        return True
    return False


@dataclass
class _GlobalBest:
    auc: float = float('-inf')
    consistency: float = 0.0
    weights: tuple[float, float, float] | None = None
    idx: int = -1
    grilla: int = -1
    bloque: int = -1
    thread: int = -1


@dataclass
class _ImproveMsg:
    """Mensaje de mejora encolado (análogo a MPI ImproveMsg)."""
    iter: int
    auc: float
    consistency: float
    weights: tuple[float, float, float]
    grilla: int
    bloque: int
    thread: int


class _CudaTracker:
    """Live mode: un launch CUDA por bloque (256 candidatos); progreso continuo."""

    def __init__(self, log: Log | None, block_size: int, k_total: int,
                 log_interval: int = LOG_INTERVAL) -> None:
        self.log = log
        self.block_size = block_size
        self.k_total = k_total
        self.log_interval = max(1, log_interval)
        self.global_best = _GlobalBest()
        self.grilla_id = 0
        self._live_best_auc = float('-inf')
        self._live_best_cons = -1.0
        self._pending: list[_ImproveMsg] = []
        self._processed = 0

    def set_k_total(self, k_total: int) -> None:
        self.k_total = k_total
        if self.log is not None:
            self.log.k = k_total

    def _try_log_global_improvement(self, msg: _ImproveMsg) -> None:
        if self.log is None:
            return
        if msg.auc > self._live_best_auc or (
            msg.auc == self._live_best_auc and msg.consistency > self._live_best_cons
        ):
            prev = self._live_best_auc if self._live_best_auc != float('-inf') else -1.0
            self.log.cuda_improvement(
                msg.iter, msg.auc, prev, msg.consistency, msg.weights,
                msg.grilla, msg.bloque, msg.thread,
            )
            self._live_best_auc = msg.auc
            self._live_best_cons = msg.consistency

    def _notify_improvement(self, gi: int, auc: float, cons: float,
                            w: tuple[float, float, float],
                            grilla_id: int, bloque_id: int, thread_id: int) -> None:
        if self.log is None:
            return
        self._pending.append(_ImproveMsg(
            gi, auc, cons, w, grilla_id, bloque_id, thread_id,
        ))

    def _drain_improvements(self) -> None:
        while self._pending:
            self._try_log_global_improvement(self._pending.pop(0))
        sys.stdout.flush()

    def process_range(self, ctx: GPUContext, host_weights: np.ndarray,
                      offset: int, length: int) -> None:
        """Launch por bloque CUDA (256 cand.); sync y progreso tras cada bloque."""
        if length == 0:
            return

        for block_off in range(0, length, self.block_size):
            chunk = min(self.block_size, length - block_off)
            global_off = offset + block_off
            bloque_id = global_off // self.block_size

            ctx.evaluate(chunk, index_offset=global_off)
            aucs, cons = ctx.fetch_chunk_results(chunk, index_offset=global_off)

            for i in range(chunk):
                gi = global_off + i
                thread_id = gi % self.block_size
                auc = float(aucs[i])
                c = float(cons[i])
                w = tuple(float(x) for x in host_weights[block_off + i].tolist())

                g = self.global_best
                if _is_better(auc, c, gi, g.auc, g.consistency, g.idx):
                    g.auc, g.consistency, g.weights, g.idx = auc, c, w, gi
                    g.grilla, g.bloque, g.thread = bloque_id, bloque_id, thread_id
                    self._notify_improvement(
                        gi, auc, c, w, bloque_id, bloque_id, thread_id,
                    )

                self._processed = gi + 1
                if self.log is not None and gi % self.log_interval == 0:
                    self._drain_improvements()

            if self.log is not None:
                self._drain_improvements()

            self.grilla_id = bloque_id + 1

    def report_locals(self) -> None:
        if self.log is not None:
            self._drain_improvements()

    def result_tuple(self) -> tuple[float, float, tuple[float, float, float], int]:
        g = self.global_best
        if g.weights is None:
            return -1.0, 0.0, (1.0 / 3, 1.0 / 3, 1.0 / 3), -1
        return g.auc, g.consistency, g.weights, g.idx


# ── Weight generation ──────────────────────────────────────────────────
def _generate_grid(resolution):
    weights = []
    for i in range(resolution + 1):
        for j in range(resolution + 1 - i):
            k = resolution - i - j
            weights.append([i / resolution, j / resolution, k / resolution])
    return np.array(weights, dtype=np.float32), len(weights)


def _resolution_for_K(K):
    return max(int(math.sqrt(2.0 * K)), 1)


def _generate_random_weights(K, seed):
    rng = np.random.default_rng(seed)
    return rng.dirichlet(np.ones(3), size=K).astype(np.float32)


def _generate_hybrid_weights(K, seed):
    """Genera pesos para las tres fases; retorna (weights, K_grid, K_random, K_local)."""
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

    random_w = rng.dirichlet(np.ones(3), size=K_random).astype(np.float32)

    return grid_w, random_w, K_grid, K_random, K_local, rng


# ── Search strategies ──────────────────────────────────────────────────
def _run_random(ctx: GPUContext, K, seed, tracker: _CudaTracker | None, fast: bool):
    weights = _generate_random_weights(K, seed)
    ctx.upload_weights(weights)

    if fast:
        ctx.evaluate(K)
        return ctx.reduce_best(K), weights, K

    assert tracker is not None
    tracker.process_range(ctx, weights, 0, K)
    tracker.report_locals()
    return tracker.result_tuple(), weights, K


def _run_grid(ctx: GPUContext, K_hint, tracker: _CudaTracker | None, fast: bool,
              grid_resolution=0):
    resolution = grid_resolution if grid_resolution > 0 else _resolution_for_K(K_hint)
    weights, actual_K = _generate_grid(resolution)
    ctx.upload_weights(weights)

    if fast:
        ctx.evaluate(actual_K)
        return ctx.reduce_best(actual_K), weights, actual_K

    assert tracker is not None
    tracker.set_k_total(actual_K)
    tracker.process_range(ctx, weights, 0, actual_K)
    tracker.report_locals()
    return tracker.result_tuple(), weights, actual_K


def _run_hybrid(ctx: GPUContext, K, seed, tracker: _CudaTracker | None, fast: bool):
    grid_w, random_w, K_grid, K_random, K_local, rng = _generate_hybrid_weights(K, seed)

    ctx.upload_weights(grid_w, offset=0)
    ctx.upload_weights(random_w, offset=K_grid)

    if not fast:
        assert tracker is not None
        tracker.process_range(ctx, grid_w, 0, K_grid)
        tracker.process_range(ctx, random_w, K_grid, K_random)

        if K_local > 0:
            _, _, best_w, _ = tracker.result_tuple()
            alpha = np.maximum(np.array(best_w, dtype=np.float32) * 100.0, 1e-3)
            local_w = rng.dirichlet(alpha, size=K_local).astype(np.float32)
            ctx.upload_weights(local_w, offset=K_grid + K_random)
            tracker.process_range(ctx, local_w, K_grid + K_random, K_local)

        tracker.report_locals()
        return tracker.result_tuple(), None, K

    ctx.evaluate(K_grid, index_offset=0)
    ctx.evaluate(K_random, index_offset=K_grid)

    if K_local > 0:
        _, _, idx = ctx.reduce_best(K_grid + K_random)
        best_w = ctx.weights_gpu[idx * 3:idx * 3 + 3].get()
        alpha = np.maximum(np.array(best_w, dtype=np.float32) * 100.0, 1e-3)
        local_w = rng.dirichlet(alpha, size=K_local).astype(np.float32)
        ctx.upload_weights(local_w, offset=K_grid + K_random)
        ctx.evaluate(K_local, index_offset=K_grid + K_random)

    actual_K = K_grid + K_random + K_local
    return ctx.reduce_best(actual_K), None, actual_K


def _result_from_gpu_best(ctx: GPUContext, auc: float, cons: float, idx: int,
                          host_weights: np.ndarray | None):
    if host_weights is not None:
        w = tuple(float(x) for x in host_weights[idx].tolist())
    else:
        w = tuple(float(x) for x in ctx.weights_gpu[idx * 3:idx * 3 + 3].get())
    return auc, cons, w, idx


def _log_search_mode(search_mode: str, step: float, fast: bool) -> None:
    mode_tag = 'fast' if fast else 'live'
    if search_mode in ('grid', 'hybrid'):
        print(f'  search={search_mode}  step={step:.4f}  mode={mode_tag}', file=sys.stderr)
    else:
        print(f'  search={search_mode}  mode={mode_tag}', file=sys.stderr)


def _query_gpu_info(block_size: int, k: int) -> tuple[int, int, int]:
    """Retorna (sm_count, cuda_cores, blocks_per_launch)."""
    import pycuda.driver as cuda
    attrs = cuda.Device(0).get_attributes()
    sm_count = int(attrs[cuda.device_attribute.MULTIPROCESSOR_COUNT])
    max_threads = int(attrs[cuda.device_attribute.MAX_THREADS_PER_MULTIPROCESSOR])
    cuda_cores = sm_count * max_threads
    blocks_per_launch = _grid_size(k, block_size)
    return sm_count, cuda_cores, blocks_per_launch


# ── Timed search ───────────────────────────────────────────────────────
def timed_search(name, sm_count, A, y, profiles, K, seed,
                 log=None, search_mode='random', block_size=DEFAULT_BLOCK_SIZE,
                 mode='full', grid_resolution=0, step=0.02, fast=False):
    """Ejecuta búsqueda con medición de tiempo. El AUC se calcula en GPU."""
    ctx = GPUContext.create(A, y, profiles, k_max=K, mode=mode, block_size=block_size)

    tracker = None
    if not fast:
        tracker = _CudaTracker(log, block_size, K)

    if log is not None:
        _log_search_mode(search_mode, step, fast)

    start = time.perf_counter()

    if search_mode == 'grid':
        if fast:
            gpu_best, host_weights, actual_k = _run_grid(
                ctx, K, None, fast=True, grid_resolution=grid_resolution)
            best_auc, best_cons, best_weights, best_iter = \
                _result_from_gpu_best(ctx, *gpu_best, host_weights)
            K = actual_k
        else:
            (best_auc, best_cons, best_weights, best_iter), _, actual_k = \
                _run_grid(ctx, K, tracker, fast=False, grid_resolution=grid_resolution)
            K = actual_k
    elif search_mode == 'hybrid':
        if fast:
            gpu_best, _, actual_k = _run_hybrid(ctx, K, seed, None, fast=True)
            best_auc, best_cons, best_weights, best_iter = \
                _result_from_gpu_best(ctx, *gpu_best, None)
            K = actual_k
        else:
            (best_auc, best_cons, best_weights, best_iter), _, actual_k = \
                _run_hybrid(ctx, K, seed, tracker, fast=False)
            K = actual_k
    else:
        if fast:
            gpu_best, host_weights, actual_k = _run_random(ctx, K, seed, None, fast=True)
            best_auc, best_cons, best_weights, best_iter = \
                _result_from_gpu_best(ctx, *gpu_best, host_weights)
            K = actual_k
        else:
            (best_auc, best_cons, best_weights, best_iter), _, actual_k = \
                _run_random(ctx, K, seed, tracker, fast=False)
            K = actual_k

    elapsed = time.perf_counter() - start

    result = SearchResult(
        implementation=name,
        parallel_units=sm_count,
        n_items=ctx.n_items,
        k=K,
        time_sec=elapsed,
        auc=best_auc,
        consistency=best_cons,
        weights=best_weights,
        seed=seed,
        search_mode=search_mode,
        iterations_until_best=best_iter,
    )

    if log is not None:
        log.complete(result)

    return result


# ── Main ───────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description='Scoring metagenómico — PyCUDA')
    ap.add_argument('--k', type=int, default=10000, help='Número de candidatos')
    ap.add_argument('--seed', type=int, default=42, help='Semilla RNG')
    ap.add_argument('--search', choices=['random', 'grid', 'hybrid'],
                    default='random', help='Estrategia de búsqueda')
    ap.add_argument('--step', type=float, default=0.02,
                    help='Paso del grid (solo logging; grid usa resolución automática)')
    ap.add_argument('--data-dir', type=Path, default=Path('data/npy'),
                    help='Directorio de datos')
    ap.add_argument('--block-size', type=int, default=DEFAULT_BLOCK_SIZE,
                    help='Tamaño de bloque CUDA')
    ap.add_argument('--mode', choices=['full', 'precompute'], default='full',
                    help='Modo de evaluación')
    ap.add_argument('--grid-resolution', type=int, default=0,
                    help='Resolución del grid (0 = automática)')
    ap.add_argument('--fast', action='store_true',
                    help='Benchmark: un launch + reduction GPU, sin logging en vivo')
    ap.add_argument('--csv', action='store_true', help='Salida en CSV')
    args = ap.parse_args()

    _setup_cuda_toolkit()

    try:
        import pycuda.autoinit  # noqa
        import pycuda.driver as cuda
    except Exception as e:
        print(f'PyCUDA no disponible: {e}', file=sys.stderr)
        raise SystemExit(2)

    global evaluate_full, evaluate_precompute
    global reduce_best_stage1, reduce_best_stage1_from, reduce_best_stage2
    (evaluate_full, evaluate_precompute,
     reduce_best_stage1, reduce_best_stage1_from, reduce_best_stage2) = _compile_kernels()

    A, y, profiles = load_data(args.data_dir)
    _validate(A, y, profiles)

    n_samples, n_items = A.shape
    log = None if args.csv or args.fast else Log('pycuda', n_items, args.k)

    sm_count, cuda_cores, blocks_per_launch = _query_gpu_info(args.block_size, args.k)
    if log is not None:
        dev_name = cuda.Device(0).name()
        log.cuda_info(dev_name, sm_count, cuda_cores, args.block_size, blocks_per_launch)

    result = timed_search('pycuda', sm_count, A, y, profiles,
                          args.k, args.seed, log=log,
                          search_mode=args.search,
                          block_size=args.block_size,
                          mode=args.mode,
                          grid_resolution=args.grid_resolution,
                          step=args.step,
                          fast=args.fast)

    if args.csv:
        print(result.csv_row())
    else:
        w1, w2, w3 = result.weights
        print(f'implementation={result.implementation}')
        print(f'search_mode={result.search_mode}')
        print(f'N={result.n_items}')
        print(f'K={result.k}')
        print(f'sms={result.parallel_units}')
        print(f'block_size={args.block_size}')
        print(f'grid_blocks={blocks_per_launch}')
        print(f'fast={args.fast}')
        print(f'best_auc={result.auc:.6f}')
        print(f'best_w=[{w1:.8f}, {w2:.8f}, {w3:.8f}]')
        print(f'consistency={result.consistency:.4f}')
        print(f'time_sec={result.time_sec:.6f}')


if __name__ == '__main__':
    main()
