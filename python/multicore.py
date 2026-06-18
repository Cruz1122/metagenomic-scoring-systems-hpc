from __future__ import annotations
import argparse
import os
import sys
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from time import perf_counter
import numpy as np
from common import SearchResult, load_data, evaluate, auc_vector, consistency
from logger import Log

# Tamaño de chunk para logging en vivo (como MPI: drain cada 32 iteraciones)
LOG_INTERVAL = 32


def _consistency_at_threshold(scores: np.ndarray, y: np.ndarray,
                              theta: float) -> float:
    """Balanced accuracy (TPR + TNR) / 2 en un umbral fijo θ."""
    pred = (scores > theta).astype(int)
    tp = ((pred == 1) & (y == 1)).sum()
    tn = ((pred == 0) & (y == 0)).sum()
    tpr = tp / (y == 1).sum()
    tnr = tn / (y == 0).sum()
    return float((tpr + tnr) / 2.0)


def _is_better(auc: float, cons: float, idx: int,
               best_auc: float, best_cons: float, best_idx: int) -> bool:
    if auc > best_auc:
        return True
    if auc == best_auc and cons > best_cons:
        return True
    if auc == best_auc and cons == best_cons and idx < best_idx:
        return True
    return False


def _eval_chunk(task: tuple, A: np.ndarray, y: np.ndarray,
                profiles: np.ndarray) -> tuple:
    """Evalúa un chunk de pesos y retorna el mejor resultado local.

    task: (chunk_id, base_offset, worker_id, chunk_array)
    Returns: (best_auc, best_consistency, best_w, global_index, worker_id)
    """
    _chunk_id, base_offset, worker_id, chunk = task
    best_auc_local = -np.inf
    best_consistency_local = 0.0
    best_w_local = None
    best_i_local = -1

    for i, w in enumerate(chunk):
        auc_val, cons_val = evaluate(A, y, profiles, w)
        if _is_better(auc_val, cons_val, base_offset + i,
                      best_auc_local, best_consistency_local, best_i_local):
            best_auc_local = auc_val
            best_consistency_local = cons_val
            best_w_local = w.copy()
            best_i_local = i

    return (best_auc_local, best_consistency_local, best_w_local,
            base_offset + best_i_local, worker_id)


def _parallel_eval(weights: np.ndarray, workers: int, pool: Pool,
                   A: np.ndarray, y: np.ndarray, profiles: np.ndarray,
                   log: Log | None = None,
                   index_offset: int = 0) -> tuple:
    """Evalúa pesos en chunks pequeños; loguea mejoras globales al completar cada uno.

    Con logger: chunks de LOG_INTERVAL candidatos + imap_unordered (como drain MPI).
    Sin logger: un chunk por worker para máximo rendimiento.
    """
    n = len(weights)
    if n == 0:
        return -np.inf, 0.0, (0.0, 0.0, 0.0), -1

    if log is not None:
        chunk_size = LOG_INTERVAL
    else:
        chunk_size = max(1, (n + workers - 1) // workers)

    tasks = []
    for off in range(0, n, chunk_size):
        sub = weights[off:off + chunk_size]
        chunk_id = off // chunk_size
        wid = chunk_id % max(workers, 1)
        tasks.append((chunk_id, off, wid, sub))

    worker_fn = partial(_eval_chunk, A=A, y=y, profiles=profiles)

    best_auc = -np.inf
    best_consistency = 0.0
    best_w = None
    best_iter = -1

    for result in pool.imap_unordered(worker_fn, tasks):
        auc_val, cons_val, w, gidx, wid = result
        if w is None:
            continue
        gidx_global = index_offset + gidx
        if _is_better(auc_val, cons_val, gidx_global,
                      best_auc, best_consistency, best_iter):
            best_auc = auc_val
            best_consistency = cons_val
            best_w = w
            best_iter = gidx_global
            if log is not None:
                log.improvement(gidx_global, auc_val, cons_val, tuple(w),
                                worker_id=wid)
                sys.stdout.flush()

    if best_w is None:
        return -np.inf, 0.0, (0.0, 0.0, 0.0), -1
    return best_auc, best_consistency, tuple(best_w), best_iter


def random_search(A, y, profiles, k: int, seed: int,
                  workers: int = 1, pool: Pool | None = None,
                  log: Log | None = None):
    """Búsqueda aleatoria de pesos Dirichlet — versión multi-core."""
    rng = np.random.default_rng(seed)
    weights = rng.dirichlet(np.ones(3), size=k)
    return _parallel_eval(weights, workers, pool, A, y, profiles, log=log)


def grid_search(A, y, profiles, step: float = 0.02,
                workers: int = 1, pool: Pool | None = None,
                log: Log | None = None):
    """Búsqueda sistemática sobre el simplex 2D con paso fijo — multi-core."""
    grid_points = []
    for w1 in np.arange(0, 1 + step, step):
        for w2 in np.arange(0, 1 - w1 + step, step):
            w3 = 1.0 - w1 - w2
            if w3 < -1e-12:
                continue
            grid_points.append([w1, w2, w3])

    grid_array = np.array(grid_points, dtype=np.float64)
    total = len(grid_array)

    if total == 0:
        return -np.inf, 0.0, (0.0, 0.0, 0.0), -1, 0

    if log is not None:
        log.k = total

    best_auc, best_cons, best_w, best_iter = _parallel_eval(
        grid_array, workers, pool, A, y, profiles, log=log)
    return best_auc, best_cons, best_w, best_iter, total


def hybrid_search(A, y, profiles, k: int, seed: int,
                  step: float = 0.02,
                  workers: int = 1, pool: Pool | None = None,
                  log: Log | None = None):
    """Búsqueda híbrida en tres fases: grid + random + local — multi-core."""
    rng = np.random.default_rng(seed)

    best_auc, best_consistency, best_w, best_iter, iteration = grid_search(
        A, y, profiles, step=step, workers=workers, pool=pool, log=log
    )

    remaining = k - iteration
    if remaining <= 0:
        return best_auc, best_consistency, best_w, best_iter

    random_n = remaining // 2
    local_n = remaining - random_n

    if random_n > 0:
        random_weights = rng.dirichlet(np.ones(3), size=random_n)
        auc_val, cons_val, w, gidx, _ = _parallel_eval(
            random_weights, workers, pool, A, y, profiles,
            log=log, index_offset=iteration)
        if _is_better(auc_val, cons_val, gidx,
                      best_auc, best_consistency, best_iter):
            best_auc, best_consistency, best_w, best_iter = auc_val, cons_val, w, gidx
        iteration += random_n

    if best_w is not None and local_n > 0:
        local_splits = [local_n // 2, local_n - local_n // 2]
        for conc, count in zip([300, 1000], local_splits):
            if count <= 0:
                continue
            alpha = np.maximum(np.array(best_w) * conc, 1e-3)
            local_weights = rng.dirichlet(alpha, size=count)
            auc_val, cons_val, w, gidx, _ = _parallel_eval(
                local_weights, workers, pool, A, y, profiles,
                log=log, index_offset=iteration)
            if _is_better(auc_val, cons_val, gidx,
                          best_auc, best_consistency, best_iter):
                best_auc, best_consistency, best_w, best_iter = auc_val, cons_val, w, gidx
            iteration += count

    return best_auc, best_consistency, best_w, best_iter


def timed_search(name: str, p: int, A, y, profiles, k: int, seed: int,
                 log: Log | None = None,
                 search_mode: str = 'random',
                 step: float = 0.02) -> SearchResult:
    """Ejecuta búsqueda de pesos con medición de tiempo — multi-core."""
    with Pool(p) as pool:
        start = perf_counter()

        if search_mode == 'grid':
            best_auc, best_consistency, best_weights, best_iter, actual_k = \
                grid_search(A, y, profiles, step=step, workers=p, pool=pool, log=log)
            k = actual_k
        elif search_mode == 'hybrid':
            best_auc, best_consistency, best_weights, best_iter = \
                hybrid_search(A, y, profiles, k, seed, step=step,
                              workers=p, pool=pool, log=log)
        else:
            best_auc, best_consistency, best_weights, best_iter = \
                random_search(A, y, profiles, k, seed, workers=p, pool=pool, log=log)

        elapsed = perf_counter() - start

    result = SearchResult(
        implementation=name,
        parallel_units=p,
        n_items=A.shape[1],
        k=k,
        time_sec=elapsed,
        auc=best_auc,
        consistency=best_consistency,
        weights=best_weights,
        seed=seed,
        search_mode=search_mode,
        iterations_until_best=best_iter,
    )

    if log is not None:
        log.complete(result)

    return result


def main():
    """CLI: distribuye k candidatos entre workers multi-core."""
    ap = argparse.ArgumentParser(description='Scoring metagenómico — versión multi-core')
    ap.add_argument('--k', type=int, default=10000, help='Número de candidatos')
    ap.add_argument('--seed', type=int, default=42, help='Semilla RNG')
    ap.add_argument('--search', choices=['random', 'grid', 'hybrid'],
                    default='random', help='Estrategia de búsqueda')
    ap.add_argument('--step', type=float, default=0.02,
                    help='Paso del grid (default 0.02)')
    ap.add_argument('--workers', type=int, default=max(1, os.cpu_count() or 1),
                    help='Número de procesos workers')
    ap.add_argument('--theta', type=float, default=None,
                    help='Umbral para consistencia (default: mediana de scores del mejor W)')
    ap.add_argument('--data-dir', type=Path, default=Path('data'), help='Directorio de datos')
    ap.add_argument('--benchmark', action='store_true',
                    help='Modo benchmark: sin logging, salida CSV')
    ap.add_argument('--csv', action='store_true', help='Salida en CSV (formato benchmark)')
    args = ap.parse_args()

    A, y, profiles = load_data(args.data_dir)
    quiet = args.benchmark or args.csv
    log = None if quiet else Log('python_multicore', A.shape[1], args.k)

    result = timed_search('python_multicore', args.workers, A, y, profiles,
                          args.k, args.seed, log=log, search_mode=args.search,
                          step=args.step)

    w = np.array(result.weights)
    scores = A @ (profiles @ w)
    theta_val = args.theta if args.theta is not None else float(np.median(scores))
    cons_theta = _consistency_at_threshold(scores, y, theta_val)

    if quiet:
        print(result.csv_row())
    else:
        w1, w2, w3 = result.weights
        print(f'implementation={result.implementation}')
        print(f'N={result.n_items}')
        print(f'K={result.k}')
        print(f'workers={args.workers}')
        print(f'best_auc={result.auc:.6f}')
        print(f'best_w=[{w1:.8f}, {w2:.8f}, {w3:.8f}]')
        print(f'consistency={cons_theta:.4f}')
        print(f'theta={theta_val:.6f}')
        print(f'time_sec={result.time_sec:.6f}')


if __name__ == '__main__':
    main()
