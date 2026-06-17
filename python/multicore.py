from __future__ import annotations
import argparse
import os
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from time import perf_counter
import numpy as np
from common import SearchResult, load_data, evaluate, auc_vector, consistency
from logger import Log


def _consistency_at_threshold(scores: np.ndarray, y: np.ndarray,
                              theta: float) -> float:
    """Balanced accuracy (TPR + TNR) / 2 en un umbral fijo θ.

    Args:
        scores: (n_samples,) scores del mejor W.
        y: (n_samples,) etiquetas binarias {0, 1}.
        theta: umbral de decisión.

    Returns:
        float: (TPR + TNR) / 2
    """
    pred = (scores > theta).astype(int)
    tp = ((pred == 1) & (y == 1)).sum()
    tn = ((pred == 0) & (y == 0)).sum()
    tpr = tp / (y == 1).sum()
    tnr = tn / (y == 0).sum()
    return float((tpr + tnr) / 2.0)


def _eval_chunk(worker_id_and_chunk: tuple, A: np.ndarray,
                y: np.ndarray, profiles: np.ndarray) -> tuple:
    """Evalúa un chunk de pesos y retorna el mejor resultado local.

    Función module-level (pickleable) para Pool.map.
    Cada worker recibe un subconjunto de los pesos globales y retorna
    solo su mejor local, etiquetado con su worker_id.

    Criterio: AUC primario, consistencia desempate.

    Args:
        worker_id_and_chunk: (worker_id, (M, 3) array de pesos W)
        A: (n_samples, n_items)
        y: (n_samples,)
        profiles: (n_items, 3)

    Returns:
        tuple: (best_auc, best_consistency, best_w_copy, local_best_index, worker_id)
    """
    worker_id, chunk = worker_id_and_chunk
    best_auc_local = -np.inf
    best_consistency_local = 0.0
    best_w_local = None
    best_i_local = -1

    for i, w in enumerate(chunk):
        auc_val, cons_val = evaluate(A, y, profiles, w)
        if auc_val > best_auc_local or \
           (auc_val == best_auc_local and cons_val > best_consistency_local):
            best_auc_local = auc_val
            best_consistency_local = cons_val
            best_w_local = w.copy()
            best_i_local = i

    return best_auc_local, best_consistency_local, best_w_local, best_i_local, worker_id


def random_search(A, y, profiles, k: int, seed: int,
                  workers: int = 1, pool: Pool | None = None,
                  log: Log | None = None):
    """Búsqueda aleatoria de pesos Dirichlet — versión multi-core.

    Genera K pesos en el proceso principal, los divide en chunks
    y evalúa en paralelo con Pool.map:

      1. W_k ~ Dirichlet(1,1,1) para k in [0, K)
      2. np.array_split(weights, workers)
      3. Pool.map(_eval_chunk, chunks)
      4. Recolecta mejores locales y escoge el mejor global.

    Args:
        A: (n_samples, n_items)
        y: (n_samples,)
        profiles: (n_items, 3)
        k: número de candidatos a evaluar
        seed: semilla RNG
        workers: número de procesos workers
        pool: Pool de multiprocessing (creado en timed_search)
        log: instancia Log opcional

    Returns:
        tuple: (best_auc, best_consistency, best_weights, best_iter)
    """
    rng = np.random.default_rng(seed)
    weights = rng.dirichlet(np.ones(3), size=k)

    n_chunks = min(workers, k)
    raw_chunks = np.array_split(weights, n_chunks)
    chunks = list(enumerate(raw_chunks))  # (worker_id, chunk_array)

    worker_fn = partial(_eval_chunk, A=A, y=y, profiles=profiles)
    results = pool.map(worker_fn, chunks)

    # Reportar mejor local de cada worker
    if log is not None:
        best_idx = max(range(len(results)),
                       key=lambda i: (results[i][0], results[i][1]))
        for i, ((wid, chunk), (auc_val, cons_val, w, local_i, wid2)) \
                in enumerate(zip(chunks, results)):
            log.worker_report(wid, auc_val, cons_val, tuple(w), len(chunk),
                              is_best=(i == best_idx))

    best_auc = -np.inf
    best_consistency = 0.0
    best_w = None
    best_iter = -1
    offset = 0

    for (wid, chunk), (auc_val, cons_val, w, local_i, wid2) in zip(chunks, results):
        assert wid == wid2
        if auc_val > best_auc or \
           (auc_val == best_auc and cons_val > best_consistency):
            best_auc = auc_val
            best_consistency = cons_val
            best_w = w
            best_iter = offset + local_i
            if log is not None:
                log.improvement(best_iter, auc_val, cons_val, tuple(best_w),
                                worker_id=wid)
        offset += len(chunk)

    return best_auc, best_consistency, tuple(best_w), best_iter


def grid_search(A, y, profiles, step: float = 0.02,
                workers: int = 1, pool: Pool | None = None,
                log: Log | None = None):
    """Búsqueda sistemática sobre el simplex 2D con paso fijo — multi-core.

    Barre w1, w2 con np.arange(0, 1 + step, step) y deriva w3 = 1 - w1 - w2.
    Genera todos los puntos como array (N, 3), divide en chunks y evalúa
    en paralelo con Pool.map.

    Args:
        A: (n_samples, n_items)
        y: (n_samples,)
        profiles: (n_items, 3)
        step: granularidad del grid (default 0.02 → ~1326 puntos)
        workers: número de procesos workers
        pool: Pool de multiprocessing
        log: Log opcional

    Returns:
        tuple: (best_auc, best_consistency, best_weights, best_iter, total)
    """
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

    n_chunks = min(workers, total)
    raw_chunks = np.array_split(grid_array, n_chunks)
    chunks = list(enumerate(raw_chunks))

    worker_fn = partial(_eval_chunk, A=A, y=y, profiles=profiles)
    results = pool.map(worker_fn, chunks)

    # Reportar mejor local de cada worker
    if log is not None:
        best_idx = max(range(len(results)),
                       key=lambda i: (results[i][0], results[i][1]))
        for i, ((wid, chunk), (auc_val, cons_val, w, local_i, wid2)) \
                in enumerate(zip(chunks, results)):
            log.worker_report(wid, auc_val, cons_val, tuple(w), len(chunk),
                              is_best=(i == best_idx))

    best_auc = -np.inf
    best_consistency = 0.0
    best_w = None
    best_iter = -1
    offset = 0

    for (wid, chunk), (auc_val, cons_val, w, local_i, wid2) in zip(chunks, results):
        assert wid == wid2
        if auc_val > best_auc or \
           (auc_val == best_auc and cons_val > best_consistency):
            best_auc = auc_val
            best_consistency = cons_val
            best_w = w
            best_iter = offset + local_i
            if log is not None:
                log.improvement(best_iter, auc_val, cons_val, tuple(best_w),
                                worker_id=wid)
        offset += len(chunk)

    return best_auc, best_consistency, tuple(best_w), best_iter, total


def _sample_local_dirichlet(best_w, n: int, rng,
                            concentration: float = 300):
    """Muestrea alrededor de best_w con Dirichlet concentrada."""
    alpha = np.maximum(np.array(best_w) * concentration, 1e-3)
    return rng.dirichlet(alpha, size=n)


def hybrid_search(A, y, profiles, k: int, seed: int,
                  workers: int = 1, pool: Pool | None = None,
                  log: Log | None = None):
    """Búsqueda híbrida en tres fases: grid + random + local — multi-core.

    Fase 1 — Grid step=0.02 (~1326 puntos)
    Fase 2 — Random Dirichlet(1,1,1) global  (~50% del resto)
    Fase 3 — Local Dirichlet concentrada alrededor del mejor W (~50% del resto)
             dividida entre concentration=300 y concentration=1000.

    Cada fase genera sus pesos de forma secuencial (determinista);
    la evaluación se distribuye entre workers con Pool.map.

    Args:
        A: (n_samples, n_items)
        y: (n_samples,)
        profiles: (n_items, 3)
        k: presupuesto total de candidatos
        seed: semilla RNG
        workers: número de procesos workers
        pool: Pool de multiprocessing
        log: Log opcional

    Returns:
        tuple: (best_auc, best_consistency, best_weights, best_iter)
    """
    rng = np.random.default_rng(seed)
    worker_fn = partial(_eval_chunk, A=A, y=y, profiles=profiles)

    # ── Fase 1: Grid grueso ────────────────────────────────────────────
    best_auc, best_consistency, best_w, best_iter, iteration = grid_search(
        A, y, profiles, workers=workers, pool=pool, log=log
    )

    remaining = k - iteration
    if remaining <= 0:
        return best_auc, best_consistency, best_w, best_iter

    random_n = remaining // 2
    local_n = remaining - random_n

    # ── Fase 2: Random global ──────────────────────────────────────────
    if random_n > 0:
        random_weights = rng.dirichlet(np.ones(3), size=random_n)
        random_raw = np.array_split(random_weights, min(workers, random_n))
        random_chunks = list(enumerate(random_raw))
        results = pool.map(worker_fn, random_chunks)

        if log is not None:
            best_idx = max(range(len(results)),
                           key=lambda i: (results[i][0], results[i][1]))
            for i, ((wid, chunk), (auc_val, cons_val, w, local_i, wid2)) \
                    in enumerate(zip(random_chunks, results)):
                log.worker_report(wid, auc_val, cons_val, tuple(w), len(chunk),
                                  is_best=(i == best_idx))

        offset = iteration
        for (wid, chunk), (auc_val, cons_val, w, local_i, wid2) \
                in zip(random_chunks, results):
            assert wid == wid2
            if auc_val > best_auc or \
               (auc_val == best_auc and cons_val > best_consistency):
                best_auc = auc_val
                best_consistency = cons_val
                best_w = w
                best_iter = offset + local_i
                if log is not None:
                    log.improvement(best_iter, auc_val, cons_val, tuple(best_w),
                                    worker_id=wid)
            offset += len(chunk)
        iteration = offset

    # ── Fase 3: Refinamiento local adaptativo ──────────────────────────
    if best_w is not None and local_n > 0:
        local_splits = [local_n // 2, local_n - local_n // 2]
        for conc, count in zip([300, 1000], local_splits):
            if count <= 0:
                continue
            alpha = np.maximum(np.array(best_w) * conc, 1e-3)
            local_weights = rng.dirichlet(alpha, size=count)
            local_raw = np.array_split(local_weights, min(workers, count))
            local_chunks = list(enumerate(local_raw))
            results = pool.map(worker_fn, local_chunks)

            if log is not None:
                best_idx = max(range(len(results)),
                               key=lambda i: (results[i][0], results[i][1]))
                for i, ((wid, chunk), (auc_val, cons_val, w, local_i, wid2)) \
                        in enumerate(zip(local_chunks, results)):
                    log.worker_report(wid, auc_val, cons_val, tuple(w), len(chunk),
                                      is_best=(i == best_idx))

            offset = iteration
            for (wid, chunk), (auc_val, cons_val, w, local_i, wid2) \
                    in zip(local_chunks, results):
                assert wid == wid2
                if auc_val > best_auc or \
                   (auc_val == best_auc and cons_val > best_consistency):
                    best_auc = auc_val
                    best_consistency = cons_val
                    best_w = w
                    best_iter = offset + local_i
                    if log is not None:
                        log.improvement(best_iter, auc_val, cons_val, tuple(best_w),
                                        worker_id=wid)
                offset += len(chunk)
            iteration = offset

    return best_auc, best_consistency, tuple(best_w), best_iter


def timed_search(name: str, p: int, A, y, profiles, k: int, seed: int,
                 log: Log | None = None,
                 search_mode: str = 'random') -> SearchResult:
    """Ejecuta búsqueda de pesos con medición de tiempo — multi-core.

    Crea un Pool de `p` workers y lo inyecta en las funciones de búsqueda.
    El cronómetro cubre solo la búsqueda (no la carga de datos, no la
    creación del Pool).

    Args:
        name: etiqueta de implementación
        p: unidades paralelas (workers)
        A, y, profiles: datos
        k: candidatos
        seed: semilla
        log: Log opcional para logging en vivo
        search_mode: 'random' | 'grid' | 'hybrid'

    Returns:
        SearchResult con métricas y pesos.
    """
    with Pool(p) as pool:
        start = perf_counter()

        if search_mode == 'grid':
            best_auc, best_consistency, best_weights, best_iter, actual_k = \
                grid_search(A, y, profiles, workers=p, pool=pool, log=log)
            k = actual_k
        elif search_mode == 'hybrid':
            best_auc, best_consistency, best_weights, best_iter = \
                hybrid_search(A, y, profiles, k, seed, workers=p, pool=pool, log=log)
        else:  # 'random'
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
    ap.add_argument('--workers', type=int, default=max(1, os.cpu_count() or 1),
                    help='Número de procesos workers')
    ap.add_argument('--theta', type=float, default=None,
                    help='Umbral para consistencia (default: mediana de scores del mejor W)')
    ap.add_argument('--data-dir', type=Path, default=Path('data'), help='Directorio de datos')
    ap.add_argument('--csv', action='store_true', help='Salida en CSV (formato benchmark)')
    args = ap.parse_args()

    # Cargar datos
    A, y, profiles = load_data(args.data_dir)

    # Logger colorido (solo si no es modo CSV)
    log = None if args.csv else Log('python_multicore', A.shape[1], args.k)

    # Búsqueda con Pool
    result = timed_search('python_multicore', args.workers, A, y, profiles,
                          args.k, args.seed, log=log, search_mode=args.search)

    # --- Consistencia con theta (validación post-hoc) ---
    w = np.array(result.weights)
    scores = A @ (profiles @ w)
    theta_val = args.theta if args.theta is not None else float(np.median(scores))
    cons_theta = _consistency_at_threshold(scores, y, theta_val)

    # --- Salida ---
    if args.csv:
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
