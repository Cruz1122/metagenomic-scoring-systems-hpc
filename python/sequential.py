from __future__ import annotations
import argparse
from pathlib import Path
from time import perf_counter
import numpy as np
from common import SearchResult, load_data, evaluate, auc_vector, consistency
from logger import Log


def random_search(A, y, profiles, k: int, seed: int,
                  log: Log | None = None):
    """Búsqueda aleatoria de pesos Dirichlet — versión secuencial.

    Itera K veces:
      - W ~ Dirichlet(1,1,1)
      - P = profiles @ W
      - scores = A @ P
      - AUC = auc(y, scores)
      - guarda el mejor

    Args:
        A: (n_samples, n_items)
        y: (n_samples,)
        profiles: (n_items, 3)
        k: número de candidatos a evaluar
        seed: semilla RNG
        log: instancia Log opcional; si se pasa, reporta cada mejora

    Returns:
        tuple: (best_auc, best_consistency, best_weights, best_iter)
    """
    rng = np.random.default_rng(seed)
    best_auc = -np.inf
    best_consistency = 0.0
    best_w = None
    best_iter = -1

    for i in range(k):
        w = rng.dirichlet(np.ones(3))
        auc_val, cons_val = evaluate(A, y, profiles, w)
        if auc_val > best_auc:
            best_auc = auc_val
            best_consistency = cons_val
            best_w = w.copy()
            best_iter = i
            if log is not None:
                log.improvement(i, auc_val, cons_val, tuple(best_w))

    return best_auc, best_consistency, tuple(best_w), best_iter


def grid_search(A, y, profiles, step: float = 0.02,
                log: Log | None = None):
    """Búsqueda sistemática sobre el simplex 2D con paso fijo — secuencial.

    Barre w1, w2 con np.arange(0, 1+step, step) y deriva w3 = 1-w1-w2.

    Args:
        A: (n_samples, n_items)
        y: (n_samples,)
        profiles: (n_items, 3)
        step: granularidad del grid (default 0.02 → ~1326 puntos)
        log: Log opcional

    Returns:
        tuple: (best_auc, best_consistency, best_weights, best_iter, total)
    """
    best_auc = -np.inf
    best_consistency = 0.0
    best_w = None
    best_iter = -1
    iteration = 0

    for w1 in np.arange(0, 1 + step, step):
        for w2 in np.arange(0, 1 - w1 + step, step):
            w3 = 1.0 - w1 - w2
            if w3 < -1e-12:
                continue
            w = np.array([w1, w2, w3])
            auc_val, cons_val = evaluate(A, y, profiles, w)
            if auc_val > best_auc:
                best_auc = auc_val
                best_consistency = cons_val
                best_w = w.copy()
                best_iter = iteration
                if log is not None:
                    log.improvement(iteration, auc_val, cons_val, tuple(best_w))
            iteration += 1

    return best_auc, best_consistency, tuple(best_w), best_iter, iteration


def _sample_local_dirichlet(best_w, n: int, rng,
                            concentration: float = 300):
    """Muestrea alrededor de best_w con Dirichlet concentrada.

    Args:
        best_w: vector de pesos (3,) centro de la distribución
        n: número de muestras
        rng: generador aleatorio numpy
        concentration: qué tan cerca del centro (mayor = más concentrado)

    Returns:
        np.ndarray: (n, 3) muestras en el simplex
    """
    alpha = np.maximum(np.array(best_w) * concentration, 1e-3)
    return rng.dirichlet(alpha, size=n)


def hybrid_search(A, y, profiles, k: int, seed: int,
                  log: Log | None = None):
    """Búsqueda híbrida en tres fases: grid + random + local — secuencial.

    Fase 1 — Grid step=0.02 (~1326 puntos)
    Fase 2 — Random Dirichlet(1,1,1) global  (~50% del resto)
    Fase 3 — Local Dirichlet concentrada alrededor del mejor W (~50% del resto)
             dividida entre concentration=300 y concentration=1000.

    Args:
        A: (n_samples, n_items)
        y: (n_samples,)
        profiles: (n_items, 3)
        k: presupuesto total de candidatos
        seed: semilla RNG
        log: Log opcional

    Returns:
        tuple: (best_auc, best_consistency, best_weights, best_iter)
    """
    rng = np.random.default_rng(seed)

    # ── Fase 1: Grid grueso ────────────────────────────────────────────
    best_auc, best_consistency, best_w, best_iter, iteration = grid_search(
        A, y, profiles, log=log
    )

    remaining = k - iteration
    if remaining <= 0:
        return best_auc, best_consistency, best_w, best_iter

    random_n = remaining // 2
    local_n = remaining - random_n

    # ── Fase 2: Random global ──────────────────────────────────────────
    for _ in range(random_n):
        w = rng.dirichlet(np.ones(3))
        auc_val, cons_val = evaluate(A, y, profiles, w)
        if auc_val > best_auc:
            best_auc = auc_val
            best_consistency = cons_val
            best_w = w.copy()
            best_iter = iteration
            if log is not None:
                log.improvement(iteration, auc_val, cons_val, tuple(best_w))
        iteration += 1

    # ── Fase 3: Refinamiento local adaptativo ──────────────────────────
    if best_w is not None and local_n > 0:
        local_splits = [local_n // 2, local_n - local_n // 2]
        for conc, count in zip([300, 1000], local_splits):
            if count <= 0:
                continue
            alpha = np.maximum(np.array(best_w) * conc, 1e-3)
            for w in rng.dirichlet(alpha, size=count):
                auc_val, cons_val = evaluate(A, y, profiles, w)
                if auc_val > best_auc:
                    best_auc = auc_val
                    best_consistency = cons_val
                    best_w = w.copy()
                    best_iter = iteration
                    if log is not None:
                        log.improvement(iteration, auc_val, cons_val, tuple(best_w))
                iteration += 1

    return best_auc, best_consistency, tuple(best_w), best_iter


def timed_search(name: str, p: int, A, y, profiles, k: int, seed: int,
                 log: Log | None = None,
                 search_mode: str = 'random') -> SearchResult:
    """Ejecuta búsqueda de pesos con medición de tiempo — secuencial.

    El cronómetro cubre solo la búsqueda (no la carga de datos).
    Delega en random_search, grid_search o hybrid_search según search_mode.

    Args:
        name: etiqueta de implementación
        p: unidades paralelas
        A, y, profiles: datos
        k: candidatos
        seed: semilla
        log: Log opcional para logging en vivo
        search_mode: 'random' | 'grid' | 'hybrid'

    Returns:
        SearchResult con métricas y pesos.
    """
    start = perf_counter()

    if search_mode == 'grid':
        best_auc, best_consistency, best_weights, best_iter, actual_k = \
            grid_search(A, y, profiles, log=log)
        k = actual_k
    elif search_mode == 'hybrid':
        best_auc, best_consistency, best_weights, best_iter = \
            hybrid_search(A, y, profiles, k, seed, log=log)
    else:  # 'random'
        best_auc, best_consistency, best_weights, best_iter = \
            random_search(A, y, profiles, k, seed, log=log)

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
    """CLI: evalúa k candidatos de forma secuencial e imprime resultado."""
    ap = argparse.ArgumentParser(description='Scoring metagenómico — versión secuencial')
    ap.add_argument('--k', type=int, default=10000, help='Número de candidatos')
    ap.add_argument('--seed', type=int, default=42, help='Semilla RNG')
    ap.add_argument('--search', choices=['random', 'grid', 'hybrid'],
                    default='random', help='Estrategia de búsqueda')
    ap.add_argument('--data-dir', type=Path, default=Path('data'), help='Directorio de datos')
    ap.add_argument('--csv', action='store_true', help='Salida en CSV (formato benchmark)')
    args = ap.parse_args()

    # Cargar datos
    A, y, profiles = load_data(args.data_dir)

    # Logger colorido (solo si no es modo CSV)
    log = None if args.csv else Log('python_sequential', A.shape[1], args.k)

    # Búsqueda con tiempo
    result = timed_search('python_sequential', 1, A, y, profiles, args.k, args.seed,
                          log=log, search_mode=args.search)

    # --- Salida ---
    if args.csv:
        print(result.csv_row())
    else:
        w1, w2, w3 = result.weights
        print(f'implementation={result.implementation}')
        print(f'N={result.n_items}')
        print(f'K={result.k}')
        print(f'best_auc={result.auc:.6f}')
        print(f'best_w=[{w1:.8f}, {w2:.8f}, {w3:.8f}]')
        print(f'best_w_sum={w1 + w2 + w3:.8f}')
        print(f'time_sec={result.time_sec:.6f}')


if __name__ == '__main__':
    main()
