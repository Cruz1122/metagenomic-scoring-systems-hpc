from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING
import numpy as np
from sklearn.metrics import roc_auc_score

if TYPE_CHECKING:
    from logger import Log


@dataclass(frozen=True)
class SearchResult:
    """Resultado de una corrida de búsqueda de pesos."""
    implementation: str
    parallel_units: int
    n_items: int
    k: int
    time_sec: float
    auc: float
    consistency: float
    weights: tuple[float, float, float]
    seed: int
    search_mode: str = 'random'
    iterations_until_best: int = -1

    def csv_row(self) -> str:
        """Serializa el resultado como línea CSV."""
        w1, w2, w3 = self.weights
        return (
            f'{self.implementation},{self.parallel_units},{self.n_items},{self.k},'
            f'{self.time_sec:.9f},{self.auc:.9f},{self.consistency:.9f},'
            f'{w1:.9f},{w2:.9f},{w3:.9f},{self.seed},'
            f'{self.search_mode},{self.iterations_until_best}'
        )


def _load_npy(data_dir: Path, names: tuple[str, str, str]):
    """Intenta cargar tres .npy desde data_dir. Retorna None si falta alguno."""
    paths = [data_dir / n for n in names]
    if all(p.exists() for p in paths):
        return tuple(np.load(str(p)) for p in paths)
    return None


def load_data(data_dir: str | Path):
    """Carga matrices A, etiquetas y perfiles desde archivos .npy.

    Busca en este orden:
      1.  data_dir/npy/  con  matrix_A.npy, labels.npy, profiles_TSF.npy  (layout repo)
      2.  data_dir/      con  matrix_A.npy, labels.npy, profiles.npy      (layout plano)

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: (A, y, profiles)

    Raises:
        FileNotFoundError: si no encuentra los archivos.
        ValueError: si las dimensiones no son coherentes.
    """
    data_dir = Path(data_dir)

    # Estrategias de búsqueda ordenadas por prioridad
    strategies = [
        (data_dir / "npy", ("matrix_A.npy", "labels.npy", "profiles_TSF.npy")),
        (data_dir,          ("matrix_A.npy", "labels.npy", "profiles.npy")),
    ]

    loaded = None
    for base_dir, names in strategies:
        loaded = _load_npy(base_dir, names)
        if loaded is not None:
            break

    if loaded is None:
        raise FileNotFoundError(
            f"No se encontraron archivos .npy en {data_dir}/npy/ ni en {data_dir}/ "
            "(se busca: matrix_A.npy, labels.npy, profiles_TSF.npy o profiles.npy)"
        )

    A, y, profiles = loaded

    # --- Validación de dimensiones ---
    if A.ndim != 2:
        raise ValueError(f"A debe tener 2 dimensiones, tiene {A.ndim}")
    if y.ndim != 1:
        raise ValueError(f"y debe tener 1 dimensión, tiene {y.ndim}")
    if profiles.ndim != 2:
        raise ValueError(f"profiles debe tener 2 dimensiones, tiene {profiles.ndim}")
    if profiles.shape[1] != 3:
        raise ValueError(
            f"profiles debe tener exactamente 3 columnas, tiene {profiles.shape[1]}"
        )
    if A.shape[1] != profiles.shape[0]:
        raise ValueError(
            f"columnas de A ({A.shape[1]}) != filas de profiles ({profiles.shape[0]})"
        )
    if A.shape[0] != y.shape[0]:
        raise ValueError(
            f"filas de A ({A.shape[0]}) != largo de y ({y.shape[0]})"
        )
    classes = set(np.unique(y))
    if classes != {0, 1}:
        raise ValueError(f"y debe contener clases 0 y 1, contiene {classes}")

    return A, y, profiles


def auc_vector(scores: np.ndarray, y: np.ndarray) -> float:
    """Calcula AUC escalar.

    Args:
        scores: (n_samples,)
        y: (n_samples,)

    Returns:
        float: AUC
    """
    return float(roc_auc_score(y, scores))


def consistency(scores: np.ndarray, y: np.ndarray) -> float:
    """Calcula el mejor balanced accuracy sobre todos los umbrales.

    Barre todos los puntos de corte entre scores consecutivos y
    retorna el máximo (TPR + TNR) / 2.

    Args:
        scores: (n_samples,)
        y: (n_samples,)

    Returns:
        float: mejor balanced accuracy
    """
    order = np.argsort(scores)
    y_sorted = y[order]
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())

    best = 0.0
    # Empezamos con todos los positivos "por encima" del umbral mínimo
    tp = int(y_sorted.sum())
    tn = 0

    for i in range(len(scores)):
        tpr = tp / n_pos if n_pos > 0 else 1.0
        tnr = tn / n_neg if n_neg > 0 else 1.0
        best = max(best, (tpr + tnr) / 2.0)
        # Mover el umbral: el sample i pasa al lado "negativo" (≤ umbral)
        if y_sorted[i] == 1:
            tp -= 1
        else:
            tn += 1

    return float(best)


def evaluate(A, y, profiles, w):
    """Evalúa AUC y consistencia para un vector de pesos w.

    Args:
        A: (n_samples, n_items)
        y: (n_samples,)
        profiles: (n_items, 3)
        w: (3,)

    Returns:
        tuple[float, float]: (auc, consistency)
    """
    P = profiles @ w
    scores = A @ P
    auc_val = auc_vector(scores, y)
    cons_val = consistency(scores, y)
    return auc_val, cons_val


def random_search(A, y, profiles, k: int, seed: int,
                  log: Log | None = None):
    """Búsqueda aleatoria de pesos Dirichlet.

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
    """Búsqueda sistemática sobre el simplex 2D con paso fijo.

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
            # Saltar puntos fuera del simplex por error numérico
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
    """Búsqueda híbrida en tres fases: grid + random + local.

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
    # Se usa el mejor W actual (puede haber mejorado en Fase 2)
    if best_w is not None and local_n > 0:
        # Dividir local_n entre dos concentraciones
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
    """Ejecuta búsqueda de pesos con medición de tiempo.

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
        k = actual_k  # reportar el número real evaluado
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
