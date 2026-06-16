from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score


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

