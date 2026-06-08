from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
import numpy as np


@dataclass(frozen=True)
class SearchResult:
    """Resultado de una corrida de búsqueda aleatoria."""
    implementation: str
    parallel_units: int
    n_items: int
    k: int
    time_sec: float
    auc: float
    consistency: float
    weights: tuple[float, float, float]
    seed: int

    def csv_row(self) -> str:
        """Serializa el resultado como línea CSV."""
        w1, w2, w3 = self.weights
        return (
            f'{self.implementation},{self.parallel_units},{self.n_items},{self.k},'
            f'{self.time_sec:.9f},{self.auc:.9f},{self.consistency:.9f},'
            f'{w1:.9f},{w2:.9f},{w3:.9f},{self.seed}'
        )


def load_data(data_dir: str | Path):
    """Carga matrices A, etiquetas y perfiles desde archivos .npy.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: (A, y, profiles)
    """
    # TODO: implementar carga con validación de dimensiones
    pass


def auc_matrix(scores: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Calcula AUC por columna (para múltiples candidatos).

    Args:
        scores: (n_samples, n_candidates)
        y: (n_samples,)  1=positivo, 0=negativo

    Returns:
        vector AUC por candidato
    """
    # TODO: implementar
    pass


def auc_vector(scores: np.ndarray, y: np.ndarray) -> float:
    """Calcula AUC escalar.

    Args:
        scores: (n_samples,)
        y: (n_samples,)
    """
    # TODO: implementar
    pass


def consistency(scores: np.ndarray, y: np.ndarray) -> float:
    """Calcula el mejor promedio de sensibilidad + especificidad.

    Args:
        scores: (n_samples,)
        y: (n_samples,)
    """
    # TODO: implementar
    pass


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
    # TODO: implementar
    pass


def random_search(A, y, profiles, k: int, seed: int, batch_size: int = 8192):
    """Búsqueda aleatoria de pesos Dirichlet.

    Args:
        A: (n_samples, n_items)
        y: (n_samples,)
        profiles: (n_items, 3)
        k: número de candidatos a evaluar
        seed: semilla RNG
        batch_size: lotes para evitar OOM

    Returns:
        tuple[float, float, tuple]: (best_auc, best_consistency, best_weights)
    """
    # TODO: implementar
    pass


def timed_search(name: str, p: int, A, y, profiles, k: int, seed: int) -> SearchResult:
    """Ejecuta random_search con medición de tiempo.

    Returns:
        SearchResult con métricas y pesos.
    """
    # TODO: implementar
    pass
