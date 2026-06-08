#!/usr/bin/env python3
"""Generación de datos sintéticos para scoring metagenómico."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np


def generate_data(n_items: int = 50, seed: int = 42, signal: float = 6.0):
    """Genera matriz de abundancias, etiquetas y perfiles sintéticos.

    El modelo asume que cada item (microbioma) tiene 3 componentes (T, S, F)
    y un peso verdadero que combina riesgo lineal.

    Args:
        n_items: número de ítems (features del microbioma)
        seed: semilla RNG
        signal: fuerza de la señal (separabilidad)

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            (A, y, profiles, true_w)
    """
    # TODO: implementar generación sintética
    raise NotImplementedError('Implementar generación de datos')


def main():
    """CLI: genera datos y los guarda en --out-dir."""
    ap = argparse.ArgumentParser(description='Generar datos sintéticos de scoring')
    ap.add_argument('--n-items', type=int, default=50, help='Número de ítems')
    ap.add_argument('--seed', type=int, default=42, help='Semilla RNG')
    ap.add_argument('--signal', type=float, default=6.0, help='Fuerza de señal')
    ap.add_argument('--out-dir', type=Path, default=Path('data'), help='Directorio de salida')
    args = ap.parse_args()

    # TODO: generar y guardar .npy, .csv, metadata.json
    raise NotImplementedError('Implementar guardado de datos')


if __name__ == '__main__':
    main()
