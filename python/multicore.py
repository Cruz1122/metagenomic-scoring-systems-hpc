#!/usr/bin/env python3
"""Implementación multi-core (multiprocessing) de búsqueda aleatoria."""
from __future__ import annotations
import argparse
import json
import os
from multiprocessing import Pool
from pathlib import Path
from time import perf_counter
from common import SearchResult, load_data, random_search


def worker(payload):
    """Worker de pool: recibe (data_dir, k, seed), devuelve mejor resultado.

    Args:
        payload: tuple[str, int, int]

    Returns:
        tuple[float, float, tuple]: (auc, consistency, weights)
    """
    # TODO: cargar datos localmente y ejecutar random_search
    raise NotImplementedError('Implementar worker')


def main():
    """CLI: distribuye k candidatos entre workers y combina resultados."""
    ap = argparse.ArgumentParser(description='Scoring metagenómico — versión multi-core')
    ap.add_argument('--k', type=int, default=10000, help='Número de candidatos')
    ap.add_argument('--seed', type=int, default=42, help='Semilla RNG')
    ap.add_argument('--workers', type=int, default=max(1, os.cpu_count() or 1), help='Número de procesos')
    ap.add_argument('--data-dir', type=Path, default=Path('data'), help='Directorio de datos')
    ap.add_argument('--csv', action='store_true', help='Salida en CSV')
    args = ap.parse_args()

    # TODO: dividir k, lanzar pool, recolectar mejores pesos, imprimir
    raise NotImplementedError('Implementar búsqueda multi-core')


if __name__ == '__main__':
    main()
