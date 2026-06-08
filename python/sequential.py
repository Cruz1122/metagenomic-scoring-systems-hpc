#!/usr/bin/env python3
"""Implementación secuencial (single-core) de búsqueda aleatoria."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from common import load_data, timed_search


def main():
    """CLI: evalúa k candidatos de forma secuencial e imprime resultado."""
    ap = argparse.ArgumentParser(description='Scoring metagenómico — versión secuencial')
    ap.add_argument('--k', type=int, default=10000, help='Número de candidatos')
    ap.add_argument('--seed', type=int, default=42, help='Semilla RNG')
    ap.add_argument('--data-dir', type=Path, default=Path('data'), help='Directorio de datos')
    ap.add_argument('--csv', action='store_true', help='Salida en CSV')
    args = ap.parse_args()

    # TODO: cargar datos, ejecutar búsqueda, imprimir resultado
    raise NotImplementedError('Implementar búsqueda secuencial')


if __name__ == '__main__':
    main()
