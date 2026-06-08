#!/usr/bin/env python3
"""Implementación PyCUDA del scoring metagenómico."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
from time import perf_counter
import numpy as np


def main():
    """CLI: evalua k candidatos en GPU vía PyCUDA e imprime resultado."""
    ap = argparse.ArgumentParser(description='Scoring metagenómico — PyCUDA')
    ap.add_argument('--k', type=int, default=10000, help='Número de candidatos')
    ap.add_argument('--seed', type=int, default=42, help='Semilla RNG')
    ap.add_argument('--data-dir', type=Path, default=Path('data'), help='Directorio de datos')
    ap.add_argument('--csv', action='store_true', help='Salida en CSV')
    args = ap.parse_args()

    try:
        import pycuda.autoinit  # noqa
        import pycuda.driver as cuda
        from pycuda.compiler import SourceModule
    except Exception as e:
        print(f'PyCUDA no disponible: {e}', file=sys.stderr)
        raise SystemExit(2)

    # TODO: cargar datos, compilar kernel, lanzar, recoger mejor peso
    raise NotImplementedError('Implementar scoring con PyCUDA')


if __name__ == '__main__':
    main()
