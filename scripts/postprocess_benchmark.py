#!/usr/bin/env python3
"""Post-procesa benchmark_raw.csv añadiendo speedup y efficiency."""
from __future__ import annotations
import argparse
import csv
from pathlib import Path


def main():
    """CLI: lee raw CSV, calcula speedup/efficiency, escribe CSV procesado."""
    ap = argparse.ArgumentParser(description='Post-procesar benchmark')
    ap.add_argument('--input', type=Path, required=True, help='benchmark_raw.csv')
    ap.add_argument('--output', type=Path, required=True, help='benchmark.csv')
    args = ap.parse_args()

    # TODO: leer, identificar baseline, calcular speedup y efficiency, escribir
    raise NotImplementedError('Implementar post-procesamiento')


if __name__ == '__main__':
    main()
