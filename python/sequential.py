#!/usr/bin/env python3
"""Implementación secuencial (single-core) de búsqueda aleatoria."""
from __future__ import annotations
import argparse
from pathlib import Path
from common import load_data, timed_search
from logger import Log


def main():
    """CLI: evalúa k candidatos de forma secuencial e imprime resultado."""
    ap = argparse.ArgumentParser(description='Scoring metagenómico — versión secuencial')
    ap.add_argument('--k', type=int, default=10000, help='Número de candidatos')
    ap.add_argument('--seed', type=int, default=42, help='Semilla RNG')
    ap.add_argument('--data-dir', type=Path, default=Path('data'), help='Directorio de datos')
    ap.add_argument('--csv', action='store_true', help='Salida en CSV (formato benchmark)')
    args = ap.parse_args()

    # Cargar datos
    A, y, profiles = load_data(args.data_dir)

    # Logger colorido (solo si no es modo CSV)
    log = None if args.csv else Log('python_sequential', A.shape[1], args.k)

    # Búsqueda con tiempo (solo el núcleo computacional)
    result = timed_search('python_sequential', 1, A, y, profiles, args.k, args.seed, log=log)

    # --- Salida ---
    if args.csv:
        # Formato CSV estándar del proyecto (para run_all.sh / benchmark)
        print(result.csv_row())
    else:
        # Formato humano legible (complementa al logger)
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
