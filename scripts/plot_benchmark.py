#!/usr/bin/env python3
"""Genera gráficas de benchmark a partir de CSV de resultados."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def plot(df, col, out):
    """Genera un gráfico de barras para una columna del DataFrame.

    Args:
        df: DataFrame con columnas implementation, parallel_units, col
        col: nombre de la columna a graficar
        out: ruta de salida PNG
    """
    # TODO: implementar gráfico de barras
    pass


def main():
    """CLI: lee benchmark.csv y genera gráficas."""
    ap = argparse.ArgumentParser(description='Generar gráficas de benchmark')
    ap.add_argument('--input', type=Path, default=Path('results/benchmark.csv'),
                    help='CSV de entrada')
    ap.add_argument('--out-dir', type=Path, default=Path('results/plots'),
                    help='Directorio de salida')
    args = ap.parse_args()

    # TODO: leer CSV, generar gráficas
    raise NotImplementedError('Implementar generación de gráficas')


if __name__ == '__main__':
    main()
