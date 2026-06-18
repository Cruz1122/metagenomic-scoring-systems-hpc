#!/usr/bin/env python3
"""benchmark_complete.py — Ejecuta benchmark en Windows+WSL y combina resultados.

Uso:
  python scripts/benchmark_complete.py --k-list 100,1000
"""
from __future__ import annotations
import argparse
import csv
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / 'results'
RESULTS.mkdir(parents=True, exist_ok=True)
COMBINED_CSV = RESULTS / 'benchmark_complete.csv'
WSL_CSV = RESULTS / 'benchmark_wsl.csv'
WIN_CSV = RESULTS / 'benchmark_win.csv'


def run_in_git_bash(k_list: str, workers: int):
    """Ejecuta benchmark desde Git Bash (Python + CUDA)."""
    print('\n  [1/2] Ejecutando en Windows (Python + CUDA)...')
    cmd = [sys.executable, str(ROOT / 'scripts/benchmark_all.py'),
           '--k-list', k_list, '--workers', str(workers), '--no-plots']
    env = os.environ.copy()
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800, env=env)
    if r.returncode != 0:
        print(f'  ERROR en Windows: {r.stderr[:300]}')
    else:
        print(f'  OK')
    # Copy resultado
    src = ROOT / 'results/benchmark_all.csv'
    if src.exists():
        import shutil
        shutil.copy2(src, WIN_CSV)
        print(f'  Resultados guardados en: {WIN_CSV}')
    return WIN_CSV.exists()


def run_in_wsl(k_list: str, workers: int):
    """Ejecuta benchmark desde WSL (Python + C + OpenMP + MPI)."""
    print('\n  [2/2] Ejecutando en WSL (C/OpenMP/MPI)...')
    
    # Escribir script temporal en WSL
    wsl_script = (
        f'cd "/mnt/c/Users/patol/Desktop/Nueva carpeta/metagenomic-scoring-systems-hpc" && '
        f'python3 scripts/benchmark_all.py --k-list {k_list} --workers {workers} --no-plots'
    )
    
    r = subprocess.run(
        ['wsl', '-d', 'Ubuntu', 'bash', '-c', wsl_script],
        capture_output=True, text=True, timeout=1800,
        env={**os.environ, 'MSYS2_ARG_CONV_EXCL': '*'}
    )
    if r.returncode != 0:
        print(f'  ERROR en WSL: {r.stderr[:300]}')
    else:
        print(f'  OK')
        # Mostrar últimas líneas del resumen
        for line in r.stdout.split('\n'):
            if 'RESUMEN' in line or all(c in line for c in 'KImplemen'):
                print(f'  {line}')
    
    # Copiar resultado desde WSL
    src_wsl = '/mnt/c/Users/patol/Desktop/Nueva carpeta/metagenomic-scoring-systems-hpc/results/benchmark_all.csv'
    dst = str(WSL_CSV)
    r2 = subprocess.run(
        ['wsl', '-d', 'Ubuntu', 'bash', '-c', f'cat "{src_wsl}"'],
        capture_output=True, text=True, timeout=30,
        env={**os.environ, 'MSYS2_ARG_CONV_EXCL': '*'}
    )
    if r2.returncode == 0 and r2.stdout.strip():
        with open(dst, 'w') as f:
            f.write(r2.stdout)
        print(f'  Resultados guardados en: {WSL_CSV}')
        return True
    return False


def combine_results():
    """Combina los CSV de Windows y WSL en uno solo."""
    print('\n  Combinando resultados...')
    
    if not WIN_CSV.exists() and not WSL_CSV.exists():
        print('  No hay resultados para combinar')
        return
    
    seen = set()
    combined = []
    
    for csv_path in [WIN_CSV, WSL_CSV]:
        if not csv_path.exists():
            continue
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for row in reader:
                if len(row) < 12:
                    continue
                # Usar (impl, search, K) como clave única
                key = (row[0], row[1], row[4])
                if key not in seen and not row[-1]:  # solo sin error
                    seen.add(key)
                    combined.append(row)
    
    if not combined:
        print('  No hay resultados válidos')
        return
    
    with open(COMBINED_CSV, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['impl', 'search', 'K', 'actual_K', 'workers',
                     'time_sec', 'auc', 'consistency', 'w1', 'w2', 'w3', 'error'])
        for row in combined:
            w.writerow(row)
    
    print(f'  Resultados combinados: {COMBINED_CSV}')
    print(f'  Total: {len(combined)} filas')
    
    # Mostrar tabla
    print(f'\n  {"=" * 70}')
    print(f'  {"RESUMEN COMBINADO":^68}')
    print(f'  {"=" * 70}')
    for search in ['random', 'grid']:
        print(f'\n  search={search}')
        print(f'  {"K":>8s}  {"Implementacion":25s}  {"Tiempo":>10s}  {"AUC":>10s}')
        print(f'  {"-" * 60}')
        rows = [r for r in combined if r[1] == search]
        for r in sorted(rows, key=lambda x: (int(x[2]), float(x[5]))):
            print(f'  {r[2]:>8s}  {r[0]:25s}  {r[5]:>10s}s  {r[6]:>10s}')


def main():
    ap = argparse.ArgumentParser(description='Benchmark completo (Windows + WSL)')
    ap.add_argument('--k-list', type=str, default='100,1000',
                    help='Valores de K separados por coma')
    ap.add_argument('--workers', type=int, default=4,
                    help='Workers/threads/ranks')
    args = ap.parse_args()

    print(f'  Benchmark completo: K={args.k_list} workers={args.workers}')
    print(f'  Fase 1: Windows (Python + CUDA)')
    print(f'  Fase 2: WSL (Python + C + OpenMP + MPI)')
    print()

    run_in_git_bash(args.k_list, args.workers)
    run_in_wsl(args.k_list, args.workers)
    combine_results()


if __name__ == '__main__':
    main()
