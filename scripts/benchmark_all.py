#!/usr/bin/env python3
"""benchmark_all.py — Benchmark completo: 6 implementaciones × 2 búsquedas × N valores de K.

Uso:
  python scripts/benchmark_all.py
  python scripts/benchmark_all.py --k-list 1000,10000
  python scripts/benchmark_all.py --data-dir data/processed/synthetic_CRC2000x500_balanced
"""
from __future__ import annotations
import argparse
import csv
import io
import math
import os
import platform
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'python'))
from logger import Log


# ── Resultado normalizado ───────────────────────────────────────────
@dataclass
class BenchResult:
    impl: str
    search: str
    K: int
    actual_K: int
    time_sec: float
    auc: float
    consistency: float
    w1: float = 0.0
    w2: float = 0.0
    w3: float = 0.0
    workers: int = 1
    error: str = ''


# ── Detección de entorno ────────────────────────────────────────────
def _in_wsl() -> bool:
    return 'microsoft' in platform.uname().release.lower()


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _data_path() -> str:
    """Retorna data-dir para implementaciones C/Python (CSV loader)."""
    # Auto-detectar el dataset disponible
    root = _project_root()
    candidates = [
        'data/processed/synthetic_CRC2000x10000_balanced',
        'data/processed/synthetic_CRC2000x500_balanced',
    ]
    for d in candidates:
        if (root / d).exists():
            return d
    return candidates[0]  # fallback


def _cuda_data_path(data_dir: str) -> str:
    return f'{data_dir}/npy'


# ── Runners ─────────────────────────────────────────────────────────
# Cada runner recibe (K, search, workers) y retorna BenchResult.

def _run_cmd(cmd: list[str], timeout: int = 120) -> tuple[str, str, int]:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.stdout, r.stderr, r.returncode
    except FileNotFoundError:
        return '', f'Comando no encontrado: {cmd[0]}', -1
    except subprocess.TimeoutExpired:
        return '', f'Timeout ({timeout}s)', -1


def _check_exe(path: str) -> bool:
    """Verifica si un ejecutable existe (y no es directorio)."""
    p = Path(path)
    return p.exists() and not p.is_dir()




def _parse_kv(stdout: str, key: str, default: float = 0.0) -> float:
    m = re.search(rf'{key}=([-\d.e+]+)', stdout)
    return float(m.group(1)) if m else default


def _parse_kv_str(stdout: str, key: str, default: str = '') -> str:
    m = re.search(rf'{key}=(.+)', stdout)
    return m.group(1).strip() if m else default


def _parse_best_w(stdout: str) -> tuple[float, float, float]:
    m = re.search(r'best_w=\[([^\]]+)\]', stdout)
    if m:
        parts = m.group(1).split(',')
        if len(parts) >= 3:
            return float(parts[0]), float(parts[1]), float(parts[2])
    return 0.0, 0.0, 0.0


# ── Runner: Python sequential ──
def _grid_resolution(K: int) -> int:
    """Resolución del grid derivada de K: N_grid ≈ K."""
    return max(int((math.sqrt(1 + 8 * K) - 1) / 2), 2)

def _grid_step(K: int) -> str:
    """Convierte K a step para Python/OpenMP grid."""
    res = _grid_resolution(K)
    return f'{1.0 / res:.6f}'

def _grid_steps(K: int) -> int:
    """Número de steps para MPI/CUDA grid."""
    return _grid_resolution(K)

def run_python_seq(K, search, workers, data_dir, log):
    label = 'python-seq'
    cmd = [sys.executable, str(_project_root() / 'python/sequential.py'),
           '--k', str(K), '--seed', '42', '--data-dir', data_dir,
           '--search', search, '--csv']
    if search == 'grid':
        cmd += ['--step', _grid_step(K)]
    stdout, stderr, rc = _run_cmd(cmd)
    if rc != 0 or not stdout.strip():
        return BenchResult(label, search, K, 0, 0, 0, 0, error=stderr or 'no output')

    parts = stdout.strip().split(',')
    try:
        impl, par, n_items, k, t, auc_val, cons, w1, w2, w3, seed, sm, iters = parts[:13]
        return BenchResult(label, search, int(k), int(k), float(t),
                           float(auc_val), float(cons), float(w1), float(w2), float(w3), workers=1)
    except (ValueError, IndexError) as e:
        return BenchResult(label, search, K, 0, 0, 0, 0, error=str(e))


# ── Runner: C sequential (via WSL con bash) ──
def run_c_seq(K, search, workers, data_dir, log):
    label = 'c-seq'
    exe = str(_project_root() / 'C_OpenMP_MPI/scoring_sequential')
    cmd = [exe, '--k', str(K), '--seed', '42', '--data-dir', data_dir]
    stdout, stderr, rc = _run_cmd(cmd)
    if rc != 0:
        return BenchResult(label, search, K, 0, 0, 0, 0, error=stderr or f'exit={rc}')

    t = _parse_kv(stdout, 'time_sec')
    auc_val = _parse_kv(stdout, 'best_auc')
    w1, w2, w3 = _parse_best_w(stdout)
    return BenchResult(label, search, K, K, t, auc_val, 0.0, w1, w2, w3, workers=1)


# ── Runner: Python multiprocessing ──
def run_python_mp(K, search, workers, data_dir, log):
    label = f'python-mp({workers})'
    cmd = [sys.executable, str(_project_root() / 'python/multicore.py'),
           '--k', str(K), '--seed', '42', '--data-dir', data_dir,
           '--search', search, '--workers', str(workers), '--csv']
    if search == 'grid':
        cmd += ['--step', _grid_step(K)]
    stdout, stderr, rc = _run_cmd(cmd)
    if rc != 0 or not stdout.strip():
        return BenchResult(label, search, K, 0, 0, 0, 0, error=stderr or 'no output')

    parts = stdout.strip().split(',')
    try:
        impl, par, n_items, k, t, auc_val, cons, w1, w2, w3, seed, sm, iters = parts[:13]
        return BenchResult(label, search, int(k), int(k), float(t),
                           float(auc_val), float(cons), float(w1), float(w2), float(w3), workers=workers)
    except (ValueError, IndexError) as e:
        return BenchResult(label, search, K, 0, 0, 0, 0, error=str(e))


# ── Runner: OpenMP ──
def run_openmp(K, search, workers, data_dir, log):
    n_threads = workers
    label = f'openmp({n_threads})'
    if _in_wsl():
        exe = str(_project_root() / 'C_OpenMP_MPI/scoring_openmp')
    else:
        exe = str(_project_root() / 'C_OpenMP_MPI/scoring_openmp.exe')
    if not _check_exe(exe):
        return BenchResult(label, search, K, 0, 0, 0, 0, error=f'binario no encontrado: {exe}')

    if search == 'grid':
        cmd = [exe, '--k', str(K), '--seed', '42', '--data-dir', data_dir,
               '--search', 'grid', '--threads', str(n_threads), '--step', _grid_step(K)]
    else:
        cmd = [exe, '--k', str(K), '--seed', '42', '--data-dir', data_dir,
               '--search', search, '--threads', str(n_threads)]
    stdout, stderr, rc = _run_cmd(cmd)
    if rc != 0:
        return BenchResult(label, search, K, 0, 0, 0, 0, error=stderr)

    t = _parse_kv(stdout, 'time_sec')
    auc_val = _parse_kv(stdout, 'best_auc')
    k_found = int(_parse_kv(stdout, 'K'))
    w1, w2, w3 = _parse_best_w(stdout)
    return BenchResult(label, search, K, k_found if k_found > 0 else K, t, auc_val, 0.0, w1, w2, w3, workers=n_threads)


# ── Runner: MPI ──
def run_mpi(K, search, workers, data_dir, log):
    n_ranks = workers
    label = f'mpi({n_ranks})'

    # Verificar binario MPI
    if _in_wsl():
        exe = str(_project_root() / 'C_OpenMP_MPI/scoring_mpi')
    else:
        exe = str(_project_root() / 'C_OpenMP_MPI/scoring_mpi.exe')
    if not _check_exe(exe):
        return BenchResult(label, search, K, 0, 0, 0, 0, error=f'binario no encontrado: {exe}')

    # Detectar mpirun/mpiexec
    launcher = 'mpirun'
    for mp in ['mpiexec', 'mpirun']:
        if subprocess.run(f'where {mp}', shell=True, capture_output=True).returncode == 0:
            launcher = mp
            break
    else:
        return BenchResult(label, search, K, 0, 0, 0, 0, error='mpirun/mpiexec no encontrado')

    if search == 'grid':
        cmd = [launcher, '-np', str(n_ranks), exe,
               '--strategy', 'grid', '--grid-steps', str(_grid_steps(K)),
               '--data-dir', data_dir]
    else:
        cmd = [launcher, '-np', str(n_ranks), exe,
               '--strategy', 'random', '--k', str(K), '--seed', '42',
               '--data-dir', data_dir]
    stdout, stderr, rc = _run_cmd(cmd, timeout=600)
    if rc != 0:
        return BenchResult(label, search, K, 0, 0, 0, 0, error=stderr)

    # Parse CSV: c_mpi,strategy,n_samples,n_items,k,workers,time_sec,best_auc,consistency,w1,w2,w3,seed
    try:
        parts = stdout.strip().split(',')
        if len(parts) < 12:
            return BenchResult(label, search, K, 0, 0, 0, 0, error=f'parse error: {stdout[:200]}')
        # parts[0]=c_mpi, parts[1]=strategy, parts[2]=n_samples, parts[3]=n_items,
        # parts[4]=k, parts[5]=workers, parts[6]=time_sec, parts[7]=auc, parts[8]=cons,
        # parts[9]=w1, parts[10]=w2, parts[11]=w3, parts[12]=seed
        k_val = int(parts[4])
        t = float(parts[6])
        auc_val = float(parts[7])
        cons_val = float(parts[8])
        w1, w2, w3 = float(parts[9]), float(parts[10]), float(parts[11])
        return BenchResult(label, search, K, k_val, t, auc_val, cons_val, w1, w2, w3, workers=n_ranks)
    except (ValueError, IndexError) as e:
        return BenchResult(label, search, K, 0, 0, 0, 0, error=str(e))


# ── Runner: CUDA C ──
def run_cuda(K, search, workers, data_dir, log):
    label = 'cuda_c'
    exe = str(_project_root() / 'CUDA/scoring_cuda.exe')
    npy_dir = _cuda_data_path(data_dir)
    if not os.path.exists(exe):
        return BenchResult(label, search, K, 0, 0, 0, 0, error=f'binario no encontrado: {exe}')

    if search == 'grid':
        cmd = [exe, '--k', str(K), '--seed', '42', '--search', 'grid',
               '--data-dir', npy_dir, '--grid-resolution', str(_grid_steps(K))]
    else:
        cmd = [exe, '--k', str(K), '--seed', '42', '--search', 'random',
               '--data-dir', npy_dir]
    stdout, stderr, rc = _run_cmd(cmd, timeout=600)
    if rc != 0:
        return BenchResult(label, search, K, 0, 0, 0, 0, error=stderr)

    # Parse CSV: cuda_c,search,mode,K,actual_k,N,best_auc,best_consistency,w1,w2,w3,time_sec,seed,block_size
    try:
        parts = stdout.strip().split(',')
        if len(parts) < 14:
            return BenchResult(label, search, K, 0, 0, 0, 0, error=f'parse error: {stdout[:200]}')
        actual_k = int(parts[4])
        t = float(parts[11])
        auc_val = float(parts[6])
        cons_val = float(parts[7])
        w1, w2, w3 = float(parts[8]), float(parts[9]), float(parts[10])
        return BenchResult(label, search, K, actual_k, t, auc_val, cons_val, w1, w2, w3, workers=1)
    except (ValueError, IndexError) as e:
        return BenchResult(label, search, K, 0, 0, 0, 0, error=str(e))


# ── Registry ────────────────────────────────────────────────────────
RUNNERS: list[tuple[str, Callable]] = [
    ('Python sequential', run_python_seq),
    ('C sequential',      run_c_seq),
    ('Python multiproc.', run_python_mp),
    ('OpenMP',            run_openmp),
    ('MPI',               run_mpi),
    ('CUDA C',            run_cuda),
]


# ── Logger wrapper ─────────────────────────────────────────────────
class BenchLogger:
    def __init__(self):
        self.log = Log('benchmark_all', 0, 0)
        self._count = 0

    def section(self, title: str):
        print(f'\n  {"=" * 56}')
        print(f'  {title:^56}')
        print(f'  {"=" * 56}\n')

    def result_line(self, r: BenchResult, elapsed: float):
        self._count += 1
        if r.error:
            print(f'  {r.impl:25s}  ERROR: {r.error[:60]}')
        else:
            speedup = f'{elapsed / r.time_sec:.1f}x' if r.time_sec > 0 else 'N/A'
            print(f'  {r.impl:25s}  AUC={r.auc:.6f}  time={r.time_sec:.4f}s  speedup={speedup}')

    def table_header(self, k: int, search: str):
        print(f'\n  {"-" * 70}')
        print(f'  K={k:,}  search={search}')
        print(f'  {"-" * 70}')
        print(f'  {"Implementacion":25s}  {"AUC":>10s}  {"Tiempo":>10s}  {"Speedup":>10s}')
        print(f'  {"-" * 70}')

    def summary(self, results: list[BenchResult]):
        print(f'\n  {"=" * 70}')
        print(f'  {"RESUMEN FINAL":^68}')
        print(f'  {"=" * 70}')
        # Pivot table
        df = pd.DataFrame([{
            'impl': r.impl, 'search': r.search, 'K': r.K,
            'time': r.time_sec, 'auc': r.auc
        } for r in results if not r.error])

        if df.empty:
            print('  No hay resultados válidos')
            return

        for search in ['random', 'grid']:
            sub = df[df.search == search]
            if sub.empty:
                continue
            print(f'\n  search={search}')
            print(f'  {"K":>8s}  {"Implementación":25s}  {"Tiempo":>10s}  {"AUC":>10s}')
            print(f'  {"-" * 60}')
            for _, row in sub.sort_values(['K', 'time']).iterrows():
                print(f'  {row.K:>8,d}  {row.impl:25s}  {row.time:>10.4f}s  {row.auc:>10.6f}')


# ── Gráficas Plotly ────────────────────────────────────────────────
def make_plots(results: list[BenchResult], output_dir: Path):
    df = pd.DataFrame([{
        'impl': r.impl, 'search': r.search, 'K': r.K,
        'time': r.time_sec, 'auc': r.auc, 'speedup': 0.0,
        'workers': r.workers,
    } for r in results if not r.error and r.time_sec > 0])

    if df.empty:
        return

    # Calcular speedup base (Python sequential)
    py_seq = df[(df.impl == 'python-seq')].copy()
    py_seq = py_seq.rename(columns={'time': 'base_time'})[['search', 'K', 'base_time']]
    df = df.merge(py_seq, on=['search', 'K'], how='left')
    df['speedup'] = df['base_time'] / df['time']

    color_map = {
        'python-seq': '#1f77b4', 'c-seq': '#ff7f0e',
        'python-mp(4)': '#2ca02c', 'openmp(4)': '#d62728',
        'mpi(4)': '#9467bd', 'cuda_c': '#8c564b',
    }

    # 1. Tiempos por K y search
    fig1 = make_subplots(rows=1, cols=2, subplot_titles=['Random search', 'Grid search'],
                         shared_yaxes=True)
    for idx, search in enumerate(['random', 'grid'], 1):
        sub = df[df.search == search]
        for impl in sorted(sub.impl.unique()):
            d = sub[sub.impl == impl].sort_values('K')
            color = color_map.get(impl, '#333333')
            fig1.add_trace(
                go.Scatter(x=d['K'], y=d['time'], mode='lines+markers',
                           name=impl, legendgroup=impl,
                           marker=dict(color=color), line=dict(color=color)),
                row=1, col=idx
            )
        fig1.update_xaxes(title_text='K', type='log', row=1, col=idx)
        fig1.update_yaxes(title_text='Tiempo (s)', type='log', row=1, col=idx)

    fig1.write_html(str(output_dir / 'times.html'))
    try:
        fig1.write_image(str(output_dir / 'times.png'), width=1200, height=500)
    except Exception:
        pass

    # 2. Speedup
    fig2 = make_subplots(rows=1, cols=2, subplot_titles=['Random search', 'Grid search'],
                         shared_yaxes=True)
    for idx, search in enumerate(['random', 'grid'], 1):
        sub = df[(df.search == search) & (df.impl != 'python-seq')]
        for impl in sorted(sub.impl.unique()):
            d = sub[sub.impl == impl].sort_values('K')
            color = color_map.get(impl, '#333333')
            fig2.add_trace(
                go.Bar(x=[str(k) for k in d['K']], y=d['speedup'],
                       name=impl, legendgroup=impl,
                       marker_color=color),
                row=1, col=idx
            )
        fig2.update_xaxes(title_text='K', row=1, col=idx)
        fig2.update_yaxes(title_text='Speedup vs Python seq.', type='log', row=1, col=idx)

    fig2.write_html(str(output_dir / 'speedup.html'))
    try:
        fig2.write_image(str(output_dir / 'speedup.png'), width=1200, height=500)
    except Exception:
        pass

    # 3. Tabla resumen HTML
    pivot = df.pivot_table(index=['search', 'K'], columns='impl',
                           values='time', aggfunc='first')
    pivot_html = pivot.to_html(float_format='%.4f')
    with open(output_dir / 'summary.html', 'w') as f:
        f.write('<html><body><h1>Benchmark Summary</h1>\n')
        f.write('<h2>Times (seconds)</h2>\n')
        f.write(pivot_html)
        pivot_auc = df.pivot_table(index=['search', 'K'], columns='impl',
                                    values='auc', aggfunc='first')
        f.write('<h2>AUC</h2>\n')
        f.write(pivot_auc.to_html(float_format='%.6f'))
        f.write('</body></html>')

    print(f'\n  Gráficas guardadas en: {output_dir}/')


# ── Main ────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description='Benchmark completo de scoring metagenómico')
    ap.add_argument('--k-list', type=str, default='1000,10000',
                    help='Valores de K separados por coma')
    ap.add_argument('--workers', type=int, default=4,
                    help='Workers/threads/ranks para impls paralelas')
    ap.add_argument('--data-dir', type=str, default='',
                    help='Data directory (default: auto)')
    ap.add_argument('--output', type=Path, default=Path('results/plots'),
                    help='Directorio para gráficas')
    ap.add_argument('--no-plots', action='store_true',
                    help='No generar gráficas Plotly')
    args = ap.parse_args()

    K_LIST = [int(k.strip()) for k in args.k_list.split(',')]
    data_dir = args.data_dir or _data_path()
    output_dir = _project_root() / args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = BenchLogger()
    all_results: list[BenchResult] = []
    results_csv = _project_root() / 'results/benchmark_all.csv'

    logger.section('BENCHMARK COMPLETO - Scoring Metagenómico')
    print(f'  K values:     {K_LIST}')
    print(f'  Searches:     random, grid')
    print(f'  Workers:      {args.workers}')
    print(f'  Data dir:     {data_dir}')
    print(f'  Environment:  {"WSL" if _in_wsl() else "Windows"}')

    for K in K_LIST:
        for search in ['random', 'grid']:
            logger.table_header(K, search)
            start = time.perf_counter()

            for impl_name, runner in RUNNERS:
                t0 = time.perf_counter()
                r = runner(K, search, args.workers, data_dir, logger)
                elapsed = time.perf_counter() - t0
                logger.result_line(r, elapsed)
                all_results.append(r)

    # CSV output
    with open(results_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['impl', 'search', 'K', 'actual_K', 'workers',
                     'time_sec', 'auc', 'consistency', 'w1', 'w2', 'w3', 'error'])
        for r in all_results:
            w.writerow([r.impl, r.search, r.K, r.actual_K, r.workers,
                        f'{r.time_sec:.6f}', f'{r.auc:.6f}', f'{r.consistency:.6f}',
                        f'{r.w1:.6f}', f'{r.w2:.6f}', f'{r.w3:.6f}', r.error])

    # Summary
    logger.summary(all_results)
    print(f'\n  Resultados guardados en: {results_csv}')

    # Plots
    if not args.no_plots:
        print(f'\n  Generando gráficas Plotly...')
        make_plots(all_results, output_dir)


if __name__ == '__main__':
    main()
