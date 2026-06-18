#!/usr/bin/env python3
r"""benchmark_all.py - Una sola ejecucion con todas las implementaciones.

Usa el junction en C:/Users/patol/Desktop/proyecto para evitar espacios.
Ejecuta C/OpenMP/MPI via WSL, Python y CUDA nativos.
"""
from __future__ import annotations
import argparse, csv, io, math, os, platform, re, subprocess, sys, time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'python'))
from logger import Log


# ── Resultado ───────────────────────────────────────────────────────
@dataclass
class BenchResult:
    impl: str; search: str; K: int; actual_K: int
    time_sec: float; auc: float; consistency: float
    w1: float = 0; w2: float = 0; w3: float = 0
    workers: int = 1; error: str = ''


# ── Path management ─────────────────────────────────────────────────
PROJECT = Path(__file__).resolve().parent.parent

def _in_wsl() -> bool:
    return 'microsoft' in platform.uname().release.lower()

def _junction() -> Path | None:
    """Retorna la ruta del junction sin espacios si existe."""
    j = Path('C:/Users/patol/Desktop/proyecto')
    return j if j.exists() else None

def _data_dir() -> str:
    """Data directory para CSV (C/Python) y NPY (CUDA)."""
    return 'data/processed/synthetic_CRC2000x10000_balanced'

def _cuda_npy_dir() -> str:
    return f'{_data_dir()}/npy'

def _grid_res(K: int) -> int:
    return max(int((math.sqrt(1 + 8 * K) - 1) / 2), 2)
def _grid_step(K: int) -> str:
    return f'{1.0 / _grid_res(K):.6f}'
def _grid_st(K: int) -> int:
    return _grid_res(K)

# ── Runner helpers ──────────────────────────────────────────────────
def _run(cmd: list[str], tmo: int = 600) -> tuple[str, str, int]:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=tmo)
        return r.stdout, r.stderr, r.returncode
    except Exception as e:
        return '', str(e), -1

def _kv(stdout: str, k: str, d: float = 0.0) -> float:
    m = re.search(rf'{k}=([-\d.e+]+)', stdout)
    return float(m.group(1)) if m else d

def _kv_str(stdout: str, k: str, d: str = '') -> str:
    m = re.search(rf'{k}=(.+)', stdout)
    return m.group(1).strip() if m else d

def _best_w(stdout: str) -> tuple[float, float, float]:
    m = re.search(r'best_w=\[([^\]]+)\]', stdout)
    if m:
        p = m.group(1).split(',')
        if len(p) >= 3: return float(p[0]), float(p[1]), float(p[2])
    return 0, 0, 0

# ── Runners ─────────────────────────────────────────────────────────

def run_py_seq(K, search, workers, data_dir, log, **kw):
    cmd = [sys.executable, str(PROJECT / 'python/sequential.py'),
           '--k', str(K), '--seed', '42', '--data-dir', data_dir, '--search', search, '--csv']
    if search == 'grid': cmd += ['--step', _grid_step(K)]
    o, e, rc = _run(cmd)
    if rc or not o.strip(): return BenchResult('python-seq', search, K, 0, 0, 0, 0, error=e or 'no output')
    try:
        p = o.strip().split(',')
        return BenchResult('python-seq', search, int(p[3]), int(p[3]), float(p[4]),
                          float(p[5]), float(p[6]), float(p[8]), float(p[9]), float(p[10]), 1)
    except Exception as ex: return BenchResult('python-seq', search, K, 0, 0, 0, 0, error=str(ex))

def run_py_mp(K, search, workers, data_dir, log, **kw):
    cmd = [sys.executable, str(PROJECT / 'python/multicore.py'),
           '--k', str(K), '--seed', '42', '--data-dir', data_dir,
           '--search', search, '--workers', str(workers), '--csv']
    if search == 'grid': cmd += ['--step', _grid_step(K)]
    o, e, rc = _run(cmd)
    if rc or not o.strip(): return BenchResult(f'python-mp({workers})', search, K, 0, 0, 0, 0, error=e or 'no output')
    try:
        p = o.strip().split(',')
        return BenchResult(f'python-mp({workers})', search, int(p[3]), int(p[3]), float(p[4]),
                          float(p[5]), float(p[6]), float(p[8]), float(p[9]), float(p[10]), workers)
    except Exception as ex: return BenchResult(f'python-mp({workers})', search, K, 0, 0, 0, 0, error=str(ex))

def _wsl_run(bin_rel: str, args: list[str], label: str, search: str, K: int, data_dir: str, workers: int) -> BenchResult:
    """Ejecuta un binario de Linux via WSL (usa junction para path sin espacios)."""
    j = _junction()
    if not j:
        return BenchResult(label, search, K, 0, 0, 0, 0, error='junction no disponible')
    wsl_root = f'/mnt/c/Users/patol/Desktop/proyecto'
    env = {**os.environ, 'MSYS2_ARG_CONV_EXCL': '*', 'WSLENV': 'MSYS2_ARG_CONV_EXCL/w'}
    cmd = ['wsl', '-d', 'Ubuntu', '--cd', f'{wsl_root}',
           f'{wsl_root}/{bin_rel}'] + args
    o, e, rc = _run(cmd, tmo=300)
    if rc: return BenchResult(label, search, K, 0, 0, 0, 0, error=e or f'exit={rc}')

    if 'scoring_sequential' in bin_rel or 'scoring_openmp' in bin_rel:
        t = _kv(o, 'time_sec'); a = _kv(o, 'best_auc')
        w1, w2, w3 = _best_w(o)
        return BenchResult(label, search, K, K, t, a, 0, w1, w2, w3, workers)
    if 'scoring_mpi' in bin_rel:
        for line in o.split('\n'):
            if line.strip().startswith('c_mpi,'):
                p = line.strip().split(',')
                if len(p) >= 12:
                    return BenchResult(label, search, K, int(p[4]), float(p[6]),
                                      float(p[7]), float(p[8]), float(p[9]), float(p[10]), float(p[11]), workers)
        return BenchResult(label, search, K, 0, 0, 0, 0, error='MPI: no CSV line')
    return BenchResult(label, search, K, 0, 0, 0, 0, error='unknown binary')

def run_c_seq(K, search, workers, data_dir, log, **kw):
    if _in_wsl():
        exe = str(PROJECT / 'C_OpenMP_MPI/scoring_sequential')
        if not exe.startswith('/'): return BenchResult('c-seq', search, K, 0, 0, 0, 0, error='WSL path needed')
        cmd = [exe, '--k', str(K), '--seed', '42', '--data-dir', data_dir]
        o, e, rc = _run(cmd)
        if rc: return BenchResult('c-seq', search, K, 0, 0, 0, 0, error=e)
        t = _kv(o, 'time_sec'); a = _kv(o, 'best_auc'); w1, w2, w3 = _best_w(o)
        return BenchResult('c-seq', search, K, K, t, a, 0, w1, w2, w3, 1)
    else:
        return _wsl_run('C_OpenMP_MPI/scoring_sequential',
                       ['--k', str(K), '--seed', '42', '--data-dir', data_dir],
                       'c-seq', search, K, data_dir, 1)

def run_openmp(K, search, workers, data_dir, log, **kw):
    n = workers
    if _in_wsl():
        exe = str(PROJECT / 'C_OpenMP_MPI/scoring_openmp')
        args = ['--k', str(K), '--seed', '42', '--data-dir', data_dir,
                '--search', search, '--threads', str(n)]
        if search == 'grid': args += ['--step', _grid_step(K)]
        o, e, rc = _run([exe] + args)
        if rc: return BenchResult(f'openmp({n})', search, K, 0, 0, 0, 0, error=e)
        t = _kv(o, 'time_sec'); a = _kv(o, 'best_auc'); kf = int(_kv(o, 'K')); w1, w2, w3 = _best_w(o)
        return BenchResult(f'openmp({n})', search, K, kf if kf else K, t, a, 0, w1, w2, w3, n)
    else:
        args = ['--k', str(K), '--seed', '42', '--data-dir', data_dir,
                '--search', search, '--threads', str(n)]
        if search == 'grid': args += ['--step', _grid_step(K)]
        return _wsl_run('C_OpenMP_MPI/scoring_openmp', args, f'openmp({n})', search, K, data_dir, n)

def run_mpi(K, search, workers, data_dir, log, **kw):
    n = workers
    if _in_wsl():
        exe = str(PROJECT / 'C_OpenMP_MPI/scoring_mpi')
        launcher = 'mpirun'
        for mp in ['mpirun', 'mpiexec']:
            if subprocess.run(['which', mp], capture_output=True).returncode == 0:
                launcher = mp
                break
        else:
            return BenchResult(f'mpi({n})', search, K, 0, 0, 0, 0, error='mpirun not found')
        if search == 'grid':
            cmd = [launcher, '-np', str(n), exe, '--strategy', 'grid',
                   '--grid-steps', str(_grid_st(K)), '--data-dir', data_dir]
        else:
            cmd = [launcher, '-np', str(n), exe, '--strategy', 'random',
                   '--k', str(K), '--seed', '42', '--data-dir', data_dir]
        o, e, rc = _run(cmd, tmo=300)
        if rc: return BenchResult(f'mpi({n})', search, K, 0, 0, 0, 0, error=e)
        for line in o.split('\n'):
            if line.strip().startswith('c_mpi,'):
                p = line.strip().split(',')
                if len(p) >= 12:
                    return BenchResult(f'mpi({n})', search, K, int(p[4]), float(p[6]),
                                      float(p[7]), float(p[8]), float(p[9]), float(p[10]), float(p[11]), n)
        return BenchResult(f'mpi({n})', search, K, 0, 0, 0, 0, error='no CSV line')
    else:
        args = ['--strategy', 'random', '--k', str(K), '--seed', '42', '--data-dir', data_dir] if search != 'grid' \
               else ['--strategy', 'grid', '--grid-steps', str(_grid_st(K)), '--data-dir', data_dir]
        return _wsl_run('C_OpenMP_MPI/scoring_mpi', args, f'mpi({n})', search, K, data_dir, n)

def run_cuda(K, search, workers, data_dir, log, **kw):
    exe = str(PROJECT / 'CUDA/scoring_cuda.exe')
    if not os.path.exists(exe):
        return BenchResult('cuda_c', search, K, 0, 0, 0, 0, error='binario no encontrado')
    npy = _cuda_npy_dir()
    if search == 'grid':
        cmd = [exe, '--k', str(K), '--seed', '42', '--search', 'grid',
               '--data-dir', npy, '--grid-resolution', str(_grid_st(K))]
    else:
        cmd = [exe, '--k', str(K), '--seed', '42', '--search', 'random', '--data-dir', npy]
    o, e, rc = _run(cmd, tmo=600)
    if rc: return BenchResult('cuda_c', search, K, 0, 0, 0, 0, error=e)
    for line in o.split('\n'):
        if line.strip().startswith('cuda_c,'):
            p = line.strip().split(',')
            if len(p) >= 14:
                return BenchResult('cuda_c', search, K, int(p[4]), float(p[11]),
                                  float(p[6]), float(p[7]), float(p[8]), float(p[9]), float(p[10]), 1)
    return BenchResult('cuda_c', search, K, 0, 0, 0, 0, error='no CSV line')

RUNNERS = [
    ('Python sequential', run_py_seq),
    ('C sequential',      run_c_seq),
    ('Python multiproc.', run_py_mp),
    ('OpenMP',            run_openmp),
    ('MPI',               run_mpi),
    ('CUDA C',            run_cuda),
]


# ── Logger & Plots ──────────────────────────────────────────────────
def make_plots(results: list[BenchResult], out_dir: Path):
    df = pd.DataFrame([{
        'impl': r.impl, 'search': r.search, 'K': r.K,
        'time': r.time_sec, 'auc': r.auc
    } for r in results if not r.error and r.time_sec > 0])
    if df.empty: return

    py_seq = df[df.impl == 'python-seq'][['search', 'K', 'time']].rename(columns={'time': 'base'})
    df = df.merge(py_seq, on=['search', 'K'], how='left')
    df['speedup'] = df['base'] / df['time']
    df['short_impl'] = df['impl'].replace({
        'python-seq': 'Python\nSeq', 'c-seq': 'C Seq',
        'python-mp(4)': 'Python\nMP', 'openmp(4)': 'OpenMP',
        'mpi(4)': 'MPI', 'cuda_c': 'CUDA'
    })

    cmap = {'python-seq':'#1f77b4','c-seq':'#ff7f0e','python-mp(4)':'#2ca02c',
            'openmp(4)':'#d62728','mpi(4)':'#9467bd','cuda_c':'#8c564b'}
    impl_order = ['python-seq','c-seq','python-mp(4)','openmp(4)','mpi(4)','cuda_c']
    impl_labels = ['Python\nSeq','C Seq','Python\nMP','OpenMP','MPI','CUDA']

    k_str_map = {10:'10',100:'100',500:'500',1000:'1K',5000:'5K',10000:'10K',100000:'100K'}

    # ── Grafica 1: Speedup (barras agrupadas) ─────────────────────
    fig1 = go.Figure()
    ks_sorted = sorted(df.K.unique())
    k_labels = [k_str_map.get(k, str(k)) for k in ks_sorted]

    for impl in impl_order:
        d = df[(df.impl == impl) & (df.impl != 'python-seq')]
        if d.empty: continue
        vals = []
        for k in ks_sorted:
            sub = d[(d.K == k)]
            sp = sub['speedup'].values[0] if len(sub) > 0 else None
            vals.append(sp)
        fig1.add_trace(go.Bar(
            name=impl_labels[impl_order.index(impl)],
            x=k_labels, y=vals,
            marker_color=cmap[impl],
            hovertemplate='K=%{x}<br>Speedup=%{y:.1f}x<extra></extra>',
            width=0.6 / len(impl_order),
            offsetgroup=impl_order.index(impl),
        ))

    fig1.update_layout(
        barmode='group', bargap=0.15, bargroupgap=0.05,
        title=dict(text='Speedup vs Python Secuencial', x=0.5, font=dict(size=16)),
        xaxis=dict(title='Numero de candidatos (K)', tickangle=45),
        yaxis=dict(title='Speedup (veces mas rapido)',
                   type='log',
                   tickmode='array',
                   tickvals=[1,2,5,10,20,50,100,200,500,1000],
                   ticktext=['1x','2x','5x','10x','20x','50x','100x','200x','500x','1000x'],
                   gridcolor='#eee', gridwidth=1),
        legend=dict(orientation='h', y=-0.25, x=0.5, xanchor='center', font=dict(size=11)),
        template='plotly_white', height=500, width=1000,
        hovermode='x unified',
        margin=dict(b=80),
    )
    # Linea de referencia en 1x
    fig1.add_hline(y=1, line=dict(color='gray', width=1, dash='dash'))

    fig1.write_html(str(out_dir / 'speedup.html'))
    try: fig1.write_image(str(out_dir / 'speedup.png'), width=1000, height=500, scale=2)
    except: pass
    print('  Grafica 1: speedup.html/png')

    # ── Grafica 2: Tiempos + AUC combinada ────────────────────────
    fig2 = make_subplots(rows=2, cols=2,
                         subplot_titles=('Tiempos - Random', 'Tiempos - Grid',
                                         'AUC - Random', 'AUC - Grid'),
                         vertical_spacing=0.15, horizontal_spacing=0.12,
                         shared_xaxes='columns', shared_yaxes='rows')

    for idx, (srch, col) in enumerate([('random', 1), ('grid', 2)]):
        sub = df[df.search == srch]
        for impl in impl_order:
            d = sub[sub.impl == impl].sort_values('K')
            if d.empty: continue
            color = cmap[impl]
            label = impl_labels[impl_order.index(impl)]

            # Tiempos (fila 1)
            fig2.add_trace(go.Scatter(
                x=d['K'], y=d['time'], mode='lines+markers',
                name=label, legendgroup=impl,
                marker=dict(size=8, color=color, line=dict(width=1, color='white')),
                line=dict(width=2.5, color=color),
                hovertemplate=f'<b>{label}</b><br>K=%{{x}}<br>Time=%{{y:.4f}}s<extra></extra>'
            ), row=1, col=col)

            # AUC (fila 2)
            fig2.add_trace(go.Scatter(
                x=d['K'], y=d['auc'], mode='lines+markers',
                name=label, legendgroup=impl, showlegend=False,
                marker=dict(size=8, color=color, symbol='diamond',
                           line=dict(width=1, color='white')),
                line=dict(width=2, color=color, dash='dot'),
                hovertemplate=f'<b>{label}</b><br>K=%{{x}}<br>AUC=%{{y:.6f}}<extra></extra>'
            ), row=2, col=col)

        # Configurar ejes X
        for row, col in [(1,1),(1,2),(2,1),(2,2)]:
            fig2.update_xaxes(
                title_text='Candidatos (K)' if row == 2 else '',
                type='log', tickangle=45,
                tickvals=list(k_str_map.keys()),
                ticktext=list(k_str_map.values()),
                gridcolor='#eee', gridwidth=1,
                row=row, col=col)

        # Ejes Y tiempos
        fig2.update_yaxes(title_text='Tiempo (s)' if col == 1 else '',
                          type='log', row=1, col=col,
                          tickformat='.1f',
                          gridcolor='#eee', gridwidth=1)

        # Ejes Y AUC
        fig2.update_yaxes(title_text='AUC' if col == 1 else '',
                          range=[0.5, 1.0], row=2, col=col,
                          tickformat='.3f',
                          gridcolor='#eee', gridwidth=1)

    fig2.update_layout(
        title=dict(text='Rendimiento por Implementacion', x=0.5, font=dict(size=16)),
        legend=dict(orientation='h', y=-0.08, x=0.5, xanchor='center', font=dict(size=11)),
        template='plotly_white', height=700, width=1100,
        hovermode='x unified',
        margin=dict(b=80),
    )
    fig2.write_html(str(out_dir / 'times_auc.html'))
    try: fig2.write_image(str(out_dir / 'times_auc.png'), width=1100, height=700, scale=2)
    except: pass
    print('  Grafica 2: times_auc.html/png')

    # ── Grafica 3: AUC detalle (scatter con jitter) ───────────────
    fig3 = go.Figure()
    for srch, dash in [('random', 'solid'), ('grid', 'dot')]:
        sub = df[df.search == srch]
        for impl in impl_order:
            d = sub[sub.impl == impl].sort_values('K')
            if d.empty: continue
            fig3.add_trace(go.Scatter(
                x=d['K'], y=d['auc'],
                mode='lines+markers',
                name=f'{impl_labels[impl_order.index(impl)]} ({srch})',
                legendgroup=f'{impl}_{srch}',
                marker=dict(size=9, color=cmap[impl],
                           symbol='circle' if srch == 'random' else 'x',
                           line=dict(width=1, color='white')),
                line=dict(width=2, color=cmap[impl], dash=dash),
                hovertemplate='<b>%{legendgroup}</b><br>K=%{x}<br>AUC=%{y:.6f}<extra></extra>'
            ))

    fig3.add_hline(y=0.75, line=dict(color='green', width=1, dash='dash'),
                   annotation_text='AUC util (>0.75)')
    fig3.add_hline(y=0.5, line=dict(color='red', width=1, dash='dot'),
                   annotation_text='AUC aleatorio (0.5)')

    fig3.update_layout(
        title=dict(text='Comparacion de AUC entre Implementaciones', x=0.5, font=dict(size=16)),
        xaxis=dict(title='Candidatos (K)', type='log',
                   tickvals=list(k_str_map.keys()),
                   ticktext=list(k_str_map.values()),
                   tickangle=45, gridcolor='#eee'),
        yaxis=dict(title='AUC (Area Under Curve)',
                   range=[0.45, 1.0], tickformat='.3f',
                   gridcolor='#eee', gridwidth=1),
        legend=dict(orientation='h', y=-0.35, x=0.5, xanchor='center',
                   font=dict(size=10)),
        template='plotly_white', height=550, width=1100,
        hovermode='x unified',
        margin=dict(b=100),
    )
    fig3.write_html(str(out_dir / 'auc.html'))
    try: fig3.write_image(str(out_dir / 'auc.png'), width=1100, height=550, scale=2)
    except: pass
    print('  Grafica 3: auc.html/png')

    # ── Summary HTML mejorado ──────────────────────────────────────
    pivot_time = df.pivot_table(index=['search', 'K'], columns='impl',
                                 values='time', aggfunc='first')
    pivot_speedup = df.pivot_table(index=['search', 'K'], columns='impl',
                                    values='speedup', aggfunc='first')
    pivot_auc = df.pivot_table(index=['search', 'K'], columns='impl',
                                values='auc', aggfunc='first')

    html = """<!DOCTYPE html><html><head><meta charset="utf-8">
    <style>
        body { font-family: 'Segoe UI', Arial, sans-serif; margin: 40px; background: #f5f5f5; }
        h1 { color: #333; text-align: center; }
        h2 { color: #555; margin-top: 40px; border-bottom: 2px solid #ddd; padding-bottom: 8px; }
        table { border-collapse: collapse; width: 100%; margin: 20px 0; background: white; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
        th { background: #4a90d9; color: white; padding: 12px 8px; text-align: center; font-size: 13px; }
        td { padding: 8px; text-align: center; border-bottom: 1px solid #eee; font-size: 12px; }
        tr:hover { background: #f0f7ff; }
        .search-header { background: #e8f0fe; font-weight: bold; }
        .best { background: #d4edda !important; font-weight: bold; }
        .k-col { font-weight: bold; color: #333; }
        .speedup-col { color: #28a745; }
    </style></head><body>
    <h1>Benchmark de Scoring Metagenomico</h1>
    """

    for title, pivot, unit in [
        ('Tiempos de Ejecucion (segundos)', pivot_time, 's'),
        ('Speedup vs Python Secuencial', pivot_speedup, 'x'),
        ('AUC por Implementacion', pivot_auc, '')
    ]:
        html += f'<h2>{title}</h2><table><tr><th>Busqueda</th><th>K</th>'
        for impl in impl_order:
            html += f'<th>{impl_labels[impl_order.index(impl)].replace(chr(10)," ")}</th>'
        html += '</tr>'

        for search in ['random', 'grid']:
            first = True
            for k in sorted(pivot.loc[search].index) if search in pivot.index else []:
                cls = '' if not first else ' class="search-header"'
                html += f'<tr{cls}>'
                html += f'<td>{"Random" if first else ""}</td>'
                html += f'<td class="k-col">{k_str_map.get(k,k)}</td>'
                row = pivot.loc[(search, k)]
                # Find best
                best_val = row.min() if 'AUC' not in title else row.max()
                for impl in impl_order:
                    val = row.get(impl, None)
                    if pd.isna(val):
                        html += '<td>-</td>'
                    else:
                        is_best = abs(val - best_val) < 0.001 if 'Speedup' in title or 'AUC' in title else val == best_val
                        cls = ' class="best"' if is_best else ''
                        u = unit if unit else ''
                        html += f'<td{cls}>{val:.4f}{u}</td>'
                html += '</tr>'
                first = False
        html += '</table>'

    html += "</body></html>"

    with open(out_dir / 'summary.html', 'w') as f:
        f.write(html)
    print('  Summary: summary.html')


# ── Main ────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description='Benchmark completo 6 implementaciones')
    ap.add_argument('--k-list', type=str, default='100,1000')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--output', type=Path, default=Path('results/plots'))
    ap.add_argument('--no-plots', action='store_true')
    args = ap.parse_args()

    K_LIST = [int(k) for k in args.k_list.split(',')]
    data_dir = _data_dir()
    out_dir = PROJECT / args.output; out_dir.mkdir(parents=True, exist_ok=True)

    log = Log('benchmark_all', 10000, max(K_LIST))
    all_results = []
    csv_path = PROJECT / 'results/benchmark_all.csv'

    print(f'\n  K values: {K_LIST}  workers: {args.workers}  data: {data_dir}')
    print(f'  Junction: {_junction()}')
    print(f'  {"=" * 60}\n')

    for K in K_LIST:
        for search in ['random', 'grid']:
            print(f'\n  {"-" * 60}')
            print(f'  K={K:,}  search={search}')
            print(f'  {"-" * 60}')
            for name, runner in RUNNERS:
                t0 = time.perf_counter()
                r = runner(K, search, args.workers, data_dir, log)
                dt = time.perf_counter() - t0
                if r.error:
                    print(f'  {r.impl:25s}  ERROR: {r.error[:60]}')
                else:
                    sp = f'{dt / r.time_sec:.1f}x' if r.time_sec > 0 else 'N/A'
                    print(f'  {r.impl:25s}  AUC={r.auc:.6f}  time={r.time_sec:.4f}s  speedup={sp}')
                all_results.append(r)

    # CSV
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['impl','search','K','actual_K','workers','time_sec','auc','consistency','w1','w2','w3','error'])
        for r in all_results:
            w.writerow([r.impl, r.search, r.K, r.actual_K, r.workers,
                       f'{r.time_sec:.6f}', f'{r.auc:.6f}', f'{r.consistency:.6f}',
                       f'{r.w1:.6f}', f'{r.w2:.6f}', f'{r.w3:.6f}', r.error])

    # Summary table
    print(f'\n  {"=" * 60}')
    print(f'  {"RESUMEN":^58}')
    print(f'  {"=" * 60}')
    for search in ['random', 'grid']:
        print(f'\n  search={search}')
        print(f'  {"K":>8}  {"Impl":25s}  {"Time":>10s}  {"Speedup":>10s}  {"AUC":>10s}')
        print(f'  {"-" * 65}')
        for r in sorted(all_results, key=lambda x: (x.search != search, x.K, x.time_sec)):
            if r.search != search or r.error: continue
            base = next((br.time_sec for br in all_results if br.impl=='python-seq' and br.K==r.K and br.search==search and not br.error), None)
            sp = f'{base/r.time_sec:.1f}x' if base and r.time_sec > 0 else '-'
            print(f'  {r.K:>8}  {r.impl:25s}  {r.time_sec:>10.4f}s  {sp:>10s}  {r.auc:>10.6f}')

    print(f'\n  Resultados: {csv_path}')

    if not args.no_plots:
        print(f'\n  Generando graficas...')
        make_plots(all_results, out_dir)
        print(f'  Graficas en: {out_dir}/')


if __name__ == '__main__':
    main()
