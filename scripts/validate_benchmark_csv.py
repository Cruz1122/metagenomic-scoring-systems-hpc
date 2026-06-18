#!/usr/bin/env python3
"""Valida que benchmark_raw.csv solo contenga filas CSV limpias (--benchmark)."""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

HEADER = (
    'implementation,parallel_units,n_items,k,time_sec,auc,consistency,'
    'w1,w2,w3,seed,search_mode,iterations_until_best'
)
IMPLS = {
    'python_sequential', 'python_multicore', 'c_sequential',
    'c_openmp', 'c_mpi', 'pycuda', 'cuda_c',
}
SEARCH_MODES = {'random', 'grid', 'hybrid'}
ANSI_RE = re.compile(r'\x1b\[[0-9;]*[mK]')
BOX_RE = re.compile(r'[╭╰│─★]')


def validate(path: Path, expected_k: set[int] | None) -> int:
    errors: list[str] = []
    rows: list[dict[str, str]] = []

    text = path.read_text(encoding='utf-8', errors='replace')
    if ANSI_RE.search(text) or BOX_RE.search(text):
        errors.append('contiene salida ANSI/cajas de log (no es modo --benchmark)')

    lines = text.splitlines()
    if not lines:
        errors.append('archivo vacío')
        print('\n'.join(f'ERROR: {e}' for e in errors))
        return 1

    if lines[0].strip() != HEADER:
        errors.append(f'header incorrecto: {lines[0]!r}')

    for i, line in enumerate(lines[1:], start=2):
        if not line.strip():
            errors.append(f'línea {i}: vacía')
            continue
        if '=' in line and not line.startswith(tuple(IMPLS)):
            errors.append(f'línea {i}: parece key=value, no CSV')
            continue
        parts = line.split(',')
        if len(parts) != 13:
            errors.append(f'línea {i}: esperadas 13 columnas, hay {len(parts)}')
            continue
        impl, units, n_items, k, t, auc, cons, w1, w2, w3, seed, mode, iters = parts
        if impl not in IMPLS:
            errors.append(f'línea {i}: implementation desconocida {impl!r}')
        try:
            int(units); int(n_items); int(k); float(t)
            float(auc); float(cons); float(w1); float(w2); float(w3)
            int(seed); int(iters)
        except ValueError:
            errors.append(f'línea {i}: campo numérico inválido')
        if mode not in SEARCH_MODES:
            errors.append(f'línea {i}: search_mode inválido {mode!r}')
        rows.append({
            'implementation': impl, 'k': k, 'search_mode': mode,
            'parallel_units': units, 'time_sec': t, 'auc': auc,
        })

    if expected_k:
        seen_k = {int(r['k']) for r in rows}
        missing = expected_k - seen_k
        if missing:
            errors.append(f'faltan valores de K: {sorted(missing)}')

    print(f'Archivo: {path}')
    print(f'Filas válidas: {len(rows)}')
    if rows:
        by_impl: dict[str, int] = {}
        by_k: dict[str, int] = {}
        for r in rows:
            by_impl[r['implementation']] = by_impl.get(r['implementation'], 0) + 1
            by_k[r['k']] = by_k.get(r['k'], 0) + 1
        print('Por implementación:', dict(sorted(by_impl.items())))
        print('Por K:', dict(sorted(by_k.items(), key=lambda x: int(x[0]))))

    if errors:
        print(f'ERRORES ({len(errors)}):')
        for e in errors[:20]:
            print(f'  - {e}')
        if len(errors) > 20:
            print(f'  ... y {len(errors) - 20} más')
        return 1

    print('OK: CSV limpio y válido')
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', '-i', default='results/benchmark_raw.csv')
    ap.add_argument('--k', nargs='*', type=int, default=None,
                    help='Valores K esperados (opcional)')
    args = ap.parse_args()
    expected = set(args.k) if args.k else None
    return validate(Path(args.input), expected)


if __name__ == '__main__':
    sys.exit(main())
