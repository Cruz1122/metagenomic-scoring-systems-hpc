"""Logger colorido para búsqueda de scoring metagenómico.

Arquitectura:
  - Clase `Log` usada por common.py (random_search, timed_search)
  - Funciona sin dependencias extra (ANSI directo)
  - Degrada gracefulmente si la terminal no soporta color
  - Cada implementación (sequential, multicore, pycuda) crea su Log al inicio

Uso:
    from logger import Log
    log = Log('python_sequential', n_items=50, k=10000)
    ...
    log.improvement(iter, auc, prev_auc, consistency, weights)
    log.complete(result)
"""
from __future__ import annotations

import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from common import SearchResult


# ── Colores ANSI ────────────────────────────────────────────────────────────
class _Style:
    """ANSI escape codes. Detecta si la terminal los soporta."""
    _supports_color = hasattr(sys.stdout, 'isatty') and sys.stdout.isatty()

    def _c(self, code: str, text: str) -> str:
        return f'\033[{code}m{text}\033[0m' if self._supports_color else text

    # ── formato ──
    bold      = lambda self, t: self._c('1', t)
    dim       = lambda self, t: self._c('2', t)
    italic    = lambda self, t: self._c('3', t)
    underline = lambda self, t: self._c('4', t)

    # ── colores frente ──
    green  = lambda self, t: self._c('32', t)
    yellow = lambda self, t: self._c('33', t)
    blue   = lambda self, t: self._c('34', t)
    magenta= lambda self, t: self._c('35', t)
    cyan   = lambda self, t: self._c('36', t)
    white  = lambda self, t: self._c('37', t)
    red    = lambda self, t: self._c('91', t)

    # ── colores fondo ──
    bg_green  = lambda self, t: self._c('42', t)
    bg_blue   = lambda self, t: self._c('44', t)
    bg_magenta= lambda self, t: self._c('45', t)
    bg_cyan   = lambda self, t: self._c('46', t)

    # ── combinaciones ──
    gold      = lambda self, t: self._c('1;33', t)   # bold yellow
    highlight = lambda self, t: self._c('1;37;42', t) # bold white on green
    warn      = lambda self, t: self._c('1;37;41', t) # bold white on red


S = _Style()


# ── Detectar soporte Unicode ─────────────────────────────────────────────────
_SUPPORTS_UNICODE: bool = True
try:
    # Verificar si stdout puede imprimir caracteres Unicode
    '╭─'.encode(sys.stdout.encoding or 'utf-8')
except (UnicodeEncodeError, UnicodeDecodeError, LookupError):
    _SUPPORTS_UNICODE = False


# ── Logger ──────────────────────────────────────────────────────────────────
class Log:
    """Logger contextual que imprime hits de AUC con color.

    Attributes:
        impl:      nombre de la implementación (python_sequential, ...)
        n_items:   número de features
        k:         total de candidatos a evaluar
        best_auc:  mejor AUC visto hasta ahora
        best_iter: iteración donde se encontró best_auc
        _count:    contador de mejoras
    """

    def __init__(self, impl: str, n_items: int, k: int) -> None:
        self.impl = impl
        self.n_items = n_items
        self.k = k
        self.best_auc = -1.0
        self.best_iter = -1
        self._count = 0

        self._header()

    # ── privados ────────────────────────────────────────────────────────

    def _header(self) -> None:
        """Header de inicio: implementación, N, K."""
        if _SUPPORTS_UNICODE:
            tl, tr, bl, br, h, v = '╭', '╮', '╰', '╯', '─', '│'
        else:
            tl, tr, bl, br, h, v = '+', '+', '+', '+', '-', '|'
        sep = h * 58
        print()
        print(S.bold(S.cyan(f'  {tl}{h} {self.impl} ' + h * max(2, 48 - len(self.impl)) + f'{tr}')))
        print(S.cyan(f'  {v}  {S.bold("BÚSQUEDA DE PESOS ÓPTIMOS")}'))
        print(S.cyan(f'  {v}  items (N) ... {self.n_items}'))
        print(S.cyan(f'  {v}  candidatos   {self.k}'))
        print(S.cyan(f'  {bl}' + h * 56 + f'{br}'))
        print()

    def _improvement_line(self, iteration: int, auc: float, prev_auc: float,
                          consistency: float, weights: tuple,
                          worker_id: int | None = None) -> None:
        """Imprime una línea de mejora."""
        self._count += 1
        w1, w2, w3 = weights
        if self._count == 1:
            delta_str = "initial"
        else:
            delta = auc - prev_auc
            delta_str = f"+{delta:.6f}"
        arrow = '->' if not _SUPPORTS_UNICODE else '\u279c'
        tag = f'[W{worker_id}] ' if worker_id is not None else ''
        pct = 100.0 * (iteration + 1) / self.k if self.k > 0 else 0.0
        line = (
            f'  {S.gold(arrow)}  '
            f'{S.bold(f"{tag}AUC {auc:.6f}")}  '
            f'{S.green(f"({delta_str})")}  '
            f'iter {iteration:,}/{self.k:,} ({pct:.1f}%)  '
            f'consist={consistency:.4f}  '
            f'w=[{w1:.4f} {w2:.4f} {w3:.4f}]'
        )
        print(line, flush=True)

    def _summary(self, result: 'SearchResult') -> None:
        """Resumen final con recuadro."""
        w1, w2, w3 = result.weights
        if _SUPPORTS_UNICODE:
            tl, tr, bl, br, h, v = '╭', '╮', '╰', '╯', '─', '│'
        else:
            tl, tr, bl, br, h, v = '+', '+', '+', '+', '-', '|'
        sep = h * 58
        print()
        print(S.bold(S.magenta(f'  {tl}{h} MEJOR RESULTADO ' + h * max(2, 40 - len(str(self._count))) + f'{tr}')))
        ell = '...' if not _SUPPORTS_UNICODE else '...'
        print(S.bold(S.magenta(f'  {v}  {S.bold("implementacion")} {ell} {result.implementation}')))
        print(S.magenta(f'  {v}  {S.bold("mejoras")}        {ell} {self._count}'))
        print(S.magenta(f'  {v}  {S.bold("AUC")}            {ell} {S.gold(f"{result.auc:.9f}")}'))
        print(S.magenta(f'  {v}  {S.bold("consistencia")}   {ell} {result.consistency:.4f}'))
        print(S.magenta(f'  {v}  {S.bold("pesos W")}        {ell} [{w1:.9f}, {w2:.9f}, {w3:.9f}]'))
        print(S.magenta(f'  {v}  {S.bold("suma W")}         {ell} {w1 + w2 + w3:.9f}'))
        print(S.magenta(f'  {v}  {S.bold("tiempo")}         {ell} {S.cyan(f"{result.time_sec:.6f} s")}'))
        print(S.magenta(f'  {bl}' + h * 56 + f'{br}'))
        print()

    # ── API pública ─────────────────────────────────────────────────────

    def worker_report(self, worker_id: int, auc: float,
                      consistency: float, weights: tuple,
                      chunk_size: int, is_best: bool = False) -> None:
        """Reporta el mejor local encontrado por un worker.

        Args:
            worker_id:  ID del worker (0-indexed)
            auc:        AUC del mejor local del worker
            consistency: consistencia asociada
            weights:    tupla (w1, w2, w3)
            chunk_size: cantidad de candidatos evaluados en el chunk
            is_best:    True si este worker tiene el mejor global
        """
        w1, w2, w3 = weights
        marker = S.gold(' ★') if is_best else ''
        line = (
            f'  {S.dim(S.cyan(f"[W{worker_id}]"))}  '
            f'{S.bold(f"AUC {auc:.6f}")}  '
            f'consist={consistency:.4f}  '
            f'w=[{w1:.4f} {w2:.4f} {w3:.4f}]  '
            f'({chunk_size} cand.){marker}'
        )
        print(line, flush=True)

    def improvement(self, iteration: int, auc: float,
                    consistency: float, weights: tuple,
                    worker_id: int | None = None) -> None:
        """Llama en cada vez que se supera el mejor AUC.

        Args:
            iteration:  iteración actual (0-indexed)
            auc:        nuevo mejor AUC
            consistency: consistencia asociada
            weights:    tupla (w1, w2, w3)
            worker_id:  ID del worker que encontró la mejora (solo multi-core)
        """
        prev = self.best_auc
        self.best_auc = auc
        self.best_iter = iteration
        self._improvement_line(iteration, auc, prev, consistency, weights,
                               worker_id=worker_id)

    def complete(self, result: 'SearchResult') -> None:
        """Imprime resumen final.

        Args:
            result: SearchResult con métricas finales.
        """
        self.best_auc = result.auc
        self._summary(result)

    # ── API PyCUDA (grilla · bloque · thread) ───────────────────────────

    def cuda_info(self, device_name: str, sm_count: int, cuda_cores: int,
                  block_size: int, blocks_per_launch: int) -> None:
        """Imprime geometría CUDA bajo la cabecera estándar."""
        if _SUPPORTS_UNICODE:
            v = '│'
        else:
            v = '|'
        print(S.cyan(f'  {v}  GPU ............ {device_name}'))
        print(S.cyan(f'  {v}  SMs / cores .... {sm_count} / {cuda_cores}'))
        print(S.cyan(f'  {v}  grid CUDA ...... {blocks_per_launch} bloques  (ceil(K/{block_size}))'))
        print(S.cyan(f'  {v}  bloque ......... {block_size} threads'))
        print(S.cyan(f'  {v}  thread ......... 1 candidato/hilo'))
        print()
        sys.stdout.flush()

    def cuda_improvement(self, iteration: int, auc: float, prev_auc: float,
                         consistency: float, weights: tuple,
                         grilla: int, bloque: int, thread: int) -> None:
        """Mejora global en vivo con etiqueta grilla · bloque · thread."""
        self._count += 1
        w1, w2, w3 = weights
        if self._count == 1:
            delta_str = 'initial'
        else:
            delta_str = f'+{auc - prev_auc:.6f}'
        arrow = '->' if not _SUPPORTS_UNICODE else '\u279c'
        sep = ' · ' if _SUPPORTS_UNICODE else ' / '
        tag = f'[grilla {grilla}{sep}bloque {bloque}{sep}thread {thread}]'
        pct = 100.0 * (iteration + 1) / self.k if self.k > 0 else 0.0
        self.best_auc = auc
        self.best_iter = iteration
        line = (
            f'  {S.gold(arrow)}  '
            f'{S.dim(S.cyan(tag))}  '
            f'{S.bold(f"AUC {auc:.6f}")}  '
            f'{S.green(f"({delta_str})")}  '
            f'iter {iteration:,}/{self.k:,} ({pct:.1f}%)  '
            f'consist={consistency:.4f}  '
            f'w=[{w1:.4f} {w2:.4f} {w3:.4f}]'
        )
        print(line, flush=True)

    def cuda_progress(self, iteration: int, auc: float, consistency: float,
                      weights: tuple) -> None:
        """Progreso periódico cada N iteraciones (mejor global actual)."""
        w1, w2, w3 = weights
        pct = 100.0 * (iteration + 1) / self.k if self.k > 0 else 0.0
        dot = '...' if not _SUPPORTS_UNICODE else '\u2026'
        line = (
            f'  {S.dim(f"{dot}")}  '
            f'iter {iteration:,}/{self.k:,} ({pct:.1f}%)  '
            f'{S.bold(f"best_AUC {auc:.6f}")}  '
            f'consist={consistency:.4f}  '
            f'w=[{w1:.4f} {w2:.4f} {w3:.4f}]'
        )
        print(line, flush=True)

    def cuda_local_report(self, level: str, unit_id: int, auc: float,
                          consistency: float, weights: tuple,
                          chunk_size: int, is_best: bool = False,
                          grilla: int | None = None) -> None:
        """Reporta mejor AUC local por grilla o bloque CUDA."""
        w1, w2, w3 = weights
        marker = S.gold(' ★') if is_best else ''
        if level == 'grilla':
            label = f'[grilla {unit_id}]'
        elif level == 'bloque':
            if grilla is not None:
                sep = ' · ' if _SUPPORTS_UNICODE else ' / '
                label = f'[grilla {grilla}{sep}bloque {unit_id}]'
            else:
                label = f'[bloque {unit_id}]'
        else:
            label = f'[{level} {unit_id}]'
        line = (
            f'  {S.dim(S.cyan(label))}  '
            f'{S.bold(f"AUC {auc:.6f}")}  '
            f'consist={consistency:.4f}  '
            f'w=[{w1:.4f} {w2:.4f} {w3:.4f}]  '
            f'({chunk_size} cand.){marker}'
        )
        print(line, flush=True)
