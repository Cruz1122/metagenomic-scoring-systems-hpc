#!/usr/bin/env python3
"""Pipeline de benchmark para scoring metagenómico.

Ejecuta una estrategia × búsqueda × lista de K, captura resultados
y genera un CSV unificado con información de hardware.

Uso:
  python scripts/benchmark_pipeline.py \\
      --strategy sequential_python \\
      --search random \\
      --k 5000 10000 20000

  python scripts/benchmark_pipeline.py \\
      --all-strategies --search random \\
      --k 5000 10000 20000

  python scripts/benchmark_pipeline.py \\
      --strategy openmp --search hybrid \\
      --k all --workers 4

Estrategias:
  sequential_python, sequential_c, multiprocessing_python,
  openmp, mpi, pycuda

Búsqueda:
  random, grid, hybrid

K:
  Lista de enteros, o "all" → 5000 10000 20000 50000 2000000
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ── Constantes ──────────────────────────────────────────────────────────
DEFAULT_K_VALUES = [5000, 10000, 20000, 50000, 2000000]
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PYTHON = str(PROJECT_ROOT / '.venv' / 'bin' / 'python')

STRATEGY_MAP = {
    'sequential_python': {
        'binary': PYTHON,
        'script': 'python/sequential.py',
        'default_workers': 1,
    },
    'sequential_c': {
        'binary': str(PROJECT_ROOT / 'C_OpenMP_MPI' / 'scoring_sequential'),
        'script': None,
        'default_workers': 1,
    },
    'multiprocessing_python': {
        'binary': PYTHON,
        'script': 'python/multicore.py',
        'default_workers': max(1, os.cpu_count() or 4),
    },
    'openmp': {
        'binary': str(PROJECT_ROOT / 'C_OpenMP_MPI' / 'scoring_openmp'),
        'script': None,
        'default_workers': max(1, os.cpu_count() or 1),
    },
    'mpi': {
        'binary': 'mpirun',
        'script': str(PROJECT_ROOT / 'C_OpenMP_MPI' / 'scoring_mpi'),
        'default_workers': max(1, os.cpu_count() or 1),
    },
    'pycuda': {
        'binary': PYTHON,
        'script': 'CUDA/scoring_pycuda.py',
        'default_workers': 1,
    },
}

SEARCH_MODES = ('random', 'grid', 'hybrid')
PARALLEL_STRATEGIES = frozenset({'multiprocessing_python', 'openmp', 'mpi'})
ALL_STRATEGIES = (
    'sequential_python',
    'multiprocessing_python',
    'sequential_c',
    'openmp',
    'mpi',
    'pycuda',
)


# ── Hardware detection ──────────────────────────────────────────────────

@dataclass
class HardwareInfo:
    cpu_model: str = ''
    cpu_cores: int = 0
    cpu_logical_cores: int = 0
    ram_gb: float = 0.0
    gpu_model: str = ''
    gpu_cuda_cores: int = 0
    gpu_mem_gb: float = 0.0


def detect_cpu_model() -> str:
    try:
        with open('/proc/cpuinfo') as f:
            for line in f:
                if line.startswith('model name'):
                    return line.split(':', 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or 'unknown'


def detect_cpu_cores() -> int:
    try:
        with open('/proc/cpuinfo') as f:
            cores = set()
            for line in f:
                if line.startswith('cpu cores'):
                    cores.add(int(line.split(':')[1].strip()))
            if cores:
                return max(cores)
    except OSError:
        pass
    try:
        import psutil
        return psutil.cpu_count(logical=False)
    except ImportError:
        pass
    return max(1, os.cpu_count() or 1)


def detect_logical_cores() -> int:
    return max(1, os.cpu_count() or 1)


def detect_ram_gb() -> float:
    try:
        with open('/proc/meminfo') as f:
            for line in f:
                if line.startswith('MemTotal:'):
                    kb = int(line.split()[1])
                    return round(kb / (1024 * 1024), 1)
    except OSError:
        pass
    try:
        import psutil
        return round(psutil.virtual_memory().total / (1024**3), 1)
    except ImportError:
        pass
    try:
        import shutil
        total, _, _ = shutil.disk_usage('/')
        return round(total / (1024**3), 1)
    except ImportError:
        pass
    return 0.0


def _cuda_cores_from_name(model: str) -> int:
    """Estima CUDA cores a partir del nombre del GPU (0 si desconocido)."""
    model_lower = model.lower()
    if 'a100' in model_lower:
        return 6912
    if 'v100' in model_lower:
        return 5120
    if 'titan' in model_lower:
        return 3840
    if 'rtx 3090' in model_lower or 'rtx3090' in model_lower:
        return 10496
    if 'rtx 3080' in model_lower:
        return 8704
    if 'rtx 3070' in model_lower:
        return 5888
    if 'rtx 3060' in model_lower:
        return 3584
    if 'rtx 2080 ti' in model_lower:
        return 4352
    if 'rtx 2080' in model_lower:
        return 2944
    if 'rtx 2070' in model_lower:
        return 2304
    if 'rtx 2060' in model_lower:
        return 1920
    if 'gtx 1080 ti' in model_lower:
        return 3584
    if 'gtx 1080' in model_lower:
        return 2560
    if 'gtx 1660' in model_lower:
        return 1408
    if 'gtx 1650' in model_lower:
        return 896
    if 'tesla t4' in model_lower or 't4' in model_lower:
        return 2560
    if 'p100' in model_lower:
        return 3584
    return 0


def _detect_gpu_via_nvidia_smi() -> Optional[Tuple[str, int, float]]:
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,memory.total',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return None
        line = result.stdout.strip().splitlines()[0]
        name, _, mem_str = line.rpartition(',')
        name = name.strip()
        if not name:
            return None
        mem_gb = 0.0
        try:
            mem_gb = round(float(mem_str.strip()) / 1024, 1)
        except ValueError:
            pass
        return name, _cuda_cores_from_name(name), mem_gb
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None


def _detect_gpu_via_pycuda() -> Optional[Tuple[str, int, float]]:
    try:
        import pycuda.driver as drv
        drv.init()
        dev = drv.Device(0)
        raw_name = dev.name()
        name = raw_name.decode() if isinstance(raw_name, bytes) else str(raw_name)
        mem_gb = round(dev.total_memory() / (1024 ** 3), 1)
        return name, _cuda_cores_from_name(name), mem_gb
    except Exception:
        return None


def _lspci_gpu_desc(line: str) -> str:
    desc = line.split(':', 2)[-1].strip() if line.count(':') >= 2 else line.strip()
    bracket = re.search(r'\[(.+?)\]', desc)
    if bracket:
        return re.sub(r'\s+', ' ', bracket.group(1))
    return re.sub(r'\s+', ' ', desc)


def _detect_gpu_via_lspci(ram_gb: float) -> Optional[Tuple[str, int, float]]:
    try:
        result = subprocess.run(['lspci'], capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            return None
        nvidia_line = None
        integrated_line = None
        for line in result.stdout.splitlines():
            lower = line.lower()
            if not any(k in lower for k in ('vga', '3d controller', 'display controller')):
                continue
            if 'nvidia' in lower:
                nvidia_line = line
                break
            if integrated_line is None and (
                'intel' in lower or 'amd' in lower or 'vga' in lower
            ):
                integrated_line = line
        if nvidia_line:
            desc = _lspci_gpu_desc(nvidia_line)
            model = desc if desc.lower().startswith('geforce') or desc.lower().startswith('quadro') else f'NVIDIA {desc}'
            return model, _cuda_cores_from_name(model), 0.0
        if integrated_line:
            desc = _lspci_gpu_desc(integrated_line)
            return f'integrated ({desc})', 0, ram_gb
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass
    return None


def detect_gpu_info(ram_gb: float) -> Tuple[str, int, float]:
    for detector in (_detect_gpu_via_nvidia_smi, _detect_gpu_via_pycuda):
        info = detector()
        if info:
            return info
    info = _detect_gpu_via_lspci(ram_gb)
    if info:
        return info
    return 'none', 0, 0.0


def detect_hardware() -> HardwareInfo:
    ram = detect_ram_gb()
    gpu_model, gpu_cores, gpu_mem = detect_gpu_info(ram)
    return HardwareInfo(
        cpu_model=detect_cpu_model(),
        cpu_cores=detect_cpu_cores(),
        cpu_logical_cores=detect_logical_cores(),
        ram_gb=ram,
        gpu_model=gpu_model,
        gpu_cuda_cores=gpu_cores,
        gpu_mem_gb=gpu_mem,
    )


# ── Dataset shape detection ──────────────────────────────────────────────

def detect_dataset_shape(data_dir: str) -> Tuple[int, int]:
    """Lee shape de matrix_A.npy sin cargar la matriz completa."""
    for sub in ['npy', '']:
        base = Path(data_dir) / sub
        for name in ['matrix_A.npy']:
            path = base / name
            if path.exists():
                try:
                    with open(path, 'rb') as f:
                        f.read(6)
                        ver_major = ord(f.read(1))
                        f.read(1)
                        if ver_major == 1:
                            hdr_len = int.from_bytes(f.read(2), 'little')
                        else:
                            hdr_len = int.from_bytes(f.read(4), 'little')
                        header = f.read(hdr_len).decode()
                        import ast
                        shape = ast.literal_eval(header)['shape']
                        if len(shape) == 2:
                            return shape[0], shape[1]
                except Exception:
                    pass
    return 0, 0


# ── Output parsing ──────────────────────────────────────────────────────

@dataclass
class RunResult:
    implementation: str = ''
    search_mode: str = ''
    k_requested: int = 0
    k_max: int = 0
    n_items: int = 0
    n_samples: int = 0
    best_auc: float = 0.0
    best_consistency: float = 0.0
    best_w1: float = 0.0
    best_w2: float = 0.0
    best_w3: float = 0.0
    iterations_until_best: int = -1
    time_sec: float = 0.0
    parallel_units: int = 1
    seed: int = 42
    cpu_model: str = ''
    cpu_cores: int = 0
    cpu_logical_cores: int = 0
    ram_gb: float = 0.0
    gpu_model: str = ''
    gpu_cuda_cores: int = 0
    gpu_mem_gb: float = 0.0


CSV_HEADER = (
    'implementation,search_mode,k_requested,k_max,'
    'n_items,n_samples,'
    'best_auc,best_w1,best_w2,best_w3,best_consistency,'
    'iterations_until_best,time_sec,parallel_units,seed,'
    'cpu_model,cpu_cores,cpu_logical_cores,ram_gb,'
    'gpu_model,gpu_cuda_cores,gpu_mem_gb'
)


def result_to_row(r: RunResult) -> List[str]:
    return [
        r.implementation,
        r.search_mode,
        str(r.k_requested),
        str(r.k_max),
        str(r.n_items),
        str(r.n_samples),
        f'{r.best_auc:.9f}',
        f'{r.best_w1:.9f}',
        f'{r.best_w2:.9f}',
        f'{r.best_w3:.9f}',
        f'{r.best_consistency:.9f}',
        str(r.iterations_until_best),
        f'{r.time_sec:.9f}',
        str(r.parallel_units),
        str(r.seed),
        r.cpu_model,
        str(r.cpu_cores),
        str(r.cpu_logical_cores),
        str(r.ram_gb),
        r.gpu_model,
        str(r.gpu_cuda_cores),
        str(r.gpu_mem_gb),
    ]


def _extract_kv(stdout: str) -> Dict[str, str]:
    """Extrae pares clave=valor de la salida (ignora líneas ANSI)."""
    ansi_escape = re.compile(r'\x1b\[[0-9;]*[mK]')
    data = {}
    for line in stdout.splitlines():
        clean = ansi_escape.sub('', line).strip()
        if '=' in clean and not clean.startswith('╭') and not clean.startswith('╰') and not clean.startswith('│'):
            key, _, val = clean.partition('=')
            data[key.strip()] = val.strip()
    return data


def _extract_consistency_from_log(stdout: str) -> Optional[float]:
    """Extrae consistencia del log (última ocurrencia de 'consistencia' o 'consist')."""
    ansi_escape = re.compile(r'\x1b\[[0-9;]*[mK]')
    last_cons = None
    for line in stdout.splitlines():
        clean = ansi_escape.sub('', line).strip()
        m = re.search(r'consisten(?:cia|t)\s*\.{0,3}\s*([0-9]+\.[0-9]+)', clean)
        if m:
            last_cons = float(m.group(1))
    return last_cons


def _extract_iterations_until_best(stdout: str) -> int:
    """Extrae la última iteración de mejora del logger (iter X/Y)."""
    ansi_escape = re.compile(r'\x1b\[[0-9;]*[mK]')
    last_iter = -1
    for line in stdout.splitlines():
        clean = ansi_escape.sub('', line).strip()
        m = re.search(r'iter\s+(\d+)/([\d,]+)', clean)
        if m:
            last_iter = int(m.group(1))
    return last_iter


def _extract_weights(weights_str: str) -> Tuple[float, float, float]:
    """Parsea '[0.595, 0.120, 0.285]' → (0.595, 0.120, 0.285)."""
    clean = weights_str.strip().strip('[]')
    parts = [p.strip() for p in clean.split(',')]
    w = [0.0, 0.0, 0.0]
    for i, p in enumerate(parts):
        if i < 3:
            try:
                w[i] = float(p)
            except ValueError:
                pass
    return tuple(w)  # type: ignore[return-value]


def _parse_csv_line(line: str) -> Optional[List[str]]:
    """Intenta parsear una línea CSV del stdout."""
    for line_raw in line.splitlines():
        stripped = line_raw.strip()
        if not stripped:
            continue
        parts = stripped.split(',')
        if len(parts) >= 5:
            return parts
    return None


def parse_output(strategy: str, stdout: str, k_requested: int,
                 seed: int, parallel_units: int,
                 hardware: HardwareInfo,
                 search_mode: str = 'random',
                 detected_n_samples: int = 0,
                 detected_n_items: int = 0) -> RunResult:
    """Parsea stdout de la estrategia ejecutada y completa RunResult."""
    r = RunResult(
        k_requested=k_requested,
        seed=seed,
        parallel_units=parallel_units,
        n_samples=detected_n_samples,
        n_items=detected_n_items,
        cpu_model=hardware.cpu_model,
        cpu_cores=hardware.cpu_cores,
        cpu_logical_cores=hardware.cpu_logical_cores,
        ram_gb=hardware.ram_gb,
        gpu_model=hardware.gpu_model,
        gpu_cuda_cores=hardware.gpu_cuda_cores,
        gpu_mem_gb=hardware.gpu_mem_gb,
    )

    if _parse_python_csv(stdout, r):
        if not r.search_mode:
            r.search_mode = search_mode
        if r.n_samples > 0 and r.n_items > 0:
            r.k_max = r.n_samples * r.n_items
        return r

    if strategy == 'sequential_c':
        _parse_c_kv(stdout, r, 'c_sequential', 1)
    elif strategy == 'openmp':
        _parse_c_kv(stdout, r, 'c_openmp', parallel_units)
    elif strategy == 'mpi':
        _parse_mpi_output(stdout, r)
    elif strategy == 'pycuda':
        _parse_pycuda_output(stdout, r)
    elif strategy == 'cuda_c':
        _parse_cudac_output(stdout, r)
    else:
        _parse_c_kv(stdout, r, strategy, parallel_units)

    r.search_mode = search_mode

    if r.n_samples > 0 and r.n_items > 0:
        r.k_max = r.n_samples * r.n_items

    return r


def _parse_python_csv(stdout: str, r: RunResult) -> bool:
    """Parsea salida CSV estándar (--benchmark / --csv). Retorna True si encontró fila."""
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(',')
        if len(parts) >= 12:
            try:
                r.implementation = parts[0]
                r.parallel_units = int(parts[1])
                r.n_items = int(parts[2])
                r.k_max = int(parts[3])
                r.time_sec = float(parts[4])
                r.best_auc = float(parts[5])
                r.best_consistency = float(parts[6])
                r.best_w1 = float(parts[7])
                r.best_w2 = float(parts[8])
                r.best_w3 = float(parts[9])
                r.seed = int(parts[10])
                r.search_mode = parts[11]
                if len(parts) > 12:
                    r.iterations_until_best = int(parts[12])
                return True
            except (ValueError, IndexError):
                continue

    return False


def _parse_c_kv(stdout: str, r: RunResult, impl: str, units: int):
    """Parsea salida clave=valor de C sequential / OpenMP."""
    kv = _extract_kv(stdout)
    r.implementation = kv.get('implementation', impl)
    r.parallel_units = units
    if r.implementation == 'c_sequential':
        r.parallel_units = 1

    if 'N' in kv:
        try:
            r.n_items = int(kv['N'])
        except ValueError:
            pass
    if 'K' in kv:
        try:
            r.k_max = int(kv['K'])
        except ValueError:
            r.k_max = r.k_requested
    else:
        r.k_max = r.k_requested
    if 'time_sec' in kv:
        try:
            r.time_sec = float(kv['time_sec'])
        except ValueError:
            pass
    if 'best_auc' in kv:
        try:
            r.best_auc = float(kv['best_auc'])
        except ValueError:
            pass

    w_str = kv.get('best_w', '')
    if w_str:
        r.best_w1, r.best_w2, r.best_w3 = _extract_weights(w_str)

    if 'workers' in kv:
        try:
            r.parallel_units = int(kv['workers'])
        except ValueError:
            pass

    if 'search_mode' in kv:
        r.search_mode = kv['search_mode']

    if 'consistency' in kv:
        try:
            r.best_consistency = float(kv['consistency'])
        except ValueError:
            pass

    cons = _extract_consistency_from_log(stdout)
    if cons is not None and r.best_consistency == 0.0:
        r.best_consistency = cons

    if r.iterations_until_best < 0:
        r.iterations_until_best = _extract_iterations_until_best(stdout)


def _parse_mpi_output(stdout: str, r: RunResult):
    """Parsea salida de MPI (CSV + key=value)."""
    kv = _extract_kv(stdout)
    r.implementation = 'c_mpi'

    if 'workers' in kv:
        try:
            r.parallel_units = int(kv['workers'])
        except ValueError:
            pass
    if 'search_mode' in kv:
        r.search_mode = kv['search_mode']
    try:
        n = int(kv.get('N', '0'))
        r.n_items = n
    except ValueError:
        pass
    try:
        r.k_max = int(kv.get('K', str(r.k_requested)))
    except ValueError:
        r.k_max = r.k_requested

    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(',')
        if len(parts) >= 12 and parts[0] == 'c_mpi':
            try:
                r.search_mode = parts[1] if parts[1] else r.search_mode
                r.n_samples = int(parts[2])
                r.n_items = int(parts[3])
                r.k_max = int(parts[4])
                r.parallel_units = int(parts[5])
                r.time_sec = float(parts[6])
                r.best_auc = float(parts[7])
                r.best_consistency = float(parts[8])
                r.best_w1 = float(parts[9])
                r.best_w2 = float(parts[10])
                r.best_w3 = float(parts[11])
                if len(parts) > 12:
                    r.seed = int(parts[12])
                return
            except (ValueError, IndexError):
                continue

    cons = _extract_consistency_from_log(stdout)
    if cons is not None:
        r.best_consistency = cons
    w_str = kv.get('best_w', '')
    if w_str:
        r.best_w1, r.best_w2, r.best_w3 = _extract_weights(w_str)


def _parse_pycuda_output(stdout: str, r: RunResult):
    """Parsea salida de PyCUDA (siempre CSV)."""
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(',')
        if len(parts) >= 13 and parts[0] == 'pycuda':
            try:
                r.implementation = 'pycuda'
                r.search_mode = parts[1]
                _mode = parts[2]
                try:
                    r.k_requested = int(parts[3])
                except ValueError:
                    pass
                try:
                    r.k_max = int(parts[4])
                except ValueError:
                    r.k_max = r.k_requested
                try:
                    r.n_items = int(parts[5])
                except ValueError:
                    pass
                r.best_auc = float(parts[6])
                r.best_consistency = float(parts[7])
                r.best_w1 = float(parts[8])
                r.best_w2 = float(parts[9])
                r.best_w3 = float(parts[10])
                r.time_sec = float(parts[11])
                r.seed = int(parts[12])
                if len(parts) > 13:
                    r.parallel_units = int(parts[13])
                return
            except (ValueError, IndexError):
                continue


def _parse_cudac_output(stdout: str, r: RunResult):
    """Parsea salida de CUDA C (siempre CSV)."""
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(',')
        if len(parts) >= 12 and parts[0] == 'cuda_c':
            try:
                r.implementation = 'cuda_c'
                r.search_mode = parts[1]
                _mode = parts[2]
                try:
                    r.k_requested = int(parts[3])
                except ValueError:
                    pass
                try:
                    r.k_max = int(parts[4])
                except ValueError:
                    r.k_max = r.k_requested
                try:
                    r.n_items = int(parts[5])
                except ValueError:
                    pass
                r.best_auc = float(parts[6])
                r.best_consistency = float(parts[7])
                r.best_w1 = float(parts[8])
                r.best_w2 = float(parts[9])
                r.best_w3 = float(parts[10])
                r.time_sec = float(parts[11])
                r.seed = int(parts[12])
                return
            except (ValueError, IndexError):
                continue


# ── Command construction ────────────────────────────────────────────────

def build_command(strategy: str, k: int, seed: int,
                  data_dir: str, search: str,
                  step: float, workers: int,
                  block_size: int, batch_size: int) -> List[str]:
    """Construye la lista de argumentos para subprocess.run()."""
    info = STRATEGY_MAP[strategy]

    if strategy == 'sequential_python':
        return [
            info['binary'],
            str(PROJECT_ROOT / info['script']),
            '--k', str(k),
            '--seed', str(seed),
            '--data-dir', data_dir,
            '--search', search,
            '--step', str(step),
            '--benchmark',
        ]

    elif strategy == 'sequential_c':
        return [
            info['binary'],
            '--k', str(k),
            '--seed', str(seed),
            '--data-dir', data_dir,
            '--search', search,
            '--step', str(step),
            '--benchmark',
        ]

    elif strategy == 'multiprocessing_python':
        return [
            info['binary'],
            str(PROJECT_ROOT / info['script']),
            '--k', str(k),
            '--seed', str(seed),
            '--data-dir', data_dir,
            '--search', search,
            '--step', str(step),
            '--workers', str(workers),
            '--benchmark',
        ]

    elif strategy == 'openmp':
        return [
            info['binary'],
            '--k', str(k),
            '--seed', str(seed),
            '--data-dir', data_dir,
            '--search', search,
            '--step', str(step),
            '--threads', str(workers),
            '--benchmark',
        ]

    elif strategy == 'mpi':
        return [
            info['binary'],
            '--allow-run-as-root',
            '--oversubscribe',
            '-np', str(workers),
            info['script'],
            '--k', str(k),
            '--seed', str(seed),
            '--data-dir', data_dir,
            '--search', search,
            '--step', str(step),
            '--benchmark',
        ]

    elif strategy == 'pycuda':
        return [
            info['binary'],
            str(PROJECT_ROOT / info['script']),
            '--k', str(k),
            '--seed', str(seed),
            '--data-dir', data_dir,
            '--search', search,
            '--block-size', str(block_size),
            '--benchmark',
        ]

    raise ValueError(f'Estrategia desconocida: {strategy}')


# ── Pre-flight checks ───────────────────────────────────────────────────

def check_strategy_available(strategy: str) -> Optional[str]:
    """Retorna mensaje de error si la estrategia no está disponible."""
    info = STRATEGY_MAP[strategy]

    if strategy in ('sequential_python', 'multiprocessing_python', 'pycuda'):
        script = PROJECT_ROOT / info['script']
        if not script.exists():
            return f'Script no encontrado: {script}'
        if not os.access(PYTHON, os.X_OK):
            return f'Python no encontrado o no ejecutable: {PYTHON}'

    if strategy in ('sequential_c', 'openmp'):
        binary = info['binary']
        if not os.access(binary, os.X_OK):
            return (f'Binario no encontrado: {binary}\n'
                    f'Ejecuta: make -C C_OpenMP_MPI {strategy.replace("_", "-")}')

    if strategy == 'mpi':
        if not shutil.which('mpirun'):
            return 'mpirun no encontrado en PATH'
        mpi_bin = info['script']
        if not os.access(mpi_bin, os.X_OK):
            return f'Binario MPI no encontrado: {mpi_bin}'

    if strategy == 'cuda_c':
        cuda_bin = info['binary']
        if not os.access(cuda_bin, os.X_OK):
            return f'Binario CUDA no encontrado: {cuda_bin}'

    return None


# ── Main pipeline ───────────────────────────────────────────────────────

def resolve_data_dir(data_dir: Optional[str]) -> str:
    if data_dir:
        return data_dir.rstrip('/')

    # Buscar el directorio de datos procesados disponible
    processed = PROJECT_ROOT / 'data' / 'processed'
    npy_dir = PROJECT_ROOT / 'data' / 'npy'

    if npy_dir.exists() and (npy_dir / 'matrix_A.npy').exists():
        return str(PROJECT_ROOT / 'data')

    if processed.exists():
        for d in sorted(processed.iterdir()):
            if d.is_dir() and (d / 'npy').exists():
                return str(d)

    if processed.exists():
        for d in sorted(processed.iterdir()):
            if d.is_dir():
                return str(d)

    return str(PROJECT_ROOT / 'data')


def resolve_parallel_units(strategy: str, hardware: HardwareInfo,
                           workers_override: Optional[int] = None) -> int:
    """Workers/threads/ranks: CPUs lógicos para estrategias paralelas, 1 si no."""
    if workers_override is not None:
        return max(1, workers_override)
    if strategy in PARALLEL_STRATEGIES:
        return max(1, hardware.cpu_logical_cores)
    return 1


def resolve_suite_strategies() -> List[str]:
    """Estrategias disponibles para --all-strategies (omite las no instaladas)."""
    selected: List[str] = []
    for strategy in ALL_STRATEGIES:
        error = check_strategy_available(strategy)
        if error:
            print(f'[WARN] Omitiendo {strategy}: {error}', file=sys.stderr)
        else:
            selected.append(strategy)
    return selected


def parse_k_values(k_args: List[str]) -> List[int]:
    if len(k_args) == 1 and k_args[0].lower() == 'all':
        return list(DEFAULT_K_VALUES)
    k_values: List[int] = []
    for v in k_args:
        try:
            k_values.append(int(v))
        except ValueError:
            print(f'Error: valor de K inválido: {v}', file=sys.stderr)
            sys.exit(1)
    return k_values


def run_strategy_benchmark(
    *,
    strategy: str,
    k_values: List[int],
    search: str,
    seed: int,
    data_dir: str,
    step: float,
    workers: int,
    block_size: int,
    batch_size: int,
    hardware: HardwareInfo,
    n_samples: int,
    n_items: int,
    writer: csv.writer,
    verbose: bool,
) -> Tuple[int, int]:
    """Ejecuta una estrategia para todos los K. Retorna (escritos, errores)."""
    results_written = 0
    errors = 0

    for k_val in k_values:
        cmd = build_command(
            strategy=strategy,
            k=k_val,
            seed=seed,
            data_dir=data_dir,
            search=search,
            step=step,
            workers=workers,
            block_size=block_size,
            batch_size=batch_size,
        )

        label = f'{strategy} k={k_val} search={search} units={workers}'
        print(f'\n[{label}]', file=sys.stderr)
        if verbose:
            print(f'  Comando: {" ".join(cmd)}', file=sys.stderr)

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=str(PROJECT_ROOT),
            )
            stdout_lines = []
            assert proc.stdout is not None
            for line in proc.stdout:
                if verbose:
                    print(line, end='', flush=True)
                stdout_lines.append(line)
            proc.wait()
        except FileNotFoundError as e:
            print(f'  ERROR: comando no encontrado: {e}', file=sys.stderr)
            errors += 1
            continue

        stderr_text = proc.stderr.read().strip() if proc.stderr else ''
        stdout_text = ''.join(stdout_lines).strip()

        if proc.returncode != 0:
            print(f'  ERROR: código de retorno {proc.returncode}', file=sys.stderr)
            if stderr_text:
                for line in stderr_text.splitlines()[-5:]:
                    print(f'  stderr: {line}', file=sys.stderr)
            errors += 1
            continue

        run_result = parse_output(
            strategy=strategy,
            stdout=stdout_text,
            k_requested=k_val,
            seed=seed,
            parallel_units=workers,
            hardware=hardware,
            search_mode=search,
            detected_n_samples=n_samples,
            detected_n_items=n_items,
        )

        writer.writerow(result_to_row(run_result))
        results_written += 1

        if verbose:
            print(f'  ✓ AUC={run_result.best_auc:.6f} '
                  f'time={run_result.time_sec:.4f}s '
                  f'k_max={run_result.k_max} '
                  f'iter={run_result.iterations_until_best}',
                  file=sys.stderr)
        else:
            print(f'  ✓ {run_result.best_auc:.6f} AUC '
                  f'en {run_result.time_sec:.4f}s '
                  f'(K={run_result.k_max}, units={workers})',
                  file=sys.stderr)

    return results_written, errors


def main():
    ap = argparse.ArgumentParser(
        description='Pipeline de benchmark para scoring metagenómico',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument('--strategy', '-s',
                      choices=list(STRATEGY_MAP.keys()),
                      help='Estrategia a ejecutar')
    mode.add_argument('--all-strategies', action='store_true',
                      help='Ejecutar todas las implementaciones disponibles (una vez c/u)')
    ap.add_argument('--search', '--search-mode', required=True,
                    choices=SEARCH_MODES,
                    help='Estrategia de búsqueda de pesos')
    ap.add_argument('--k', '-k', nargs='+', required=True,
                    help='Valores de K (enteros) o "all" para valores default')
    ap.add_argument('--workers', '-w', type=int, default=None,
                    help='Override de workers/threads/ranks (default: CPUs lógicos)')
    ap.add_argument('--data-dir', type=str, default=None,
                    help='Directorio de datos (default: auto)')
    ap.add_argument('--seed', type=int, default=42,
                    help='Semilla RNG (default: 42)')
    ap.add_argument('--output', '-o', type=str,
                    default=str(PROJECT_ROOT / 'results' / 'benchmark_pipeline.csv'),
                    help='Archivo CSV de salida')
    ap.add_argument('--step', type=float, default=0.02,
                    help='Paso del grid (default: 0.02)')
    ap.add_argument('--block-size', type=int, default=256,
                    help='Tamaño de bloque CUDA (default: 256)')
    ap.add_argument('--batch-size', type=int, default=1000000,
                    help='Candidatos por batch CUDA (default: 1000000)')
    ap.add_argument('--append', '-a', action='store_true',
                    help='Append al CSV existente en vez de sobrescribir')
    ap.add_argument('--verbose', '-v', action='store_true',
                    help='Salida verbosa')
    args = ap.parse_args()

    k_values = parse_k_values(args.k)
    data_dir = resolve_data_dir(args.data_dir)

    if args.all_strategies:
        strategies = resolve_suite_strategies()
        if not strategies:
            print('Error: ninguna estrategia disponible', file=sys.stderr)
            sys.exit(1)
    else:
        error = check_strategy_available(args.strategy)
        if error:
            print(f'Error: {error}', file=sys.stderr)
            sys.exit(1)
        strategies = [args.strategy]

    print('Detectando hardware...', file=sys.stderr)
    hardware = detect_hardware()
    n_samples, n_items = detect_dataset_shape(data_dir)
    if n_samples == 0:
        n_items = 0

    if args.verbose:
        print(f'  Dataset: {n_samples} muestras × {n_items} items', file=sys.stderr)
        print(f'  CPU: {hardware.cpu_model}', file=sys.stderr)
        print(f'  Cores: {hardware.cpu_cores} físicos, '
              f'{hardware.cpu_logical_cores} lógicos', file=sys.stderr)
        print(f'  RAM: {hardware.ram_gb} GB', file=sys.stderr)
        gpu_label = hardware.gpu_model if hardware.gpu_model != 'none' else 'none'
        print(f'  GPU: {gpu_label}', file=sys.stderr)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    write_header = not (args.append and output_path.exists())
    file_mode = 'a' if args.append else 'w'

    total_written = 0
    total_errors = 0

    with open(output_path, file_mode, newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(CSV_HEADER.split(','))

        for strategy in strategies:
            workers = resolve_parallel_units(strategy, hardware, args.workers)
            written, errs = run_strategy_benchmark(
                strategy=strategy,
                k_values=k_values,
                search=args.search,
                seed=args.seed,
                data_dir=data_dir,
                step=args.step,
                workers=workers,
                block_size=args.block_size,
                batch_size=args.batch_size,
                hardware=hardware,
                n_samples=n_samples,
                n_items=n_items,
                writer=writer,
                verbose=args.verbose,
            )
            total_written += written
            total_errors += errs

    print(f'\n{"="*50}', file=sys.stderr)
    print(f'Pipeline completado: {total_written} resultados escritos, '
          f'{total_errors} errores', file=sys.stderr)
    print(f'CSV: {output_path}', file=sys.stderr)
    if args.all_strategies:
        print(f'Estrategias: {", ".join(strategies)}', file=sys.stderr)
    else:
        print(f'Estrategia: {strategies[0]}', file=sys.stderr)
    print(f'Búsqueda: {args.search}', file=sys.stderr)
    print(f'K values: {k_values}', file=sys.stderr)
    print(f'Hardware: {hardware.cpu_model} / '
          f'{hardware.cpu_cores}C{hardware.cpu_logical_cores}T / '
          f'{hardware.ram_gb}GB RAM',
          file=sys.stderr)
    print(f'{"="*50}', file=sys.stderr)

    return 0 if total_errors == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
