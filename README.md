<p align="center">
  <img src="./icon.svg" alt="Scoring Metagenómico HPC" width="160" />
</p>

<p align="center" style="font-size:2rem;"><strong>Scoring Metagenómico HPC</strong></p>
<p align="center" style="font-size:1.25rem; margin-top:-1em;"><em>Optimización Paralela de Scoring Metagenómico para Clasificación Binaria</em></p>

<p align="center">
  <strong>Lenguajes y plataformas</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/C-11-A8B9CC?logo=c&logoColor=white" alt="C11" />
  <img src="https://img.shields.io/badge/CUDA-12-76B900?logo=nvidia&logoColor=white" alt="CUDA 12" />
  <img src="https://img.shields.io/badge/OpenMP-4.5-000000?logo=openmp&logoColor=white" alt="OpenMP" />
  <img src="https://img.shields.io/badge/MPI-3.0-006B3F?logo=linux&logoColor=white" alt="MPI" />
  <img src="https://img.shields.io/badge/PyCUDA-2024.1-3776AB?logo=python&logoColor=white" alt="PyCUDA" />
</p>

<p align="center">
  <img src="https://img.shields.io/badge/NumPy-≥1.24-013243?logo=numpy&logoColor=white" alt="NumPy" />
  <img src="https://img.shields.io/badge/scikit--learn-≥1.3-F7931E?logo=scikit-learn&logoColor=white" alt="scikit-learn" />
  <img src="https://img.shields.io/badge/Pandas-≥2.0-150458?logo=pandas&logoColor=white" alt="Pandas" />
  <img src="https://img.shields.io/badge/Matplotlib-≥3.7-11557C?logo=python&logoColor=white" alt="Matplotlib" />
</p>

---

## Decisión de arquitectura

El sistema busca un vector de pesos `W = (W1, W2, W3)` en el simplex `W1 + W2 + W3 = 1`, `Wi ≥ 0` que maximice el **AUC** de un scoring binario sobre 10 muestras metagenómicas (5 sanas, 5 enfermas). La búsqueda es por **Random Search** sobre `K` candidatos independientes, lo que permite paralelización trivial.

La arquitectura tiene **tres niveles de abstracción**:

```
Nivel 1 — Python          → baseline, validación, portabilidad
Nivel 2 — C/OpenMP + MPI  → paralelismo CPU: memoria compartida y distribuida
Nivel 3 — CUDA + PyCUDA   → aceleración GPU masivamente paralela
```

---

## Estructura del repositorio

```text
metagenomic-scoring-systems-hpc/
├── data/
│   ├── csv/                    → symlink a processed/.../csv/
│   ├── npy/                    → symlink a processed/.../npy/
│   ├── dataset_manifest.json   → symlink a processed/.../dataset_manifest.json
│   ├── processed/              → datasets generados
│   │   ├── synthetic_CRC100x500_balanced/   → 100 muestras (desarrollo)
│   │   └── synthetic_CRC2000x500_balanced/  → 2000 muestras (default)
│   └── scripts/
│       └── generate_dataset.py → genera dataset sintético (implementado)
├── python/
│   ├── __init__.py             → docstring del paquete
│   ├── common.py               → SearchResult, load_data, evaluate, AUC, consistency
│   ├── sequential.py           → baseline: random/grid/hybrid search (implementado)
│   ├── multicore.py            → multiprocessing Pool.map (implementado)
│   └── logger.py               → logger colorido ANSI
├── C_OpenMP_MPI/
│   ├── scoring_sequential.c    → baseline C secuencial
│   ├── scoring_openmp.c        → OpenMP: random + grid + hybrid (3 estrategias)
│   ├── scoring_mpi.c           → [SCAFFOLD] MPI: reusa shared/
│   ├── shared/                 → código compartido C
│   │   ├── common.h/c          → load_data, evaluate, AUC, consistency
│   │   ├── rng.h/c             → PCG64 + Dirichlet
│   │   ├── ziggurat.h/c        → generador Ziggurat
│   │   └── logger.h/c          → logger ANSI
│   └── Makefile                → compilación gcc / mpicc
├── CUDA/
│   └── scoring_pycuda.py       → PyCUDA (kernel embebido + logger grilla/bloque/thread)
├── scripts/
│   ├── postprocess_benchmark.py → [SCAFFOLD] calcula speedup/efficiency
│   └── plot_benchmark.py        → [SCAFFOLD] gráficas de benchmark
├── results/
│   └── plots/                  → .gitkeep (gráficas se generan con make plots)
├── docs/                       → documentación técnica (13 .md + prompts/)
│   └── prompts/                → prompts de dataset analysis
├── DATASET.md                  → descripción de datasets disponibles
├── run_all.sh                  → pipeline de benchmark automatizado
├── Makefile                    → atajos data, python-seq/grid/hybrid, python-mp/grid/hybrid, seq-run, openmp-run, mpi-run, cuda, benchmark
├── PROJECT.md                  → especificación contractual del proyecto
├── icon.svg                    → logo del proyecto
├── requirements.txt            → dependencias Python
└── .gitignore                  → ignora __pycache__, compilados, datos generados
```

---

## Requisitos

| Herramienta | Versión | Propósito |
|---|---|---|
| Python | ≥ 3.10 | Baseline secuencial, multicore, PyCUDA |
| GCC | ≥ 10 | Compilación OpenMP |
| MPICH / OpenMPI | ≥ 3.0 | Compilación y ejecución MPI |
| PyCUDA + CUDA toolkit | ≥ 12 | GPU scoring (`make python-pycuda`) |
| make | ≥ 4 | Automatización |

Dependencias Python:

```bash
pip install -r requirements.txt
# Contenido: numpy>=1.24, pandas>=2.0, matplotlib>=3.7, scikit-learn>=1.3
```

---

## Instalación y ejecución

### 1. Generar datos

```bash
# 2000 muestras (default)
make data

# 100 muestras (desarrollo rápido)
make data-100

# Directo
python data/scripts/generate_dataset.py

# 100 muestras (desarrollo rápido)
python data/scripts/generate_dataset.py --n-eval 100 --n-ref 200 --allow-small
```

### 2. Python (baseline)

```bash
make python-seq K=10000         # secuencial
make python-mp K=10000 WORKERS=4  # multicore
```

### 3. C secuencial

```bash
make seq-run K=10000            # compila y ejecuta baseline C
```

### 4. C/OpenMP

```bash
make c                          # compila todos los binarios C
OMP_NUM_THREADS=4 ./C_OpenMP_MPI/scoring_openmp --k 100000 --seed 42 --data-dir data
```

### 5. C/MPI

```bash
make -C C_OpenMP_MPI scoring_mpi
mpirun -np 4 ./C_OpenMP_MPI/scoring_mpi --k 100000 --seed 42 --data-dir data
```
> ⚠️ **Scaffold:** MPI imprime salida placeholder. Falta implementar la lógica de búsqueda distribuida.

### 6. PyCUDA

```bash
make python-pycuda K=100000 SEED=42 SEARCH=random
# Requiere CUDA toolkit (p. ej. /opt/cuda) y PyCUDA instalado
```

### 7. Benchmark integral

```bash
./run_all.sh                              # 2000 muestras
DATASET_SIZE=100 ./run_all.sh             # 100 muestras (rápido)
# Sobrescribe variables de entorno para configurar:
DATASET_SIZE=2000 N_ITEMS=500 K=100000 SEED=42 THREADS_LIST="1 2 4 8" WORKERS_LIST="2 4" MPI_RANKS_LIST="2 4" ./run_all.sh
```

Resultado: se genera `results/benchmark_raw.csv` con columnas `implementation,parallel_units,n_items,k,time_sec,auc,consistency,w1,w2,w3,seed,search_mode,iterations_until_best`. Post-procesado genera `results/benchmark.csv` añadiendo `speedup, efficiency`.

---

## Formato de salida estándar

Toda implementación imprime una línea CSV con:

```text
implementation,parallel_units,n_items,k,time_sec,auc,consistency,w1,w2,w3,seed,search_mode,iterations_until_best
```

| Columna | Descripción |
|---|---|---|
| `implementation` | Identificador: `python_sequential`, `python_multicore`, `c_openmp`, `c_mpi`, `cuda_c`, `pycuda` |
| `parallel_units` | Hilos / procesos / ranks usados |
| `n_items` | Número de items (N) |
| `k` | Candidatos evaluados |
| `time_sec` | Tiempo de búsqueda (excluye carga de datos) |
| `auc` | AUC del mejor W encontrado |
| `consistency` | Balanced accuracy máxima |
| `w1..w3` | Pesos del mejor candidato |
| `seed` | Semilla de reproducibilidad |
| `search_mode` | Estrategia: `random` \| `grid` \| `hybrid` |
| `iterations_until_best` | Iteración donde se encontró el mejor AUC |

El postprocesado (`scripts/postprocess_benchmark.py`) agrega `speedup = T_seq / T_impl` y `efficiency = speedup / parallel_units`.

---

## Estrategia de paralelización

| Implementación | Paradigma | División de trabajo | RNG | Sincronización |
|---|---|---|---|---|---|
| Python secuencial | — | batches vectorizados (8192) | `numpy.random.default_rng` | — |
| Python multicore | Multiprocessing | `K / workers`, seed offset 100003 | `numpy.random.default_rng` por worker | `Pool.map` |
| C/OpenMP | Memoria compartida | `schedule(static)` sobre for loop | PCG64 por hilo (`seed + tid`) | Merge local→global post-loop vía `#pragma omp critical` |
| C/MPI | Memoria distribuida | `chunk = ceil(K/size)` por rank | PCG64 por rank (`seed + rank`) | `MPI_Gather` / `MPI_Reduce` (scaffold) |
| PyCUDA | GPU (float32) | 1 thread = 1 candidato, grilla/bloque/thread | `numpy.random.default_rng` | AUC y consistencia en GPU |

---

## Calidad y verificación

```bash
# Postprocesar resultados
python scripts/postprocess_benchmark.py --input results/benchmark_raw.csv --output results/benchmark.csv

# Generar gráficas
python scripts/plot_benchmark.py --input results/benchmark.csv --out-dir results/plots

# Limpiar datos compilados y resultados
make clean
```

### Criterios de validación

- **AUC > 0.7**: indica separabilidad entre grupos (el azar es 0.5).
- **Consistencia ≥ 0.8**: balanced accuracy satisfactoria.
- **Speedup lineal vs P**: escalabilidad fuerte ideal.
- **Repetir corridas**: una sola ejecución sirve para depurar; el informe final debe reportar promedio ± desviación.

---

## Documentación

| Archivo | Contenido |
|---|---|
| `docs/index.md` | Índice de documentación operativa |
| `docs/00_resumen_tecnico.md` | Resumen técnico del proyecto |
| `docs/01_convenciones.md` | Convenciones de carpetas, CSV y reglas |
| `docs/02_modelo_scoring.md` | Modelo matemático de scoring |
| `docs/03_datos_y_seed.md` | Datos sintéticos, señal y reproducibilidad |
| `docs/04_python_multiprocessing.md` | Python secuencial y multiprocessing |
| `docs/05_openmp.md` | OpenMP: RNG, patrón, métricas |
| `docs/06_mpi.md` | MPI: scaffold, reusa shared/ |
| `docs/07_cuda.md` | PyCUDA |
| `docs/08_benchmarks.md` | Pipeline de benchmark |
| `docs/09_amdahl_gustafson.md` | Amdahl y escalabilidad débil |
| `docs/10_entregables.md` | Entregables mínimos del proyecto |
| `PROJECT.md` | Especificación contractual completa |

---

## Suposición explícita

El contrato define perfiles `T_i`, `S_i`, `F_i` en `item_profiles.csv` y `profiles_TSF.npy`.
`T` es magnitud diferencial en `[0,1]`. Ver `docs/11_dataset.md`.

---

## Notas operativas

- **No uses `K` pequeño para medir rendimiento.** Crear procesos/hilos/contextos tiene overhead. `K ≥ 10000` para mediciones significativas.
- **No compares tiempos entre implementaciones si cambiaste N, K, seed o dataset.**
- **No optimices código que no reproduce el AUC** — si la búsqueda no encuentra el máximo, la optimización de tiempo es irrelevante.
- **MPI en una sola máquina mide overhead**, no escalabilidad real.
- **Implementado:** `python/` (sequential, multicore, grid, hybrid), `C/OpenMP` (3 estrategias), generación de datos sintéticos.
- **Scaffold con TODO:** `C/MPI` (scoring_mpi.c), `CUDA/` (kernel.cu + pycuda.py), `scripts/postprocess_benchmark.py`, `scripts/plot_benchmark.py`.
