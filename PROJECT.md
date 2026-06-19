# Optimización Paralela del Sistema de Scoring Metagenómico para Clasificación Binaria

**Proyecto de Computación de Alto Rendimiento (HPC)**  
**Tecnologías:** Python · C/OpenMP · MPI · CUDA  
**Área:** Bioinformática Computacional  
**Fecha:** 7 de mayo de 2026

---

## Índice

1. [Descripción del Problema](#1-descripción-del-problema)
2. [Modelo Matemático](#2-modelo-matemático)
   - [2.1. Score por Item](#21-score-por-item)
   - [2.2. Score por Muestra](#22-score-por-muestra)
   - [2.3. Función Objetivo](#23-función-objetivo)
   - [2.4. Validación del Scoring](#24-validación-del-scoring)
   - [2.5. Espacio de Búsqueda](#25-espacio-de-búsqueda)
3. [Arquitectura del Sistema](#3-arquitectura-del-sistema)
   - [3.1. Nivel 1 — Python: Baseline](#31-nivel-1--python-baseline)
   - [3.2. Nivel 2 — C con OpenMP y MPI](#32-nivel-2--c-con-openmp-y-mpi)
   - [3.3. Nivel 3 — CUDA: Aceleración GPU](#33-nivel-3--cuda-aceleración-gpu)
4. [Métricas de Evaluación](#4-métricas-de-evaluación)
   - [4.1. Tiempo de Ejecución](#41-tiempo-de-ejecución)
   - [4.2. Speedup](#42-speedup)
   - [4.3. Eficiencia](#43-eficiencia)
   - [4.4. Ley de Amdahl](#44-ley-de-amdahl)
   - [4.5. Tabla Comparativa](#45-tabla-comparativa)
5. [Estructura del Repositorio](#5-estructura-del-repositorio)
6. [Script de Generación de Datos](#6-script-de-generación-de-datos)
7. [Entregables](#7-entregables)

---

## 1. Descripción del Problema

Se busca desarrollar un sistema de optimización de alto rendimiento para determinar un vector de pesos $W$ que maximice el área bajo la curva ROC (**AUC**) en la clasificación binaria de muestras biológicas de tipo metagenómico.

El conjunto de datos consiste en **10 muestras** divididas simétricamente:

- **5 muestras** — población sana ($y = 0$).
- **5 muestras** — población enferma ($y = 1$).

Un scoring bien calibrado producirá una distribución de scores donde los dos grupos sean claramente distinguibles. Si la mayoría de scores altos se concentran en las muestras enfermas —o viceversa de modo consistente—, el modelo es válido; en caso contrario, la combinación de pesos $W$ no representa adecuadamente los datos.

---

## 2. Modelo Matemático

Cada muestra $j$ se describe mediante $N$ items —genomas o taxones—. Para cada item $i$ se dispone de tres perfiles:

**Cuadro 1. Perfiles por item**

| Símbolo | Perfil | Descripción |
|---|---|---|
| $T_i$ | Taxonómico | Abundancia relativa de microorganismos. |
| $S_i$ | Ecológico poblacional | Variables contextuales no genómicas de la muestra. |
| $F_i$ | Funcional | Presencia/ausencia de genes de interés —benéficos, de resistencia, etc.—. |

### 2.1. Score por Item

$$
P_i = W_1T_i + W_2S_i + W_3F_i
$$

Donde:

$$
W = (W_1, W_2, W_3)^\top
$$

es el vector de pesos a optimizar.

### 2.2. Score por Muestra

$$
\text{Score} = A \cdot P
$$

Donde:

- $A \in \mathbb{R}^{10 \times N}$: matriz de contribución de dimensiones fijas. La entrada $a_{ji}$ pondera el item $i$ en la muestra $j$.
- $P \in \mathbb{R}^{N}$: vector de scores por item.
- $\text{Score} \in \mathbb{R}^{10}$: un score escalar por muestra.

### 2.3. Función Objetivo

$$
\max_W \operatorname{AUC}(y, \text{Score}(W))
$$

El AUC mide la probabilidad de que una muestra enferma reciba un score mayor que una muestra sana. El rango de interés práctico es:

$$
\operatorname{AUC} \in [0{,}5, 1{,}0]
$$

### 2.4. Validación del Scoring

Para un umbral de decisión $\theta$:

$$
\text{Consistencia} =
\frac{|\{j : \text{Score}_j > \theta,\ y_j = 1\}|}{5}
+
\frac{|\{j : \text{Score}_j \leq \theta,\ y_j = 0\}|}{5}
$$

Se considera satisfactorio:

$$
\text{Consistencia} \geq 0{,}8
$$

### 2.5. Espacio de Búsqueda

$$
W_1 + W_2 + W_3 = 1, \qquad W_i \geq 0
$$

---

## 3. Arquitectura del Sistema

**Cuadro 2. Niveles de implementación**

| Nivel | Tecnología | Paradigma | Propósito |
|---:|---|---|---|
| 1 | Python — NumPy / multiprocessing | Secuencial / Multicore | Baseline y validación |
| 2 | C + OpenMP / C + MPI | Memoria compartida / distribuida | Paralelismo de CPU |
| 3 | PyCUDA / CUDA C | GPU — SIMD masivo | Aceleración máxima |

### 3.1. Nivel 1 — Python: Baseline

**Librerías:** NumPy, scikit-learn, multiprocessing.  
**Estrategias:** `random`, `grid` e `hybrid` sobre $K$ candidatos de $W$ en el simplex.

**Flujo (random):**

1. Cargar $A$, $y$ y `profiles`.
2. Muestrear $K$ vectores $W$ sobre el simplex (secuencial: uno por iteración; multicore: bloque con `dirichlet(..., size=K)`).
3. Para cada $W$: calcular $P \rightarrow \text{Score} = AP \rightarrow \text{AUC}$.
4. Retornar $W^* = \arg\max \text{AUC}$ (desempate: mayor consistencia, luego menor índice).

**Versión multicore:** generar los $K$ pesos **una vez** en el proceso principal; `_parallel_eval()` divide en tareas y evalúa con `pool.imap_unordered(partial(_eval_chunk, A, y, profiles), tasks)`. Cada worker retorna su mejor local; el proceso principal selecciona el mejor global. Sin variables compartidas ni sincronización explícita.

### 3.2. Nivel 2 — C con OpenMP y MPI

#### 3.2.1. C + OpenMP — Memoria Compartida

- Paralelizar el bucle de candidatos con `#pragma omp parallel` y `#pragma omp for schedule(static)`.
- Cada hilo mantiene `local_best` y lo guarda en `threads[tid].best`.
- Mejor global: merge serial post-loop sobre `threads[t].best` (sin `critical` ni `reduction`).
- `evaluate()` (incluida $A \cdot P$) corre secuencialmente dentro de cada hilo.

#### 3.2.2. C + MPI — Memoria Distribuida

- El proceso root genera los $K$ candidatos y los distribuye con `MPI_Scatterv`.
- Cada proceso evalúa su subconjunto de forma independiente.
- Recolección del óptimo con `MPI_Gather` + merge en rank 0 + `MPI_Bcast`.
- `MPI_Reduce(MPI_MAX)` solo para el tiempo de ejecución global.

### 3.3. Nivel 3 — CUDA: Aceleración GPU

- Cada hilo evalúa un candidato $W_k$.
- El kernel usa memoria compartida para cachear **columnas** de $A$ por bloque.
- Los datos estáticos (A, profiles, labels) se transfieren H2D una vez; los pesos por fase de búsqueda; en modo live hay D2H parcial por bloque.
- Reducción del mejor candidato con kernels `reduce_best_stage1` / `reduce_best_stage2` (multi-paso si $K$ es grande).

$$
\text{Grid} = \left\lceil \frac{K}{\text{BLOCK\_SIZE}} \right\rceil,
\qquad
\text{BLOCK\_SIZE} = 256
$$

---

## 4. Métricas de Evaluación

### 4.1. Tiempo de Ejecución

$$
T = t_{\text{fin}} - t_{\text{inicio}}
$$

Medido desde el inicio de la búsqueda hasta obtener $W^*$ —excluye carga de datos—.

### 4.2. Speedup

$$
S = \frac{T_{\text{Python}}}{T_{\text{impl}}}
$$

### 4.3. Eficiencia

$$
E = \frac{S}{P}
$$

Donde $P$ es el número de núcleos/hilos. $E = 1$ indica escalabilidad ideal.

### 4.4. Ley de Amdahl

$$
S_{\max} = \frac{1}{(1 - f) + \frac{f}{P}}
$$

$f$ se estima empíricamente como la fracción del tiempo total que ocupa la parte paralela.

### 4.5. Tabla Comparativa

**Cuadro 3. Resumen de desempeño — completar con resultados**

| Implementación | $T$ (s) | $S$ | $E$ | AUC |
|---|---:|---:|---:|---:|
| Python secuencial | — | 1.00 | — | — |
| Python multicore | — | — | — | — |
| C + OpenMP | — | — | — | — |
| C + MPI | — | — | — | — |
| CUDA | — | — | — | — |

---

## 5. Estructura del Repositorio

**Listing 1. Organización de carpetas**

```text
metagenomic-scoring-systems-hpc/
├── data/
│   ├── processed/               → datasets generados
│   │   ├── synthetic_CRC100x500_balanced/
│   │   └── synthetic_CRC2000x500_balanced/
│   └── scripts/
│       └── generate_dataset.py  → genera A, y, profiles sintéticos
├── python/
│   ├── __init__.py
│   ├── common.py                → [IMPLEMENTADO] SearchResult, load_data, evaluate
│   ├── sequential.py            → [IMPLEMENTADO] random/grid/hybrid search
│   ├── multicore.py             → [IMPLEMENTADO] multiprocessing con imap_unordered
│   └── logger.py                → [IMPLEMENTADO] logger ANSI
├── C_OpenMP_MPI/
│   ├── scoring_sequential.c     → [IMPLEMENTADO] baseline C secuencial
│   ├── scoring_openmp.c         → [IMPLEMENTADO] OpenMP: 3 estrategias
│   ├── scoring_mpi.c            → [IMPLEMENTADO] MPI: random/grid/hybrid
│   ├── shared/                  → [IMPLEMENTADO] common, RNG, ziggurat, logger
│   └── Makefile
├── CUDA/
│   └── scoring_pycuda.py        → [IMPLEMENTADO] PyCUDA: 3 estrategias, kernel embebido
├── scripts/
│   ├── benchmark_pipeline.py    → [IMPLEMENTADO] pipeline multi-K + detección hardware
│   └── validate_benchmark_csv.py→ [IMPLEMENTADO] valida formato CSV
├── results/
│   ├── benchmark.csv            → mediciones completas (random, K hasta 2M)
│   └── plots/                   → fig1_runtime … fig5_throughput (PNG)
├── docs/                       → especificación técnica (12 .md + prompts/)
├── run_all.sh                  → pipeline benchmark
├── Makefile
├── PROJECT.md
├── icon.svg
└── requirements.txt
```

---

## 6. Script de Generación de Datos

**Listing 2. `data/scripts/generate_dataset.py` (implementado)**

El generador produce datasets sintéticos con señal controlada basada en:

- **Abundancia relativa:** modelo gamma + ruido lognormal + factor de clase + metadata.
- **Perfiles T/S/F:** T desde log2FC de cohorte REF, S desde correlación con metadata, F desde marcadores funcionales.
- **REF/EVAL split:** T y S se estiman de REF independiente para evitar fuga de etiqueta.

Uso:

```bash
# 2000 muestras, 500 items (default)
python data/scripts/generate_dataset.py

# 100 muestras (desarrollo rápido)
python data/scripts/generate_dataset.py --n-eval 100 --n-ref 200 --allow-small
```

Parámetros clave: `--signal`, `--t-strength`, `--metadata-strength`, `--zero-inflation`, `--noise-sigma`.

Ver `docs/03_datos_y_seed.md` y `docs/11_dataset.md` para detalles completos.

---

## 7. Entregables

1. Código fuente organizado en `python/`, `C_OpenMP_MPI/` y `CUDA/`.
2. Script de generación de datos con $N$ configurable para pruebas de escalabilidad.
3. Script de benchmark `run_all.sh` que genera `results/benchmark.csv`.
4. Informe técnico en PDF con:
   - Estrategias de sincronización por nivel.
   - Análisis de gestión de memoria.
   - Gráficas comparativas de Speedup y Eficiencia vs. $P$.
   - Análisis de Amdahl con $f$ empírico.
   - Discusión de separabilidad de grupos mediante el score óptimo.
