# PyCUDA (GPU)

## Rol en la arquitectura

PyCUDA es la implementacion que utiliza una GPU NVIDIA para acelerar la busqueda de pesos mediante paralelismo masivo. Mientras que las implementaciones CPU tienen decenas de nucleos (OpenMP, MPI) o procesos (multiprocessing), una GPU moderna tiene miles de nucleos SIMT que pueden evaluar miles de candidatos simultaneamente.

## Archivo

`CUDA/scoring_pycuda.py`

## Paradigma: paralelismo masivo GPU (SIMT)

Las GPUs NVIDIA organizan la ejecucion en una jerarquia de tres niveles:

1. **Grid:** conjunto de bloques que ejecutan un kernel. El grid se dimensiona segun el numero total de hilos necesarios.
2. **Bloque:** conjunto de hilos que se ejecutan en el mismo multiprocesador (SM) y pueden comunicarse mediante memoria compartida. Tamanio tipico: 256 hilos.
3. **Hilo (thread):** unidad minima de ejecucion. Cada hilo ejecuta el mismo codigo del kernel pero sobre datos diferentes.

Para este problema, la correspondencia es directa:

- **1 hilo = 1 candidato W**.
- El grid se dimensiona como `ceil(K / BLOCK_SIZE)` bloques de `BLOCK_SIZE = 256` hilos.
- Cada hilo evalua su candidato completo: genera scores, calcula AUC y consistencia.

## Kernel CUDA

El kernel CUDA esta escrito en C++ (embebido como string en el codigo Python) y compilado en runtime por PyCUDA (JIT compilation). El kernel se compila especificamente para la arquitectura de la GPU presente (ej. `sm_89` para RTX 4090).

### Kernel evaluate_full (modo completo)

Modo por defecto. Cada hilo:

1. Lee sus tres pesos W desde el arreglo global `weights`.
2. Calcula `P_i = W1*T_i + W2*S_i + W3*F_i` para cada item i.
3. Acumula `scores[s] += A[s, i] * P_i` para cada muestra s.
4. Calcula AUC y consistencia con funciones device.

Para optimizar el acceso a memoria, el kernel:

- Cachea columnas de A en memoria compartida (`__shared__`) para reducir accesos a memoria global.
- Usa acumulacion en double precision para los scores, aunque A y profiles esten en float32.
- Los hilos inactivos (en el ultimo bloque cuando K no es multiplo de BLOCK_SIZE) participan en `__syncthreads()` durante la carga cooperativa.

### Kernel evaluate_precompute (modo precompute)

Variante optimizada que precalcula `B = A @ profiles` en el host y transfiere la matriz B (n_muestras x 3) a la GPU. El kernel se reduce a:

```
scores[s] = B[s, 0] * W1 + B[s, 1] * W2 + B[s, 2] * W3
```

Ventaja: elimina las iteraciones sobre items dentro del kernel. Desventaja: requiere almacenar B en la GPU (n_muestras * 3 floats, despreciable para 2000 muestras).

## Reduccion en GPU

Para encontrar el mejor candidato entre todos los hilos, no se puede simplemente escribir en una variable global (condicion de carrera). En su lugar, se implementa una reduccion en dos etapas:

### Etapa 1: reduccion intra-bloque

Cada bloque reduce sus 256 candidatos a un `BestVal` parcial:

- Cada hilo escribe su AUC, consistencia e indice en memoria compartida.
- Se realiza una reduccion paralela con el patron de arbol binario (cada iteracion divide el numero de participantes por 2).
- El hilo 0 de cada bloque escribe el mejor valor del bloque en un arreglo global `partial`.

### Etapa 2: reduccion inter-bloque

Los valores parciales de todos los bloques se reducen a un solo resultado global:

- Si hay pocos bloques (<= BLOCK_SIZE), se usa un segundo kernel `reduce_best_stage2` que combina los parciales.
- Si hay muchos bloques, se itera con `reduce_best_stage1_from` antes de `reduce_best_stage2`.

### Pseudocodigo de la reduccion:

```
// Etapa 1: cada bloque -> 1 valor parcial
reduce_best_stage1<<<grid, block>>>(aucs, consistencies, partial, K)

// Etapa 2: parciales -> mejor global
reduce_best_stage2<<<1, block>>>(partial, &best_auc, &best_cons, &best_idx, n_blocks)
```

## Transferencias Host-Device

Las transferencias entre CPU (host) y GPU (device) son costosas y deben minimizarse:

| Buffer | Momento de transferencia |
|---|---|
| A (2000 x 10000) | Una vez al inicio |
| profiles (10000 x 3) | Una vez al inicio |
| labels (2000) | Una vez al inicio |
| B (2000 x 3) solo modo precompute | Una vez al inicio |
| weights (K x 3) | Una vez por fase de busqueda (`upload_weights`); en modo live no hay re-upload por bloque |
| Resultados AUC/consistency | Solo el mejor global (modo fast) o por bloques (modo live) |

## Modos de ejecucion

### Modo fast (benchmark)

Un solo launch del kernel con todos los K candidatos, seguido de reduccion completa en GPU. Solo el mejor resultado (AUC, consistencia, pesos, indice) se transfiere de vuelta al host. Es el modo apropiado para medir rendimiento bruto.

### Modo live (default)

Un launch por bloque de 256 candidatos. Despues de cada bloque, los resultados parciales se transfieren al host para mostrar progreso en vivo. Esto permite:

- Ver la evolucion del mejor AUC en tiempo real.
- Detectar problemas de convergencia temprana.
- Identificar en que bloque/thready/grilla aparece el mejor resultado.

El costo es que las transferencias parciales reducen el rendimiento total.

## Logger CUDA

El logger (`python/logger.py`) tiene un modo especifico para CUDA que muestra la jerarquia de paralelismo:

```
Mejora en iter 2560/100000: AUC 0.7823 (antes -1.0000) consistencia 0.95 W=[0.42,0.31,0.27] grilla#10 bloque#10 thread#0
```

Cada mejora reporta la grilla, bloque y thread donde se encontro, permitiendo correlacionar el resultado con la organizacion del paralelismo GPU.

## Compilacion y ejecucion

```bash
make python-pycuda K=100000 SEED=42 SEARCH=random
```

O directamente:

```bash
python CUDA/scoring_pycuda.py --k 100000 --seed 42 --data-dir data --search random --mode full
```

Requisitos: CUDA toolkit (>= 12), PyCUDA (>= 2024.1), GPU NVIDIA con capacidad de computo >= 5.0.

## Rendimiento esperado

La GPU evalua miles de hilos simultaneamente, pero cada hilo es mas lento que un nucleo de CPU (menor frecuencia de reloj, sin ejecucion out-of-order). La ventaja proviene del paralelismo masivo:

- Una GPU tipica tiene 100+ SMs, cada uno ejecutando 256 hilos en grupos de 32 (warps).
- El throughput teorico (candidatos/segundo) puede ser 10-50x superior al de una CPU secuencial.
- El speedup respecto a Python secuencial puede ser de 100x o mas para K suficientemente grande.

## Nota sobre eficiencia

La metrica de eficiencia tradicional (speedup / unidades de paralelismo) no es directamente aplicable a GPU porque las unidades de paralelismo (SMs) no son comparables a hilos o procesos de CPU. Se recomienda reportar throughput y speedup, y solo usar eficiencia GPU como metrica heuristica con la aclaracion correspondiente.
