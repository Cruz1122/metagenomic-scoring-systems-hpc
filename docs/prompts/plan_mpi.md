# Chat — Plan para implementar MPI en Metagenomic Scoring Systems HPC

## Usuario

### Contexto general del proyecto

**Metagenomic Scoring Systems HPC** — Implementaciones paralelas de búsqueda de pesos óptimos para scoring metagenómico.

### Problema matemático

Dados:

- `A`: abundancias, dimensión `100 × 500`
- `profiles_TSF`: perfiles taxonómicos, dimensión `500 × 3`
- `labels`: sano/enfermo, dimensión `100`

Se busca el vector:

```text
W = [w1, w2, w3]
```

que maximiza el AUC, donde:

```text
P = profiles @ W
scores = A @ P
auc = AUC(labels, scores)
```

### Implementaciones disponibles

| # | Implementación | Estrategias | Estado |
|---:|---|---|---|
| 1 | Python Sequential (`python/sequential.py`) | `random`, `grid`, `hybrid` | ✅ Completo |
| 2 | Python Multicore (`python/multicore.py`) | `random`, `grid`, `hybrid` | ✅ Completo |
| 3 | C OpenMP (`C_OpenMP_MPI/scoring_openmp.c`) | `random`, `grid`, `hybrid` | ✅ Completo |
| 4 | C Sequential (`C_OpenMP_MPI/scoring_sequential.c`) | `random` | ✅ Completo |
| 5 | CUDA C (`CUDA/scoring_kernel.cu`) | `random`, `grid`, `hybrid` + modo `precompute` | ✅ Completo |
| 6 | PyCUDA (`CUDA/scoring_pycuda.py`) | `random`, `grid`, `hybrid` + modo `precompute` | ✅ Completo |
| 7 | MPI (`C_OpenMP_MPI/scoring_mpi.c`) | ninguno | ❌ Scaffold, solo estructura |

### Componentes compartidos

#### Python

- `python/common.py`: AUC, consistencia, carga de datos.
- `python/logger.py`: logger ANSI.

#### C

- `C_OpenMP_MPI/shared/common.c`: AUC, consistencia, carga CSV.
- `C_OpenMP_MPI/shared/rng.c`: PCG64 + Dirichlet.
- `C_OpenMP_MPI/shared/ziggurat.c`
- `C_OpenMP_MPI/shared/logger.c`

#### Datos

- `data/npy/`: 3 archivos `.npy`.
- `data/csv/`: 6 archivos CSV.

#### Dataset

Dataset sintético generado con:

```text
data/scripts/generate_dataset.py
```

Características:

- 100 muestras.
- 50 sanas.
- 50 enfermas.
- 500 items.

### Pipeline completo

`run_all.sh` ejecuta todas las implementaciones secuencialmente y recolecta resultados.

El `Makefile` raíz tiene targets para cada implementación.

### Documentación

Hay 12 documentos en `docs/` cubriendo:

- Modelo matemático.
- Cada implementación.
- Benchmarks.
- Amdahl/Gustafson.
- Entregables.

### Último implementado

CUDA C y PyCUDA fueron completados en esta sesión:

- Kernels con AUC/consistencia en GPU.
- Reducción por shared memory.
- Tres estrategias de búsqueda.
- Modos `full` y `precompute`.
- Salida CSV.

Esto es un resumen de lo que se lleva.

---

## Usuario

Dame el plan para pasarle al agente.

---

## Asistente

Pégale esto al agente. Está pensado para que no improvise ni rompa lo que ya está funcionando.

---

# Plan para implementar MPI en `C_OpenMP_MPI/scoring_mpi.c`

Necesito implementar correctamente `C_OpenMP_MPI/scoring_mpi.c` en el proyecto **Metagenomic Scoring Systems HPC**.

## Estado actual del repo

- `python/sequential.py`: completo con estrategias `random`, `grid`, `hybrid`.
- `python/multicore.py`: completo con estrategias `random`, `grid`, `hybrid`.
- `C_OpenMP_MPI/scoring_openmp.c`: completo con estrategias `random`, `grid`, `hybrid`.
- `C_OpenMP_MPI/scoring_sequential.c`: completo para `random`.
- `CUDA/scoring_kernel.cu`: completo con `random`, `grid`, `hybrid` y modo `precompute`.
- `CUDA/scoring_pycuda.py`: completo con `random`, `grid`, `hybrid` y modo `precompute`.
- `C_OpenMP_MPI/scoring_mpi.c`: scaffold incompleto. Este es el objetivo principal.

## Datos reales del proyecto actual

- `A`: matriz de abundancias `100 x 500`.
- `profiles_TSF`: matriz de perfiles `500 x 3`.
- `labels`: vector binario de tamaño `100`, con clases sano/enfermo.

La matemática obligatoria es:

```text
P = profiles_TSF @ W
scores = A @ P
auc = AUC(labels, scores)
```

Se debe maximizar `auc` buscando:

```text
W = [w1, w2, w3]
```

con las restricciones:

```text
w1 + w2 + w3 = 1
wi >= 0
```

## Objetivo

Completar `C_OpenMP_MPI/scoring_mpi.c` para soportar las estrategias:

- `random`
- `grid`
- `hybrid`

La implementación MPI debe ser comparable con Python, multicore, OpenMP, C sequential, CUDA C y PyCUDA.

No debe cambiar la matemática ni generar resultados incompatibles.

---

# Reglas técnicas

1. No duplicar lógica si ya existe en `shared/`.
2. Reutilizar:

```text
C_OpenMP_MPI/shared/common.c
C_OpenMP_MPI/shared/common.h
C_OpenMP_MPI/shared/rng.c
C_OpenMP_MPI/shared/rng.h
C_OpenMP_MPI/shared/ziggurat.c
C_OpenMP_MPI/shared/logger.c
C_OpenMP_MPI/shared/logger.h
```

3. Si falta una función común para evaluar un candidato, crearla en `shared/common.c`, no solo dentro de MPI.
4. MPI debe paralelizar por candidatos `W`, no por filas de `A`, no por columnas, no por muestras.
5. Cada rank debe evaluar candidatos completos contra todo `A`, todo `profiles_TSF` y todo `labels`.

Dividir por muestras es una mala partición para este problema, porque el AUC necesita comparar scores globales de sanos contra enfermos.

---

# Arquitectura MPI esperada

1. `MPI_Init`.
2. Obtener:
   - `world_rank`
   - `world_size`
3. `rank 0` carga los datos desde `data/csv/` usando las funciones compartidas existentes.
4. `rank 0` transmite dimensiones:
   - `n_samples`
   - `n_items`
   - cualquier otro metadato necesario.
5. Todos los ranks reservan memoria para:
   - `A`
   - `profiles_TSF`
   - `labels`
6. Usar `MPI_Bcast` para enviar desde `rank 0` a todos:
   - `A`
   - `profiles_TSF`
   - `labels`
7. Generar candidatos `W` según estrategia.
8. Repartir candidatos entre ranks.
9. Cada rank evalúa su bloque local.
10. Reducir o reunir los mejores locales.
11. `rank 0` decide el mejor global.
12. Imprimir salida compatible con `run_all.sh` y `results/benchmark.csv`.
13. `MPI_Finalize`.

---

# Estructura de resultado

Crear una estructura de resultado como mínimo:

```c
typedef struct {
    double auc;
    double consistency;
    long candidate_idx;
    float w1;
    float w2;
    float w3;
} BestResult;
```

## Criterio de desempate

1. Mayor `auc` gana.
2. Si hay empate de `auc`, usar mayor `consistency`.
3. Si persiste empate, usar menor `candidate_idx` para reproducibilidad.

---

# Funciones internas recomendadas en `scoring_mpi.c`

Implementar estas funciones:

```text
parse_args
broadcast_problem
generate_candidates_random
generate_candidates_grid
generate_candidates_hybrid
scatter_candidates
evaluate_local_candidates
gather_best_results
select_global_best
print_csv_result
```

---

# Estrategia `random`

1. `rank 0` genera exactamente `K` candidatos `W`.
2. Usar el mismo RNG/Dirichlet que usa C/OpenMP para que sea comparable.
3. No permitir que cada rank genere su propia secuencia aleatoria independiente, porque eso rompe la validación contra secuencial.
4. Repartir candidatos con `MPI_Scatterv`, preferiblemente.
5. Cada rank evalúa su bloque.
6. Reunir mejores locales con `MPI_Gather`.
7. `rank 0` selecciona el mejor global.

---

# Estrategia `grid`

1. `rank 0` genera todos los puntos del simplex igual que OpenMP/Python.
2. Patrón recomendado:

```c
for (int i = 0; i <= grid_steps; i++) {
    for (int j = 0; j <= grid_steps - i; j++) {
        w1 = (float)i / grid_steps;
        w2 = (float)j / grid_steps;
        w3 = 1.0f - w1 - w2;
    }
}
```

3. El número real de candidatos grid no necesariamente es igual a `K`. Guardar y reportar el `actual_k`.
4. Repartir candidatos con `MPI_Scatterv`.
5. Evaluar localmente.
6. Reunir mejores locales.
7. `rank 0` selecciona mejor global.

---

# Estrategia `hybrid`

Debe tener dos fases.

## Fase 1: búsqueda global

1. Generar candidatos random.
2. Distribuir candidatos.
3. Evaluar en paralelo.
4. Obtener `best_random`.

## Fase 2: refinamiento local alrededor de `best_random`

1. `rank 0` genera candidatos alrededor de `best_random`.
2. Respetar simplex:
   - `wi >= 0`
   - `w1 + w2 + w3 = 1`
3. Distribuir candidatos de refinamiento.
4. Evaluar en paralelo.
5. Obtener `best_refined`.
6. Resultado final: mejor entre `best_random` y `best_refined`.

No confundir esta estrategia `hybrid` con MPI+OpenMP. Aquí `hybrid` es estrategia de búsqueda, no paralelismo híbrido.

---

# Distribución de candidatos

Usar `MPI_Scatterv`, porque `K` no siempre es divisible por `world_size`.

Calcular:

```c
base = K / world_size;
rem = K % world_size;

local_count = base + (rank < rem ? 1 : 0);
```

En `rank 0`, construir:

```c
sendcounts[rank] = local_count_rank * 3;
displs[rank] = offset_en_floats;
```

Cada candidato tiene tres `float`:

```text
w1, w2, w3
```

---

# Reducción del mejor resultado

No usar solamente:

```c
MPI_Reduce(..., MPI_MAX, ...)
```

sobre el AUC.

Eso está incompleto porque se pierde:

- `W`
- consistencia
- índice del candidato

Usar opción simple y robusta:

1. Cada rank produce un `BestResult local_best`.
2. Usar `MPI_Gather` para reunir todos los `BestResult` en `rank 0`.
3. `rank 0` recorre los resultados y selecciona el mejor con el criterio:
   - mayor AUC
   - mayor consistencia si hay empate
   - menor índice si persiste empate.

Si `MPI_Gather` con struct da problemas, usar un arreglo plano de `double`:

```c
double local_pack[6] = {
    auc,
    consistency,
    (double)candidate_idx,
    w1,
    w2,
    w3
};
```

Luego:

```c
MPI_Gather(
    local_pack,
    6,
    MPI_DOUBLE,
    all_packs,
    6,
    MPI_DOUBLE,
    0,
    MPI_COMM_WORLD
);
```

---

# Timing

Medir solo búsqueda/evaluación, no carga de datos.

Patrón:

```c
MPI_Barrier(MPI_COMM_WORLD);
double t0 = MPI_Wtime();

/* generar candidatos si se considera parte de la búsqueda */
/* scatter */
/* evaluación local */
/* gather/reducción */

double t1 = MPI_Wtime();
double local_elapsed = t1 - t0;
double global_elapsed = 0.0;

MPI_Reduce(
    &local_elapsed,
    &global_elapsed,
    1,
    MPI_DOUBLE,
    MPI_MAX,
    0,
    MPI_COMM_WORLD
);
```

Usar `MPI_MAX` para tiempo, porque el tiempo real del programa MPI lo determina el rank más lento.

---

# CLI requerida

`scoring_mpi` debe aceptar argumentos equivalentes a las otras implementaciones.

## Random

```bash
mpirun -np 4 C_OpenMP_MPI/scoring_mpi \
  --strategy random \
  --k 100000 \
  --seed 42 \
  --data-dir data/csv
```

## Grid

```bash
mpirun -np 4 C_OpenMP_MPI/scoring_mpi \
  --strategy grid \
  --grid-steps 100 \
  --data-dir data/csv
```

## Hybrid

```bash
mpirun -np 4 C_OpenMP_MPI/scoring_mpi \
  --strategy hybrid \
  --k 100000 \
  --seed 42 \
  --refine-steps 50 \
  --refine-radius 0.05 \
  --data-dir data/csv
```

## Explicación de los comandos

- `mpirun -np 4`: lanza 4 procesos MPI.
- `C_OpenMP_MPI/scoring_mpi`: ejecutable MPI.
- `--strategy`: selecciona la estrategia de búsqueda.
- `--k`: define el número de candidatos para `random` o `hybrid`.
- `--seed`: fija reproducibilidad.
- `--grid-steps`: define la resolución del simplex en `grid`.
- `--refine-steps` y `--refine-radius`: controlan la fase de refinamiento de `hybrid`.
- `--data-dir data/csv`: indica desde dónde cargar los CSV.

---

# Makefile

Actualizar `C_OpenMP_MPI/Makefile` para incluir target `mpi`.

Debe compilar con `mpicc`, enlazando los archivos compartidos:

```make
mpi:
	mpicc -O3 -Wall -Wextra -std=c11 \
		-o scoring_mpi \
		scoring_mpi.c \
		shared/common.c \
		shared/rng.c \
		shared/ziggurat.c \
		shared/logger.c \
		-lm
```

Desde la raíz del repo, el comando de compilación debe ser:

```bash
make -C C_OpenMP_MPI mpi
```

## Explicación

- `make`: ejecuta reglas del Makefile.
- `-C C_OpenMP_MPI`: entra a esa carpeta antes de ejecutar.
- `mpi`: selecciona el target que compila `scoring_mpi`.

---

# Integración con `run_all.sh`

Agregar ejecución MPI para:

- `np=1`
- `np=2`
- `np=4`
- si el equipo lo permite, `np=8`

Ejemplo:

```bash
mpirun -np 1 C_OpenMP_MPI/scoring_mpi \
  --strategy random \
  --k "$K" \
  --seed "$SEED" \
  --data-dir data/csv

mpirun -np 2 C_OpenMP_MPI/scoring_mpi \
  --strategy random \
  --k "$K" \
  --seed "$SEED" \
  --data-dir data/csv

mpirun -np 4 C_OpenMP_MPI/scoring_mpi \
  --strategy random \
  --k "$K" \
  --seed "$SEED" \
  --data-dir data/csv
```

---

# Salida CSV esperada

Debe ser compatible con las demás implementaciones.

Formato recomendado:

```csv
implementation,strategy,n_samples,n_items,k,workers,time_sec,best_auc,consistency,w1,w2,w3,seed
c_mpi,random,100,500,100000,4,0.123456,0.987654,0.940000,0.200000,0.500000,0.300000,42
```

---

# Validación obligatoria

Antes de hacer benchmarks grandes, correr:

```bash
make -C C_OpenMP_MPI mpi
```

Esto compila la versión MPI.

Luego:

```bash
mpirun -np 1 C_OpenMP_MPI/scoring_mpi \
  --strategy random \
  --k 10000 \
  --seed 42 \
  --data-dir data/csv
```

Esto prueba MPI con un solo proceso. Debe coincidir con C sequential/OpenMP para la misma seed y `K`.

Luego:

```bash
mpirun -np 2 C_OpenMP_MPI/scoring_mpi \
  --strategy random \
  --k 10000 \
  --seed 42 \
  --data-dir data/csv
```

Esto prueba partición entre dos procesos.

Luego:

```bash
mpirun -np 4 C_OpenMP_MPI/scoring_mpi \
  --strategy random \
  --k 10000 \
  --seed 42 \
  --data-dir data/csv
```

Esto prueba partición entre cuatro procesos.

---

# Criterios de aceptación

1. Compila sin errores.
2. No hay warnings graves de tipos, memoria o MPI.
3. `random` funciona con `np=1,2,4`.
4. `grid` funciona con `np=1,2,4`.
5. `hybrid` funciona con `np=1,2,4`.
6. `np=1` coincide con C sequential/OpenMP para `random`.
7. `np=2` y `np=4` deben producir el mismo mejor AUC y el mismo `W` que `np=1`, salvo diferencias flotantes mínimas.
8. Si `np=4` da otro mejor resultado, no asumir que es mejora por paralelismo: probablemente es bug de RNG, partición, scatter/gather o criterio de reducción.
9. `run_all.sh` debe incluir MPI y escribir filas en `results/benchmark.csv`.
10. La documentación debe explicar:
    - Por qué se paraleliza por candidatos `W`.
    - Por qué cada rank necesita copia completa de `A`, `profiles_TSF` y `labels`.
    - Uso de `MPI_Bcast`.
    - Uso de `MPI_Scatterv` o `MPI_Scatter`.
    - Uso de `MPI_Gather` para recolectar mejores locales.
    - Medición con `MPI_Wtime`.
    - Speedup y eficiencia usando `workers = world_size`.

---

# Puntos que NO se deben hacer

1. No partir `A` por filas.
2. No hacer que cada rank genere candidatos random diferentes con seeds independientes.
3. No usar solo `MPI_Reduce(MPI_MAX)` sobre AUC perdiendo el `W`.
4. No cambiar el cálculo de AUC respecto a `shared/common.c`.
5. No crear una salida distinta a la del resto del pipeline.
6. No tocar CUDA/PyCUDA/OpenMP si no es necesario.
7. No implementar MPI+OpenMP todavía; eso sería otra variante, no el objetivo actual.

---

# Entregable final

- `C_OpenMP_MPI/scoring_mpi.c` completo.
- `C_OpenMP_MPI/Makefile` actualizado.
- `run_all.sh` actualizado con MPI.
- Documentación mínima en `docs/` sobre la implementación MPI.
- Validación con `np=1,2,4`.
