# C OpenMP

## Rol en la arquitectura

OpenMP (Open Multi-Processing) es el primer nivel de paralelizacion en CPU que utiliza **memoria compartida**. Todos los hilos acceden a las mismas matrices en memoria y colaboran para evaluar los K candidatos. Es la opcion natural para aprovechar los multiples nucleos de un procesador moderno en una sola maquina.

## Archivo

`C_OpenMP_MPI/scoring_openmp.c`

## Paradigma: memoria compartida con directivas de compilacion

OpenMP extiende C (y otros lenguajes) con **directivas de compilacion** (pragma) que indican al compilador que regiones del codigo deben ejecutarse en paralelo. El modelo de ejecucion es **fork/join**:

1. El programa comienza con un solo hilo (master thread).
2. Al encontrar una region paralela (`#pragma omp parallel`), el hilo master crea un equipo de hilos (fork).
3. Cada hilo ejecuta el codigo de la region paralela de forma independiente.
4. Al finalizar la region, los hilos se sincronizan y se unen al master (join).
5. La ejecucion continua secuencialmente.

Este modelo es especialmente adecuado para bucles cuyas iteraciones son independientes (paralelismo de datos).

## Algoritmo

```
1. Cargar datos (un solo hilo)
2. #pragma omp parallel
   {
       tid = omp_get_thread_num()
       pcg64_seed(pcg, seed + tid)
       local_best = inicializar

       #pragma omp for schedule(static)
       for (i = 0; i < K; i++)
           - Generar W_i con RNG propio (random) o leer del grid
           - evaluate(): P = profiles @ W_i, scores = A @ P, AUC, consistencia
           - Actualizar local_best si AUC > local_best.auc

       threads[tid].best = local_best
   }
3. Merge serial (fuera del parallel):
   for (t = 0; t < n_threads; t++)
       if (threads[t].best.auc > global_best.auc)
           global_best = threads[t].best
4. Imprimir resultado
```

`evaluate()` y `matvec()` en `shared/common.c` corren secuencialmente dentro de cada hilo; no hay `reduction` ni paralelismo en la multiplicacion matriz-vector.

## Division del trabajo

Se utiliza `schedule(static)` para la distribucion de iteraciones: el compilador divide el rango `[0, K)` en bloques contiguos de aproximadamente `K / num_threads` iteraciones y asigna un bloque a cada hilo. Esto minimiza el overhead de scheduling porque no hay dependencia entre iteraciones.

## RNG por hilo

Cada hilo tiene su propia instancia del generador PCG64, inicializada con semilla `seed + thread_id`. Esto es fundamental por dos razones:

1. **No hay contencion:** cada hilo genera numeros aleatorios sin necesidad de mutex ni exclusion mutua.
2. **Reproducibilidad:** con la misma semilla base y el mismo numero de hilos, los resultados son identicos entre ejecuciones.

## Sincronizacion

El merge del mejor global no usa `critical` ni `reduction`. Cada hilo escribe solo en su slot `threads[tid].best` (sin condicion de carrera), y fuera de la region paralela el hilo master itera el array para encontrar el mejor.

En modo `--verbose`, `#pragma omp critical(log_improve)` serializa unicamente el logging de mejoras en vivo; no afecta al merge del resultado.

## Estrategias de busqueda

OpenMP implementa las tres estrategias (random, grid, hybrid). En hybrid, la variante coincide con Python/C secuencial: grid `step=0.02`, luego 50/50 random/local (concentraciones 300 y 1000). Ver [10_estrategias_busqueda.md](10_estrategias_busqueda.md).

## Rendimiento esperado

Para K suficientemente grande (K >= 10000), el speedup debe ser aproximadamente lineal con el numero de hilos, porque:

- La fraccion paralela del algoritmo es alta (>99% del tiempo se gasta en evaluar candidatos).
- No hay dependencias entre iteraciones.
- El merge post-loop no introduce contencion entre hilos.

Para K pequeno, el overhead de crear el equipo de hilos y el desbalanceo de carga pueden reducir la eficiencia.

## Compilacion y ejecucion

```bash
# Compilar (el flag -fopenmp activa OpenMP)
gcc -O3 -std=c11 -Wall -Wextra -fopenmp scoring_openmp.c shared/*.c -o scoring_openmp -lm

# Ejecutar con 4 hilos
export OMP_NUM_THREADS=4
./C_OpenMP_MPI/scoring_openmp --k 100000 --seed 42 --data-dir data
```

La variable de entorno `OMP_NUM_THREADS` controla el numero de hilos en runtime sin recompilar.

## Consideraciones tecnicas

- **False sharing:** las variables privadas de cada hilo deben estar alineadas para evitar que caigan en la misma linea de cache.
- **Localidad:** el acceso a las matrices A y profiles es de solo lectura y compartido, por lo que la cache L2/L3 se aprovecha eficientemente.
- **Overhead de creacion:** la primera region paralela tiene mayor overhead porque el equipo de hilos se crea "lazy". En este caso solo hay una region paralela, por lo que el impacto es despreciable.
