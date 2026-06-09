# OpenMP

OpenMP se usa para memoria compartida en CPU. Todos los hilos leen `A`, `profiles` e `y`, y cada hilo evalúa candidatos `W` independientes.

Archivo:

```text
C_OpenMP_MPI/scoring_openmp.c
```

Compilar y ejecutar:

```bash
make -C C_OpenMP_MPI scoring_openmp
./C_OpenMP_MPI/scoring_openmp --k 100000 --threads 4 --seed 42 --data-dir data
```

## Estrategia de RNG

Cada hilo usa un generador Xorshift independiente inicializado con `seed + 0x9e3779b97f4a7c15ULL * (thread_id + 1)`. Los candidatos se generan mediante tres exponenciales normalizadas para obtener vectores en el simplex.

## Lectura de datos

A diferencia de Python (que lee `.npy` binario), el código C lee archivos CSV (`matrix_A.csv`, `profiles_TSF.npy` se leería como `item_profiles.csv`) con `fscanf`. Esto es más lento pero portable y evita dependencias binarias.

## Patrón usado

```c
#pragma omp parallel
{
    // RNG local + Best local
    #pragma omp for schedule(static)
    for (long k = 0; k < K; k++) { ... }
    #pragma omp critical
    if (local.auc > global.auc) global = local;
}
```

- `#pragma omp parallel` crea el equipo de hilos.
- `#pragma omp for schedule(static)` distribuye iteraciones estáticamente.
- La sección crítica solo consolida el mejor local de cada hilo. Meter locks por candidato sería una mala decisión.
- No se usa `reduction` porque la estructura `Best` no es un tipo nativo de C.

## Métricas

El tiempo se mide con `omp_get_wtime()`. La salida incluye `c_openmp` como nombre de implementación, `parallel_units` = número de hilos usado.
