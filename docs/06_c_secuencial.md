# C secuencial

## Rol en la arquitectura

La implementacion C secuencial es el **baseline nativo**. Al eliminar la maquina virtual de Python y la interpretacion dinamica, muestra el rendimiento teorico maximo del algoritmo en un solo nucleo. Sirve para:

- Cuantificar el overhead de Python (tipicamente un factor de 10x a 50x).
- Proporcionar una base de comparacion justa para las implementaciones paralelas en C (OpenMP, MPI).
- Verificar que los resultados numericos coinciden con Python (prueba de correccion).

## Archivo

`C_OpenMP_MPI/scoring_sequential.c`

## Algoritmo

Identico al de Python secuencial, pero implementado en C:

```
1. load_data(data_dir, &ds): carga A, y, profiles desde archivos CSV
2. Generar K vectores W en el simplex via RNG propio (un candidato por iteracion en random)
3. Para cada W:
   a. evaluate(): P = profiles @ W, scores = A @ P
   b. AUC = auc(scores, y, n_samples)
   c. consistency = consistency(scores, y, n_samples)
   d. Actualizar mejor global si AUC > mejor_auc
4. Imprimir resultado
```

Soporta `random`, `grid` e `hybrid` (hybrid: grid `step=0.02`, luego 50/50 random/local con concentraciones 300 y 1000).

## RNG: PCG64

El generador de numeros aleatorios es PCG64 (Permuted Congruential Generator), implementado en `shared/rng.c`. Se eligio PCG64 por:

- Buena calidad estadistica (pasa pruebas BigCrush de TestU01).
- Periodo largo (2^128).
- Estado pequeno (128 bits + 128 bits de secuencia).
- Rapido: ~2-3 cycles/byte en hardware moderno.
- Facil de inicializar con semillas diferentes para cada hilo/rank.

El muestreo en el simplex se realiza generando tres valores gamma con el algoritmo de Marsaglia-Tsang y normalizando:

```
gamma_i = sample_gamma(1.0, rng)   para i = 1, 2, 3
W_i = gamma_i / (gamma_1 + gamma_2 + gamma_3)
```

Esto produce una distribucion Dirichlet(1,1,1) uniforme en el simplex.

## AUC en C

La funcion `auc()` en `shared/common.c` implementa el estadistico U de Mann-Whitney:

1. Ordenar scores de menor a mayor con `qsort()`.
2. Asignar rangos promedio a scores empatados.
3. Sumar rangos de las muestras positivas.
4. Calcular U = sum_ranks_pos - n_pos*(n_pos+1)/2.
5. AUC = U / (n_pos * n_neg).

Este calculo es identico al de `sklearn.metrics.roc_auc_score` para el caso binario.

## Consistencia en C

La funcion `consistency()` en `shared/common.c`:

1. Ordenar scores.
2. Inicializar TP = n_pos, TN = 0.
3. Recorrer scores ordenados moviendo el umbral:
   - En cada paso, calcular balanced accuracy.
   - Si el sample actual es positivo: TP--; si es negativo: TN++.
4. Retornar el maximo balanced accuracy encontrado.

## Estructura de datos

Para maximizar la localidad de cache, las matrices se almacenan como arreglos planos en orden row-major:

```c
A[s * n_items + i]       // elemento (s, i)
profiles[i * 3 + p]      // perfil p del item i
labels[s]                // etiqueta de la muestra s
```

## Compilacion

```bash
gcc -O3 -std=c11 -Wall -Wextra scoring_sequential.c shared/common.c shared/rng.c shared/ziggurat.c shared/logger.c -o scoring_sequential -lm
```

La flag `-O3` activa optimizaciones agresivas del compilador, incluyendo auto-vectorizacion, inlining y reordenamiento de instrucciones.

## Diferencia con Python secuencial

| Aspecto | Python | C |
|---|---|---|
| Ejecucion | Interpretada (CPython VM) | Compilada (codigo maquina) |
| Tipos | Dinamicos, check en runtime | Estaticos, check en compile-time |
| RNG | PCG64 (NumPy default) | PCG64 (implementacion propia) |
| AUC | sklearn (Cython optimizado) | Implementacion propia en C |
| Memoria | Arreglos NumPy con overhead | Arreglos planos sin overhead |
| Overhead tipico | 10x - 50x respecto a C | 1x (referencia) |
