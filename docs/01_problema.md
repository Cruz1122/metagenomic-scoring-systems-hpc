# Planteamiento del problema

## Contexto biologico

El analisis de datos metagenomicos permite caracterizar comunidades microbianas a partir de muestras biologicas. Un problema frecuente es determinar si existe un patron de abundancia diferencial entre dos poblaciones: por ejemplo, individuos sanos versus individuos con cancer colorrectal (CRC).

Cada muestra se describe mediante un vector de abundancias relativas de items (especies, genes, taxones). El desafio no es solo clasificar, sino entender que combinacion de perfiles biologicos (taxonomicos, ecologicos, funcionales) maximiza la separacion entre grupos.

## Formulacion del problema de scoring

Se dispone de:

- Una matriz de contribucion `A` de dimensiones `n_muestras x N`, donde cada entrada `A_ji` representa la contribucion del item i en la muestra j.
- Un vector de etiquetas `y` de longitud `n_muestras`, donde `y_j = 0` para la poblacion sana y `y_j = 1` para la poblacion enferma.
- Una matriz de perfiles `profiles` de dimensiones `N x 3`, donde la fila i contiene los perfiles `T_i`, `S_i` y `F_i` del item i.

El problema consiste en encontrar un vector de pesos `W = (W1, W2, W3)` que combine los tres perfiles para maximizar la capacidad discriminante del scoring.

## Relevancia computacional

La funcion objetivo debe evaluarse para cada candidato W propuesto. Cada evaluacion requiere:

1. Calcular `P = profiles @ W`: producto matriz-vector de dimension `N x 3` por `3 x 1`, con costo O(N).
2. Calcular `scores = A @ P`: producto matriz-vector de dimension `n_muestras x N` por `N x 1`, con costo O(n_muestras * N).
3. Calcular AUC de `scores` contra `y`: requiere ordenar los scores, con costo O(n_muestras * log n_muestras).

Para un dataset de 2000 muestras y 10000 items, cada evaluacion implica ~20 millones de operaciones de punto flotante solo en el producto matriz-vector. Con K=100000 candidatos, el costo total es ~2x10^12 operaciones. Esto justifica el uso de paralelizacion en multiples niveles.

## Restricciones del espacio de busqueda

Los pesos W deben cumplir:

```
W1 + W2 + W3 = 1
Wi >= 0 para i = 1, 2, 3
```

Este es un simplex de dimension 2 embebido en R^3. La restriccion de suma unitaria refleja que los pesos representan una contribucion relativa de cada perfil al score total.

## Por que paralelizar

La evaluacion de cada candidato W es independiente de los demas. Esto hace que el problema sea **embarazosamente paralelo**: no hay dependencia de datos entre iteraciones, no se requiere comunicacion durante la evaluacion, y la sincronizacion solo es necesaria para identificar el mejor resultado global. Esta estructura es ideal para paralelizacion tanto en CPU (memoria compartida con OpenMP, memoria distribuida con MPI, procesos con multiprocessing) como en GPU (SIMT con CUDA).
