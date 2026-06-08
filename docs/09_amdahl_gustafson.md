# Amdahl y Gustafson

Amdahl aplica a escalabilidad fuerte: `N` y `K` fijos, más recursos.

```text
S = T1 / TP
E = S / P
```

Donde `T1` es el tiempo del baseline **python_sequential** con 1 unidad de paralelismo. `TP` es el tiempo de la implementación con `P` unidades (hilos, procesos o ranks).

El `benchmark.csv` ya incluye `speedup` y `efficiency` calculados por `scripts/postprocess_benchmark.py`.

Modelo de Amdahl:

```text
Smax = 1 / ((1 - f_parallel) + f_parallel/P)
```

La fracción paralela `f_parallel` se estima empíricamente. Para extraerla del benchmark:

```text
f_parallel = (1 - 1/S_max) / (1 - 1/P_max)
```

Donde `S_max` es el speedup observado con `P_max` unidades.

Gustafson aplica a escalabilidad débil: al aumentar recursos también crece `K` o `N`. Se puede probar escalabilidad débil variando `K` proporcionalmente a `P` y observando si el tiempo se mantiene constante.

Usa Amdahl para responder "¿cuánto baja el tiempo con el mismo problema?". Usa Gustafson para responder "¿cuánto problema adicional puedo procesar con más recursos?".
