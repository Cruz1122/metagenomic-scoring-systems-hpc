# Estrategias de busqueda

Cada implementacion del sistema soporta tres estrategias para explorar el simplex de pesos W. La eleccion de estrategia afecta la calidad del AUC encontrado, el costo computacional y la reproducibilidad de los resultados.

## 1. Random search

### Descripcion

Genera K vectores W independientes mediante muestreo uniforme sobre el simplex con la distribucion Dirichlet.

### Algoritmo

```
Para iter = 1..K:
    W = Dirichlet(alpha=(1,1,1))
    scores = A @ (profiles @ W)
    auc = AUC(y, scores)
    si auc > mejor_auc: guardar W
```

### Propiedades estadisticas

- Cada candidato es independiente de los anteriores.
- La probabilidad de que un punto cualquiera del simplex sea muestreado es uniforme.
- Con K candidatos, la probabilidad de que el mejor punto encontrado este dentro de un factor epsilon del optimo global crece con K, pero no hay garantia determinista.
- La varianza del AUC obtenido depende de K (a mayor K, menor varianza).

### Costo computacional

O(K * (N * n_muestras + n_muestras * log n_muestras)). Lineal en K.

### Reproducibilidad

Depende de la semilla del RNG. Con la misma semilla y el mismo K, se obtienen los mismos candidatos en el mismo orden. Esto permite comparaciones justas entre implementaciones.

### Ventajas

- Extremadamente simple de implementar.
- Paralelizable sin restricciones (no hay dependencia entre candidatos).
- El costo total se controla directamente con K.
- No requiere conocimiento previo del espacio de busqueda.

### Desventajas

- No hay garantia de cubrir regiones especificas del simplex.
- Para K pequeno, la probabilidad de omitir el optimo global es alta.
- No hay un criterio de parada natural (salvo agotar K).

## 2. Grid search

### Descripcion

Define una malla regular sobre el simplex. Dada una resolucion R, se generan todos los puntos de la forma:

```
W1 = i / R
W2 = j / R
W3 = (R - i - j) / R
```

para i, j >= 0 con i + j <= R.

### Numero de puntos

El numero de puntos en el grid es:

```
K_grid = (R + 1) * (R + 2) / 2
```

Esto crece cuadraticamente con R:

| R | K_grid |
|---|---|
| 10 | 66 |
| 50 | 1326 |
| 100 | 5151 |
| 200 | 20301 |
| 500 | 125751 |

### Resolucion automatica

Cuando se solicita un K nominal, la resolucion se ajusta como:

```
R = max(1, int(sqrt(2 * K)))
```

Esto produce un K_grid aproximadamente igual a K, pero puede ser mayor o menor dependiendo de la discrecion del grid.

### Propiedades

- Cobertura determinista y uniforme del simplex.
- Los puntos del grid son siempre los mismos, independientemente de la semilla.
- Cada punto del simplex tiene un punto del grid a distancia Manhattan <= 1/R.

### Costo computacional

O(K_grid * (N * n_muestras + n_muestras * log n_muestras)). Similar a random search para el mismo numero de puntos, pero el numero real de puntos puede diferir del K solicitado.

### Reproducibilidad

Total. Sin intervencion del azar, el resultado es deterministico para una resolucion R dada.

### Ventajas

- Cobertura uniforme y predecible.
- Resultados deterministicos.
- No requiere RNG.

### Desventajas

- La granularidad puede omitir el optimo si este se encuentra entre puntos del grid.
- K_grid no es continuo: pequenos cambios en R pueden duplicar o triplicar el numero de puntos.
- Para alta resolucion, K_grid puede ser mucho mayor que el K nominal solicitado.

## 3. Hybrid search

### Descripcion

Combina grid, random y refinamiento local en fases secuenciales. **La particion de fases depende de la implementacion** (ver tabla abajo).

### Variante PyCUDA

Usada en `CUDA/scoring_pycuda.py`:

**Fase 1 — Grid (20% de K, max 2000):**

```
K_grid = min(int(K * 0.2), 2000)
R = resolucion_para_K(K_grid)
grid_points = generar_grid(R)
```

**Fase 2 — Random (60% de K):**

```
K_random = int(K * 0.6)
random_points = Dirichlet((1,1,1), size=K_random)
```

**Fase 3 — Local (resto de K):**

```
alpha = max(W_mejor * 100, 1e-3)
local_points = Dirichlet(alpha, size=K_local)
```

Donde `K_local = K - K_grid - K_random`.

### Variante Python / C secuencial / C OpenMP

Usada en `python/sequential.py`, `python/multicore.py`, `scoring_sequential.c`, `scoring_openmp.c`:

1. **Grid fijo** con `--step 0.02` (~1326 puntos, independiente del presupuesto K).
2. Del resto (`K - grid_total`), **50% random** Dirichlet(1,1,1) y **50% local** con dos concentraciones:
   - `alpha = max(w * 300, 1e-3)`
   - `alpha = max(w * 1000, 1e-3)`

### Variante C MPI

Usada en `scoring_mpi.c`:

1. **Fase random:** K candidatos (todo el presupuesto en random).
2. **Fase refinamiento:** `--refine-steps` o 20% de K, semilla `seed + 9999`.
3. **Sin fase grid.** Entre fases: `gather_best()` para sincronizar el mejor W.

### Tabla resumen

| Implementacion | Grid | Random | Local |
|---|---|---|---|
| Python / C sec / C OpenMP | step=0.02 fijo | 50% del resto | 50% del resto (conc 300/1000) |
| C MPI | — | K candidatos | refine (20% K o --refine-steps) |
| PyCUDA | 20% K (max 2000) | 60% K | resto (alpha = w*100) |

### Flujo completo (variante PyCUDA)

```
1. Generar puntos del grid
2. Evaluar puntos del grid, encontrar mejor W_grid
3. Generar puntos aleatorios
4. Evaluar puntos aleatorios, encontrar mejor W_random
5. W_mejor = mejor(W_grid, W_random)
6. Si K_local > 0:
   a. Generar puntos locales alrededor de W_mejor
   b. Evaluar puntos locales
   c. Actualizar W_mejor si corresponde
7. Retornar W_mejor
```

### Propiedades

- Combina la cobertura determinista del grid con la exploracion estocastica del random search.
- La fase de refinamiento permite mejorar la solucion sin aumentar significativamente K.
- Las fases 1 y 2 pueden ejecutarse en paralelo internamente (todos los candidatos de una fase son independientes).
- La fase 3 depende del resultado de las fases 1 y 2, por lo que es inherentemente secuencial.

### Costo computacional

O(K * (N * n_muestras + n_muestras * log n_muestras)). Similar a random search.

### Reproducibilidad

Depende de la semilla para las fases aleatorias (2 y 3). Las fases 1 (grid) y 2 (random con semilla conocida) son reproducibles. La fase 3 depende del resultado de las fases previas, pero con la misma semilla se obtiene el mismo resultado.

## Comparacion

| Aspecto | Random | Grid | Hybrid |
|---|---|---|---|
| Tipo de busqueda | Estocastica | Determinista | Mixta |
| Cobertura del simplex | Probabilistica (uniforme) | Uniforme (discreta) | Grid + aleatoria + local |
| Dependencia entre candidatos | Ninguna | Ninguna | Fases 1-2 independientes, fase 3 depende de 1-2 |
| Reproducibilidad | Por semilla | Total | Por semilla |
| Riesgo de omitir optimo | Alto para K bajo | Depende de resolucion | Bajo (fase local reduce riesgo) |
| Calidad tipica de AUC | Buena para K grande | Buena para R suficiente | Generalmente la mejor |
| Complejidad de implementacion | Baja | Baja | Media |
| Uso recomendado | Benchmarks, K grande | Exploracion inicial, K conocido | Produccion, maxima calidad |
