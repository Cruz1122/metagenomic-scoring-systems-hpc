# Python secuencial

## Rol en la arquitectura

La implementacion Python secuencial es el **baseline** del proyecto. Es la primera implementacion que se escribe, la mas portatil (no requiere compilacion) y contra la que se miden todas las demas. Su proposito no es la velocidad, sino la correccion y la reproducibilidad.

## Archivo

`python/sequential.py`

## Algoritmo (modo random)

```
1. Cargar A, y, profiles desde archivos .npy
2. Para i = 1..K:
   a. W = rng.dirichlet((1,1,1))     # un candidato por iteracion
   b. P = profiles @ W               (N x 3) @ (3,)
   c. scores = A @ P                 (n_muestras x N) @ (N,)
   d. AUC = roc_auc_score(y, scores)
   e. consistency = max_bal_accuracy(y, scores)
   f. Si AUC > mejor_auc: guardar W
3. Retornar mejor W encontrado
```

Tambien soporta `grid` (malla con `--step`) e `hybrid` (grid + random + local; ver [10_estrategias_busqueda.md](10_estrategias_busqueda.md)).

## Evaluacion por candidato

Cada candidato se evalua de forma independiente mediante `evaluate()` en `python/common.py`. No hay batching: un solo `W` por iteracion. Las operaciones matriciales (`profiles @ w`, `A @ P`) aprovechan BLAS/NumPy para ese candidato, pero no se vectorizan multiples W a la vez.

## Funcion objetivo

```python
def evaluate(A, y, profiles, w):
    P = profiles @ w
    scores = A @ P
    auc_val = auc_vector(scores, y)
    cons_val = consistency(scores, y)
    return auc_val, cons_val
```

Donde `auc_vector` utiliza `sklearn.metrics.roc_auc_score` y `consistency` implementa el barrido de umbrales para encontrar el balanced accuracy maximo.

## RNG

Se utiliza `numpy.random.default_rng(seed)` como generador. Este usa el algoritmo PCG64 por defecto, que ofrece buena calidad estadistica y periodo largo. La semilla se pasa como argumento (default 42) para garantizar reproducibilidad.

## Formato de salida

En modo normal, imprime informacion formateada con el logger ANSI. En modo benchmark (flag `--benchmark`), imprime una linea CSV con el formato estandar del proyecto.

## Limitaciones

- No aprovecha multiples nucleos del procesador.
- El overhead de Python (maquina virtual, interpretacion, type checking) es significativo respecto a C nativo.
- La memoria utilizada escala con el tamano del dataset (A, profiles, y).
