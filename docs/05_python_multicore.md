# Python multicore (multiprocessing)

## Rol en la arquitectura

Esta implementacion introduce paralelismo a nivel de CPU utilizando el modulo `multiprocessing` de Python. Representa el primer paso de paralelizacion sobre el baseline secuencial.

## Archivo

`python/multicore.py`

## Paradigma: multiprocessing con memoria independiente

A diferencia de threading (donde los hilos comparten memoria y estan limitados por el GIL en CPython), `multiprocessing` crea procesos separados, cada uno con su propio espacio de direcciones. Esto permite:

- Ejecucion genuinamente paralela en multiples nucleos (sin GIL).
- Aislamiento de memoria: no hay condiciones de carrera sobre las variables de los workers.
- Facilidad de programacion: los workers son funciones puras que reciben datos y retornan resultados.

El costo es que cada proceso tiene su propia copia de los datos en memoria. Si el dataset es grande, el consumo de memoria se multiplica por el numero de workers.

## Algoritmo (modo random)

```
1. Cargar A, y, profiles (proceso principal)
2. Generar K vectores W: W = rng.dirichlet((1,1,1), size=K)
3. _parallel_eval():
   a. Construir tareas (start, end, weights_slice, worker_id)
   b. chunk_size = 32 si hay logging (LOG_INTERVAL), else ceil(K/workers)
   c. pool.imap_unordered(partial(_eval_chunk, A=..., y=..., profiles=...), tasks)
4. Fusionar mejores locales (desempate: consistencia, luego indice)
5. Retornar mejor W encontrado
```

`_eval_chunk(task, A, y, profiles)` evalua su subconjunto de candidatos llamando a `evaluate()` y retorna `(best_auc, best_consistency, best_weights, best_iter)`.

## Flujo de datos

```
Proceso principal                    Workers (procesos separados)

Genera W (K x 3)
Construye tareas por chunks
                    ----- task 0 --> Worker: evalua, retorna mejor local
                    ----- task 1 --> Worker: evalua, retorna mejor local
                    ...
Recolecta resultados (imap_unordered)
Selecciona mejor global
```

Cada worker recibe una copia de `A`, `y`, `profiles` (por herencia de memoria al hacer fork en Linux, o serializacion en Windows). Los datos de entrada son de solo lectura, por lo que no hay riesgo de modificacion accidental.

## Sincronizacion

No se requiere sincronizacion explicita durante la ejecucion. Cada worker opera de forma independiente y solo al final se comparan los resultados locales. No se usan `Lock`, `Queue`, `Manager` ni `Value`. Esto es posible porque el problema es embarazosamente paralelo.

## RNG

En modo random, todos los candidatos W se generan en el proceso principal con `rng.dirichlet(..., size=k)`. Cada worker evalua su subconjunto pero no genera numeros aleatorios. Esto garantiza que la secuencia de candidatos sea identica a la del secuencial, lo que permite comparar directamente los resultados.

## Estrategias de busqueda

Al igual que el secuencial, esta implementacion soporta tres estrategias: random, grid y hybrid. La generacion de pesos se realiza siempre en el proceso principal; solo la evaluacion se distribuye entre los workers.

## Ventajas y limitaciones

**Ventajas:**
- Escala linealmente con el numero de nucleos para K suficientemente grande.
- Codigo simple y directo, sin primitivas de sincronizacion.
- Reutiliza `evaluate()` de `python/common.py`.

**Limitaciones:**
- El overhead de crear procesos y serializar datos no es despreciable para K pequenos.
- Cada worker tiene copia completa de las matrices: el consumo de RAM es `workers` veces el dataset.
- No escala a multiples maquinas (a diferencia de MPI).
- En Windows, la herencia de memoria no funciona como en Linux (fork), lo que puede requerir serializacion explicita.
