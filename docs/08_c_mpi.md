# C MPI

## Rol en la arquitectura

MPI (Message Passing Interface) es el nivel de paralelizacion en CPU que utiliza **memoria distribuida**. A diferencia de OpenMP (donde todos los hilos comparten la misma memoria fisica), en MPI cada proceso (rank) tiene su propio espacio de direcciones y se comunica con los demas mediante llamadas explicitas a funciones de paso de mensajes.

MPI es necesario cuando el problema no cabe en la memoria de una sola maquina o cuando se dispone de un cluster de multiples nodos.

## Archivo

`C_OpenMP_MPI/scoring_mpi.c`

## Paradigma: memoria distribuida con paso de mensajes

En el modelo de memoria distribuida:

- Cada proceso tiene su propia copia de los datos en memoria local.
- No hay variables globales compartidas entre procesos.
- La comunicacion ocurre solo cuando un proceso envia o recibe datos explicitamente mediante llamadas MPI.
- El paralelismo es MIMD (Multiple Instruction, Multiple Data): cada proceso puede ejecutar instrucciones diferentes sobre datos diferentes.

Para este problema, que es embarazosamente paralelo, MPI se usa con un patron SPMD (Single Program, Multiple Data): todos los procesos ejecutan el mismo codigo pero sobre diferentes subconjuntos de candidatos.

## Algoritmo (modo random)

```
Rank 0:
1. Cargar A, y, profiles desde disco (CSV)
2. broadcast_problem(): MPI_Bcast de dimensiones y matrices a todos los ranks
3. Generar K candidatos W con PCG64 (semilla seed, sin offset de rank)
4. MPI_Scatterv: distribuir bloques de candidatos a cada rank

Todos los ranks:
5. evaluate_local(): evaluar bloque recibido, mantener mejor local
6. gather_best(): MPI_Gather de mejores locales -> merge en rank 0 -> MPI_Bcast

Rank 0:
7. Imprimir resultado
```

Los bloques tienen tamano casi uniforme (diferencia de a lo sumo 1 candidato entre ranks).

## Comunicacion

### MPI_Bcast (broadcast)

El rank 0 transmite las matrices A, profiles, y y las dimensiones a todos los ranks:

```c
MPI_Bcast(&n_samples, 1, MPI_INT, 0, MPI_COMM_WORLD);
MPI_Bcast(&n_items, 1, MPI_INT, 0, MPI_COMM_WORLD);
MPI_Bcast(A, n_samples * n_items, MPI_DOUBLE, 0, MPI_COMM_WORLD);
MPI_Bcast(profiles, n_items * 3, MPI_DOUBLE, 0, MPI_COMM_WORLD);
MPI_Bcast(y, n_samples, MPI_INT, 0, MPI_COMM_WORLD);
```

Todos los procesos, incluido el rank 0, participan en el broadcast. Al finalizar, cada rank tiene copia completa de los datos.

### MPI_Scatterv (distribucion de candidatos)

El rank 0 genera todos los vectores W y los reparte con `MPI_Scatterv`. Cada rank recibe su bloque en memoria local y lo evalua sin generar pesos propios.

### gather_best() (recoleccion del mejor)

Cada rank empaqueta su mejor local (AUC, consistencia, indice, pesos) y participa en:

```c
MPI_Gather(pack, 6, MPI_DOUBLE, all, 6, MPI_DOUBLE, 0, MPI_COMM_WORLD);
```

El rank 0 fusiona en serial (`merge_from_gather`) y difunde el resultado con `MPI_Bcast`. No se usa `MPI_MAXLOC` ni `MPI_Reduce` para el mejor candidato.

`MPI_Reduce(MPI_MAX)` se usa solo para reportar el tiempo de ejecucion global (el maximo entre ranks).

## RNG

El rank 0 genera todos los candidatos con `pcg64_seed(pcg, seed)`. En la fase de refinamiento del hybrid, usa `seed + 9999`. Los ranks no generan pesos independientemente; reciben los suyos via scatter.

## Estrategias de busqueda

- **random / grid:** rank 0 genera candidatos, scatter, evaluacion local, `gather_best()`.
- **hybrid (variante MPI):** fase 1 random (K candidatos) + fase 2 refinamiento (`--refine-steps` o 20% de K). **No incluye fase grid.** Entre fases se llama a `gather_best()` para que todos los ranks conozcan el mejor W antes del refinamiento.

Ver [10_estrategias_busqueda.md](10_estrategias_busqueda.md) para las variantes de hybrid en otras implementaciones.

## Diferencia clave con OpenMP

| Aspecto | OpenMP | MPI |
|---|---|---|
| Memoria | Compartida (un solo espacio) | Distribuida (cada proceso tiene la suya) |
| Comunicacion | Automatica (variables compartidas) | Explicita (MPI_Send/Recv/Gather/Bcast) |
| Escalabilidad | Un solo nodo (limite fisico de RAM) | Multiples nodos (cluster) |
| Inicio | Fork de hilos al entrar en paralelo | Lanzamiento de procesos con mpirun |
| Sincronizacion | Merge serial post-loop | gather_best() (Gather + Bcast) |
| Generacion de W | Cada hilo con `seed + tid` | Rank 0 con `seed`, distribuye con Scatterv |
| Tolerancia a fallos | No (todos los hilos en el mismo proceso) | Potencial (procesos independientes) |

## Compilacion y ejecucion

```bash
# Compilar con mpicc
mpicc -O3 -std=c11 -Wall -Wextra scoring_mpi.c shared/*.c -o scoring_mpi -lm

# Ejecutar con 4 procesos
mpirun -np 4 ./C_OpenMP_MPI/scoring_mpi --k 100000 --seed 42 --data-dir data
```

## Consideraciones tecnicas

- **Cada rank tiene copia completa de los datos.** Con datasets grandes, esto limita el numero maximo de ranks por nodo (limitacion de RAM).
- **MPI en una sola maquina mide overhead de comunicacion**, no escalabilidad real. El beneficio de MPI se observa en clusters donde el cuello de botella no es la memoria sino el tiempo de computo.
- **El broadcast de matrices grandes puede ser costoso.** Sin embargo, solo ocurre una vez al inicio, por lo que el costo se amortiza con K grande.
