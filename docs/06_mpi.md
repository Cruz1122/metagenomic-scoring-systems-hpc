# MPI

MPI se usa para memoria distribuida. Cada rank evalúa un bloque de candidatos.

Archivo:

```text
C_OpenMP_MPI/scoring_mpi.c
```

Compilar y ejecutar:

```bash
make -C C_OpenMP_MPI scoring_mpi
mpirun -np 4 ./C_OpenMP_MPI/scoring_mpi --k 100000 --seed 42 --data-dir data
```

Estrategia:

1. **Rank 0** genera todos los candidatos usando Xorshift y los almacena en `all[]`.
2. `MPI_Barrier` sincroniza antes de comenzar la medición de tiempo.
3. `MPI_Scatter` reparte `chunk = ceil(K/size)` candidatos a cada rank (broadcast de W).
4. Cada rank evalúa su bloque de forma completamente independiente.
5. `MPI_Reduce` con `MPI_MAXLOC` identifica el rank que tiene el mejor AUC (operación de par `{value, rank}`).
6. `MPI_Gather` recolecta los 5 valores `[auc, cons, w1, w2, w3]` de todos los ranks.
7. Rank 0 imprime la línea CSV tomando los pesos del rank ganador.

## Detalles técnicos

- El tiempo se mide con `MPI_Wtime()` después del barrier, antes del scatter. Esto aísla el cómputo paralelo de la generación y distribución.
- Todos los ranks cargan los datos completos de CSV (cada rank tiene copia propia de `A`, `profiles`, `y`). Esto es necesario porque cada rank evalúa candidatos.
- Los candidatos extras (cuando `chunk * size > K`) se generan pero se ignoran en la evaluación.

## Advertencia

MPI puede no acelerar con `K` pequeño o en una sola máquina. Eso no invalida MPI; invalida el experimento. MPI brilla en clústeres con alta latencia de red pero muchos nodos, donde `K` es suficientemente grande para que el cómputo domine sobre la comunicación.
