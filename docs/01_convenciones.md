# Convenciones

## Carpetas

- `data/`: generación y datos reproducibles.
- `python/`: baseline secuencial y multicore.
- `C_OpenMP_MPI/`: OpenMP y MPI en C.
- `CUDA/`: CUDA C y PyCUDA.
- `results/`: CSV y gráficas.
- `report/`: informe final.
- `docs/`: documentación técnica.

## CSV estándar

Toda implementación imprime:

```text
implementation,parallel_units,n_items,k,time_sec,auc,consistency,w1,w2,w3,seed
```

`postprocess_benchmark.py` agrega:

```text
speedup = T_python_sequential / T_impl
efficiency = speedup / parallel_units
```

## Reglas duras

No compares tiempos si cambiaste `N`, `K`, seed o dataset. No optimices código que no reproduce el AUC. No declares una tecnología “lenta” con `K` pequeño: probablemente estás midiendo overhead, no cómputo.
