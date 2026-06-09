# Resumen técnico

El proyecto optimiza un scoring metagenómico binario. Hay 10 muestras: 5 sanas (`y=0`) y 5 enfermas (`y=1`). Cada muestra usa una matriz de contribución `A` de tamaño `10 x N`. Cada item tiene perfiles `[T,S,F]`. El vector `W=(W1,W2,W3)` combina los perfiles para producir `P`, y el score de muestras se calcula como `Score = A @ P`.

Los datos se generan sintéticamente con `data/generate_dataset.py`. El
parámetro `--signal` (default 2.0) controla la separabilidad. Cada item
tiene perfiles `T` (taxonómico, `[0,1]`), `S` (ecológico/poblacional) y
`F` (funcional) en `profiles_TSF.npy`.

Arquitectura del repo:

| Nivel | Carpeta | Tecnología | Rol |
|---|---|---|---|
| 1 | `python/` | NumPy + multiprocessing | baseline y validación |
| 2 | `C_OpenMP_MPI/` | C/OpenMP y C/MPI | CPU memoria compartida y distribuida |
| 3 | `CUDA/` | CUDA C y PyCUDA | GPU |
| — | `scripts/` | Python | postprocesado y gráficas |

La salida medible es `results/benchmark.csv`, con tiempo, speedup, eficiencia, AUC, consistencia y pesos. Un script de postprocesado (`scripts/postprocess_benchmark.py`) agrega las columnas `speedup` y `efficiency` usando el baseline secuencial de Python como referencia.
