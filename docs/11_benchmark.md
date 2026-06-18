# Benchmark

## Objetivo

El benchmark busca comparar las seis implementaciones del sistema en terminos de:

- **Tiempo de ejecucion:** cuanto tarda cada implementacion en evaluar K candidatos.
- **Speedup:** ganancia de velocidad respecto al baseline Python secuencial.
- **Eficiencia:** que tan bien escala cada implementacion al aumentar las unidades de paralelismo.
- **Throughput:** candidatos evaluados por segundo.
- **Calidad de la solucion:** AUC del mejor W encontrado.
- **Convergencia:** iteracion en la que aparece el mejor resultado.

## Metricas base

### Tiempo (T)

Tiempo de busqueda medido desde el inicio de la generacion/evaluacion de candidatos hasta la obtencion del mejor resultado. Excluye el tiempo de carga de datos desde disco, que es comun a todas las implementaciones.

### Speedup (S)

```
S = T_python_sequential / T_impl
```

Donde `T_python_sequential` es el tiempo del baseline Python secuencial con la misma configuracion (K, seed, estrategia, dataset). El speedup mide la ganancia relativa respecto a la implementacion mas simple.

### Eficiencia (E)

```
E = S / P
```

Donde P es el numero de unidades de paralelismo:

- Para Python multicore: numero de workers (procesos).
- Para C OpenMP: numero de hilos.
- Para C MPI: numero de ranks (procesos MPI).

La eficiencia ideal es 1.0 (speedup lineal). Valores menores indican overhead de paralelizacion (comunicacion, sincronizacion, desbalanceo de carga).

### Throughput

```
Throughput = K / T
```

Mide cuantos candidatos se evaluan por segundo. Es util para comparar implementaciones con diferentes K, aunque la relacion no es perfectamente lineal debido a overheads fijos.

## Consideraciones especificas por implementacion

### GPU (PyCUDA)

Para la implementacion GPU, la metrica P corresponde al numero de multiprocesadores (SM) de la GPU. Sin embargo, la eficiencia calculada como `S / SM_count` no es directamente comparable con la eficiencia de CPU por las siguientes razones:

- Cada SM ejecuta multiples warps (32 hilos) simultaneamente, no un solo hilo.
- El paralelismo GPU es masivo (miles de hilos) y no saturado con P unidades.
- La frecuencia de reloj de los nucleos GPU es menor que la de CPU.

Por lo tanto, para GPU se recomienda:

- Reportar **throughput** (candidatos/segundo) como metrica principal.
- Reportar **speedup** respecto a Python secuencial como metrica secundaria.
- La eficiencia puede reportarse como metrica heuristica con la nota de que `P = SM_count` no representa unidades de ejecucion equivalentes a hilos/procesos CPU.

### MPI en una sola maquina

Ejecutar MPI en una sola maquina (modo `--allow-run-as-root`) mide principalmente el overhead de comunicacion MPI (buffers, sockets locales) y no la escalabilidad real en un cluster. Los resultados de speedup y eficiencia en una sola maquina tienden a ser peores que en un cluster debido a que la comunicacion compite con el computo por los mismos recursos.

## Parametros del benchmark

### Fijos

| Parametro | Valor | Razon |
|---|---|---|
| Dataset | synthetic_CRC2000x10000_balanced | Dataset principal del proyecto |
| Seed | 42 | Reproducibilidad |
| K minimo | 5000 | Overhead de paralelizacion se amortiza |

### Variables

| Parametro | Valores tipicos | Proposito |
|---|---|---|
| K | 5000, 10000, 20000, 50000, 100000 | Evaluar escalabilidad con la carga de trabajo |
| Workers/Threads/Ranks | 1, 2, 4, 8 | Evaluar escalabilidad con el paralelismo |
| Estrategia | random, grid, hybrid | Comparar calidad vs. costo |
| Implementacion | todas las disponibles | Comparar paradigmas |

## Pipeline de ejecucion

### Metodo 1: run_all.sh

```bash
./run_all.sh
```

Ejecuta todas las implementaciones disponibles con configuracion por defecto y genera `results/benchmark_raw.csv`.

### Metodo 2: make benchmark

```bash
make benchmark K="5000 10000 20000" SEARCH=random
```

Ejecuta el pipeline `scripts/benchmark_pipeline.py` con deteccion de hardware y genera `results/benchmark.csv` con columnas enriquecidas (CPU, GPU, RAM).

### Metodo 3: benchmark_pipeline.py

```bash
python scripts/benchmark_pipeline.py \
    --all-strategies \
    --search random \
    --k 5000 10000 20000 \
    --output results/benchmark_pipeline.csv
```

Pipeline avanzado que permite seleccionar estrategias, valores de K, workers, y detecta automaticamente el hardware (modelo de CPU, cores fisicos/logicos, RAM, modelo de GPU, cores CUDA, memoria GPU).

## Validacion de resultados

El script `scripts/validate_benchmark_csv.py` verifica que el archivo CSV de salida cumpla con:

- Cabecera correcta (13 columnas).
- Valores numericos validos en todas las columnas.
- Nombres de implementacion dentro del conjunto esperado.
- Modos de busqueda dentro del conjunto esperado.
- Ausencia de caracteres ANSI o de caja (indicadores de salida de log en lugar de modo benchmark).

## Graficas

Las siguientes graficas se generaran a partir de los datos del benchmark y se almacenaran en `results/plots/`:

| Grafica | Eje X | Eje Y | Series |
|---|---|---|---|
| Tiempo vs K | K | Tiempo (s) | Una curva por implementacion |
| Speedup vs K | K | Speedup | Una curva por implementacion paralela |
| Eficiencia vs K | K | Eficiencia | Una curva por implementacion CPU paralela |
| Throughput vs K | K | Candidatos/s | Una curva por implementacion |
| AUC vs K | K | AUC | Una curva por implementacion |
| Mejor W por implementacion | Implementacion | W1, W2, W3 | Barras apiladas |
| Convergencia vs K | K | Iteracion del mejor | Una curva por implementacion |
| Heatmap tiempo | Implementacion x K | Tiempo (color) | Mapa de calor |
| Heatmap speedup | Implementacion x K | Speedup (color) | Mapa de calor |

## Criterios de validacion de resultados

Los resultados del benchmark deben cumplir:

- **AUC > 0.7:** el scoring debe ser significativamente mejor que el azar (0.5).
- **Consistencia >= 0.8:** debe existir un umbral con balanced accuracy satisfactorio.
- **Speedup creciente con P:** a mas unidades de paralelismo, mayor velocidad (o al menos no empeorar).
- **Repetibilidad:** ejecuciones con la misma configuracion deben producir AUC similares.
- **Correccion numerica:** con la misma estrategia, K y semilla, `random` y `grid` deben coincidir entre implementaciones dentro de tolerancia (~1e-6). En `hybrid`, los resultados pueden diferir porque cada implementacion usa una variante distinta de particion de fases (ver [10_estrategias_busqueda.md](10_estrategias_busqueda.md)).
