<p align="center">
  <img src="./icon.svg" alt="Scoring Metagenomico HPC" width="160" />
</p>

<p align="center" style="font-size:2rem;"><strong>Scoring Metagenomico HPC</strong></p>
<p align="center" style="font-size:1.25rem; margin-top:-1em;"><em>Optimizacion Paralela de Scoring Metagenomico para Clasificacion Binaria</em></p>

<p align="center">
  <strong>Lenguajes y plataformas</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/C-11-A8B9CC?logo=c&logoColor=white" alt="C11" />
  <img src="https://img.shields.io/badge/CUDA-12-76B900?logo=nvidia&logoColor=white" alt="CUDA 12" />
  <img src="https://img.shields.io/badge/OpenMP-4.5-000000?logo=openmp&logoColor=white" alt="OpenMP" />
  <img src="https://img.shields.io/badge/MPI-3.0-006B3F?logo=linux&logoColor=white" alt="MPI" />
  <img src="https://img.shields.io/badge/PyCUDA-2024.1-3776AB?logo=python&logoColor=white" alt="PyCUDA" />
</p>

<p align="center">
  <img src="https://img.shields.io/badge/NumPy-1.24+-013243?logo=numpy&logoColor=white" alt="NumPy" />
  <img src="https://img.shields.io/badge/scikit--learn-1.3+-F7931E?logo=scikit-learn&logoColor=white" alt="scikit-learn" />
  <img src="https://img.shields.io/badge/Pandas-2.0+-150458?logo=pandas&logoColor=white" alt="Pandas" />
</p>

---

# Contenido

1. [Planteamiento del problema](#1-planteamiento-del-problema)
2. [Dataset](#2-dataset)
3. [Modelo matematico](#3-modelo-matematico)
4. [Arquitectura del sistema](#4-arquitectura-del-sistema)
5. [Estrategias de busqueda](#5-estrategias-de-busqueda)
6. [Benchmark](#6-benchmark)
7. [Formato de salida estandar](#7-formato-de-salida-estandar)
8. [Estructura del repositorio](#8-estructura-del-repositorio)
9. [Requisitos y ejecucion](#9-requisitos-y-ejecucion)
10. [Documentacion](#10-documentacion)

---

## 1. Planteamiento del problema

Dado un conjunto de $N$ perfiles taxonomicos, ecologicos y funcionales para cada item de interes biologico, se busca un vector de pesos $W = (W_1, W_2, W_3)$ que maximice la capacidad de discriminar entre dos poblaciones: sana ($y=0$) y enferma ($y=1$). La discriminacion se mide mediante el area bajo la curva ROC (AUC).

Cada muestra j se describe mediante una matriz de contribucion $A$ de dimensiones $n_{\text{muestras}} \times N$. Cada item $i$ tiene tres perfiles: $T_i$ (taxonomico), $S_i$ (ecologico-poblacional) y $F_i$ (funcional). El score de una muestra se obtiene combinando estos perfiles con los pesos:

$$
P_i = W_1 T_i + W_2 S_i + W_3 F_i
$$

$$
\text{Score}_j = \sum_i A_{ji} P_i
$$

El objetivo es maximizar $\mathrm{AUC}(y, \mathrm{Score}(W))$ variando $W$ dentro del simplex tridimensional.

El problema es computacionalmente relevante porque por cada candidato $W$ se debe evaluar la expresion $A \cdot (\mathrm{profiles} \cdot W)$, cuyo costo depende de $n_{\text{muestras}} \cdot N$. Con datasets realistas (2000 muestras, 10000 items), evaluar miles o millones de candidatos requiere estrategias de paralelizacion eficientes.

---

## 2. Dataset

### 2.1 Origen y referencia biologica

El dataset sintetico se diseno tomando como referencia el problema de clasificacion de cancer colorrectal (CRC) versus healthy, reportado en el preprint:

> Haldar, Stein-Thoeringer, Borisov. *Interpreting Microbiome Relative Abundance Data Using Symbolic Regression*. arXiv:2410.16109, 2024.

Ese trabajo utiliza 71 estudios de `curatedMetagenomicData`, con 11,137 muestras healthy/CRC y 749 features de especies microbianas. Por limitaciones de acceso a los datos reales (HTTP 403 al descargar `relative_abundance`), este proyecto genera un dataset sintetico compatible con el problema real.

El archivo [`fuente_real_dataset_sintetico_crc.md`](fuente_real_dataset_sintetico_crc.md) documenta en detalle la fuente real, las decisiones metodologicas y las diferencias con el dataset sintetico.

### 2.2 Dimensiones y estructura

El dataset principal se denomina `synthetic_CRC2000x10000_balanced`:

| Componente | Dimension | Contenido |
|---|---|---|
| `matrix_A.npy` | 2000 x 10000 | Contribucion de cada item por muestra (abundancia relativa normalizada) |
| `labels.npy` | 2000 | Clase: 0 (healthy), 1 (CRC) |
| `profiles_TSF.npy` | 10000 x 3 | Perfiles T (taxonomico), S (ecologico), F (funcional) por item |

Composicion de etiquetas:

- 1000 muestras healthy (y=0)
- 1000 muestras CRC (y=1)

El dataset esta balanceado por construccion para evitar sesgos en la metrica AUC.

### 2.3 Separacion REF/EVAL

El generador separa las muestras en dos cohortes independientes:

- **REF** (1000 muestras): se usa exclusivamente para estimar los perfiles T y S.
- **EVAL** (2000 muestras): se exporta como `matrix_A.npy` y `labels.npy`, y es sobre el que se mide el AUC.

Esta separacion evita **fuga de etiqueta (label leakage)**: si T se calculara sobre las mismas muestras que se evaluan, el perfil taxonomico podria codificar indirectamente la clase y producir un AUC artificialmente alto.

### 2.4 Generacion sintetica

El script `data/scripts/generate_dataset.py` produce las matrices mediante:

1. Abundancias base con distribucion gamma.
2. Efecto de clase sobre grupos de items enriquecidos en CRC o healthy.
3. Efecto de metadata sintetica (edad, BMI).
4. Ruido lognormal por muestra.
5. Zero-inflation para simular escasez (sparsity).
6. Normalizacion composicional (cada fila suma 1).

Los perfiles T se derivan del log fold-change entre CRC y healthy en REF. Los perfiles S se derivan de la correlacion con metadata de riesgo. Los perfiles F son marcadores funcionales sinteticos (resistencia, virulencia, inflamacion, metabolicos, beneficos).

### 2.5 Reproducibilidad

Todo el proceso usa `seed=42` como semilla base. El generador acepta parametros para controlar senal, ruido, zero-inflation y dimensiones, documentados en la ayuda del script.

---

## 3. Modelo matematico

### 3.1 Score por item

Cada item $i$ aporta un score escalar que combina sus tres perfiles mediante el vector de pesos $W$:

$$
P_i = W_1 T_i + W_2 S_i + W_3 F_i
$$

Donde $T_i, S_i, F_i \in [0, 1]$ representan respectivamente la senal taxonomica, ecologica y funcional del item $i$.

### 3.2 Score por muestra

El score de la muestra $j$ se obtiene como combinacion lineal de los scores por item, ponderada por la matriz de contribucion $A$:

$$
\text{Score}_j = \sum_{i=1}^{N} A_{ji} P_i
$$

En forma matricial:

$$
\mathbf{Score} = A \mathbf{P}
$$

donde $A \in \mathbb{R}^{n_{\text{muestras}} \times N}$ y $\mathbf{P} \in \mathbb{R}^{N}$.

### 3.3 Funcion objetivo

$$
\max_{W} \; \mathrm{AUC}\bigl(y, \mathrm{Score}(W)\bigr)
$$

El AUC mide la probabilidad de que una muestra enferma reciba un score mayor que una muestra sana. Su rango teorico es $[0, 1]$, donde $0.5$ indica azar y $1.0$ separacion perfecta.

### 3.4 Espacio de busqueda

Los pesos $W$ deben pertenecer al simplex de dimension 2 (simplex tridimensional):

$$
W_1 + W_2 + W_3 = 1, \qquad W_i \geq 0 \quad \text{para } i = 1, 2, 3
$$

Este espacio es continuo y de dimension 2 (tres coordenadas con una restriccion de suma). La distribucion de Dirichlet con parametros $\alpha = (1, 1, 1)$ es uniforme sobre este simplex, lo que la convierte en la opcion natural para muestrear candidatos en Random Search. Cada muestra de $\mathrm{Dirichlet}(1, 1, 1)$ produce un vector $W$ que cumple automaticamente ambas restricciones.

### 3.5 Validacion del scoring

Para un umbral de decision $\theta$, la consistencia se define como el balanced accuracy maximo sobre todos los umbrales posibles:

$$
\text{Consistencia} = \max_{\theta} \; 0.5 \cdot \bigl(\mathrm{TPR}(\theta) + \mathrm{TNR}(\theta)\bigr)
$$

donde $\mathrm{TPR}$ es sensibilidad (tasa de positivos correctos) y $\mathrm{TNR}$ es especificidad (tasa de negativos correctos). Se considera satisfactorio un valor $\geq 0.8$.

---

## 4. Arquitectura del sistema

El sistema se organiza en tres niveles de abstraccion computacional, cada uno implementado con paradigmas de paralelismo distintos. Todos resuelven el mismo problema matematico y comparten el mismo formato de entrada y salida.

### 4.1 Python secuencial

**Archivo:** `python/sequential.py`

Paradigma: ejecucion secuencial clasica de von Neumann. Un solo nucleo de CPU evalua los K candidatos uno tras otro. No hay paralelismo.

Funcionamiento:
- En modo `random`, genera un vector W por iteracion con `rng.dirichlet(np.ones(3))` en un bucle `for i in range(k)`.
- Para cada W, `evaluate()` calcula `P = profiles @ W`, `scores = A @ P`, `AUC = roc_auc_score(y, scores)` y consistencia.
- La evaluacion de cada candidato usa operaciones matriciales de NumPy (BLAS) sobre un solo W; no hay batching ni paralelismo explicito.
- Soporta tambien `grid` y `hybrid` (ver seccion 5).

Rol: sirve como **baseline** para medir speedup. Es la implementacion de referencia contra la que se comparan todas las demas. Es tambien la unica implementacion que no requiere compilacion ni dependencias exoticas, lo que la hace portatil.

### 4.2 Python multicore (multiprocessing)

**Archivo:** `python/multicore.py`

Paradigma: paralelismo por **multiples procesos** con memoria independiente (MIMD). Cada proceso worker tiene su propio espacio de direcciones y no comparte variables con los demas.

Fundamento teorico: el problema es **embarazosamente paralelo**: evaluar K candidatos independientes no requiere comunicacion entre workers. Cada worker recibe un subconjunto de candidatos, los evalua y retorna su mejor resultado local. El proceso principal recolecta los resultados y selecciona el mejor global.

Funcionamiento:
- El proceso principal genera los K vectores W una sola vez con `rng.dirichlet(np.ones(3), size=k)` (en modo random).
- `_parallel_eval()` divide los candidatos en tareas: chunks de 32 si hay logging en vivo (`LOG_INTERVAL`), o `ceil(n/workers)` en modo benchmark.
- Distribuye las tareas con `pool.imap_unordered(partial(_eval_chunk, A=..., y=..., profiles=...), tasks)`.
- Cada worker ejecuta `_eval_chunk(task, A, y, profiles)` sobre su bloque y retorna el mejor resultado local.
- El proceso principal fusiona los mejores locales y selecciona el mejor global (desempate: mayor consistencia, luego menor indice).

Ventaja: no requiere sincronizacion ni exclusividad mutua. Cada worker es una funcion pura que retorna un valor. No se usa `Manager`, `Queue`, `Lock` ni `Value`.

Consideracion: al ser procesos independientes, la memoria total utilizada es aproximadamente `workers` veces la memoria de un solo proceso (cada worker tiene copia de A, profiles, y). Esto limita la escalabilidad cuando el dataset es grande.

### 4.3 C secuencial

**Archivo:** `C_OpenMP_MPI/scoring_sequential.c`

Paradigma: ejecucion secuencial en C nativo. Sin paralelismo.

Funcionamiento:
- Implementa el mismo algoritmo que Python secuencial pero en C compilado con `-O3`.
- Usa libreria matematica estandar (`libm`) para las operaciones.
- El RNG es PCG64 (Permuted Congruential Generator) con generacion de numeros en el simplex mediante transformacion de Dirichlet via algoritmo de muestreo gamma.
- Las matrices se almacenan en arreglos planos (row-major) para localidad de cache.

Rol: sirve como **baseline nativo** para medir el overhead de Python. Al eliminar la maquina virtual e interpretacion, muestra el rendimiento crudo del hardware en secuencial.

### 4.4 C OpenMP

**Archivo:** `C_OpenMP_MPI/scoring_openmp.c`

Paradigma: paralelismo por **memoria compartida** (SIMD + MIMD) mediante **directivas de compilacion**. OpenMP sigue el modelo **fork/join**: al entrar en una region paralela, el hilo principal (master) crea un equipo de hilos que ejecutan el codigo en paralelo; al finalizar, los hilos se unen y la ejecucion continua con un solo hilo.

Fundamento teorico: todos los hilos comparten el mismo espacio de memoria (A, profiles, y son globales y de solo lectura). Cada hilo tiene su propio RNG (PCG64 con semilla `seed + thread_id`) para generar candidatos sin contencion. La division del trabajo usa `#pragma omp parallel` con `#pragma omp for schedule(static)` sobre el bucle de candidatos.

Mecanismo de sincronizacion:
- Cada hilo mantiene su mejor resultado local (`local_best`) y lo guarda en `threads[tid].best`.
- Fuera de la region paralela, un merge serial compara `threads[t].best` y actualiza el mejor global (sin `critical` ni `reduction`).
- `evaluate()` (incluida la multiplicacion matriz-vector) corre secuencialmente dentro de cada hilo.
- En modo `--verbose`, `#pragma omp critical(log_improve)` serializa solo el logging de mejoras en vivo.

Ventaja: el merge no introduce contencion entre hilos. OpenMP permite escalar desde 1 hilo (equivalente a C secuencial) hasta N hilos sin cambiar el codigo fuente.

### 4.5 C MPI

**Archivo:** `C_OpenMP_MPI/scoring_mpi.c`

Paradigma: paralelismo por **paso de mensajes** y **memoria distribuida** (MIMD). Cada proceso (rank) tiene su propio espacio de memoria y se comunica con los demas mediante llamadas a la libreria MPI.

Fundamento teorico: a diferencia de OpenMP (memoria compartida), MPI no presupone que los procesos compartan memoria fisica. Esto permite escalar a multiples nodos en un cluster, pero introduce costo de comunicacion explicita. En este problema, la comunicacion es minima porque el problema es embarazosamente paralelo.

Funcionamiento:
- El rank 0 carga los datos desde disco.
- El rank 0 transmite (broadcast) las matrices A, profiles, y y las dimensiones a todos los ranks mediante `MPI_Bcast`.
- El rank 0 genera todos los candidatos W (PCG64 con semilla `seed`) y los distribuye con `MPI_Scatterv`.
- Cada rank evalua su bloque recibido (tamano casi uniforme, diferencia de a lo sumo 1).
- Al finalizar cada fase, `gather_best()` recolecta los mejores locales con `MPI_Gather`, el rank 0 fusiona en serial y difunde el resultado con `MPI_Bcast`.
- `MPI_Reduce(MPI_MAX)` se usa solo para el tiempo de ejecucion global, no para el mejor candidato.
- Solo el rank 0 imprime la salida.

Diferencia clave con OpenMP: en MPI cada proceso tiene copia completa de los datos. No hay memoria compartida, por lo que no hay condiciones de carrera sobre las matrices. La sincronizacion ocurre al inicio (broadcast), entre fases (gather/bcast) y al final (reduce de tiempo).

### 4.6 PyCUDA (GPU)

**Archivo:** `CUDA/scoring_pycuda.py`

Paradigma: paralelismo **masivamente paralelo en GPU** (SIMT - Single Instruction Multiple Threads). Cada hilo GPU evalua un candidato W independiente.

Fundamento teorico: las GPUs de NVIDIA organizan los hilos en una jerarquia de **grid** (conjunto de bloques), **bloques** (conjunto de hilos que comparten memoria) y **hilos** (unidad minima de ejecucion). Todos los hilos ejecutan el mismo kernel (codigo) pero sobre datos diferentes.

Arquitectura del kernel:
- **1 hilo = 1 candidato W.** Cada hilo recibe el indice global de su candidato, calcula `P = profiles @ W`, `scores = A @ P`, y luego `AUC` y `consistencia` en GPU.
- El kernel calcula AUC y consistencia completamente en la GPU mediante el metodo Mann-Whitney por rangos, con acumulacion en double precision para evitar perdida numerica.
- El tamanio de bloque es 256 hilos. La grilla se dimensiona como `ceil(K / 256)` bloques.

Memoria:
- **Memoria compartida:** en modo `full`, cada bloque cachea una columna de A en memoria compartida (`__shared__`) para reducir accesos a memoria global. Tamano: `n_muestras * sizeof(float)` (~8 KB para 2000 muestras).
- **Memoria global:** A, profiles, labels se copian una sola vez al device al inicio. Los pesos se suben con `upload_weights()` por fase de busqueda; en modo live no hay re-upload por bloque de 256, solo D2H parcial de resultados.

Reduccion:
- El mejor global se obtiene mediante dos etapas de reduction kernel:
  - `reduce_best_stage1`: cada bloque reduce sus candidatos a un `BestVal` parcial.
  - `reduce_best_stage2`: combina los parciales de todos los bloques en un solo resultado global.

Modos de ejecucion:
- **Modo live** (default): un launch CUDA por bloque (256 candidatos). Permite logging de progreso en vivo con mejoras parciales.
- **Modo fast** (`--fast` o `--benchmark`): un solo launch sobre todos los K candidatos seguido de reduction completa en GPU. Solo transfiere el mejor resultado de vuelta al host.

Variante precompute:
- En modo `precompute`, se calcula `B = A @ profiles` una vez en el host y se transfiere a la GPU. Esto reduce el kernel a `scores = B @ W`, eliminando las iteraciones sobre items dentro del kernel. Es mas rapido cuando K es grande, pero requiere almacenar la matriz B (n_muestras x 3).

---

## 5. Estrategias de busqueda

Cada implementacion soporta tres estrategias para explorar el simplex de pesos. La eleccion de estrategia afecta la calidad del AUC encontrado, el costo computacional y la reproducibilidad.

### 5.1 Random search

Genera $K$ vectores $W$ independientes mediante muestreo uniforme sobre el simplex con $\mathrm{Dirichlet}(1, 1, 1)$.

Propiedades:
- Cada candidato es independiente de los anteriores.
- La cobertura del simplex es probabilistica: con K suficientemente grande, la probabilidad de aproximarse al optimo global tiende a 1.
- Es la estrategia mas simple y la que mejor se presta a paralelizacion (no hay dependencia entre candidatos).
- El costo computacional es lineal en K.
- No hay riesgo de sobreadaptacion al grid.

Limitacion: no garantiza cubrir regiones especificas del simplex. Dependiendo de K, puede dejar zonas sin explorar.

### 5.2 Grid search

Define una malla regular sobre el simplex. Dada una resolucion $R$, se generan todos los puntos de la forma:

$$
W_1 = \frac{i}{R}, \qquad W_2 = \frac{j}{R}, \qquad W_3 = \frac{R - i - j}{R}
$$

para $i, j \geq 0$ con $i + j \leq R$. El numero de puntos es $(R+1)(R+2)/2$.

Propiedades:
- Cobertura determinista y uniforme del simplex.
- Reproducible por definicion (no interviene el azar).
- El numero de puntos crece cuadraticamente con $R$: $K \sim R^2 / 2$.
- Puede generar mas o menos candidatos que el $K$ solicitado. La resolucion se ajusta automaticamente como $R = \max(1, \lfloor\sqrt{2K}\rfloor)$.

Limitacion: para R pequeno, la granularidad puede omitir el optimo global si este se encuentra entre puntos del grid. Para R grande, K crece rapidamente y el costo computacional puede superar al de random search con el mismo K nominal.

### 5.3 Hybrid search

Combina grid, random y refinamiento local. **La particion de fases depende de la implementacion:**

| Implementacion | Fases |
|---|---|
| Python / C secuencial / C OpenMP | Grid fijo `step=0.02` (~1326 pts); luego 50/50 del resto entre random Dirichlet(1,1,1) y local con `alpha = max(w * {300, 1000}, 1e-3)` |
| C MPI | Random (K candidatos) + refinamiento (`--refine-steps` o 20% de K, semilla `seed+9999`); sin fase grid |
| PyCUDA | 20% grid (max 2000) + 60% random + resto local con `alpha = max(w * 100, 1e-3)` |

Propiedades:
- Aprovecha cobertura determinista (grid) y exploracion estocastica (random).
- La fase local refina alrededor del mejor W encontrado.
- El costo total es comparable al de random search con el mismo K nominal.
- La reproducibilidad depende de la semilla en las fases aleatorias.

### 5.4 Comparacion entre estrategias

| Aspecto | Random | Grid | Hybrid |
|---|---|---|---|
| Cobertura del simplex | Probabilistica | Determinista uniforme | Grid + aleatoria + local |
| Reproducibilidad | Depende de seed | Total | Depende de seed (fases 2 y 3) |
| Costo computacional | $O(K)$ | $O(R^2) \sim O(K)$ | $O(K)$ |
| Riesgo de omitir optimo | Disminuye con K | Depende de resolucion | Bajo (fase local) |
| Paralelizacion | Trivial | Trivial | Trivial (fases secuenciales pero cada fase es paralela) |
| Calidad tipica de AUC | Buena para K grande | Buena para R suficiente | Generalmente la mejor |

---

## 6. Benchmark

Comparacion pareada **random vs grid** sobre el mismo dataset, semilla y presupuestos K. El objetivo no es solo medir velocidad, sino mostrar como cambia el costo computacional y la calidad del scoring al variar la estrategia de busqueda y el paradigma de implementacion (Python, C, GPU).

### 6.1 Metricas base

Cada ejecucion de benchmark produce las siguientes metricas por implementacion y configuracion:

| Metrica | Descripcion | Formula |
|---|---|---|
| Tiempo (s) | Tiempo de busqueda excluyendo carga de datos | $t_{\mathrm{fin}} - t_{\mathrm{inicio}}$ |
| Speedup | Ganancia respecto al baseline de su familia | $S_{\mathrm{py}} = T_{\mathrm{python\_sequential}} / T_{\mathrm{impl}}$; $S_{\mathrm{c}} = T_{\mathrm{c\_sequential}} / T_{\mathrm{impl}}$ |
| Eficiencia | Speedup normalizado por unidades de paralelismo | $E = S / P$ |
| Throughput | Candidatos evaluados por segundo | $K / T_{\mathrm{impl}}$ |
| AUC | Area bajo la curva ROC del mejor W encontrado | $\max \mathrm{AUC}(y, \mathrm{Score}(W))$ |
| Consistencia | Balanced accuracy maximo del mejor W | $\max_{\theta} \; 0.5 \cdot (\mathrm{TPR}+\mathrm{TNR})$ |

### 6.2 Consideraciones sobre la eficiencia en GPU

Para las implementaciones GPU (PyCUDA), la metrica `parallel_units` corresponde al numero de multiprocesadores (SM) de la GPU, no a hilos o procesos de CPU. Comparar eficiencia GPU como si `parallel_units=14` fuera equivalente a 8 procesos CPU no es tecnicamente valido, porque:

- La GPU ejecuta miles de hilos simultaneamente en cada SM, no un hilo por SM.
- La metrica $\mathrm{efficiency} = \mathrm{speedup} / \mathrm{parallel\_units}$ para GPU infraestima artificialmente el rendimiento real porque el denominador no refleja la capacidad de paralelismo masivo.

Para GPU se recomienda reportar:
- **Throughput** (candidatos/segundo) como metrica principal.
- **Speedup** respecto al baseline secuencial como metrica secundaria.
- **Eficiencia** solo si se aclara que es una metrica heuristica referida a SMs, no a unidades de ejecucion equivalentes a CPU.

### 6.3 Configuracion de la medicion

Medicion principal en [`results/benchmark.csv`](results/benchmark.csv): **88 filas validas** (11 configuraciones x 8 valores de K). Cada implementacion se evalua en **random** y **grid** bajo las mismas condiciones, lo que permite una comparacion pareada entre estrategias de busqueda.

| Parametro | Valor |
|---|---|
| Dataset | `synthetic_CRC2000x10000_balanced` (10 000 items, 2 000 muestras) |
| Estrategias | `random` y `grid` (comparacion pareada) |
| Seed | 42 |
| Valores de K | 5 000, 10 000, 20 000, 50 000, 250 000, 500 000, 1 000 000, 2 000 000 |
| Implementaciones | Python secuencial, Python multicore, C secuencial (solo random), C + OpenMP, C + MPI, PyCUDA |
| Paralelismo | Python multicore, C OpenMP y C MPI: 8 unidades; PyCUDA: 14 SMs |
| Hardware | Intel Core i5-10300H (4C/8T), 23.3 GB RAM, NVIDIA GTX 1650 (896 CUDA cores) |

**Nota sobre baselines:** el CSV incluye `c_sequential` solo en random. Para calcular speedup de implementaciones C en grid, se usa `c_sequential` random como referencia de lenguaje; esto se declara explicitamente en las graficas para no mezclar estrategia con paralelizacion.

### 6.4 Resultados destacados (K = 2 000 000)

#### Random search

| Implementacion | Tiempo (s) | Tiempo (h) | AUC | Throughput (cand/s) | Speedup vs baseline propio |
|---|---|---|---|---|---|
| C OpenMP | 5 650.1 | 1.57 | 0.792837 | 354.0 | 7.09x vs C secuencial |
| PyCUDA | 7 160.2 | 1.99 | 0.792827 | 279.3 | 7.27x vs Python secuencial |
| C MPI | 9 434.9 | 2.62 | 0.792845 | 212.0 | 4.25x vs C secuencial |
| C secuencial | 40 083.8 | 11.13 | 0.792827 | 49.9 | 1.00x (baseline C) |
| Python multicore | 46 586.5 | 12.94 | 0.792827 | 42.9 | 1.12x vs Python secuencial |
| Python secuencial | 52 033.0 | 14.45 | 0.792827 | 38.4 | 1.00x (baseline Python) |

#### Grid search

| Implementacion | Tiempo (s) | Tiempo (h) | AUC | Throughput (cand/s) | Speedup vs baseline propio |
|---|---|---|---|---|---|
| PyCUDA | 4 665.5 | 1.30 | 0.792825 | 428.7 | 20.9x vs Python secuencial grid |
| C OpenMP | 10 359.5 | 2.88 | 0.792817 | 193.1 | 3.87x vs C secuencial random* |
| C MPI | 15 222.2 | 4.23 | 0.792817 | 131.4 | 2.63x vs C secuencial random* |
| Python multicore | 87 968.3 | 24.44 | 0.792817 | 22.7 | 1.11x vs Python secuencial grid |
| Python secuencial | 97 410.3 | 27.06 | 0.792817 | 20.5 | 1.00x (baseline Python grid) |

\* Baseline C tomado de `c_sequential` random (no existe `c_sequential` grid en el benchmark).

**Lectura integrada:**

- **Tiempo:** crece casi proporcionalmente con K en ambas estrategias. En random, C + OpenMP es el mas rapido en CPU (5 650.1 s); en grid, PyCUDA lidera (4 665.5 s). La GPU aporta mas ventaja cuando la exploracion es sistematica (grid) que cuando es estocastica (random).
- **Calidad:** el AUC se concentra alrededor de 0.7928 en todos los casos. Random alcanza el mejor valor global (0.792845, C + MPI); grid converge rapido pero se estanca cerca de 0.792825. La diferencia importante no es la calidad, sino el costo computacional para llegar a ella.
- **Speedup:** en K = 2 000 000, PyCUDA-grid logra el mayor speedup relativo (20.9x frente a Python secuencial grid), mientras C + OpenMP-random obtiene el mejor speedup CPU (7.1x frente a C secuencial).
- **Recomendacion practica:** si solo hay CPU, C + OpenMP con random search ofrece el mejor equilibrio tiempo-AUC. Si hay GPU disponible y la estrategia es grid, PyCUDA domina en tiempo y throughput sin sacrificar calidad de forma relevante.

### 6.5 Graficas

Las graficas se generan a partir de `results/benchmark.csv` y se almacenan en `results/plots/`:

| Figura | Archivo | Contenido |
|---|---|---|
| Fig. 1 | [`fig1_runtime_random_grid.png`](results/plots/fig1_runtime_random_grid.png) | Tiempo de ejecucion vs K (random y grid) |
| Fig. 2 | [`fig2_speedup_random_grid.png`](results/plots/fig2_speedup_random_grid.png) | Speedup vs baseline propio por familia |
| Fig. 3 | [`fig3_auc_random_grid.png`](results/plots/fig3_auc_random_grid.png) | Convergencia del AUC (random y grid) |
| Fig. 4 | [`fig4_pareto_random_grid.png`](results/plots/fig4_pareto_random_grid.png) | Relacion costo-calidad en K = 2 000 000 |
| Fig. 5 | [`fig5_throughput_random_grid.png`](results/plots/fig5_throughput_random_grid.png) | Throughput (candidatos/s) en K = 2 000 000 |

#### Figura 1. Tiempo de ejecucion por estrategia

![Tiempo vs K — random y grid](results/plots/fig1_runtime_random_grid.png)

El tiempo crece casi proporcionalmente con K en ambas estrategias. En random, C + OpenMP es el mas rapido al mayor presupuesto (5 650.1 s en K = 2 000 000); en grid, PyCUDA queda primero (4 665.5 s). Lectura practica: para CPU gana OpenMP, mientras que grid se beneficia especialmente de la GPU.

#### Figura 2. Speedup con baseline propio

![Speedup vs baseline propio — random y grid](results/plots/fig2_speedup_random_grid.png)

El speedup se calcula contra el baseline de su propia familia: Python multicore frente a Python secuencial, PyCUDA frente a Python secuencial y C/OpenMP-MPI frente a C secuencial random (porque el benchmark no incluye C secuencial grid). En K = 2 000 000, PyCUDA-grid logra el mayor speedup relativo (20.9x), mientras C + OpenMP-random alcanza el mejor speedup CPU (7.1x).

#### Figura 3. Convergencia de AUC

![Convergencia AUC — random y grid](results/plots/fig3_auc_random_grid.png)

La calidad del scoring cambia muy poco: todos los valores quedan alrededor de 0.7928. Random alcanza el mejor AUC final (0.792845, C + MPI), mientras grid llega rapido a una zona competitiva pero se estanca cerca de 0.792825 al mayor K. La leccion pedagogica: paralelizar o cambiar de lenguaje acelera la busqueda sin alterar sustancialmente la calidad del mejor W encontrado.

#### Figura 4. Pareto tiempo vs AUC

![Pareto tiempo-AUC — random y grid](results/plots/fig4_pareto_random_grid.png)

El grafico Pareto separa rendimiento y calidad. En random, C + OpenMP ofrece el mejor equilibrio tiempo-AUC; C + MPI obtiene el AUC mas alto, pero tarda mas. En grid, PyCUDA domina en tiempo, aunque no alcanza el mejor AUC global de random. Esta es la figura mas util para justificar la implementacion recomendada en una sustentacion.

#### Figura 5. Throughput final

![Throughput final — random y grid](results/plots/fig5_throughput_random_grid.png)

El throughput confirma la misma historia en una metrica mas intuitiva. PyCUDA-grid procesa mas candidatos por segundo (428.7 cand/s) y C + OpenMP-random es el mejor caso CPU (354.0 cand/s). Python queda muy por debajo, incluso con multiprocessing, lo que evidencia overhead de procesos y menor eficiencia de ejecucion.

---

## 7. Formato de salida estandar

Toda implementacion imprime una linea CSV con las siguientes columnas:

```text
implementation,parallel_units,n_items,k,time_sec,auc,consistency,w1,w2,w3,seed,search_mode,iterations_until_best
```

| Columna | Descripcion |
|---|---|
| `implementation` | Identificador: `python_sequential`, `python_multicore`, `c_sequential`, `c_openmp`, `c_mpi`, `pycuda` |
| `parallel_units` | Hilos (OpenMP), procesos (multicore/MPI) o SMs (GPU) usados |
| `n_items` | Numero de items (N) del dataset |
| `k` | Candidatos evaluados |
| `time_sec` | Tiempo de busqueda (excluye carga de datos) |
| `auc` | AUC del mejor W encontrado |
| `consistency` | Balanced accuracy maxima |
| `w1..w3` | Pesos del mejor candidato |
| `seed` | Semilla de reproducibilidad |
| `search_mode` | Estrategia: `random`, `grid` o `hybrid` |
| `iterations_until_best` | Iteracion donde se encontro el mejor AUC |

El pipeline `scripts/benchmark_pipeline.py` genera un CSV enriquecido con datos de hardware (CPU, GPU, RAM). El script `scripts/validate_benchmark_csv.py` valida el formato y contenido del CSV de salida.

---

## 8. Estructura del repositorio

```text
metagenomic-scoring-systems-hpc/
├── data/
│   ├── processed/                  datasets generados
│   │   └── synthetic_CRC2000x10000_balanced/
│   ├── scripts/
│   │   └── generate_dataset.py     generacion de dataset sintetico
│   └── dataset_manifest.json~HEAD  manifiesto del dataset
├── python/
│   ├── __init__.py                 docstring del paquete
│   ├── common.py                   SearchResult, load_data, evaluate, AUC, consistency
│   ├── sequential.py               busqueda secuencial (baseline)
│   ├── multicore.py                busqueda con multiprocessing
│   └── logger.py                   logger ANSI
├── C_OpenMP_MPI/
│   ├── scoring_sequential.c        C secuencial
│   ├── scoring_openmp.c            OpenMP: 3 estrategias de busqueda
│   ├── scoring_mpi.c               MPI: 3 estrategias de busqueda
│   ├── shared/
│   │   ├── common.h/c              load_data, evaluate, AUC, consistency
│   │   ├── rng.h/c                 PCG64 + Dirichlet
│   │   ├── ziggurat.h/c            generador Ziggurat
│   │   └── logger.h/c              logger ANSI
│   └── Makefile                    compilacion gcc / mpicc
├── CUDA/
│   └── scoring_pycuda.py           PyCUDA: kernel embebido + reduction GPU
├── scripts/
│   ├── benchmark_pipeline.py       pipeline de benchmark multi-K con deteccion hardware
│   └── validate_benchmark_csv.py   valida formato CSV de salida
├── docs/                           documentacion tecnica
├── results/
│   ├── benchmark.csv               mediciones (11 configs x 8 K, random + grid)
│   └── plots/                      fig1–fig5 (random vs grid: tiempo, speedup, AUC, pareto, throughput)
├── run_all.sh                      pipeline de benchmark automatizado
├── Makefile                        automatizacion: data, python-*, c-*, benchmark, clean
├── PROJECT.md                      especificacion contractual del proyecto
├── fuente_real_dataset_sintetico_crc.md  origen y referencia del dataset
├── icon.svg                        logo del proyecto
├── requirements.txt                dependencias Python
└── .gitignore
```

---

## 9. Requisitos y ejecucion

### 9.1 Requisitos

| Herramienta | Version | Proposito |
|---|---|---|
| Python | >= 3.10 | Implementaciones Python, PyCUDA |
| GCC | >= 10 | Compilacion OpenMP |
| MPICH / OpenMPI | >= 3.0 | Compilacion y ejecucion MPI |
| CUDA toolkit | >= 12 | PyCUDA: compilacion JIT de kernels |
| make | >= 4 | Automatizacion |

Dependencias Python:

```
pip install -r requirements.txt
# numpy>=1.24, pandas>=2.0, matplotlib>=3.7, scikit-learn>=1.3, pycuda>=2024.1
```

### 9.2 Generar datos

```bash
# 2000 muestras x 10000 items (default)
make data

# 100 muestras (desarrollo rapido)
python data/scripts/generate_dataset.py --n-eval 100 --n-ref 200 --allow-small
```

### 9.3 Ejecutar implementaciones

```bash
# Python secuencial (baseline)
make python-sequential K=10000

# Python multicore
make python-multicore K=10000 WORKERS=4

# Compilar binarios C
make c

# C secuencial
make c-sequential K=10000

# C OpenMP
make c-openmp K=10000 THREADS=4

# C MPI
make c-mpi K=10000 MPI_RANKS=4

# PyCUDA
make python-pycuda K=100000 SEED=42 SEARCH=random
```

### 9.4 Benchmark completo

```bash
# Pipeline automatizado
./run_all.sh

# Pipeline parametrizable (Makefile)
make benchmark K="5000 10000 20000" SEARCH=random

# Pipeline avanzado con deteccion de hardware
python scripts/benchmark_pipeline.py --all-strategies --search random --k 5000 10000
```

---

## 10. Documentacion

La documentacion detallada se encuentra en el directorio `docs/`:

| Archivo | Contenido |
|---|---|
| `docs/index.md` | Indice de documentacion |
| `docs/01_problema.md` | Planteamiento del problema en profundidad |
| `docs/02_dataset.md` | Dataset: origen, generacion, estructura |
| `docs/03_modelo_matematico.md` | Modelo matematico detallado |
| `docs/04_python_secuencial.md` | Python secuencial: implementacion y analisis |
| `docs/05_python_multicore.md` | Python multicore: multiprocessing |
| `docs/06_c_secuencial.md` | C secuencial: implementacion nativa |
| `docs/07_c_openmp.md` | OpenMP: directivas, fork/join, compartido |
| `docs/08_c_mpi.md` | MPI: memoria distribuida, paso de mensajes |
| `docs/09_pycuda.md` | PyCUDA: GPU, kernel, reduccion |
| `docs/10_estrategias_busqueda.md` | Estrategias de busqueda: random, grid, hybrid |
| `docs/11_benchmark.md` | Benchmark: metricas, metodologia, graficas |
| `PROJECT.md` | Especificacion contractual completa del proyecto |