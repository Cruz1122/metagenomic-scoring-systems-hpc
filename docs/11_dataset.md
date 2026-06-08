# 11. Dataset del proyecto: `cMD_CRC100_balanced`

## 1. Resumen ejecutivo

El dataset de trabajo del proyecto se denomina:

```text
cMD_CRC100_balanced
```

Su objetivo es alimentar el sistema de scoring metagenómico para clasificación binaria:

```text
healthy/control vs CRC
```

donde `CRC` significa cáncer colorrectal (*colorectal cancer*).

La versión incluida en el paquete actual es un **dataset de trabajo cMD-compatible**, congelado y reproducible, generado con una semilla fija (`seed = 42`) para que todas las implementaciones HPC trabajen sobre exactamente las mismas matrices. Está diseñado para validar lectura de datos, cálculo de scores, búsqueda de pesos, AUC, benchmarks, speedup y eficiencia.

La configuración final del dataset es:

```text
Dataset: cMD_CRC100_balanced
Tipo: dataset de trabajo compatible con curatedMetagenomicData
Tarea: clasificación binaria healthy/control vs CRC
Muestras totales: 100
Healthy/control: 50
CRC/disease: 50
Items/taxones: 500
Seed: 42
matrix_A: 100 x 500
labels: 100
profiles_TSF: 500 x 3
```

Para la presentación final del proyecto se buscará y preparará un **dataset real reconstruido desde curatedMetagenomicData**, usando el script `data/scripts/build_final_dataset.R` como ruta de reconstrucción en un entorno R/Bioconductor con acceso a `ExperimentHub`.

---

## 2. Relación con el modelo matemático del proyecto

El proyecto define un sistema de scoring donde cada muestra biológica se representa por una matriz de abundancias o contribuciones `A`, y cada item/taxón tiene tres perfiles:

| Símbolo | Perfil | Significado |
|---|---|---|
| `T_i` | Taxonómico | Señal diferencial del item entre sanos y enfermos. |
| `S_i` | Ecológico/poblacional | Asociación del item con variables contextuales o poblacionales. |
| `F_i` | Funcional | Resumen de funciones biológicas relevantes del item. |

El score por item se calcula como:

```text
P_i = W1*T_i + W2*S_i + W3*F_i
```

con:

```text
W = (W1, W2, W3)
W1 + W2 + W3 = 1
Wi >= 0
```

Después, el score por muestra se obtiene mediante:

```text
Score = A · P
```

En este dataset:

```text
A              -> matrix_A.npy / matrix_A.csv
P              -> resultado de combinar T,S,F con W
T,S,F          -> profiles_TSF.npy / item_profiles.csv
labels y       -> labels.npy / samples.csv
metadata       -> metadata.csv
funciones F    -> functional_matrix.csv + item_profiles.csv
```

Por tanto, el dataset está organizado directamente para el flujo computacional del proyecto:

```text
1. Cargar A, y, TSF.
2. Generar K candidatos de pesos W.
3. Para cada W, calcular P = TSF · W.
4. Calcular Score = A · P.
5. Medir AUC(y, Score).
6. Retornar W* con mayor AUC.
```

---

## 3. Fuente biológica de referencia

La fuente de referencia elegida para la versión real del dataset es **curatedMetagenomicData**, un paquete de Bioconductor que proporciona datos metagenómicos humanos curados y estandarizados.

Según la documentación oficial, `curatedMetagenomicData` proporciona datos humanos del microbioma ya procesados para análisis nuevos. Incluye:

- `relative_abundance`
- `gene_families`
- `marker_abundance`
- `marker_presence`
- `pathway_abundance`
- `pathway_coverage`
- metadata curada manualmente

También documenta que las abundancias taxonómicas bacterianas, fúngicas y arqueales se calcularon con **MetaPhlAn3**, mientras que el potencial funcional metabólico se calculó con **HUMAnN3**.

La motivación de usar `curatedMetagenomicData` es evitar partir de lecturas crudas FASTQ. El proyecto no necesita ensamblar genomas ni perfilar metagenomas desde cero. Necesita matrices ya procesadas, limpias y convertibles al esquema del scoring HPC.

Referencia principal:

```text
Pasolli E., Schiffer L., Manghi P. et al.
Accessible, curated metagenomic data through ExperimentHub.
Nature Methods 14, 1023–1024 (2017).
DOI: 10.1038/nmeth.4468
```

Esa publicación presenta `curatedMetagenomicData` como un recurso para hacer reutilizables datos públicos de metagenómica shotgun humana, reduciendo la fricción técnica asociada a descargar, reprocesar y armonizar estudios independientes.

---

## 4. Referencia biológica para la tarea CRC vs healthy

La tarea elegida para el proyecto es:

```text
healthy/control vs colorectal cancer (CRC)
```

La justificación biológica viene de estudios que han reportado firmas taxonómicas y funcionales asociadas al cáncer colorrectal en microbioma fecal.

Una referencia central es:

```text
Wirbel J. et al.
Meta-analysis of fecal metagenomes reveals global microbial signatures that are specific for colorectal cancer.
Nature Medicine 25, 679–689 (2019).
DOI: 10.1038/s41591-019-0406-6
```

Ese trabajo integró ocho estudios fecales shotgun metagenómicos de cáncer colorrectal con un total de `n = 768` muestras. El objetivo fue identificar firmas microbianas globales y generalizables asociadas a CRC, controlando confusores entre estudios. El estudio reportó un conjunto central de especies enriquecidas en CRC y señales funcionales asociadas, incluyendo genes relacionados con catabolismo de proteínas/mucina y cambios en rutas metabólicas vinculadas a ácidos biliares secundarios.

Esta referencia es relevante para el proyecto porque el modelo no usa solo abundancia taxonómica. También separa tres niveles de señal:

```text
T -> señal taxonómica
S -> señal ecológica/poblacional
F -> señal funcional
```

Esto está alineado con el tipo de análisis usado en estudios reales de microbioma y CRC: abundancias microbianas, metadatos/confusores y perfiles funcionales.

---

## 5. Tamaño del recurso original y tamaño usado en el proyecto

Hay dos niveles de referencia:

### 5.1. curatedMetagenomicData como recurso general

La documentación de `curatedMetagenomicData` versión 3.0.0 reporta:

```text
20,283 muestras
86 estudios
metadata curada manualmente
procesamiento estandarizado
```

Además, esa versión migró al procesamiento con:

```text
MetaPhlAn3
HUMAnN3
```

### 5.2. Metaanálisis CRC de referencia

El metaanálisis de Wirbel et al. 2019 reporta:

```text
8 estudios fecales shotgun metagenómicos
n = 768 muestras
29 especies significativamente enriquecidas en CRC
```

### 5.3. Tamaño adoptado para este proyecto HPC

Para el proyecto se usa una versión de trabajo balanceada:

```text
100 muestras totales
50 healthy/control
50 CRC
500 items/taxones
```

La razón de esta decisión es computacional y metodológica. Una versión de 10 muestras, como la mínima del enunciado original, sirve para demostrar el flujo, pero es demasiado pequeña para benchmarks HPC. Con `100 x 500` ya se tiene una matriz suficiente para probar lectura, vectorización, paralelización y comparación de implementaciones, especialmente cuando se evalúan muchos candidatos de pesos `W`.

El costo aproximado de la búsqueda escala como:

```text
O(K * M * N)
```

donde:

```text
K = número de candidatos W evaluados
M = número de muestras
N = número de items/taxones
```

Con este dataset:

```text
M = 100
N = 500
```

Si se usa:

```text
K = 100,000 o 1,000,000
```

ya se genera una carga suficiente para comparar:

```text
Python secuencial
Python multiprocessing
C/OpenMP
C/MPI
CUDA/PyCUDA
```

---

## 6. Estado actual del dataset incluido

El paquete actual contiene un dataset congelado, reproducible y cMD-compatible. Su estado es:

```text
Nombre: cMD_CRC100_balanced
Tipo: dataset de trabajo cMD-compatible
Generación: sintética-controlada con semilla fija
Seed: 42
Uso principal: desarrollo, validación y benchmark HPC
```

La ruta del paquete generado es:

```text
scoring_metagenomico_dataset/
```

Estructura:

```text
scoring_metagenomico_dataset/
├── data/
│   ├── samples.csv
│   ├── matrix_A.csv
│   ├── metadata.csv
│   ├── functional_matrix.csv
│   ├── item_profiles.csv
│   ├── item_mapping.csv
│   ├── matrix_A.npy
│   ├── labels.npy
│   ├── profiles_TSF.npy
│   ├── build_final_dataset.R
│   └── validate_dataset.py
├── README.md
├── manifest.json
└── SHA256SUMS.txt
```

Validación ejecutada:

```text
OK: dataset integrity validated
A shape: (100, 500)
labels: healthy=50, CRC=50
TSF: (500, 3)
```

---

## 7. Cómo se generó el dataset compatible

El dataset se generó siguiendo una simulación controlada con estructura metagenómica plausible. El objetivo fue producir datos que respeten el esquema matemático y computacional del proyecto.

El proceso fue:

### 7.1. Definición de clases

Se crearon 100 muestras:

```text
CTRL_001 ... CTRL_050 -> label 0 -> healthy/control
CRC_001  ... CRC_050  -> label 1 -> CRC/disease
```

La tabla `samples.csv` define esa asignación.

### 7.2. Definición de items/taxones

Se definieron 500 items:

```text
item_000
item_001
...
item_499
```

Cada item representa un taxón/especie. El archivo `item_mapping.csv` conserva el mapeo entre:

```text
item_id
taxon_name
feature_type
source_feature_id
crc_effect_log2fc
```

### 7.3. Generación de abundancias relativas

Se generó una matriz `A` de dimensión:

```text
100 x 500
```

Cada fila representa una muestra. Cada columna representa un taxón. Los valores son abundancias relativas normalizadas.

Propiedad principal:

```text
sum(A[j, :]) ≈ 1.0 para cada muestra j
```

Esto imita la naturaleza composicional de los perfiles metagenómicos: una muestra fecal o ambiental se representa como proporciones relativas de múltiples microorganismos.

Validación del dataset actual:

```text
A shape: (100, 500)
A dtype: float32
A min: 0.0
A max: 0.089373544
row_sums_min: 0.9999999
row_sums_max: 1.0000001
```

### 7.4. Inyección de señal diferencial CRC/healthy

No todos los taxones reciben la misma señal. El dataset introduce variación diferencial entre los grupos:

```text
algunos items quedan más asociados a CRC
algunos items quedan más asociados a healthy/control
muchos items tienen señal débil o moderada
```

Esto permite que el AUC no sea puramente aleatorio, pero tampoco obliga a que todas las variables separen perfectamente las clases.

### 7.5. Generación de metadatos

Se generaron metadatos por muestra para simular variables ecológicas/poblacionales:

```text
age
sex
bmi
diet
antibiotic_use
environment
location
country
study_name
disease
```

Estas variables permiten construir o aproximar el perfil `S_i`, que mide asociación del item con el contexto de la muestra.

Distribuciones del dataset actual:

```text
Sexo:
female: 56
male: 44

Dieta:
omnivore: 46
western: 20
mediterranean: 20
high_fiber: 14

Antibióticos:
no: 90
yes: 10

País:
Denmark: 23
Spain: 22
Italy: 22
France: 20
Germany: 13

Estudio de referencia:
ZellerG_2014: 55
FengQ_2015: 29
WirbelJ_2019: 16

Edad:
min: 35
max: 82
mean: 57.72

BMI:
min: 18.00
max: 33.49
mean: 25.05
```

### 7.6. Generación de marcadores funcionales

Para cada item se generó una fila funcional en `functional_matrix.csv` con cinco columnas:

```text
resistance_marker
virulence_marker
inflammation_marker
metabolic_marker
beneficial_marker
```

Conteos actuales:

```text
resistance_marker: 35 items
virulence_marker: 63 items
inflammation_marker: 79 items
metabolic_marker: 308 items
beneficial_marker: 107 items
```

Estos marcadores alimentan el perfil funcional `F_i`.

### 7.7. Construcción de perfiles T, S y F

Para cada item se construyeron tres perfiles:

```text
T = perfil taxonómico
S = perfil ecológico/poblacional
F = perfil funcional
```

Estos perfiles se guardan en:

```text
item_profiles.csv
profiles_TSF.npy
```

Dimensiones:

```text
item_profiles.csv: 500 x 5
profiles_TSF.npy: 500 x 3
```

Estadísticas actuales:

```text
T:
min: -0.883213
max: 0.778841
mean: -0.194095
positivos: 83
negativos: 417

S:
min: 0.0
max: 1.0
mean: 0.351041

F:
min: 0.0
max: 1.0
mean: 0.343679
```

Interpretación de `T`:

```text
T > 0  -> item más asociado a CRC
T < 0  -> item más asociado a healthy/control
T ≈ 0  -> item con baja diferencia entre grupos
```

---

## 8. Cómo funciona la seed

La semilla usada es:

```text
seed = 42
```

Una seed controla el generador pseudoaleatorio. Su función es garantizar reproducibilidad.

En términos prácticos:

```text
mismo algoritmo + mismos parámetros + misma seed = mismos archivos generados
```

La seed controla componentes como:

```text
variación individual entre muestras
asignación de metadatos sintéticos
ruido de abundancias
marcadores funcionales proxy
orden o composición de efectos simulados
```

La seed no convierte un dato simulado en una medición clínica real. Su función es que el experimento computacional sea repetible. Esto es clave para HPC porque permite comparar implementaciones sin que cambie la entrada.

Ejemplo conceptual en Python:

```python
import numpy as np

rng = np.random.default_rng(42)
x = rng.normal(size=5)
```

Si se repite el código con la misma seed, el vector `x` será idéntico. Si se cambia la seed, cambia la secuencia pseudoaleatoria.

Para el proyecto, fijar `seed = 42` permite:

```text
comparar tiempos entre implementaciones
repetir benchmarks
validar que Python, OpenMP, MPI y CUDA leen la misma entrada
hacer debugging determinista
reconstruir resultados
```

---

## 9. Los cinco subsets principales

El dataset se organiza en cinco tablas principales. Estas tablas son los cinco subconjuntos lógicos que alimentan el modelo.

---

### 9.1. `samples.csv`

Ruta:

```text
data/csv/samples.csv
```

Dimensión:

```text
100 filas x 3 columnas
```

Columnas:

```text
sample_id,label,group
```

Ejemplo conceptual:

```csv
sample_id,label,group
CTRL_001,0,healthy
CTRL_002,0,healthy
CRC_001,1,CRC
CRC_002,1,CRC
```

Contenido:

```text
50 healthy/control -> label 0
50 CRC/disease     -> label 1
```

Uso en el modelo:

`label` es la variable objetivo. Se usa para calcular AUC comparando los scores generados por el modelo contra la clase real.

Relación con archivos binarios:

```text
samples.csv -> labels.npy
```

---

### 9.2. `matrix_A.csv`

Ruta:

```text
data/csv/matrix_A.csv
```

Dimensión CSV:

```text
100 filas x 501 columnas
```

La primera columna es `sample_id`. Las 500 columnas restantes son items:

```text
item_000,item_001,...,item_499
```

Dimensión NumPy:

```text
matrix_A.npy -> shape (100, 500)
```

Contenido:

Cada fila es una muestra. Cada columna representa un taxón. Cada valor indica la abundancia relativa del item en esa muestra.

Ejemplo conceptual:

```csv
sample_id,item_000,item_001,item_002
CTRL_001,0.012,0.000,0.031
CRC_001,0.044,0.010,0.006
```

Uso en el modelo:

```text
A = matrix_A
Score = A · P
```

Validación principal:

```text
cada fila debe sumar aproximadamente 1.0
no debe haber valores negativos
shape esperado: 100 x 500
```

---

### 9.3. `metadata.csv`

Ruta:

```text
data/csv/metadata.csv
```

Dimensión:

```text
100 filas x 11 columnas
```

Columnas:

```text
sample_id
age
sex
bmi
diet
antibiotic_use
environment
location
country
study_name
disease
```

Contenido:

Variables poblacionales, clínicas o ecológicas por muestra.

Uso en el modelo:

Esta tabla alimenta el concepto de:

```text
S_i = ecological/population profile
```

La idea es medir si la abundancia de un item/taxón se asocia con variables de contexto como edad, BMI, sexo, dieta, antibióticos, país o estudio.

Importancia:

En microbioma, muchas asociaciones pueden estar afectadas por variables externas. Edad, dieta, uso de antibióticos, país o cohorte pueden cambiar la composición microbiana. Por eso `metadata.csv` es necesario para construir una señal `S_i` y no depender solo de abundancia taxonómica.

---

### 9.4. `functional_matrix.csv`

Ruta:

```text
data/csv/functional_matrix.csv
```

Dimensión:

```text
500 filas x 6 columnas
```

Columnas:

```text
item_id
resistance_marker
virulence_marker
inflammation_marker
metabolic_marker
beneficial_marker
```

Ejemplo conceptual:

```csv
item_id,resistance_marker,virulence_marker,inflammation_marker,metabolic_marker,beneficial_marker
item_000,0,0,1,1,0
item_001,0,1,0,1,0
```

Uso en el modelo:

Esta tabla alimenta:

```text
F_i = functional profile
```

Interpretación de columnas:

| Columna | Interpretación |
|---|---|
| `resistance_marker` | Señal funcional asociada a resistencia. |
| `virulence_marker` | Señal asociada a potencial de virulencia o patogenicidad. |
| `inflammation_marker` | Señal asociada a inflamación. |
| `metabolic_marker` | Señal de función metabólica relevante. |
| `beneficial_marker` | Señal funcional asociada a perfil protector o beneficioso. |

En la reconstrucción real desde `curatedMetagenomicData`, esta tabla debe derivarse de recursos funcionales como:

```text
gene_families
pathway_abundance
pathway_coverage
marker_presence
marker_abundance
```

---

### 9.5. `item_profiles.csv`

Ruta:

```text
data/csv/item_profiles.csv
```

Dimensión:

```text
500 filas x 5 columnas
```

Columnas:

```text
item_id,taxon_name,T,S,F
```

Ejemplo conceptual:

```csv
item_id,taxon_name,T,S,F
item_000,Fusobacterium nucleatum,0.591504,0.317046,0.544003
item_001,Peptostreptococcus stomatis,0.764724,0.505475,0.571068
```

Uso en el modelo:

Esta es la tabla más directa para construir `P`.

Para cada item:

```text
P_i = W1*T_i + W2*S_i + W3*F_i
```

Relación con NumPy:

```text
item_profiles.csv -> profiles_TSF.npy
```

`profiles_TSF.npy` tiene forma:

```text
(500, 3)
```

Columnas:

```text
columna 0 = T
columna 1 = S
columna 2 = F
```

Carga típica:

```python
import numpy as np

TSF = np.load("data/npy/profiles_TSF.npy")
T = TSF[:, 0]
S = TSF[:, 1]
F = TSF[:, 2]
```

---

## 10. Archivos auxiliares

Además de los cinco subsets principales, el paquete contiene archivos auxiliares.

### 10.1. `matrix_A.npy`

Misma matriz que `matrix_A.csv`, pero en formato binario NumPy.

```text
shape = (100, 500)
```

Uso:

```python
A = np.load("data/npy/matrix_A.npy")
```

Ventaja:

```text
carga más rápida
menos parsing de texto
mejor para benchmarks HPC
```

### 10.2. `labels.npy`

Vector binario de etiquetas.

```text
shape = (100,)
valores: 0 y 1
conteo: [50, 50]
```

Uso:

```python
y = np.load("data/npy/labels.npy")
```

### 10.3. `profiles_TSF.npy`

Matriz compacta de perfiles por item.

```text
shape = (500, 3)
```

Uso:

```python
profiles = np.load("data/npy/profiles_TSF.npy")
```

### 10.4. `item_mapping.csv`

Tabla de mapeo entre `item_id` y nombre taxonómico.

Columnas:

```text
item_id
taxon_name
feature_type
source_feature_id
crc_effect_log2fc
```

Uso:

```text
trazabilidad biológica
interpretación de resultados
ranking de taxones por T
informe técnico
```

### 10.5. `validate_dataset.py`

Script de validación de integridad.

Comprueba:

```text
matrix_A.npy shape (100, 500)
labels.npy shape (100,)
profiles_TSF.npy shape (500, 3)
50 muestras healthy
50 muestras CRC
filas de A suman aproximadamente 1
alineación básica entre CSV y NPY
```

Ejecución:

```bash
python data/scripts/validate_dataset.py .
```

Salida esperada:

```text
OK: dataset integrity validated
A shape: (100, 500); labels: healthy=50, CRC=50; TSF: (500, 3)
```

### 10.6. `build_final_dataset.R`

Script de reconstrucción desde `curatedMetagenomicData`.

Uso previsto:

```text
1. Instalar R + Bioconductor.
2. Instalar curatedMetagenomicData.
3. Consultar recursos CRC/control.
4. Seleccionar 50 controles y 50 CRC con seed=42.
5. Tomar 500 taxones/species-level items.
6. Exportar los cinco CSV.
7. Exportar NPY equivalentes.
```

Este script es la base para preparar el **dataset real reconstruido desde curatedMetagenomicData** que se buscará para la presentación final.

---

## 11. Por qué el dataset se comporta como datos metagenómicos plausibles

El dataset reproduce varias propiedades estructurales típicas de matrices metagenómicas procesadas:

### 11.1. Composicionalidad

Las abundancias por muestra suman aproximadamente 1:

```text
sum(A[j, :]) ≈ 1.0
```

Esto es consistente con matrices de abundancia relativa.

### 11.2. Alta dimensionalidad

Hay más variables que clases y una cantidad considerable de features:

```text
100 muestras
500 items
```

En estudios ómicos y metagenómicos es común trabajar con matrices donde el número de features es alto en relación con el número de muestras.

### 11.3. Señal parcial

No todos los taxones separan sanos y CRC. El dataset contiene:

```text
items con T positivo
items con T negativo
items con efecto moderado o bajo
```

Esto evita que el problema sea trivial.

### 11.4. Variabilidad individual

Muestras dentro de la misma clase no son idénticas. Existe ruido y variación de abundancia entre individuos.

### 11.5. Metadatos contextuales

Incluye variables como:

```text
edad
sexo
BMI
dieta
antibióticos
país
estudio
```

Esto permite construir `S_i`, una señal poblacional/ecológica por item.

### 11.6. Perfil funcional

Incluye marcadores funcionales por item, que alimentan `F_i`.

En la reconstrucción real, esta señal debe venir de recursos funcionales reales de `curatedMetagenomicData`, como `gene_families` o `pathway_abundance`.

---

## 12. Validaciones recomendadas antes de usarlo en benchmarks

Antes de ejecutar benchmarks, correr:

```bash
python data/scripts/validate_dataset.py .
```

Además, se recomienda crear un diagnóstico estadístico adicional con:

```text
data/diagnose_dataset.py
```

Ese script debería calcular:

```text
shape de A, y, TSF
conteo de labels
suma por fila de A
sparsity o proporción de ceros
histograma de abundancias
histograma de T, S, F
AUC con W = (1/3, 1/3, 1/3)
AUC con W optimizado
distribución de scores por grupo
top taxones CRC-like
top taxones healthy-like
PCA de A o de scores
```

Criterios esperados:

```text
A no debe tener valores negativos
cada fila de A debe sumar ~1
labels debe estar balanceado 50/50
TSF debe tener shape 500 x 3
S y F deben estar en [0, 1]
T debe tener valores positivos y negativos
AUC baseline debe ser mayor que azar si la señal está bien inyectada
```

---

## 13. Integración con el código HPC

El código debe cargar preferentemente los archivos binarios para evitar overhead innecesario de CSV.

Carga Python:

```python
import numpy as np

A = np.load("data/npy/matrix_A.npy")
y = np.load("data/npy/labels.npy")
profiles = np.load("data/npy/profiles_TSF.npy")
```

Evaluación de un candidato `W`:

```python
W = np.array([0.4, 0.4, 0.2], dtype=np.float32)
P = profiles @ W
scores = A @ P
```

Búsqueda aleatoria:

```python
rng = np.random.default_rng(42)
W_candidates = rng.dirichlet(np.ones(3), size=K)
```

Cada implementación debe mantener la misma lógica:

```text
Python secuencial -> baseline de exactitud
Python multicore  -> divide K entre procesos
OpenMP            -> divide K entre hilos
MPI               -> divide K entre ranks
CUDA/PyCUDA       -> un hilo evalúa uno o varios candidatos W
```

La comparación de rendimiento debe hacerse manteniendo constantes:

```text
mismo dataset
misma seed
mismo K
mismo número de items
misma función de AUC
```

---

## 14. Plan para la presentación: dataset real reconstruido desde curatedMetagenomicData

Para la presentación del proyecto se buscará y preparará un **dataset real reconstruido desde curatedMetagenomicData**.

La ruta esperada es:

```text
data/scripts/build_final_dataset.R
```

Objetivo del script:

```text
1. Descargar/consultar curatedMetagenomicData desde Bioconductor.
2. Identificar muestras healthy/control y CRC.
3. Priorizar estudios de CRC/metagenómica fecal.
4. Filtrar muestras con metadata suficiente.
5. Seleccionar 50 healthy/control y 50 CRC con seed=42.
6. Extraer abundancia relativa species-level.
7. Seleccionar los 500 taxones más abundantes o más informativos.
8. Construir matrix_A.csv y matrix_A.npy.
9. Construir samples.csv y labels.npy.
10. Construir metadata.csv.
11. Derivar funcional_matrix.csv desde gene families/pathways/markers.
12. Calcular item_profiles.csv con T, S y F.
13. Exportar profiles_TSF.npy.
14. Validar shapes, balance y row sums.
```

Criterios de selección para el dataset real:

```text
muestras fecales/metagenómicas
healthy/control claramente etiquetado
CRC claramente etiquetado
excluir adenoma si se quiere clasificación estricta healthy vs CRC
evitar muestras longitudinales duplicadas
preferir metadata con edad, sexo, país/estudio
usar abundancia relativa species-level
usar recursos funcionales HUMAnN3 cuando estén disponibles
mantener seed=42 para selección reproducible
```

La versión real reconstruida debe producir exactamente el mismo esquema de salida:

```text
samples.csv
matrix_A.csv
metadata.csv
functional_matrix.csv
item_profiles.csv
matrix_A.npy
labels.npy
profiles_TSF.npy
item_mapping.csv
```

---

## 15. Redacción técnica sugerida para el informe

Texto recomendado:

```text
Para el desarrollo y benchmarking inicial se utilizó el dataset cMD_CRC100_balanced, una versión de trabajo compatible con el esquema de curatedMetagenomicData. El dataset contiene 100 muestras balanceadas, 50 healthy/control y 50 CRC, descritas por 500 items taxonómicos. Cada muestra se representa mediante una matriz de abundancias relativas A, y cada item posee tres perfiles: taxonómico (T), ecológico/poblacional (S) y funcional (F). La seed global se fijó en 42 para garantizar reproducibilidad entre implementaciones.

Para la presentación final se buscará reconstruir una versión real desde curatedMetagenomicData/Bioconductor, preservando el mismo esquema de salida y los mismos archivos requeridos por el pipeline HPC.
```

---

## 16. Referencias

1. curatedMetagenomicData — sitio oficial del paquete.  
   https://waldronlab.io/curatedMetagenomicData/

2. curatedMetagenomicData — página oficial en Bioconductor.  
   https://www.bioconductor.org/packages/release/data/experiment/html/curatedMetagenomicData.html

3. Pasolli E., Schiffer L., Manghi P. et al. *Accessible, curated metagenomic data through ExperimentHub*. Nature Methods 14, 1023–1024 (2017).  
   https://doi.org/10.1038/nmeth.4468

4. curatedMetagenomicData — Version Three notes.  
   https://waldronlab.io/curatedMetagenomicData/articles/version-three.html

5. Wirbel J. et al. *Meta-analysis of fecal metagenomes reveals global microbial signatures that are specific for colorectal cancer*. Nature Medicine 25, 679–689 (2019).  
   https://doi.org/10.1038/s41591-019-0406-6
