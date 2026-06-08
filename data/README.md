# Paquete de dataset `cMD_CRC100_balanced`

Este directorio contiene el dataset congelado para el proyecto de scoring metagenómico HPC.

## Estado importante

Las tablas incluidas son un **dataset de trabajo cMD-compatible**, congelado y reproducible, generado con `seed = 42` para uso directo en las implementaciones HPC. El entorno usado para crear este paquete no tiene acceso a Bioconductor/ExperimentHub en vivo, por lo que el paquete también incluye `data/scripts/build_final_dataset.R`, que es el script de reconstrucción para regenerar el mismo esquema desde recursos `curatedMetagenomicData` cuando se ejecute en un entorno completo de R/Bioconductor.

No confundir con datos FASTQ crudos. Este paquete está intencionalmente pre-procesado en las matrices de scoring esperadas por el proyecto.

## Identidad del dataset

- Nombre interno: `cMD_CRC100_balanced`
- Fuente prevista: `curatedMetagenomicData` / Bioconductor
- Tarea clínica: healthy/control vs cáncer colorrectal (CRC)
- Muestras: 100 total
  - 50 healthy/control
  - 50 CRC/disease
- Items/features: 500 items a nivel de especie
- Seed: 42
- Matriz de abundancias: abundancia relativa normalizada por fila

## Estructura de directorios

```text
data/
├── __init__.py
├── README.md
├── manifest.json
├── SHA256SUMS.txt
├── dataset_manifest.json
├── scripts/
│   ├── generate_synthetic_dataset.py
│   ├── build_final_dataset.R
│   └── validate_dataset.py
├── csv/
│   ├── samples.csv
│   ├── matrix_A.csv
│   ├── metadata.csv
│   ├── functional_matrix.csv
│   ├── item_profiles.csv
│   └── item_mapping.csv
└── npy/
    ├── matrix_A.npy
    ├── labels.npy
    └── profiles_TSF.npy
```

## Mapeo de esquema

### `csv/samples.csv`

Columnas:

```text
sample_id,label,group
```

- `label = 0`: healthy/control
- `label = 1`: CRC/disease

### `csv/matrix_A.csv` y `npy/matrix_A.npy`

Filas son muestras. Columnas son items taxonómicos:

```text
sample_id,item_000,item_001,...,item_499
```

La matriz numérica tiene forma:

```text
A: 100 x 500
```

Las filas están normalizadas para sumar aproximadamente 1.0.

### `csv/metadata.csv`

Contiene variables ecológicas/poblacionales:

```text
sample_id,age,sex,bmi,diet,antibiotic_use,environment,location,country,study_name,disease
```

Estas variables se usan para calcular el perfil de asociación ecológica/poblacional `S_i`.

### `csv/functional_matrix.csv`

Contiene marcadores funcionales proxy a nivel de item:

```text
item_id,resistance_marker,virulence_marker,inflammation_marker,metabolic_marker,beneficial_marker
```

Se usan para calcular `F_i`.

### `csv/item_profiles.csv`

Contiene los tres escalares de perfil requeridos por el modelo de scoring:

```text
item_id,taxon_name,T,S,F
```

- `T`: perfil taxonómico diferencial, derivado de la diferencia de abundancia CRC vs healthy.
- `S`: perfil de asociación ecológica/poblacional, derivado de la asociación abundancia-metadatos.
- `F`: perfil funcional, derivado de las categorías de marcadores funcionales.

### `npy/profiles_TSF.npy`

Matriz binaria NumPy con forma:

```text
500 x 3
```

Orden de columnas:

```text
T, S, F
```

## Compatibilidad matemática

El paquete está listo para el modelo de scoring del proyecto:

```text
P_i = W1*T_i + W2*S_i + W3*F_i
Score = A · P
Objective = maximize AUC(y, Score)
```

La carga de trabajo esperada para benchmarking no es solo `M x N`; es aproximadamente:

```text
O(K * M * N)
```

donde:

- `K`: número de vectores de peso candidatos W
- `M`: número de muestras = 100
- `N`: número de items = 500

Tamaños de benchmark recomendados:

```text
K = 100000 a 1000000
```

## Validación

Desde la raíz del repositorio:

```bash
python data/scripts/validate_dataset.py
```

Salida esperada:

```text
OK: dataset integrity validated
A shape: (100, 500); labels: healthy=50, CRC=50; TSF: (500, 3)
```

## Reconstrucción desde curatedMetagenomicData en vivo

```bash
Rscript data/scripts/build_final_dataset.R
```

Paquetes R/Bioconductor requeridos:

```r
BiocManager
curatedMetagenomicData
SummarizedExperiment
TreeSummarizedExperiment
reticulate
```

El script consulta recursos CRC/control, selecciona 50 controles y 50 CRC con `seed = 42`, conserva los 500 taxones principales por abundancia media relativa, calcula perfiles T/S/F y exporta archivos a `data/csv/` y `data/npy/`.

## Generación de datos sintéticos para desarrollo

Para pruebas, debugging y benchmarks preliminares sin depender de Bioconductor, se incluye:

```bash
python data/scripts/generate_synthetic_dataset.py \
    --out-dir data \
    --n-samples 100 \
    --n-items 500 \
    --seed 42 \
    --signal 2.2
```

O para pruebas rápidas:

```bash
python data/scripts/generate_synthetic_dataset.py \
    --out-dir data \
    --n-samples 20 \
    --n-items 50 \
    --seed 42
```

Este script genera datos sintéticos con el mismo esquema de archivos (CSV en `csv/`, NPY en `npy/`), pero no corresponde a mediciones clínicas reales. Está diseñado para desarrollo, validación de I/O y benchmarks HPC preliminares.

Parámetros clave:

| Parámetro | Default | Descripción |
|---|---|---|
| `--seed` | 42 | Controla reproducibilidad. Mismo seed + mismos parámetros = mismos archivos. |
| `--signal` | 2.2 | Controla separabilidad healthy vs CRC. Mayor valor = dataset más fácil. |
| `--concentration` | 240.0 | Controla variabilidad composicional tipo Dirichlet. Menor valor = muestras más ruidosas. |
| `--dropout-rate` | 0.30 | Controla proporción de ceros en la matriz de abundancias. |

## Nota importante

El dataset congelado en este paquete es apto para desarrollo HPC, validación de scoring, I/O, benchmarks y estructura de informes. Si el profesor exige explícitamente que cada valor de abundancia provenga de una cohorte pública real, ejecute `data/scripts/build_final_dataset.R` en R/Bioconductor y reemplace los archivos congelados. No presente la tabla congelada como un dump directo de una cohorte específica de cMD.
