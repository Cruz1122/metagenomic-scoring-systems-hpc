# Dataset

## Origen y referencia biologica

El problema de clasificacion biologica que inspira este proyecto es la discriminacion entre pacientes con cancer colorrectal (CRC) y pacientes sanos (healthy) a partir de datos de abundancia relativa del microbioma intestinal.

La referencia bibliografica principal es:

> Haldar, Stein-Thoeringer, Borisov. *Interpreting Microbiome Relative Abundance Data Using Symbolic Regression*. arXiv:2410.16109, 2024.

Ese articulo utiliza datos de `curatedMetagenomicData` (71 estudios, 11,137 muestras healthy/CRC, 749 especies) para evaluar tecnicas de regresion simbolica en clasificacion de CRC. Aunque no fue posible acceder directamente a los datos reales (descarga HTTP 403), el problema biologico, la estructura de los datos y los taxones asociados a CRC documentados en el articulo sirvieron como referencia para disenar un dataset sintetico compatible.

El archivo [`fuente_real_dataset_sintetico_crc.md`](../fuente_real_dataset_sintetico_crc.md) documenta en detalle la fuente real, las especies reportadas, las limitaciones de acceso y las decisiones metodologicas adoptadas.

## Dataset sintetico

### Dimensiones

El dataset principal se denomina `synthetic_CRC2000x10000_balanced`. Sus componentes son:

| Archivo | Dimension | Tipo | Descripcion |
|---|---|---|---|
| `matrix_A.npy` | 2000 x 10000 | float32 | Matriz de contribucion: abundancia relativa de cada item por muestra |
| `labels.npy` | 2000 | int32 | Etiquetas: 0 (healthy), 1 (CRC) |
| `profiles_TSF.npy` | 10000 x 3 | float32 | Perfiles T (taxonomico), S (ecologico), F (funcional) por item |

### Composicion de clases

Dataset balanceado por construccion:

- 1000 muestras healthy (y = 0)
- 1000 muestras CRC (y = 1)
- Semilla: 42

El balanceo es intencional: el AUC no se ve afectado por el desbalance de clases en la misma medida que el accuracy, pero tener clases balanceadas simplifica el analisis y evita sesgos en la interpretacion de la consistencia.

## Generacion sintetica

El script `data/scripts/generate_dataset.py` (776 lineas) implementa el generador. El proceso tiene las siguientes etapas:

### 1. Abundancias base

Cada item i tiene una abundancia base muestreada de una distribucion gamma:

```
base_i ~ Gamma(shape=0.8, scale=1.0)
```

### 2. Efecto de clase

Se definen dos grupos de items:

- `CRC_enriched`: items cuya abundancia aumenta en presencia de CRC.
- `healthy_enriched`: items cuya abundancia aumenta en la poblacion sana.
- Items neutros: sin efecto de clase.

El factor de clase multiplica la abundancia base:

```
class_factor(item, label) = 
    1 + signal * t_strength   si item es CRC_enriched y label=CRC
    1 - signal * t_strength   si item es healthy_enriched y label=CRC
    1                          en otro caso
```

### 3. Efecto de metadata

Se genera metadata sintetica (edad, BMI) para cada muestra. La edad se muestrea uniformemente en [20, 80] y el BMI en [18, 35]. Se construye un riesgo sintetico:

```
metadata_risk = 0.65 * zscore(edad) + 0.35 * zscore(BMI)
```

Las abundancias se modulan por `1 + metadata_strength * metadata_risk` para items sensibles a metadata.

### 4. Ruido y zero-inflation

- Ruido lognormal: `exp(N(0, noise_sigma))` por muestra-item.
- Zero-inflation: una fraccion configurable de las entradas se fuerza a cero (parametro `zero_inflation`, default 0.05).

### 5. Normalizacion composicional

Cada fila de la matriz de abundancias se normaliza para que sume 1:

```
A[j, :] = raw_abundances[j, :] / sum(raw_abundances[j, :])
```

Esto produce un perfil de abundancia relativa, consistente con la naturaleza de los datos metagenomicos.

## Perfiles T, S, F

Los perfiles se calculan exclusivamente sobre la cohorte REF para evitar fuga de etiqueta.

### Perfil T (taxonomico)

Se calcula como el log fold-change de CRC vs. healthy en REF:

```
log2fc_i = log2((mean_crc_i + eps) / (mean_healthy_i + eps))
T_i = clip(0.5 + t_strength * tanh(log2fc_i / 2) / 2, 0, 1)
```

Interpretacion:
- `T_i > 0.5`: item mas abundante en CRC.
- `T_i < 0.5`: item mas abundante en healthy.
- `T_i = 0.5`: item neutro.

### Perfil S (ecologico-poblacional)

Mide la correlacion entre la abundancia del item y la metadata de riesgo sintetica:

```
S_i = 0.5 + 0.5 * corr(A_ref[:, i], metadata_risk_ref)
```

No usa las etiquetas y directamente, lo que evita fuga de informacion.

### Perfil F (funcional)

Representa un proxy funcional sintetico. Cada item puede portar marcadores funcionales (resistencia, virulencia, inflamacion, metabolicos, beneficos). El valor F_i se incrementa para items con marcadores de dano (virulencia, resistencia, inflamacion) y disminuye para items beneficiosos.

## Separacion REF/EVAL

La separacion entre REF y EVAL es una decision de diseno critica:

- **REF** (1000 muestras = 500 healthy + 500 CRC): se usa exclusivamente para estimar los perfiles T y S.
- **EVAL** (2000 muestras = 1000 healthy + 1000 CRC): se exporta como `matrix_A.npy` y `labels.npy`. Es el conjunto sobre el que se mide el AUC.

Si T se calculara sobre las mismas muestras que se evaluan, el perfil taxonomico podria codificar las etiquetas del conjunto de evaluacion, produciendo un AUC artificialmente alto. Esto se conoce como **label leakage** y es una de las causas mas frecuentes de resultados sobreoptimistas en pipelines bioinformaticos.

## Reproducibilidad

El generador acepta los siguientes parametros para controlar la generacion:

| Parametro | Default | Descripcion |
|---|---|---|
| `--seed` | 42 | Semilla del RNG |
| `--n-eval` | 2000 | Muestras en EVAL |
| `--n-ref` | 1000 | Muestras en REF |
| `--n-items` | 10000 | Numero de items (N) |
| `--signal` | 0.35 | Fuerza de la senal de clase |
| `--t-strength` | 0.80 | Peso del perfil T |
| `--metadata-strength` | 0.07 | Peso del efecto de metadata |
| `--zero-inflation` | 0.05 | Fraccion de ceros |
| `--noise-sigma` | 0.30 | Desviacion del ruido lognormal |

## Limitaciones metodologicas

1. El dataset es **sintetico**, no contiene abundancias reales de pacientes.
2. Los nombres de especies son del tipo `Genus synthetic_species_XXX`; no representan taxones reales observados.
3. La escala de 10000 items supera la escala real del articulo (749 especies) para generar carga computacional HPC.
4. Las conclusiones del proyecto son sobre rendimiento computacional, no sobre validez biologica.
