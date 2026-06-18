# Transcripción del chat — Dataset metagenómico HPC

## Contexto general

Proyecto: `metagenomic-scoring-systems-hpc` / `scoring_metagenomico`.

Objetivo del bloque de conversación: intentar construir un dataset real con `curatedMetagenomicData`, diagnosticar errores `HTTP 403` en `ExperimentHub`, descartar esa fuente por bloqueo remoto y pasar a un dataset sintético escalado para benchmarks HPC.

---

## 1. Aclaración de REF/EVAL

### Usuario

> qué es REF en este contexto

### Asistente

REF es la cohorte de referencia.

En el proyecto se separan dos subconjuntos:

```text
REF  = muestras usadas para calcular los perfiles T y S
EVAL = muestras usadas para exportar matrix_A.npy y evaluar AUC
```

La razón es evitar leakage. Si `T` se calcula con las mismas muestras usadas para evaluar AUC, el perfil taxonómico termina mirando las etiquetas del conjunto evaluado.

Ejemplo con el censo disponible:

```text
healthy = 9742
CRC     = 651
```

Para un dataset de 1000 muestras exportadas:

```text
EVAL:
  500 healthy
  500 CRC

REF:
  151 healthy
  151 CRC
```

Entonces:

```text
T y S se calculan usando REF.
El benchmark se mide usando EVAL.
```

---

## 2. Decisión inicial después del censo

### Usuario

> Vale me gusta esa desición, qué hago a partir de acá (senso)

### Asistente

Se propone construir:

```text
REF:
  151 healthy
  151 CRC

EVAL:
  500 healthy
  500 CRC

Dataset exportado:
  cMD_CRC1000x500_balanced
```

Comando sugerido:

```bash
Rscript data/scripts/build_final_dataset.R \
  --dataset benchmark \
  --target-eval 1000 \
  --n-items 500 \
  --n-ref-per-class 151 \
  --ref-fraction 0.0 \
  --out-root data/processed \
  --seed 42
```

Validaciones esperadas:

```text
n_eval_samples = 1000
n_eval_healthy = 500
n_eval_crc = 500
n_ref_samples = 302
n_ref_healthy = 151
n_ref_crc = 151
n_items_final = 500
```

---

## 3. Primer error de descarga en curatedMetagenomicData

### Usuario

```text
Census after filters: healthy=9742, CRC=651, dedup_total=10393 
Census files written to /home/cruz1122/data/processed 
Selected REF=302 (151/class), EVAL=1000 (500/class) 
Downloading/returning relative_abundance for 1302 samples... 

Error: failed to load resource
  name: EH5458
  title: 2021-03-31.AsnicarF_2021.relative_abundance
  reason: 1 resources failed to download
Además: Avisos:
1: download failed
  web resource path: ‘https://experimenthub.bioconductor.org/fetch/5501’
  local file path: ‘/home/cruz1122/.cache/R/ExperimentHub/2a73f3a149816_5501’
  reason: HTTP 403 Forbidden.
```

### Asistente

Diagnóstico: el fallo no está en el código de scoring, sino en la descarga de un recurso de `ExperimentHub`.

Recomendación inicial: excluir `AsnicarF_2021` y volver a generar.

Parche conceptual:

```r
bad_studies <- c(
  "AsnicarF_2021"
)

md_clean <- md_clean[!(md_clean$study_name %in% bad_studies), , drop = FALSE]
```

---

## 4. Solicitud de script Bash para parchear

### Usuario

> dame el script en bash para hacer este parche

### Asistente

Se entrega un script Bash que parchea `data/scripts/build_final_dataset.R` para agregar:

```text
--exclude-studies AsnicarF_2021
```

y filtrar estudios problemáticos antes de seleccionar REF/EVAL.

Uso propuesto:

```bash
mkdir -p scripts
nano scripts/patch_exclude_bad_studies.sh
chmod +x scripts/patch_exclude_bad_studies.sh

./scripts/patch_exclude_bad_studies.sh data/scripts/build_final_dataset.R AsnicarF_2021
```

Luego ejecutar:

```bash
Rscript data/scripts/build_final_dataset.R \
  --dataset benchmark \
  --target-eval 1000 \
  --n-items 500 \
  --n-ref-per-class 151 \
  --ref-fraction 0.0 \
  --out-root data/processed \
  --seed 42 \
  --exclude-studies AsnicarF_2021
```

---

## 5. Solicitud de reemplazo completo del script R

### Usuario

> dame el script de python para poner en bash. el script que reemplace todo el contenido del script de R

### Asistente

Se propone un bloque de Python que escribe un nuevo `data/scripts/build_final_dataset.R` completo.

Características del script R generado:

```text
- Usa curatedMetagenomicData::sampleMetadata.
- Filtra healthy/control vs CRC.
- Excluye estudios problemáticos con --exclude-studies.
- Selecciona REF/EVAL balanceado.
- Descarga relative_abundance.
- Calcula T desde REF.
- Calcula S desde metadata REF.
- Calcula F como proxy taxonómico.
- Exporta CSV y NPY.
```

Comando de uso:

```bash
python - <<'PY'
from pathlib import Path
# ...contenido que reemplaza build_final_dataset.R...
PY
```

Generación esperada:

```bash
Rscript data/scripts/build_final_dataset.R \
  --dataset benchmark \
  --target-eval 1000 \
  --n-items 500 \
  --n-ref-per-class 151 \
  --out-root data/processed \
  --seed 42 \
  --functional-mode taxon_proxy \
  --exclude-studies AsnicarF_2021
```

---

## 6. Nuevos errores HTTP 403 por otros estudios

### Usuario

```text
Rscript data/scripts/build_final_dataset.R \
  --dataset benchmark \
  --target-eval 1000 \
  --n-items 500 \
  --n-ref-per-class 151 \
  --out-root data/processed \
  --seed 42 \
  --functional-mode taxon_proxy \
  --exclude-studies AsnicarF_2021,AsnicarF_2017,BackhedF_2015,BedarfJR_2017

Loading sampleMetadata...
Excluded unavailable/problematic studies: AsnicarF_2021, AsnicarF_2017, BackhedF_2015, BedarfJR_2017; removed 1542 samples
Census after filters: healthy=8168, CRC=666, dedup_total=8834
Census files written to /home/cruz1122/data/processed
Selected REF=302 (151/class), EVAL=1000 (500/class)
Downloading/returning relative_abundance for 1302 samples...

Error: Falló returnSamples(relative_abundance).
Mensaje original:
failed to load resource
  name: EH7103
  title: 2021-10-14.Bengtsson-PalmeJ_2015.relative_abundance
  reason: 1 resources failed to download

reason: HTTP 403 Forbidden.

Yo creo que literalmente todos van a salir 403
```

### Asistente

Diagnóstico: ya no conviene excluir estudios manualmente. Hay que hacer un preflight por estudio.

Se propone script:

```bash
scripts/probe_cmd_relative_abundance.sh
```

Objetivo:

```text
1. Probar relative_abundance por estudio.
2. Marcar OK/FAILED.
3. Escribir CSV, JSON y lista de estudios fallidos.
4. Construir dataset solo con estudios accesibles.
```

Ejecución:

```bash
./scripts/probe_cmd_relative_abundance.sh \
  data/processed \
  AsnicarF_2021,AsnicarF_2017,BackhedF_2015,BedarfJR_2017,Bengtsson-PalmeJ_2015
```

---

## 7. Probe confirma que todos fallan

### Usuario

```text
Probe terminado.
CSV:      data/processed/cMD_relative_abundance_probe_by_study.csv
JSON:     data/processed/cMD_relative_abundance_probe_summary.json
Exclude:  data/processed/cMD_failed_relative_abundance_studies.txt

OK after probe: healthy=0 CRC=0 total=0 max_balanced_no_ref=0

FAILED studies:
BritoIL_2016,BrooksB_2017,ChuDM_2017,CosteaPI_2017,DavidLA_2015,DeFilippisF_2019,DhakanDB_2019,FengQ_2015,FerrettiP_2018,GuptaA_2019,HallAB_2017,HanniganGD_2017,HansenLBS_2018,Heitz-BuschartA_2016,HMP_2012,HMP_2019_ibdmdb,HMP_2019_t2d,IaniroG_2022,IjazUZ_2017,JieZ_2017,KarlssonFH_2013,KaurK_2020,KeohaneDM_2020,KieserS_2018,KosticAD_2015,LeChatelierE_2013,LifeLinesDeep_2016,LiJ_2014,LiJ_2017,LiSS_2016,LiuW_2016,LokmerA_2019,LouisS_2016,MehtaRS_2018,MetaCardis_2020_a,NagySzakalD_2017,NielsenHB_2014,Obregon-TitoAJ_2015,PasolliE_2019,PehrssonE_2016,QinJ_2012,QinN_2014,RampelliS_2015,RaymondF_2016,RosaBA_2018,RubelMA_2020,SankaranarayananK_2015,SchirmerM_2016,ShaoY_2019,SmitsSA_2017,TettAJ_2019_a,TettAJ_2019_b,TettAJ_2019_c,ThomasAM_2018a,ThomasAM_2018b,ThomasAM_2019_c,VatanenT_2016,VincentC_2016,VogtmannE_2016,WampachL_2018,WirbelJ_2018,XieH_2016,YachidaS_2019,YassourM_2016,YassourM_2018,YeZ_2018,YuJ_2015,ZeeviD_2015,ZellerG_2014,ZhuF_2020
```

### Asistente

Diagnóstico final parcial:

```text
metadata disponible
matrices relative_abundance no descargables vía ExperimentHub
```

Se propone probar acceso directo:

```bash
Rscript - <<'RS'
suppressPackageStartupMessages({
  library(curatedMetagenomicData)
})

print(curatedMetagenomicData("ZellerG_2014.relative_abundance"))

res <- tryCatch(
  curatedMetagenomicData("ZellerG_2014.relative_abundance", dryrun = FALSE, rownames = "short"),
  error = function(e) e
)

print(res)
RS
```

Conclusión: si también falla, abandonar `returnSamples()` y probar TSV flat exports o Docker.

---

## 8. Opción A: CLI TSV

### Usuario

> vamos con opcion a no pasa nada

### Asistente

Se propone probar `curatedMetagenomicDataTerminal`.

Instalación:

```bash
Rscript -e 'utils::install.packages("BiocManager", repos = "https://cloud.r-project.org/")'
Rscript -e 'BiocManager::install("waldronlab/curatedMetagenomicDataTerminal", ask=FALSE, update=FALSE)'
Rscript -e 'curatedMetagenomicDataTerminal::install()'
```

Prueba:

```bash
curatedMetagenomicData "ZellerG_2014.relative_abundance" --dryrun

curatedMetagenomicData "ZellerG_2014.relative_abundance" \
  > data/raw_cmd_tsv/test/ZellerG_2014.relative_abundance.tsv
```

---

## 9. Error instalando la CLI por falta de remotes

### Usuario

```text
Installing github package(s) 'waldronlab/curatedMetagenomicDataTerminal'
Error: package 'remotes' not installed in library path(s)
install with 'BiocManager::install("remotes")'
Ejecución interrumpida
Error en loadNamespace(x): 
  no hay paquete llamado ‘curatedMetagenomicDataTerminal’
```

### Asistente

Solución:

```bash
Rscript -e 'install.packages("remotes", repos="https://cloud.r-project.org")'

Rscript -e 'BiocManager::install("waldronlab/curatedMetagenomicDataTerminal", ask=FALSE, update=FALSE)'
Rscript -e 'curatedMetagenomicDataTerminal::install()'
```

Validación:

```bash
which curatedMetagenomicData
curatedMetagenomicData --help
```

---

## 10. La CLI no queda en PATH

### Usuario

```text
which curatedMetagenomicData
curatedMetagenomicData --help
which: no curatedMetagenomicData in (...)
bash: curatedMetagenomicData: orden no encontrada
```

### Asistente

Diagnóstico: el paquete puede estar instalado, pero el ejecutable no quedó en `PATH`.

Buscar script:

```bash
find "$HOME/R" "$HOME" -type f -name curatedMetagenomicData 2>/dev/null | head -n 20
```

Agregar al `PATH`:

```bash
CMD_DIR="$(dirname "$(find "$HOME/R" -type f -path "*/curatedMetagenomicDataTerminal/exec/curatedMetagenomicData" 2>/dev/null | head -n 1)")"

chmod +x "$CMD_DIR/curatedMetagenomicData"
export PATH="$CMD_DIR:$PATH"
```

---

## 11. Error de symlink por permisos

### Usuario

```text
* DONE (curatedMetagenomicDataTerminal)
ln: fallo al crear el enlace simbólico '/usr/local/bin/curatedMetagenomicData': Permiso denegado
```

### Asistente

Diagnóstico: el paquete se instaló, pero no pudo crear symlink global en `/usr/local/bin`.

Solución con `sudo`:

```bash
CMD_BIN="$(find "$HOME/R" -type f -path "*/curatedMetagenomicDataTerminal/exec/curatedMetagenomicData" 2>/dev/null | head -n 1)"

chmod +x "$CMD_BIN"
sudo ln -sf "$CMD_BIN" /usr/local/bin/curatedMetagenomicData
```

Validación:

```bash
which curatedMetagenomicData
curatedMetagenomicData --help
```

---

## 12. Dryrun funciona, descarga real no

### Usuario

```text
curatedMetagenomicData "ZellerG_2014.relative_abundance" --dryrun
2021-03-31.ZellerG_2014.relative_abundance
```

### Asistente

El `dryrun` solo confirma que la CLI resuelve el nombre del recurso. Falta confirmar descarga real:

```bash
mkdir -p data/raw_cmd_tsv/test

curatedMetagenomicData "ZellerG_2014.relative_abundance" \
  > data/raw_cmd_tsv/test/ZellerG_2014.relative_abundance.tsv \
  2> data/raw_cmd_tsv/test/ZellerG_2014.relative_abundance.log
```

---

## 13. Descarga real prohibida

### Usuario

> forbidden

### Asistente

Conclusión: la CLI también usa el backend bloqueado. Estado:

```text
sampleMetadata: funciona
dryrun: funciona
relative_abundance real: bloqueado por HTTP 403
returnSamples(): bloqueado
curatedMetagenomicData(): bloqueado
CLI TSV: bloqueado
```

Se propone probar Docker.

---

## 14. Limpieza local antes de Docker

### Usuario

> pero si hacemos docker ya no importa nada de lo que descargue en mi máquina, verdad? Cómo borro todo eso desde R hasta lo último

### Asistente

Respuesta: con Docker no importa el R local, salvo los archivos montados con `-v "$PWD":/work`.

Script de limpieza segura:

```bash
cat > cleanup_r_cmd_host.sh <<'BASH'
#!/usr/bin/env bash
set -euo pipefail

rm -rf "$HOME/R"
rm -rf "$HOME/.cache/R"
rm -rf "$HOME/.cache/BiocFileCache"
rm -rf "$HOME/.cache/ExperimentHub"
rm -rf "$HOME/.local/share/R"
rm -rf "$HOME/.config/R"
rm -rf /tmp/Rtmp* 2>/dev/null || true

if [[ -L /usr/local/bin/curatedMetagenomicData || -f /usr/local/bin/curatedMetagenomicData ]]; then
  sudo rm -f /usr/local/bin/curatedMetagenomicData
fi

if [[ -d "$HOME/scoring_metagenomico" ]]; then
  cd "$HOME/scoring_metagenomico"
  rm -rf data/raw_cmd_tsv
  rm -f data/processed/cMD_relative_abundance_probe_by_study.csv
  rm -f data/processed/cMD_relative_abundance_probe_summary.json
  rm -f data/processed/cMD_failed_relative_abundance_studies.txt
fi
BASH

chmod +x cleanup_r_cmd_host.sh
./cleanup_r_cmd_host.sh
```

---

## 15. Imagen Docker de WaldronLab denegada

### Usuario

```text
docker pull ghcr.io/waldronlab/curatedmetagenomicanalyses:latest
Error response from daemon: error from registry: denied
denied
```

### Asistente

Cambio a imagen oficial de Bioconductor:

```bash
docker pull bioconductor/bioconductor_docker:RELEASE_3_23
```

Crear imagen propia:

```bash
cat > Dockerfile.cmd <<'DOCKER'
FROM bioconductor/bioconductor_docker:RELEASE_3_23

RUN Rscript -e 'BiocManager::install(c( \
  "curatedMetagenomicData", \
  "SummarizedExperiment", \
  "TreeSummarizedExperiment", \
  "S4Vectors", \
  "jsonlite" \
), ask=FALSE, update=FALSE)'
DOCKER

docker build -f Dockerfile.cmd -t scoring-cmd-bioc:3.23 .
```

---

## 16. Error de Docker con `-it` y heredoc

### Usuario

```text
docker run --rm -it \
  -v "$PWD":/work \
  -w /work \
  scoring-cmd-bioc:3.23 \
  Rscript - <<'RS'
...
RS

cannot attach stdin to a TTY-enabled container because stdin is not a terminal
```

### Asistente

Diagnóstico: no se debe usar `-t` con heredoc. Usar `-i` sin `-t`.

Comando corregido:

```bash
docker run --rm -i \
  -v "$PWD":/work \
  -w /work \
  scoring-cmd-bioc:3.23 \
  Rscript - <<'RS'
suppressPackageStartupMessages({
  library(curatedMetagenomicData)
})

res <- tryCatch(
  curatedMetagenomicData(
    "ZellerG_2014.relative_abundance",
    dryrun = FALSE,
    rownames = "short"
  ),
  error = function(e) e
)

print(res)
RS
```

---

## 17. Docker también da 403

### Usuario

> 403 también

### Asistente

Conclusión definitiva:

```text
R local              -> HTTP 403
CLI TSV             -> HTTP 403
Docker Bioconductor -> HTTP 403
```

Decisión: cerrar `curatedMetagenomicData` para este proyecto y pasar a dataset alternativo o sintético calibrado.

Se aclara que usar metadata real de cMD con abundancias simuladas sería metodológicamente falso.

---

## 18. Reclamo sobre las keys

### Usuario

> cómo es posible que me digas que no se requieren keys pero todo es 403

### Asistente

Corrección:

```text
No requiere una API key que el usuario tenga que configurar.
Pero eso no garantiza que el servidor remoto permita descargar el archivo.
```

Diferencia:

```text
sin API key requerida ≠ descarga garantizada
```

Diagnóstico honesto:

```text
No faltaba una key pública documentada.
Falló el acceso HTTP al recurso remoto.
El paquete permite consultar metadata, pero el backend de archivos niega la descarga.
```

Justificación para el informe:

```text
Se intentó usar curatedMetagenomicData como fuente primaria. La metadata y el dryrun funcionaron, pero la descarga de relative_abundance falló con HTTP 403 mediante R local, CLI TSV y contenedor Bioconductor. Por reproducibilidad, se cambió a una fuente alternativa descargable.
```

---

## 19. Cambio a dataset sintético escalado mínimo 2000 muestras

### Usuario

> En ese caso, generemos un dataset de MÍNIMO 2000 muestras
>
> Te adjunto el que tenemos, lo ideal es escalar para que generate-dataset.py pueda generarlo y no hayan problemas de distribución

### Asistente

Decisión: usar dataset sintético escalado, no semi-real falso.

Propuesta inicial:

```text
EVAL = 2000 muestras = 1000 healthy + 1000 CRC
REF  = 1000 muestras = 500 healthy + 500 CRC
items = 500
```

Separación:

```text
REF  -> calcula T y S
EVAL -> exporta matrix_A.npy, labels.npy, profiles_TSF.npy
```

Salida esperada:

```text
data/processed/synthetic_CRC2000x500_balanced/
├── csv/
├── npy/
└── dataset_manifest.json
```

Validación reportada:

```text
A_eval: (2000, 500) float32
labels: (2000,) healthy=1000 CRC=1000
profiles_TSF: (500, 3) float32
row_sum_min=0.99999988 row_sum_max=1.00000012
```

AUC sanity:

```text
T=0.7965
S=0.6455
F=0.7097
equal=0.7777
best≈0.8144
best_w≈[0.7874, 0.0726, 0.1400]
```

---

## 20. Solicitud final: 2000 muestras y 10000 items

### Usuario

> puedes contruir uno de 2000 muestras y 10000 items? Valida el script antes de dármelo

### Asistente

Se construye y valida un generador para:

```text
EVAL = 2000 muestras = 1000 healthy + 1000 CRC
REF  = 1000 muestras = 500 healthy + 500 CRC
items = 10000
```

Archivo entregado:

```text
generate_dataset_2000x10000.py
```

Validación reportada:

```text
A_eval: (2000, 10000) float32
labels: (2000,) int32
healthy=1000
CRC=1000
profiles_TSF: (10000, 3) float32

row_sum_min = 0.99999988
row_sum_max = 1.00000012
profiles_min = 0.10158631
profiles_max = 0.87307715
```

Sanity AUC:

```text
T_only_auc = 0.7312
S_only_auc = 0.6934
F_only_auc = 0.7225
equal_auc  = 0.7711
best_auc   = 0.7918
best_w     = [0.4902, 0.3496, 0.1602]
```

Comando de generación:

```bash
python data/scripts/generate_dataset.py \
  --name synthetic_CRC2000x10000_balanced \
  --n-eval 2000 \
  --n-ref 1000 \
  --n-items 10000 \
  --seed 42 \
  --quick-k 500 \
  --write-root-copy
```

Versión recomendada sin CSV gigante:

```bash
python data/scripts/generate_dataset.py \
  --name synthetic_CRC2000x10000_balanced \
  --n-eval 2000 \
  --n-ref 1000 \
  --n-items 10000 \
  --seed 42 \
  --quick-k 500 \
  --write-root-copy \
  --no-matrix-csv
```

Validación posterior:

```bash
python - <<'PY'
import numpy as np
from pathlib import Path

root = Path("data/processed/synthetic_CRC2000x10000_balanced/npy")

A = np.load(root / "matrix_A.npy")
y = np.load(root / "labels.npy")
P = np.load(root / "profiles_TSF.npy")

print("A:", A.shape, A.dtype)
print("labels:", y.shape, y.dtype, "healthy:", int((y == 0).sum()), "CRC:", int((y == 1).sum()))
print("profiles_TSF:", P.shape, P.dtype)
print("row sums:", float(A.sum(axis=1).min()), float(A.sum(axis=1).max()))
print("profiles range:", float(P.min()), float(P.max()))

assert A.shape == (2000, 10000)
assert y.shape == (2000,)
assert P.shape == (10000, 3)
assert int((y == 0).sum()) == 1000
assert int((y == 1).sum()) == 1000
assert np.allclose(A.sum(axis=1), 1.0, atol=1e-5)
assert np.all(P >= 0.0)
assert np.all(P <= 1.0)

print("OK")
PY
```

Nota técnica: para el sanity check se precalcula:

```text
Z = A @ profiles_TSF
```

Luego cada peso se evalúa como:

```text
scores = Z @ W
```

Esto acelera el sanity check, pero los benchmarks reales deben mantener la ruta original:

```text
P = profiles_TSF @ W
scores = A @ P
```

---

## 21. Solicitud de transcripción

### Usuario

> transcribe este chat en formato mark down

### Asistente

Se genera esta transcripción en Markdown.
