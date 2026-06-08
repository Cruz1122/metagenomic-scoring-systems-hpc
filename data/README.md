# Datos

`generate_data.py` (scaffold) produce al ejecutarse:

- `matrix_A.npy` / `matrix_A.csv`: matriz `A` de dimensión `10 x N` (contribución por muestra×item).
- `labels.npy` / `labels.csv`: etiquetas binarias (5 sanas = 0, 5 enfermas = 1).
- `profiles.npy` / `profiles.csv`: perfiles `[T, S, F]` por item — `N x 3`.
- `metadata.json`: seed, `N`, señal sintética y `true_w_synthetic` usado para inducir separabilidad.

> Estos archivos se regeneran con `make data` o `python data/generate_data.py`. No están trackeados en git (ver `.gitignore`).
