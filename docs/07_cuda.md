# CUDA y PyCUDA

CUDA asigna un candidato `W` por hilo GPU. Esto encaja con random search porque los candidatos son independientes.

Archivos:

```text
CUDA/scoring_kernel.cu   → kernel CUDA C + host
CUDA/scoring_pycuda.py   → wrapper Python con kernel embebido
CUDA/Makefile            → compilación con nvcc
```

## CUDA C (`scoring_kernel.cu`)

```bash
make -C CUDA scoring_cuda
./CUDA/scoring_cuda --k 100000 --seed 42 --data-dir data
```

- Lee datos desde CSV (como las implementaciones C).
- Genera candidatos en host con `std::mt19937_64` + distribución exponencial.
- El kernel `kernel()` lanza `K` hilos, cada uno calcula `scores = A @ (profiles @ W)` y su AUC.
- **Limitación**: la reducción del máximo se hace en host (`std::max_element`). La consistencia se reporta como `0.0` (scaffold).
- Un solo `cudaMemcpy` transfiere `A`, `profiles`, `y`, `W` al device. Cada candidato es independiente, no hay copias por iteración.

```cuda
kernel<<<(K+255)/256, 256>>>(dA, dP, dy, dW, dauc, ac, K);
```

## PyCUDA (`scoring_pycuda.py`)

```bash
python CUDA/scoring_pycuda.py --k 100000 --seed 42 --data-dir data
```

- Lee datos desde `.npy` (comparte formato con Python).
- Genera candidatos con `np.random.default_rng(seed).dirichlet(...)`.
- El kernel CUDA se embebe como string en el código Python.
- PyCUDA usa `float32` (vs `double` en CUDA C).
- El AUC se calcula en **host** (no en kernel) con `scoring_pycuda.py` línea 39.
- La consistencia se reporta como `0.0` (scaffold, como CUDA C).

## Reglas de memoria

- Copiar `A`, `profiles`, `y` y `W` una sola vez al device. Si copias por candidato, destruyes el rendimiento.
- Para versión final fuerte, agrega un kernel de reducción en GPU en lugar de reducir en host.
