### 3.3. Nivel 3 — CUDA: Aceleración GPU

- Cada hilo evalúa un candidato \(W_k\).
- Kernel usa **memoria compartida** para cachear filas de \(A\) por bloque.
- Transferencias **Host-to-Device** realizadas una única vez antes del kernel.
- Reducción del AUC con *reduction kernel* estándar.

\[
\text{Grid} =
\left\lceil
\frac{K}{\text{BLOCK\_SIZE}}
\right\rceil,
\qquad
\text{BLOCK\_SIZE} = 256
\]