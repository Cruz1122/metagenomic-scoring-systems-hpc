# Modelo de scoring

Variables:

- `A`: matriz `10 x N`.
- `profiles`: matriz `N x 3`, columnas `[T,S,F]`.
- `y`: etiquetas binarias.
- `W`: pesos en el simplex.

Score por item:

```text
P_i = W1*T_i + W2*S_i + W3*F_i
```

Score por muestra:

```python
scores = A @ (profiles @ W)
```

Objetivo:

```text
max_W AUC(y, scores(W))
```

El AUC se implementa por conteo exacto de pares positivo-negativo. Con 5 sanos y 5 enfermos hay 25 pares. La fórmula usada en código:

```python
auc = ((pos[:,None] > neg[None,:]).sum() + 0.5*(pos[:,None] == neg[None,:]).sum()) / (pos.size * neg.size)
```

La consistencia se reporta como balanced accuracy máxima normalizada en `[0,1]`. Se evalúan todos los thresholds relevantes (puntos medios entre scores únicos) y se retorna el mejor:

```python
best = max(0.5*(sensibilidad + especificidad)) para todo threshold θ
```

Una consistencia ≥ 0.8 se considera satisfactoria. Buscar alejamiento del azar (0.5) con AUC > 0.7 suele indicar que los pesos encontrados separan bien los grupos.
