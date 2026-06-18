# Modelo matematico

## Score por item

Cada item biologico i (especie, gen, taxon) se describe mediante tres perfiles numericos en el intervalo [0, 1]:

- `T_i`: perfil taxonomico. Representa la abundancia relativa diferencial del item entre poblaciones.
- `S_i`: perfil ecologico-poblacional. Representa la correlacion del item con variables contextuales (edad, BMI).
- `F_i`: perfil funcional. Representa la presencia de marcadores funcionales (resistencia, virulencia, beneficio).

Dado un vector de pesos `W = (W1, W2, W3)`, el score del item i se define como:

```
P_i = W1 * T_i + W2 * S_i + W3 * F_i
```

En forma matricial, para todos los N items:

```
P = profiles @ W
```

donde `profiles` es una matriz de dimension `N x 3` y W es un vector columna de dimension 3.

## Score por muestra

Cada muestra j tiene asociado un vector de abundancias relativas `A[j, :]` de dimension N, donde `A_ji` es la contribucion del item i a la muestra j. El score de la muestra j es:

```
Score_j = sum_{i=1}^{N} A_ji * P_i
```

O en forma matricial:

```
Score = A @ P
```

donde A tiene dimensiones `n_muestras x N` y P tiene dimension N.

## Funcion objetivo

El objetivo es encontrar W que maximice el area bajo la curva ROC (AUC) entre los scores predichos y las etiquetas reales:

```
max_{W} AUC(y, Score(W))
```

El AUC mide la probabilidad de que una muestra enferma elegida al azar tenga un score mayor que una muestra sana elegida al azar. Su rango es:

- AUC = 0.5: el scoring no es mejor que el azar.
- AUC = 1.0: separacion perfecta (todos los enfermos tienen score mayor que todos los sanos).
- AUC > 0.7: tipicamente considerado como discriminacion aceptable.
- AUC > 0.9: discriminacion excelente.

## Consistencia (balanced accuracy)

Para un umbral de decision theta, la muestra j se clasifica como enferma si `Score_j > theta`. La consistencia se define como el balanced accuracy maximo sobre todos los umbrales posibles:

```
Consistencia = max_{theta} 0.5 * (TPR(theta) + TNR(theta))
```

Donde:
- TPR (sensibilidad) = proportion de enfermos correctamente clasificados.
- TNR (especificidad) = proporcion de sanos correctamente clasificados.

La consistencia complementa al AUC: mientras que el AUC mide la discriminacion global, la consistencia indica si existe un umbral que produzca una clasificacion util. Se considera satisfactorio un valor >= 0.8.

## El simplex de pesos

Los pesos deben cumplir dos restricciones:

1. **Suma unitaria:** W1 + W2 + W3 = 1. Esto asegura que el score combinado sea una combinacion convexa de los perfiles, preservando la escala.
2. **No negatividad:** Wi >= 0 para i = 1, 2, 3. Esto evita que un perfil contribuya negativamente al score, lo que no tendria sentido biologico (un perfil no deberia invertir la direccion de la clasificacion).

El conjunto de puntos que satisfacen ambas restricciones es un **simplex de dimension 2** embebido en R^3. Geometricamente, es un triangulo equilatero cuyos vertices son:

- (1, 0, 0): solo importa el perfil taxonomico.
- (0, 1, 0): solo importa el perfil ecologico.
- (0, 0, 1): solo importa el perfil funcional.

Los puntos interiores representan combinaciones de los tres perfiles.

## Muestreo con Dirichlet

La distribucion de Dirichlet con parametros `alpha = (1, 1, 1)` es uniforme sobre el simplex. Esto significa que cada muestra de `Dirichlet(1, 1, 1)` produce un vector W que:

- Cumple automaticamente W1 + W2 + W3 = 1.
- Cumple automaticamente Wi >= 0.
- Tiene igual probabilidad de caer en cualquier region del simplex.

Por esta razon, Dirichlet es la opcion natural para Random Search sobre el simplex. Otras opciones serian:

- Muestrear dos coordenadas uniformemente en [0, 1] y derivar la tercera como 1 - W1 - W2 (con rechazo si es negativa), pero esto produce una distribucion no uniforme.
- Usar metodos de optimizacion basados en gradiente, pero requieren que la funcion objetivo sea diferenciable, lo que no es el caso del AUC (depende de ordenamientos).

## Costo computacional

La evaluacion de la funcion objetivo para un solo W requiere:

| Operacion | Dimensiones | Costo (FLOPs) |
|---|---|---|
| P = profiles @ W | N x 3 @ 3 x 1 | ~ 3N |
| scores = A @ P | n_muestras x N @ N | ~ 2 * n_muestras * N |
| AUC(y, scores) | n_muestras | ~ n_muestras * log n_muestras (sort) + n_muestras |

Para el dataset estandar (n_muestras=2000, N=10000), cada evaluacion cuesta aproximadamente 40 millones de FLOPs. Con K=100000 candidatos, el costo total es del orden de 4x10^12 FLOPs, lo que justifica la necesidad de paralelizacion.
