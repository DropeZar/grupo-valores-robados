# El reranker aprendido (MaxEnt)

Una de las formas de combinar las señales para producir el ranking final. En vez de escribir la
fórmula de fusión a mano, un modelo **aprende** cómo combinar las features mirando quién ganó en
el pasado. MaxEnt = regresión logística = modelo log-lineal.

## Qué hace, en una frase

Construye **su propio puntaje** para cada par (licitación, empresa) como una **suma pesada de
features**, convierte ese puntaje en una probabilidad de ganar, y ordena a las candidatas por esa
probabilidad. No parte de la fusión a mano ni la retoca: arma un puntaje nuevo desde las features
(que incluyen los mismos ingredientes crudos que usaba la fusión, más otros).

## Cómo funciona, paso a paso

Para cada par (licitación `q`, empresa `e` del top-200):

1. **Features.** Un vector de números que describe ese par (ver el diccionario abajo).

2. **Puntaje lineal (la "energía").** Combina las features con una suma pesada:
   ```
   z(q,e) = λ₀ + λ₁·feature₁ + λ₂·feature₂ + … + λₙ·featureₙ
   ```
   Los `λ` (los pesos) son lo único que el modelo aprende.

3. **Puntaje → probabilidad (sigmoide).**
   ```
   P(gana | q,e) = 1 / (1 + e^(−z)) = sigmoid(z)
   ```
   `z` alto → probabilidad cerca de 1; `z` bajo → cerca de 0. Es la forma de Boltzmann (`P ∝ e^z`).

4. **Rankear.** Se calcula `P` de cada candidata de la licitación y se las ordena de mayor a menor.

## Qué aprende

Se le dan muchos ejemplos: pares (q,e) con etiqueta `ganó` (1) o `no ganó` (0), de las
licitaciones de train. El modelo **ajusta los `λ`** para que `P(gana)` sea alta en los ganadores
reales y baja en el resto (maximiza la verosimilitud / minimiza el log-loss). Es un problema
convexo: hay una solución única, sin mínimos locales.

El nombre "máxima entropía" (Jaynes): entre todas las distribuciones que respetan lo observado en
los datos, se elige la **más desinformada posible**; la solución de ese principio tiene
exactamente esta forma exponencial/logística.

## Diccionario de features

Cada fila es una feature. "Por candidata" = cambia entre las empresas de una misma licitación;
"por licitación" = es la misma para las 200 candidatas de esa licitación.

| feature | qué es | alcance |
|---|---|---|
| `rank_v2` | puesto de la empresa por **incumbencia** (ranking del B0-v2) dentro del shortlist, 1 = la de más incumbencia | por candidata |
| `p_v2` | el **puntaje** de incumbencia del B0-v2 (cuánta "masa" de probabilidad le asigna) | por candidata |
| `sim` | **similitud de texto**: coseno entre el embedding de la licitación y el de la empresa | por candidata |
| `rank_txt` | puesto de la empresa por `sim` dentro del shortlist, 1 = la más parecida por texto | por candidata |
| `n_awards` | cantidad de **contratos previos** de la empresa en la familia (tamaño / experiencia) | por candidata |
| `renyi2` | **entropía de Rényi-2** de la distribución `p_v2` de la licitación: cuán concentrada está la incumbencia (0 = una empresa domina → predecible; 1 = chata → abierta) | por licitación |
| `renyi3` | **entropía de Rényi-3** de `p_v2`: lo mismo, pero enfocándose aún más en los dominantes | por licitación |
| `min_ent` | **min-entropía** de `p_v2` (`−log(máx p)`): mira solo al candidato más fuerte | por licitación |
| `sim_x_r2` | **interacción texto × dificultad** (`sim · renyi2`): deja que el modelo pese el texto distinto según cuán abierta sea la licitación | mixta |
| `pv2_x_r2` | **interacción incumbencia × dificultad** (`p_v2 · renyi2`) | mixta |

Las tres de Rényi y la min-entropía son las medidas de **concentración** de la incumbencia (la
"dificultad" de la licitación). Entran como features para que el modelo pueda, por ejemplo,
aprender a pesar más el texto cuando la incumbencia está chata — el "dial", pero aprendido, vía
las interacciones `sim_x_r2` y `pv2_x_r2`.

## Los pesos que aprendió (en `j`, coeficientes estandarizados)

```
rank_v2    -0.986   ← menos puesto de incumbencia (= más arriba) sube MUCHO la prob. La señal dominante.
rank_txt   -0.568   ← mejor puesto por texto → más probable
sim        +0.413   ← más similitud de texto → más probable
pv2_x_r2   +0.353   ← interacción incumbencia × dificultad (el "dial" aprendido)
sim_x_r2   +0.207   ← interacción texto × dificultad
renyi2     -0.145
p_v2       -0.111
min_ent    +0.102
n_awards   +0.029
renyi3     -0.006
```

El modelo **descubrió solo** que la incumbencia (`rank_v2`) es lo que más empuja, el texto
segundo, y que las interacciones con la dificultad aportan algo. Esos `λ` son, en lenguaje de
física, los **acoplamientos** de Boltzmann: cuánto tira cada feature hacia "ganar".

## Relación con la fusión a mano

Las dos combinan `rank_v2` y `rank_txt`, pero de forma distinta:

| | fusión a mano | MaxEnt |
|---|---|---|
| forma | `(1−α)/(K+rank_v2) + α/(K+rank_txt)` | `λ₁·rank_v2 + λ₂·rank_txt + …` (lineal) |
| pesos | puestos a mano (α, K) | **aprendidos de los datos** |
| nº de señales | 2 | 10 (con `n_awards`, Rényi, interacciones) |

MaxEnt es un combinador **más general y aprendido** de los mismos ingredientes (más algunos
extra), no una modificación del `s`.

## Resultado

En dev `j`, MaxEnt dio **Δ MRR +0,048** (vs +0,051 de la fusión a mano), con **mejor Hit@5
(0,321 vs 0,303)**: ordena mejor el top-5 aunque el MRR global quede un pelín abajo. En la
práctica **empató** a la fusión (dentro del ruido), lo que confirma que la combinación simple ya
era casi óptima con estas features. Por eso el modelo final llevado al test es la fusión a mano;
MaxEnt queda como la versión interpretable (los `λ` cuentan qué pesa) y la base teórica
(máxima entropía / Boltzmann).
