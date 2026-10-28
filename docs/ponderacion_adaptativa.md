# Ponderación adaptativa por dificultad

Experimento sobre cómo combinar la señal de incumbencia (B0-v2) con la de texto (embeddings).
Vertical `j`, dev = 2023S2, métrica Δ MRR sobre el B0.

---

## La idea

El modelo de texto reordena el top-200 del B0-v2 mezclando dos señales con un peso. Hasta acá
ese peso era **fijo e igual para todas** las licitaciones. La hipótesis: el peso del texto no
debería ser fijo, porque las licitaciones no son todas iguales de predecibles.

- Licitación **predecible**: una empresa domina la oficina → la incumbencia ya acierta → **poco texto**.
- Licitación **abierta**: sin dueño claro → la incumbencia flota → **mucho texto**.

Entonces el peso del texto debería **adaptarse** a cada licitación según cuán predecible sea.

## Los pesos (la fórmula)

El puntaje de una empresa `e` para una licitación `q`, reordenando el shortlist del v2:

```
s(q,e) = (1 − α)/(K + r_cod)  +  α/(K + r_txt)
```

- `r_cod` = puesto de `e` por incumbencia (B0-v2); `r_txt` = puesto por similitud de texto.
- `K` = constante de amortiguación del rank fusion (óptimo K=5): cuanto más chica, más premia
  estar en el tope de cada lista.
- **`α` = el peso del texto**, entre 0 y 1:
  - `α = 0` → solo incumbencia (= el v2).
  - `α = 1` → solo texto (empíricamente malo, Δ MRR −0,037).
  - `α = 0,5` → el punto fijo óptimo (Δ MRR +0,052).

La ponderación adaptativa hace que **`α` varíe por licitación**, con una interpolación lineal
entre un piso y un techo:

```
α(q) = α_lo + (α_hi − α_lo) · dificultad(q)          con dificultad(q) ∈ [0, 1]
```

- **`α_lo` (piso):** el peso del texto cuando la licitación es predecible (dificultad ≈ 0).
- **`α_hi` (techo):** el peso cuando es abierta (dificultad ≈ 1).
- En el medio, `α` se desliza en línea recta entre los dos.

`α_lo` y `α_hi` son las **dos perillas** que se tunean en dev. No se fijan en 0 y 1 porque los
extremos rinden mal; la banda se elige donde el texto suma sin romper.

**Propiedad clave:** si `α_lo = α_hi`, se recupera el peso fijo. O sea, la versión adaptativa
**generaliza** la fija (la contiene como caso particular), así que en el barrido nunca puede
quedar peor en teoría: en el peor caso empata.

## La medida de dificultad

`dificultad(q)` mide cuán **predecible** es la licitación. Se mide con la **entropía** de la
distribución de puntajes del B0-v2 sobre las candidatas:

- distribución **picuda** (una empresa se lleva casi toda la masa) → entropía baja → predecible.
- distribución **chata** (todas parejas) → entropía alta → abierta.

Para diagnosticar dónde ayuda el texto, partimos las licitaciones en dos (con el B0):
`S_inc` = el B0 pone un ganador en su top-10 (fáciles); `S_new` = el resto (difíciles). Medición
con peso fijo: el texto **lastima muy poco las fáciles (−0,006)** y **ayuda mucho las difíciles
(+0,077)**. O sea, la heterogeneidad existe.

## Qué significa `q` (la familia de entropías de Rényi)

La entropía de Rényi es una **familia** parametrizada por un orden `q`:

```
H_q = 1/(1 − q) · log( Σ pᵢ^q )
```

El orden `q` controla **cuánto pesa cada estado según su probabilidad**:

| orden `q` | nombre | qué mira |
|---|---|---|
| q → 1 | Shannon | `−Σ p log p` — sensible a TODA la distribución, incluida la cola |
| q = 2 | colisión | `−log Σ p²` — pesa por `p²`, la cola de valores ínfimos casi no cuenta |
| q = 3 | — | pesa por `p³`, se enfoca AÚN MÁS en los dominantes |
| q → ∞ | min-entropía | `−log(máx p)` — solo importa el estado más grande |

Intuición: **a mayor `q`, la medida mira cada vez más solo a los que dominan** y menos a la cola
de estados poco probables. (Es como subir el exponente: `p²`, `p³`… agrandan la diferencia entre
el grande y los chicos.)

Por qué importa acá: la distribución `p_v2` tiene una **cola larga de valores casi iguales y
chiquititos** (todas las empresas reciben una masa mínima parecida). Shannon (`q=1`) es muy
sensible a esa cola, así que daba entropía alta **para casi todas** las licitaciones. Subir el
orden (`q ≥ 2`) **ignora esa cola** y mide la concentración entre los pocos que de verdad pelean.

## Resultados

| medida de dificultad | ¿discrimina? | mejor adaptativo | vs peso fijo |
|---|---|---|---|
| Shannon (q→1) | NO (mediana 0,90, saturada) | α_lo=α_hi=0,5 | empata +0,053 |
| Rényi-2 (q=2) | SÍ (0,13–0,60) | α_lo=0,3, α_hi=0,8 | empata +0,053 |
| Rényi-3 (q=3) | SÍ (0,10–0,54) | α_lo=α_hi=0,5 | empata +0,053 |

Con Rényi-2 el adaptativo **se comportó como debía** (recuperó las fáciles: S_inc pasó de −0,006
a +0,009), pero el **MRR total empató**, porque lo ganado en las fáciles lo cedió en las difíciles.

## Dos conceptos a no confundir

- **Saturar** (problema de la *medida*): da casi el mismo valor a todas → no discrimina. Shannon
  saturó; Rényi-2/3 no.
- **Techo estructural** (problema del *problema*): aunque la medida fuera perfecta, el adaptativo
  solo podría recuperar `(270/910)·0,006 ≈ +0,002` (bajo el ruido ±0,018), porque el daño que hay
  para arreglar (−0,006 en las fáciles) es ínfimo.

El giro: primero le echamos la culpa a la saturación. Rényi arregló la saturación y el método
**igual empató** → se confirma que la causa real es el **techo estructural**.

## Conclusión

Barrimos la familia de Rényi (`q = 1, 2, 3`) como medida de dificultad: **ninguna supera al peso
fijo**. La ponderación adaptativa queda como **experimento negativo bien documentado** — su techo
es estructural, no de la medida. El modelo vigente sigue siendo el de **peso fijo** (α=0,5, K=5,
Δ MRR +0,052 en dev `j`).

La física no se descarta: las medidas de concentración de Rényi (q=2, q=3, min-entropía) son
buenas **features para el reranker aprendido**, donde se pueden usar de forma no lineal, sin el
techo que las ata a un peso lineal.

*(Reproducibilidad: `dial_dificultad.py --fuente renyi --orden q`.)*
