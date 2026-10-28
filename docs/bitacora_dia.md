# Bitácora — Ranking de licitaciones (Grupo Valores Robados)

Resumen del trabajo del día. Vertical de referencia: `j`. Validación: dev = 2023S2,
test = 2024S1 (sellado). Métrica: Δ MRR sobre el baseline B0, con IC 95 %.

---

## Resumen ejecutivo

- Diagnosticamos **por qué falla el B0** y construimos un modelo que lo supera.
- **Mejor modelo hoy:** rerank del top-200 del B0-v2 con una señal de **texto por embeddings**,
  fusión tuneada (α=0,5, K=5) → **Δ MRR = +0,052 sobre el B0 en dev `j`** (significativo,
  IC95 (+0,035, +0,071)), con Hit@5 de 0,303.
- Probamos un **dial de dificultad** (idea propia, con entropía): **experimento negativo bien
  documentado** — el techo es estructural, no de la medida.
- Próximo paso: **reranker aprendido (MaxEnt)** con todas las señales como features.

---

## 1. El problema y el baseline B0

Dada una licitación, rankear las empresas candidatas (~30.000 por licitación) por probabilidad
de ganarla, usando solo información anterior a su publicación. El **B0** es el baseline dado:
rankea por **incumbencia** (quién ganó antes el mismo trabajo, sobre todo en la misma oficina),
sin leer texto. Se compite por **cuánto se mejora el MRR sobre el B0**.

## 2. Diagnóstico: el B0 entierra ganadores

En dev `j`, el B0 deja al ganador **fuera del top-100 el 47 % de las veces** (puesto mediano 73;
Hit@5 ~18 %). Mecanismo: el nivel más fuerte del B0 es "mismo PSC + misma oficina", así que
**entierra a empresas capaces que no son el proveedor histórico de esa oficina** (el 61 % de los
enterrados nunca ganó en esa oficina).

Ejemplo real — "IBM Hardware Repair" (oficina naval N00178): ganó *BETTER DIRECT*, cuyo historial
es puro "HP/Dell/IBM server maintenance" — el proveedor obvio — pero el B0 la puso en el puesto
**4.753** por no haber vendido antes a esa oficina. La señal estaba en el **texto**.

Conclusión: **códigos = cobertura (incumbencia); el texto es la señal de discriminación que falta.**

## 3. B0-v2: backoff de códigos

Agregamos tres niveles al B0 (`b0_v2.py`): familia de PSC (primeros 2 caracteres) y jerarquía de
oficina (subagencia). Resultado: **sube el recall de forma consistente** (Hit@100 +2–3 pp en las 5
verticales) **pero no mueve el MRR** — mete más ganadores a la lista, pero caen hondo.

## 4. El modelo de texto

### Factores latentes y embeddings
Representamos cada empresa (su `texto_awards`) y cada licitación (`title` + descripciones de sus
códigos) como **vectores**, y puntuamos por **alineación** (coseno = capacidad · requerimiento).
Los vectores los da un **embedder** pre-entrenado (`all-MiniLM-L6-v2`, 384 dimensiones):
tokeniza el texto, lo pasa por capas de atención que mezclan el contexto, promedia y entrega un
vector cuya **dirección codifica el significado**. Nota: esos 384 ejes son **genéricos** (del
modelo, no de nuestros datos) — útiles porque el conocimiento del lenguaje transfiere.

### Qué funcionó y qué no
- **TF-IDF + SVD (LSA)** (`latente.py`): **falló** (Δ MRR −0,006). El espacio latente casero no
  superó al texto crudo (saturado por ruido / primer componente dominante).
- **Embeddings + rerank del top-200 del v2** (`latente_emb.py`): **Δ MRR +0,025**, significativo.
  El texto por fin paga.
- **Tuneo de la fusión** (`tune_fusion.py`): barriendo el peso α y la constante K → **Δ MRR +0,052**
  en (α=0,5, K=5). Chequeamos que el óptimo no estuviera en el borde (extendimos K hacia abajo y
  en fino): **es un pico interior genuino**, resultado robusto en el vecindario.

### Las 4 variantes comparadas (dev `j`, B0 MRR = 0,174)
| variante | qué señal usa | Δ MRR |
|---|---|---|
| v2 | solo códigos | +0,001 |
| **rerank** | códigos + texto (fusión) | **+0,052** |
| rerank_txt | solo texto (sobre el shortlist) | −0,037 |
| fus_full | códigos + texto, sin shortlist | +0,052 |

Lección clave: **fusionar es lo que paga, no el texto solo.** El texto reordenado por sí mismo
(`rerank_txt`) **empeora** — suma cuando acompaña a la incumbencia, no cuando la reemplaza.

## 5. El dial de dificultad (experimento negativo)

Idea propia: el peso del texto α no debería ser fijo. En licitaciones **predecibles** (una empresa
domina) poco texto; en las **abiertas** mucho. Se mide la dificultad y se hace
`α(q) = α_lo + (α_hi − α_lo)·dificultad(q)`.

Diagnóstico con `S_inc` / `S_new` (el B0 pone o no un ganador en su top-10 = fáciles / difíciles):
el texto **lastima muy poco las fáciles (−0,006)** y **ayuda mucho las difíciles (+0,077)**.

Dos conceptos que conviene distinguir:
- **Saturar** (problema de la *medida*): da casi el mismo valor a todas → no discrimina. La
  **entropía de Shannon** saturó (dificultad ~0,90 para casi todas).
- **Techo estructural** (problema del *problema*): aunque la medida fuera perfecta, el dial solo
  podría recuperar `(270/910)·0,006 ≈ +0,002` (bajo el ruido ±0,018), porque el daño a arreglar
  es ínfimo.

Probamos **la familia de entropías de Rényi** como medida (idea de mecánica estadística):
- Shannon (q→1): **saturó** → el dial empató +0,053.
- Rényi-2 (colisión) y Rényi-3: **no saturaron** (pesan por pᵠ, ignoran la cola) → dificultad
  bien esparcida (0,1–0,6), el dial se comportó como debía (recuperó las fáciles) **pero igual
  empató +0,053**.

Resultado negativo completo y limpio: **barrimos la familia de Rényi; ninguna supera al peso fijo
→ el techo del dial es estructural, no de la medida.** La física no se tira: estas medidas de
concentración (Rényi-2/3, min-entropía) quedan como **buenas features para el reranker**.

## 6. Conceptos documentados (glosario para el equipo)

- **Incumbencia:** la señal de los códigos (quién gana en esa oficina para ese trabajo).
- **S_inc / S_new:** licitaciones donde el B0 ya acierta (fáciles) vs. donde falla (difíciles).
- **Rerankear aprendido (MaxEnt):** en vez de escribir la fórmula de fusión a mano, un modelo
  (log-lineal / LightGBM) **aprende** a combinar muchas features (v2, cada nivel, texto, Rényi,
  n_awards, recencia…) mirando quién ganó. Un modelo log-lineal de learning-to-rank **es** un
  modelo de máxima entropía (Boltzmann): score = −energía, pesos = acoplamientos aprendidos.
- **Física:** el producto punto capacidad·requerimiento es una energía de acoplamiento (tipo
  Ising); la fusión se puede escribir como distribución de Gibbs P ∝ exp(−E/T) con temperatura.

## 7. Estado actual y próximos pasos

- **Modelo vigente:** rerank con fusión fija (α=0,5, K=5) → Δ MRR +0,052 en dev `j`.
- **Próximos pasos:**
  1. **Reranker aprendido (MaxEnt/LightGBM)** sobre el shortlist, con todas las señales como
     features (incluidas las de Rényi). El que más puede mover la aguja.
  2. **Confirmar el +0,052 en las otras 4 verticales** (rd, d, r, it).
  3. **Medición final en test (2024S1)**, una sola vez, para el número honesto.

## 8. Metodología

- Cortes temporales por vertical: **train ≤ 2023-06-30 · dev = 2023S2 · test = 2024S1 (sellado)**.
- **Regla del futuro:** para una licitación en `t`, solo se usa información anterior a `t`.
- El embedder es externo (pre-entrenado) y el B0 tiene pesos fijos; en dev se ajustan solo
  hiperparámetros de fusión, no pesos aprendidos (eso recién llega con el reranker).

## 9. Scripts (reproducibilidad, en el repo)

- `b0_v2.py` — backoff de códigos.
- `latente.py` — texto TF-IDF/SVD (falló).
- `latente_emb.py` — texto por embeddings (MiniLM) + rerank del top-200. Cachea en `.emb_cache/`.
- `tune_fusion.py` — barrido de la fusión (α, K).
- `dial_dificultad.py` — dial de dificultad (`--fuente v2|renyi|hist`, `--orden q`).
- `viz_b0.py` / `viz_b0_multi.py` — figuras del diagnóstico del B0.
