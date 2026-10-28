# Resultados finales

## El modelo

Para cada licitación, reordenamos el **top-200 del B0-v2 optimizado** (historial con backoff de
códigos + pesos y τ tuneados por vertical) combinándolo con una **señal de texto por embeddings**:
representamos la licitación y cada empresa como vectores con `bge-small-en-v1.5`, y la similitud
coseno es la señal de capacidad. La combinación es una fusión de rangos con peso fijo:

```
s(q,e) = (1 − α)/(K + rango_historial)  +  α/(K + rango_texto)        α = 0,5 ; K = 5
```

## Validación

Cortes temporales por vertical: **train ≤ 2023-06-30 · dev = 2023S2 (ajuste) · test = 2024S1
(sellado, una sola pasada)**. Cada licitación usa solo información anterior a su fecha. Δ MRR sobre
el B0, con IC95 por bootstrap pareado.

## Resultado en el test sellado (2024S1)

| vertical | N | B0 MRR | modelo MRR | Hit@5 | Δ MRR | IC95 |
|---|---|---|---|---|---|---|
| rd | 57 | 0,091 | 0,145 | 0,263 | +0,054 | (−0,004, +0,119) |
| d | 161 | 0,090 | 0,202 | 0,292 | **+0,112** | (+0,068, +0,157) |
| j | 995 | 0,136 | 0,202 | 0,269 | **+0,067** | (+0,052, +0,081) |
| r | 468 | 0,194 | 0,266 | 0,353 | **+0,071** | (+0,046, +0,098) |
| it | 319 | 0,088 | 0,165 | 0,223 | **+0,077** | (+0,048, +0,106) |

**Mejora significativa en 4 de 5 verticales** (el intervalo no cruza el cero); `rd` queda a un
pelo (IC −0,004 a +0,119), por tener solo 57 licitaciones. **Promedio ponderado por N: ≈ +0,073.**

## La magnitud, en contexto

El MRR del B0 es bajo, así que en términos **relativos** la mejora es grande:

| vertical | B0 → modelo | mejora relativa |
|---|---|---|
| d | 0,090 → 0,202 | +124 % |
| it | 0,088 → 0,165 | +88 % |
| rd | 0,091 → 0,145 | +59 % |
| j | 0,136 → 0,202 | +49 % |
| r | 0,194 → 0,266 | +37 % |

Subimos el MRR entre **37 % y 124 %** sobre el baseline.

## Robustez: dev vs test (no es sobreajuste)

| vertical | dev (2023S2) | test (2024S1) |
|---|---|---|
| rd | +0,049 | +0,054 |
| d | +0,091 | +0,112 |
| j | +0,078 | +0,067 |
| r | +0,093 | +0,071 |
| it | +0,059 | +0,077 |

El promedio pasa de **+0,074** en dev a **+0,073** en test — prácticamente idénticos. Los
hiperparámetros elegidos en dev generalizan a datos que el modelo nunca vio: **no hay sobreajuste.**

## Por qué funciona

El B0 premia al proveedor histórico de cada oficina, así que entierra a empresas capaces que no
son "de la casa" de esa oficina (el 61 % de los ganadores enterrados nunca había ganado allí). El
texto rescata justo a esas: reconoce que el historial de una empresa habla del mismo trabajo que
pide la licitación, aunque no compartan códigos. En una frase: **códigos = cobertura, texto =
discriminación; el modelo los combina.**

## Lo que probamos y no sumó (también es resultado)

- **TF-IDF + SVD** como vector de texto: por debajo del baseline (−0,006). El salto vino de los
  **embeddings**.
- **Ponderación adaptativa por dificultad** (entropías de Rényi): ninguna superó al peso fijo; el
  margen era estructural, no de la medida.
- **Reranker aprendido** (MaxEnt y LightGBM): empató a la fusión a mano — la combinación simple ya
  era casi óptima.
- **Embedders más grandes** (bge-base, Gemma): no superaron a bge-small sobre texto corto de dominio.

## Conclusión

Partimos de "el B0 entierra ganadores" y llegamos a un modelo que **mejora el MRR sobre el B0 en
+0,073 (37–124 % relativo), significativo en 4 de 5 verticales, validado sobre el test sellado**,
con dev y test casi idénticos. Modelo final: **bge-small para la capacidad + B0-v2 optimizado para
el historial + fusión simple** — el más preciso y el más barato de correr (sin GPU).
