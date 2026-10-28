# Resultados finales

## El modelo

Para cada licitación, reordenamos el **top-200 del B0-v2** (incumbencia con backoff de códigos)
combinándolo con una **señal de texto por embeddings**: representamos la licitación y cada empresa
como vectores (`all-MiniLM-L6-v2`, 384 dim) y medimos su alineación (coseno). La combinación es
una fusión de rangos con peso fijo:

```
s(q,e) = (1 − α)/(K + rango_incumbencia)  +  α/(K + rango_texto)        α = 0,5 ; K = 5
```

Los hiperparámetros (α, K) se eligieron en dev; el modelo de texto es pre-entrenado y la fusión
no aprende pesos de los datos.

## Validación

Cortes temporales por vertical, respetando que cada licitación solo usa información anterior a su
fecha: **train ≤ 2023-06-30 · dev = 2023S2 (ajuste) · test = 2024S1 (sellado, una sola pasada)**.
Métrica: variación del MRR sobre el baseline B0 (Δ MRR), con intervalos de confianza al 95 %.

## Resultado en el test sellado (2024S1)

| vertical | N | B0 MRR | modelo MRR | Hit@5 | Δ MRR | IC95 |
|---|---|---|---|---|---|---|
| j | 995 | 0,136 | 0,178 | 0,236 | **+0,042** | (+0,028, +0,057) |
| rd | 57 | 0,091 | 0,123 | 0,228 | +0,032 | (−0,032, +0,099) |
| d | 161 | 0,090 | 0,163 | 0,242 | **+0,073** | (+0,033, +0,114) |
| r | 468 | 0,194 | 0,253 | 0,331 | **+0,058** | (+0,034, +0,083) |
| it | 319 | 0,088 | 0,138 | 0,191 | **+0,050** | (+0,027, +0,076) |

**Mejora significativa en 4 de 5 verticales** (el intervalo no cruza el cero). `rd` da positivo
(+0,032) pero no significativo, por tener solo 57 licitaciones en el test. **Promedio ponderado
por N: ≈ +0,049.**

## La magnitud, en contexto

El Δ MRR absoluto (~+0,05) parece chico, pero el MRR del B0 ya es bajo. En términos **relativos**
la mejora es grande:

| vertical | B0 → modelo | mejora relativa |
|---|---|---|
| d | 0,090 → 0,163 | +81 % |
| it | 0,088 → 0,138 | +57 % |
| rd | 0,091 → 0,123 | +35 % |
| j | 0,136 → 0,178 | +31 % |
| r | 0,194 → 0,253 | +30 % |

Subimos el MRR entre **30 % y 80 %** sobre el baseline. Una mejora de este tamaño, significativa
y estable, es lo que se espera de una mejora legítima contra un baseline fuerte; un salto enorme
sería sospechoso de fuga de datos.

## Robustez: dev vs test

| vertical | dev (2023S2) | test (2024S1) |
|---|---|---|
| j | +0,052 | +0,042 |
| rd | +0,012 | +0,032 |
| d | +0,051 | +0,073 |
| r | +0,081 | +0,058 |
| it | +0,037 | +0,050 |

Algunas suben y otras bajan, pero **todas siguen positivas** — variación normal entre dev y test,
sin colapso. Que el +0,05 de dev se sostenga en el test sellado confirma que **no era sobreajuste**:
el modelo mejora de verdad sobre licitaciones futuras que nunca vio.

## Por qué funciona

El B0 premia al proveedor histórico de cada oficina, así que **entierra a empresas capaces que no
son "de la casa"** de esa oficina (el 61 % de los ganadores enterrados nunca había ganado allí). El
texto rescata justo a esas: reconoce que el historial de una empresa habla del mismo trabajo que
pide la licitación, aunque no compartan códigos. Medido: el texto **casi no toca** las licitaciones
que el B0 ya resolvía y **aporta fuerte** donde el B0 fallaba (Δ MRR +0,077 en ese subconjunto en
dev `j`). En una frase: **códigos = cobertura, texto = discriminación; el modelo los combina.**

## Lo que probamos y no sumó (también es resultado)

- **TF-IDF + SVD** como vector de texto: por debajo del baseline (−0,006). El salto vino de cambiar
  a **embeddings**.
- **Ponderación adaptativa por dificultad** (peso del texto variable por licitación, medido con
  entropía de Rényi): barriendo la familia de Rényi, **ninguna superó al peso fijo** — el margen
  era estructural, no de la medida.
- **Reranker aprendido** (MaxEnt y LightGBM con más features): **empató** la fusión a mano
  (LightGBM +0,055 vs fija +0,051 en dev, dentro del ruido), confirmando que la fusión simple ya
  era casi óptima. Dato útil: `n_awards` resultó la feature más usada por LightGBM.

## Conclusión

Partimos de un diagnóstico ("el B0 entierra ganadores") y llegamos a un modelo que **mejora el MRR
sobre el B0 de forma significativa y robusta en 4 de 5 verticales, validado sobre el test sellado**
(`rd` positivo pero con pocos datos). El número honesto es **≈ +0,05 de Δ MRR (30–80 % relativo)**.
Resultado modesto en absoluto, pero real, significativo, estable y con una explicación clara de por
qué funciona.
