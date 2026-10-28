# El modelo de machine learning

## Qué es

El problema es un **learning to rank**: dada una licitación, ordenar sus empresas candidatas
para que el ganador quede lo más arriba posible. No es clasificar una empresa como "gana / no
gana" en aislamiento, sino **ordenar** el conjunto de candidatas de cada licitación.

Nuestro sistema usa machine learning en dos lugares distintos, y conviene no confundirlos:

1. **Los embeddings de texto (`all-MiniLM-L6-v2`).** Es una red neuronal **pre-entrenada** (por
   terceros, con texto general de internet) que convierte una frase en un vector de 384 números
   cuya dirección codifica el significado. Acá el ML es **transfer learning**: aprovechamos
   conocimiento del lenguaje aprendido afuera, sin entrenar nada nosotros. Con eso medimos cuán
   parecido es lo que una empresa hizo antes a lo que pide la licitación.

2. **El combinador / reranker.** Es lo que decide el orden final a partir de varias señales. Lo
   probamos en dos versiones:
   - **Fusión a mano:** una fórmula fija que mezcla incumbencia y texto con un peso (α=0,5).
   - **Reranker aprendido:** un modelo (regresión logística = MaxEnt, y LightGBM con objetivo
     *lambdarank*) que **aprende de los datos** cómo combinar las señales, entrenado sobre
     licitaciones pasadas con la etiqueta de quién ganó. Features: incumbencia (puesto y puntaje
     del B0-v2), texto (similitud de embedding), concentración de la incumbencia (entropías de
     Rényi), `n_awards`, e interacciones.

El modelo log-lineal (MaxEnt) tiene, además, una lectura de mecánica estadística: su forma es
`P(gana) ∝ exp(Σ λᵢ·featureᵢ)` — una distribución de Boltzmann donde el puntaje es menos la
energía y los pesos aprendidos son los acoplamientos.

## Cómo se relaciona con lo anterior

El modelo **no reemplaza** lo que veníamos construyendo: se para **encima** y lo combina. Todo lo
anterior se convierte en una **señal (feature)** que entra al combinador:

```
B0  (incumbencia base, dado)
  └─> B0-v2  (le agregamos backoff de códigos)  →  da el SHORTLIST (top-200) y la señal de incumbencia
Texto  (embeddings MiniLM)                       →  da la señal de discriminación (coseno)
Concentración de la incumbencia (Rényi)          →  da la "dificultad" de la licitación
n_awards, etc.                                   →  señales de la empresa
        │
        ▼
   COMBINADOR  (fusión a mano  o  reranker aprendido)  →  ranking final
```

Dicho de otro modo: el B0-v2 aporta la **cobertura** (mete a los candidatos razonables en el
shortlist), el texto aporta la **discriminación** (distingue, entre esos candidatos, al que
realmente corresponde), y el combinador los junta. El reranker aprendido es la versión que
**aprende** esa combinación en vez de que la escribamos nosotros.

## Dónde quedó (honesto)

El reranker aprendido se construyó y se midió, pero **empató** a la fusión a mano (LightGBM
+0,055 vs fusión fija +0,051 en dev `j`, diferencia dentro del ruido; MaxEnt +0,048 pero con
mejor Hit@5). Conclusión: con este set de features, **la fusión simple ya era casi óptima**, así
que el modelo final que llevamos al test es esa fusión (que igual usa los embeddings — o sea,
igual es ML por el lado del texto). El reranker, además, dejó un dato útil: la feature más usada
por LightGBM fue `n_awards`.

## Resultados (test sellado 2024S1)

El modelo final (embeddings + fusión, α=0,5, K=5) evaluado una sola vez sobre el test:

| vertical | N | B0 MRR | modelo MRR | Δ MRR | IC95 |
|---|---|---|---|---|---|
| j | 995 | 0,136 | 0,178 | **+0,042** | (+0,028, +0,057) |
| rd | 57 | 0,091 | 0,123 | +0,032 | (−0,032, +0,099) |
| d | 161 | 0,090 | 0,163 | **+0,073** | (+0,033, +0,114) |
| r | 468 | 0,194 | 0,253 | **+0,058** | (+0,034, +0,083) |
| it | 319 | 0,088 | 0,138 | **+0,050** | (+0,027, +0,076) |

**Mejora significativa en 4 de 5 verticales** (el intervalo no cruza el cero); `rd` positivo pero
con pocos datos. Promedio ponderado ≈ **+0,049**, que en términos relativos es una mejora del
**30–80 %** sobre el B0. El resultado se sostiene del dev al test sellado, así que **no es
sobreajuste**: el modelo mejora de verdad sobre licitaciones futuras que nunca vio.

## En una frase

El machine learning entra como **un combinador que ordena a las empresas candidatas usando todas
las señales que fuimos construyendo** —incumbencia, texto, concentración, experiencia— y cuyo
aporte neto sobre el baseline es modesto en absoluto (~+0,05) pero significativo, robusto y
explicable.
