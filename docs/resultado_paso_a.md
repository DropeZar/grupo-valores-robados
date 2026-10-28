# Paso A — Texto por embeddings, fusionado con el B0-v2

> Sección para el informe. Primer experimento de texto que **supera al baseline** en la
> métrica del concurso (MRR), medido en dev.

## La propuesta

Representar cada licitación y cada empresa como vectores en un mismo espacio y rankear por
alineación, pero **construyendo los vectores con un modelo de embeddings pre-entrenado**
(`all-MiniLM-L6-v2`, 384 dimensiones) en vez de TF-IDF, y usando el texto como **afinador de
la punta sobre la incumbencia**, no como ranker independiente.

Tres decisiones concretas:

1. **Vector-builder = embeddings.** El primer intento con TF-IDF/SVD había rendido flojo
   (Δ MRR −0,006); los embeddings capturan el significado, no solo las palabras. Ejemplo real
   del modelo:

   ```
   sim("IBM hardware repair", "HP dell server maintenance") = 0.40   ← alto
   sim("IBM hardware repair", "auto body collision")        = 0.06   ← bajo
   ```

   Reconoce la capacidad compartida **sin que los textos compartan una sola palabra**.

2. **Documentos limpios + puente de códigos.**
   - licitación: `title` + descripción de su PSC + NAICS + nombre de oficina (se descarta la
     `description` cruda, llena de boilerplate: contactos, teléfonos, fechas).
   - empresa: `texto_awards` + descripciones de los PSC de su historial previo al semestre.

3. **Rerank del shortlist, no fusión total.** Se toma el **top-200 del B0-v2** y se **reordena**
   combinando el orden de incumbencia con el puntaje de texto (rank fusion). Así el texto afina
   la punta sin poder hundir a los ganadores que la incumbencia ya tenía bien.

## Cómo se mide

- Validación en **dev = 2023S2** (test 2024S1 sellado), vertical `j`, 910 licitaciones.
- Métrica: **mejora del MRR sobre el B0** (Δ MRR), con intervalo de confianza al 95%.
- Variantes comparadas: `v2` (solo códigos), `rerank` (v2 + texto, la apuesta), `rerank_txt`
  (reordenar el shortlist solo por texto), `fus_full` (fusión total, de contraste).

## Resultados (dev `j` 2023S2, B0 de referencia MRR = 0,174)

| modelo | MRR | Hit@5 | Δ MRR (IC95) | Hit@100 |
|---|---|---|---|---|
| v2 (códigos) | 0,176 | 0,233 | +0,001 (−0,004, +0,006) | 0,568 |
| **rerank (v2 + texto)** | **0,200** | **0,282** | **+0,025 (+0,006, +0,045)** | 0,610 |
| rerank_txt (texto puro) | 0,138 | 0,177 | −0,037 (−0,062, −0,011) | 0,573 |
| fus_full (fusión total) | 0,200 | 0,275 | +0,026 (+0,003, +0,049) | 0,604 |

## Lectura

1. **El texto por fin mueve el MRR.** El `rerank` da **Δ MRR = +0,025 sobre el B0, y es
   significativo** (el intervalo +0,006…+0,045 no cruza el cero). Es el primer modelo que
   supera al baseline en la métrica que define el concurso. El Hit@5 sube de 0,233 a **0,282**
   — justo en la punta, donde se decide el MRR.

2. **El salto fue el vector-builder.** Con TF-IDF/SVD habíamos sacado −0,006; con embeddings,
   +0,025. Mismo pipeline, distinta caja de "construir vectores".

3. **Fusionar es lo que paga, no el texto solo.** `rerank_txt` (reordenar por texto puro) da
   **−0,037, negativo**: el texto por sí mismo *empeora* el ranking. La ganancia viene de
   **combinar incumbencia + texto**. Esto confirma la tesis del trabajo: *códigos = cobertura,
   texto = discriminación; el MRR sale de mezclarlos.*

4. **El shortlist no cuesta nada.** `rerank` (top-200) y `fus_full` (fusión total) empatan
   (+0,025 vs +0,026), así que se adopta el `rerank` por ser más barato y controlado.

## Qué nos diferencia (medido)

Un enfoque de **texto standalone** (embeddings sin incumbencia, como el de otros grupos)
equivale, en el mejor caso, a `rerank_txt` → **Δ MRR negativo**. La diferencia que nos pone por
encima **no es el embedding** (esa herramienta es compartida) sino **la fusión sobre una base de
incumbencia**, más la medición rigurosa en dev con test sellado.

## Advertencias honestas

- Es sobre **dev** (2023S2); el test (2024S1) sigue sellado. Falta confirmar que aguanta.
- La ganancia es **modesta en absoluto** (+0,025), pero **real, significativa y en la dirección
  correcta** (la punta del ranking).
- Falta ver si se sostiene en las otras cuatro verticales.

## Próximos pasos

- **Tunear la fusión** (cuánto pesa el texto vs. el v2; el RRF plano no es el óptimo).
- **Dial de dificultad** (Paso C): dar más texto a las licitaciones "abiertas" y menos a las
  predecibles.
- **Confirmar en las otras verticales** (rd, d, r, it).
- Más adelante, **reranker aprendido** (Paso D) que combine todas las señales optimizando la punta.

## Reproducibilidad

- Script: `latente_emb.py` (embeddings + docs limpios/códigos + rerank del top-200 del v2).
- Los embeddings se cachean en `.emb_cache/`, así que los experimentos siguientes son
  instantáneos.
