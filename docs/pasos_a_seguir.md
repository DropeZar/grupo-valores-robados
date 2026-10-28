# De dónde salen los vectores y pasos a seguir

> Sección para el informe. Continúa la de factores latentes: explica cómo se construyen los
> vectores de capacidad/requerimiento y el plan de trabajo, con el punto de control (MRR en
> dev) en cada paso.

## De dónde salen los vectores

La idea del modelo es representar **cada empresa** (su capacidad) y **cada licitación** (su
requerimiento) como vectores en un **mismo espacio**, y rankear por qué tan alineados están
(producto punto / coseno). La pregunta es: ¿cómo convertimos texto en esos vectores?

La cadena siempre es la misma:

```
texto  →  [algo lo convierte en números]  →  vector  →  comparar por alineación
```

Hay tres formas de hacer ese "algo". Las dos primeras son alternativas; la tercera las combina.

### Qué texto entra (los dos lados)

- **Empresa:** su `texto_awards` (descripciones de sus contratos pasados, ya "congeladas" al
  semestre → sin fuga de futuro) **+** las descripciones de sus códigos (PSC/NAICS).
- **Licitación:** su `title` **+** las descripciones de PSC/NAICS/oficina, bajando o limpiando
  la `description` cruda (está llena de boilerplate: contactos, teléfonos, fechas).

Las descripciones de los **códigos** son el **vocabulario estandarizado que ambos lados
comparten** — el puente que empareja "lo que se pide" con "lo que la empresa hizo".

### Ruta A — TF-IDF + SVD (LSA)

1. **TF-IDF** convierte cada texto en un vector de palabras, pesando cada palabra por cuán
   informativa es (las comunes como "the", "maintenance" pesan poco; las raras como "ibm",
   "lokomat" pesan mucho).
2. **SVD** comprime esos vectores enormes (~40.000 palabras) a ~200 **factores latentes**,
   agrupando palabras que aparecen juntas en un mismo eje (p. ej. "server, dell, warranty"
   → eje "IT").

Ventajas: baratísimo, corre en cualquier lado, **interpretable** (se ve qué palabras cargan
cada factor). Desventaja: entiende sinónimos solo de forma indirecta.

### Ruta B — Embeddings (red neuronal pre-entrenada)

Un modelo ya entrenado (por ejemplo `all-MiniLM-L6-v2`) convierte cada texto **directamente**
en un vector denso (384 números) que **ya captura el significado**: entiende que "diesel
generator" y "power unit" son lo mismo aunque no compartan palabras, sin que nadie se lo
explique.

```
texto  →  [modelo de embeddings]  →  vector denso  (sin TF-IDF, sin SVD)
```

Ventajas: más potente semánticamente. Desventajas: más pesado (hay que descargar el modelo),
menos interpretable.

### Ruta C — El híbrido (lo que nos distingue)

Usamos el **texto para construir/inicializar los vectores** (resuelve el "arranque en frío":
hasta una empresa con un solo contrato tiene descripción que la ubica) y después dejamos que
las **etiquetas de quién ganó** los afinen. Texto para ubicar, victorias para corregir.

### Comparación

| | Ruta A: TF-IDF + SVD | Ruta B: Embeddings |
|---|---|---|
| potencia semántica | buena | mejor (entiende significado de fábrica) |
| costo | muy bajo, sin descargas | más alto (modelo externo) |
| interpretable | sí | no |
| qué probamos | primer experimento (rindió flojo) | próximo paso |

**Decisión:** el vector-builder es una caja **intercambiable** dentro del pipeline
(*construir vectores → puntuar → fusionar*). Empezamos con la Ruta A por rápida, pero el
primer experimento rindió flojo (el espacio latente no superó al texto crudo), así que
**pasamos a embeddings (Ruta B)** — que es, además, lo que vemos que da señal. Lo importante:
cambiar el vector-builder **no toca el resto del pipeline**.

## Lo que aprendimos hasta acá

- El **backoff de códigos (B0-v2)** sube el recall de forma consistente (Hit@100 +2–3 pp en las
  5 verticales), pero **no mueve el MRR**: mete más ganadores a la lista, pero caen hondo.
- El **texto crudo (TF-IDF/LSA) solo** rinde muy por debajo de los códigos (MRR 0,01–0,08 vs
  0,17 del B0). El texto **no sirve como ranker independiente**: sirve para **afinar la punta**
  fusionado con la incumbencia.
- Conclusión: **códigos = cobertura, texto = discriminación.** El MRR se gana combinando los dos.

## Los pasos a seguir

Cada paso se mide con **Δ MRR en dev (2023S2)**, dejando el test (2024S1) sellado.

| paso | qué | pregunta que responde |
|---|---|---|
| **A. Vector-builder** | construir los vectores con embeddings (MiniLM); documentos limpios + puente de códigos | ¿los embeddings dan mejor señal de texto que TF-IDF? |
| **B. Fusión / rerank** | reordenar el **top-200 del B0-v2** con el puntaje de texto (no fusión total, para no bajar a los ganadores que el B0 ya tenía bien) | ¿el texto sube el MRR sobre el B0-v2? |
| **C. Dial de dificultad** | pesar incumbencia vs. texto según cuán predecible es la licitación (entropía del historial de la oficina) | ¿gana más dándole el texto solo a las licitaciones "abiertas"? |
| **D. Reranker aprendido** | un ranker (LightGBM) que combina todas las señales (B0, cada nivel, texto, dificultad, n_awards, recencia) optimizando la punta | ¿aprender los pesos supera a combinarlos a mano? |

### Disciplina de validación (en todos los pasos)

- **Cortes temporales**, por vertical: train ≤ 2023-06-30 · **dev = 2023S2** (se tunea acá) ·
  **test = 2024S1** (sellado, se toca una vez al final).
- **Regla del futuro:** para una licitación en `t`, sólo se usa información anterior a `t`.
- Se reporta siempre la **mejora sobre el B0** (Δ MRR) con intervalo de confianza, no el MRR
  absoluto — porque el problema se vuelve más difícil en los semestres recientes.

### Por qué este plan nos diferencia

Un modelo de **solo texto** (sin incumbencia, sin fusión, sin validación de MRR) queda por
debajo del propio B0 — es decir, mejora negativa en la métrica del concurso. Nuestro plan usa
el texto como lo que es (afinador de la punta) **sobre** una base de incumbencia sólida, y lo
mide bien en cada paso.
