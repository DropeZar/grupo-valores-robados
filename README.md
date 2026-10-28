# Grupo Valores Robados — Ranking de licitaciones

**Y-Hat: ¿quién gana la licitación?** — dada una licitación pública, rankear las empresas
candidatas por probabilidad de ganarla, usando sólo información anterior a su publicación.
Se compite por **cuánto mejora el MRR sobre el baseline B0**.

> La consigna original y el detalle de los datos están en el repo público de la cátedra.

---

## La idea en una frase

El B0 (incumbencia pura: quién ganó antes el mismo trabajo en la misma oficina) **entierra
ganadores** — empresas capaces que no son "el proveedor de la casa" de esa oficina. Nuestro
modelo los rescata con **texto**: reconoce que el historial de una empresa habla del mismo
trabajo que pide la licitación, aunque no compartan códigos.

**Códigos = cobertura. Texto = discriminación. El modelo los combina.**

## El modelo

```
                 B0-v2  (incumbencia con backoff de códigos + pesos/tau tuneados)
                   │  da el shortlist (top-200) y el rango de incumbencia
Texto (embeddings bge-small, licitación ↔ historial de la empresa)
                   │  da el rango por similitud
                   ▼
             FUSIÓN (rank fusion, α=0.5, K=5)  →  ranking final
```

1. **B0-v2** (`b0_v2.py`): al B0 original le agregamos niveles de *backoff* — familia de PSC y
   jerarquía de oficina (subagencia) — que suben el recall sin romper la incumbencia.
2. **B0-v2 tuneado** (`b0_rocco.py`): pesos de los niveles **y** `tau` (vida media del
   decaimiento) optimizados por vertical, de forma robusta (multi-semestre, sin tocar el test).
3. **Texto** (`bge-small-en-v1.5`): embeddings de la licitación y del historial de cada empresa;
   la similitud coseno es la señal de discriminación.
4. **Fusión**: se reordena el top-200 del B0-v2 combinando incumbencia + texto por *rank fusion*.

## Resultados

**Elección de embedder** (dev 2023S2, Δ MRR vs B0, con el B0-v2 tuneado). Sorpresa: el modelo
**más chico es el mejor** — los grandes (bge-base, Gemma) no aportan sobre texto corto de dominio:

| embedder | rd | d | j | r | it | **prom** |
|---|---|---|---|---|---|---|
| MiniLM (22M) | +0,024 | +0,051 | +0,055 | +0,078 | +0,039 | +0,049 |
| **bge-small (33M)** | **+0,049** | **+0,091** | **+0,078** | **+0,093** | +0,059 | **+0,074** |
| bge-base (109M) | +0,023 | +0,072 | +0,073 | +0,075 | +0,038 | +0,056 |
| gemma (300M) | +0,027 | +0,086 | +0,073 | +0,058 | +0,063 | +0,062 |

**Validación en test sellado (2024S1):** el enfoque **mejora el MRR sobre el B0 de forma
significativa en 4 de 5 verticales** (el intervalo de confianza no cruza el cero), con una mejora
relativa del **30–80 %**. El resultado se sostiene de dev a test → **no es sobreajuste**. El
número final con bge-small + B0-v2 tuneado está en `docs/resultados_finales.md`.

## Estructura del repo

```
├── b0.py, evaluar.py        baseline y evaluador (dados por la cátedra)
├── b0_v2.py                 B0 con backoff de códigos
├── b0_rocco.py              B0-v2 con pesos y tau tuneados por vertical
├── test_final.py            corre el modelo final (bge-small + B0-v2 tuneado) en el test
├── experimentos/            scripts de análisis, tuneo y exploración (ver abajo)
├── docs/                    informes y bitácora del proyecto
└── figuras/                 figuras del diagnóstico
```

## Cómo correr

Este entorno usa Python 3.15 (sin wheels para varias libs), así que fijamos **3.12** en un
entorno efímero con `uv`:

```bash
# 1. datos (del release de la cátedra), descomprimidos en data/
curl -LO <url-del-release>/data.zip && tar -xf data.zip

# 2. modelo final sobre el test (codifica con bge-small + rerank)
PYTHONPATH="$PWD" uv run --no-project --python 3.12 \
    --with sentence-transformers --with scikit-learn --with numpy --with pandas --with pyarrow \
    python test_final.py
```

> Todos los scripts se corren **desde la raíz del repo** con `PYTHONPATH="$PWD"` (así encuentran
> los módulos `b0`, `b0_v2`, `b0_rocco` y las carpetas `data/` y `.emb_cache/`). Los embeddings se
> cachean en `.emb_cache/` (ignorado por git).

## experimentos/

El camino completo, incluidos los **experimentos negativos** (que también son resultado):

- `latente.py`, `latente_emb.py` — señal de texto con TF-IDF/SVD (falló) y con embeddings (funcionó).
- `tune_fusion.py` — barrido del peso de la fusión (α) y la constante K.
- `dial_dificultad.py` — ponderación adaptativa del texto por dificultad (entropías de Rényi); no superó al peso fijo.
- `reranker.py` — reranker aprendido (MaxEnt + LightGBM); empató a la fusión a mano.
- `tune_pesos.py`, `prop_pesos.py`, `prop_rocco.py` — tuneo de los pesos del B0-v2 y pruebas de propagación al modelo final.
- `probar_bge.py`, `confirmar_bge.py`, `comparar_modelos.py` — pruebas y comparación de embedders.
- `confirmar_verticales.py` — evaluación del modelo en las 5 verticales (dev/test).
- `viz_b0.py`, `viz_b0_multi.py` — figuras del diagnóstico del B0.
- `colab_embeddings.py` — notebook de Colab para codificar con modelos grandes en GPU.
