# Informe — Ranking de licitaciones

**Grupo Valores Robados** · análisis del baseline B0 y primer experimento (B0-v2)

> Objetivo del informe: dejar registrado qué entendimos de los datos, dónde falla el
> baseline, qué probamos y qué sigue. Los gráficos referenciados están en `figuras/`.

---

## 1. El problema

Dada una licitación, rankear las empresas candidatas por probabilidad de ganarla, usando
**sólo** información anterior a su publicación. Hay 5 verticales (rubros), cada una se
trabaja por separado:

| vertical | rubro | licitaciones |
|---|---|---|
| `rd` | investigación y desarrollo | 328 |
| `d` | IT y telecomunicaciones | 819 |
| `j` | mantenimiento y reparación de equipos | 4.504 |
| `r` | servicios profesionales | 2.180 |
| `it` | productos informáticos | 1.792 |

- **Candidatas:** para una licitación de un semestre, son todas las empresas de ese semestre
  (~30.000 en `j`). El ganador siempre está entre ellas.
- **Métrica:** MRR (1 / puesto del mejor ganador). Lo que se compara entre grupos es
  **cuánto mejora el MRR sobre el B0**. Se reportan además Hit@5, Hit@1 y, como diagnóstico,
  Hit@100 (recall: ¿el ganador está en las 100 entregadas?).
- **B0:** baseline dado. Rankea por historial de incumbencia (quién ganó antes el mismo
  trabajo, sobre todo en la misma oficina), sin leer texto.

## 2. Diagnóstico: el B0 entierra ganadores

Validación temporal (último semestre). En `j` / 2024S1 (995 licitaciones):

- El ganador queda en el **puesto 1** sólo el **8%** de las veces; en el top-5, el **18%**.
- **El 47% de los ganadores no entra siquiera en el top-100** (Hit@100 = 53%).
- Puesto mediano del ganador: **73** de ~32.800.

> **Figura:** `figuras/b0_enterrados_j_2024S1_b0.png` — (a) distribución de puestos,
> (b) curva Hit@k, (c) el mecanismo, (d) contratos previos vs. puesto.

### Por qué los entierra

Entre los ganadores enterrados (puesto > 10):

| | enterrados (>10) | encontrados (≤10) |
|---|---|---|
| nunca ganó en ESA oficina | **60,5%** | 3,1% |
| nunca ganó con ESE PSC | 29,3% | 0,0% |
| 0 contratos previos | 0,0% | 0,0% |

**Conclusión:** el ganador típico enterrado **no es una empresa desconocida** (todas tienen
historial) — es una empresa capaz que **no es el proveedor habitual de esa oficina puntual**.
El B0 premia al "de la casa" y manda al fondo a quien hizo el mismo trabajo en otro lado.

> **Ejemplo:** licitación "IBM Hardware Repair" (oficina naval N00178). Ganó *BETTER DIRECT*,
> cuyo historial es puro "HP/Dell/IBM server maintenance" — el proveedor obvio. El B0 la puso
> en el **puesto 4.753** porque nunca le había vendido a esa oficina; en el puesto 1 puso al
> proveedor habitual de la oficina. La señal estaba en el **texto**, no en los códigos.

## 3. Cuánto puede aportar cada idea (headroom)

Sobre los enterrados de `j`, qué señal tenían disponible que el B0 no usó:

| idea | dispara nuevo en… |
|---|---|
| agrupar PSC por familia (primeros 2 chars) | 28,8% de los enterrados |
| jerarquía de oficina (subagencia) | 23,0% de los enterrados |
| sólo el texto salva (ni familia PSC ni subagencia) | 0,3% |

**Lectura clave:** casi todos los ganadores tienen *alguna* señal de código disponible. Por
eso separamos dos cosas:

- **Códigos = cobertura.** Baratos, suben el recall, pero como *todas* las candidatas
  comparten familia-PSC (99,5%) y muchas la subagencia, **discriminan mal**.
- **Texto = discriminación.** Es lo único que distingue, entre las 500 empresas con el mismo
  código, a la que realmente hizo *este* trabajo. Ahí se juega el MRR (que se decide en la punta).

## 4. B0-v2: backoff de códigos

Agregamos tres niveles al B0, sin tocar los cinco originales (`b0_v2.py`):

- `(psc_familia, office)` y `(psc_familia)` — agrupar PSC por familia.
- `(psc, subagencia)` — jerarquía de oficina.

Todo lo demás es idéntico (decaimiento τ=2,5 años, corte estricto `fecha < t`, empates
deterministas). Pesos de los niveles nuevos puestos a mano y conservadores.

### Resultado: recall ↑ consistente, MRR plano

Mejora de **Hit@100** (recall) del v2 sobre el B0, promediada por vertical, en los 5 semestres:

| vertical | Δ Hit@100 promedio |
|---|---|
| `r` | +3,3 pp |
| `rd` | +3,1 pp |
| `j` | +2,7 pp |
| `d` | +2,7 pp |
| `it` | +2,4 pp |

Sube o empata en **24 de 25** combinaciones (vertical × semestre), y el puesto mediano del
ganador mejora casi siempre (p. ej. `rd` 2024S1: 249 → 56). **El backoff funciona y es robusto.**

> **Figura:** `figuras/overview_recall.png` — Hit@100 por vertical y semestre, B0 vs v2.

**Pero** el MRR **no se movió**: en `j` 2024S1, Δ MRR = −0,001 (el intervalo cruza el cero),
y un leve costo en los casos fáciles (S_inc). ¿Por qué? Porque los ganadores que rescatamos
**caen hondo** (puesto mediano ~39): meterlos a la lista sube el recall, pero `1/39 ≈ 0,026`
casi no mueve el MRR. **Subimos el piso, no la punta.**

### Observación temporal

El B0 (y el v2) **se degradan con el tiempo** en todas las verticales: el problema se vuelve
más difícil en los semestres recientes. Por eso el test (2024S1) dará números absolutos
bajos para cualquier modelo — hay que leer la **mejora relativa**, no el absoluto.

## 5. Conclusión y próximo paso

- Los **códigos ya dieron lo que podían**: suben el recall de forma consistente, pero no el
  MRR, porque no discriminan en la punta (todas las candidatas comparten código).
- El **texto** es la señal de discriminación que falta, y es el paso estructuralmente
  necesario para mover el MRR. **Próximo:** señal de texto (TF-IDF/BM25 licitación↔historial
  de cada empresa) fusionada con el v2, medida en dev.

## Metodología (reproducibilidad)

- **Validación temporal**, por vertical. Cortes: train ≤ 2023-06-30 · dev = 2023S2 ·
  **test = 2024S1 (sellado**, se toca una vez al final).
- **Regla del futuro:** para una licitación en `t`, sólo se usa historial con `fecha < t`.
  El código lo respeta por construcción (no hay fuga de futuro).
- Scripts: `b0.py` (baseline), `b0_v2.py` (backoff + comparación), `viz_b0.py` y
  `viz_b0_multi.py` (figuras). Evaluación con `evaluar.py`.
