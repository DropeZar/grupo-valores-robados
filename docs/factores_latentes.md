# El modelo de factores latentes (capacidad × requerimiento)

> Sección para el informe. Enfoque propio para rankear empresas, distinto de la similitud de
> palabras clásica (TF-IDF). Cada empresa y cada licitación se representan como vectores en un
> mismo "espacio de capacidades", y se rankea por qué tan alineados están.

## La idea en una frase

A cada **empresa** le asignamos un vector de **capacidad** (en qué es buena) y a cada
**licitación** un vector de **requerimiento** (qué pide). El puntaje de una empresa para una
licitación es qué tan **alineados** están esos dos vectores. Esos vectores no se definen a
mano: **se aprenden de los datos**.

## La intuición: el "modelo Netflix"

Netflix no sabe de antemano qué géneros te gustan. Mira **qué viste** y descubre solo unos
"ejes ocultos" (acción, romance, documental…). Ubica a cada película en esos ejes, y a vos
también. Te recomienda las películas cuyos ejes se **alinean** con los tuyos.

Nosotros hacemos lo mismo cambiando *usuarios y películas* por *empresas y licitaciones*:

| Netflix | Nuestro problema |
|---|---|
| usuario | empresa |
| película | licitación |
| "te gusta la acción" | "sos capaz en IT" |
| qué viste | qué contratos ganaste |
| recomendar películas | rankear empresas para la licitación |

Esos ejes ocultos se llaman **factores latentes** ("latente" = está ahí pero no se ve
directo). Nadie los escribe: **emergen** de los datos.

## Qué es un "factor latente", concretamente

Un factor es una **dimensión de capacidad** que agrupa cosas que van juntas. En la vertical de
mantenimiento (`j`), los datos podrían esconder factores como:

- factor 1 = "IT / servidores"
- factor 2 = "energía / generadores"
- factor 3 = "equipamiento médico"
- … (en la práctica, 100 a 300 factores, muchos sin nombre claro)

Cada empresa es un vector = una lista de números que dice **cuánto tiene de cada factor**. Lo
mismo cada licitación, pero interpretado como **cuánto pide de cada factor**.

## Ejemplo numérico, paso a paso

Supongamos (de juguete) que hay solo 3 factores: `[IT, energía, médico]`.

**Las empresas candidatas:**

```
Empresa A (una casa de hardware IT):   capacidad = [0.90, 0.10, 0.00]
Empresa B (mantenimiento de energía):  capacidad = [0.10, 0.90, 0.20]
Empresa C (equipamiento médico):       capacidad = [0.00, 0.10, 0.95]
```

(Leé la empresa A así: "0,90 de IT, 0,10 de energía, 0 de médico" → es casi pura IT.)

**La licitación** "IBM Hardware Repair" pide, sobre todo, IT:

```
Licitación:  requerimiento = [0.95, 0.00, 0.00]
```

**El puntaje** es el *producto punto*: multiplicás factor por factor y sumás.

```
Empresa A:  0.90×0.95 + 0.10×0.00 + 0.00×0.00  =  0.855   ← el más alto → va primero
Empresa B:  0.10×0.95 + 0.90×0.00 + 0.20×0.00  =  0.095
Empresa C:  0.00×0.95 + 0.10×0.00 + 0.95×0.00  =  0.000
```

La empresa A gana el ranking **aunque B pueda ser más grande**, porque su capacidad se
**alinea** con lo que pide la licitación. Ese "alinearse" es lo que un solo número (un
puntaje de "qué tan buena es la empresa" en general) nunca podría capturar: ganar es
**contextual**, depende del cruce entre lo que la empresa sabe y lo que la licitación pide.

## El producto punto, en palabras

El producto punto es alto cuando los dos vectores **apuntan en la misma dirección** (la empresa
es fuerte justo en los factores que la licitación pide) y es casi cero cuando apuntan en
direcciones distintas (la empresa es fuerte en factores que a la licitación no le importan).
No compara palabras: compara **direcciones en el espacio de capacidades**.

> **Figura:** (acá va el gráfico del espacio de factores latentes — los vectores de las
> empresas y de la licitación; la empresa cuya flecha se alinea con la de la licitación es la
> que gana).

## De dónde salen los vectores (el corazón del método)

Hay dos formas de obtenerlos. La mejor las combina.

### Ruta A — del texto (LSA / SVD)

1. Armamos, para cada empresa, el vector de palabras de su historial (`texto_awards`), y para
   cada licitación, el de su `title` + `description`. Son vectores enormes (~40.000 palabras)
   y casi todo ceros.
2. Los **comprimimos** de ~40.000 palabras a ~200 factores con una técnica llamada **SVD**
   (cuando se aplica a texto, se la conoce como **LSA**, *Latent Semantic Analysis*).

**Qué hace el SVD, en criollo:** mira qué palabras **aparecen juntas** una y otra vez y las
agrupa en un mismo factor. Si "server", "dell", "hp" y "warranty" aparecen siempre en los
mismos documentos, el SVD arma con ellas **un solo eje** = "capacidad IT". Después reescribe
cada documento no como "qué palabras tiene" sino como "cuánto tiene de cada eje".

**Por qué esto es la clave:** una licitación que dice *"diesel generator overhaul"* y una
empresa cuyo historial dice *"emergency power unit servicing"* **no comparten ni una palabra**,
así que la comparación de palabras cruda (TF-IDF) las ve como no relacionadas. Pero las dos
cargan el mismo factor "energía", así que en el espacio latente **quedan cerca**. Eso es lo
que el texto crudo no puede hacer, y es nuestra ventaja.

Ventaja extra: **no necesita etiquetas** (no usa quién ganó), así que es robusto y simple.

### Ruta B — de quién ganó (factorización de matrices)

1. Armamos una matriz gigante "empresa × tipo-de-trabajo", con un 1 donde esa empresa ganó
   ese tipo de trabajo.
2. La **factorizamos** en dos matrices chicas: una de "empresa × factores" y otra de
   "factores × tipo-de-trabajo".

Acá los factores emergen de los **patrones de victorias**, sin leer texto (es el enfoque
Netflix puro). Problema: una empresa con poco historial queda mal ubicada — es el **"cold
start"** (arranque en frío): sin datos de victorias, el modelo no sabe dónde ponerla.

### El híbrido (lo que nos distingue)

Usamos el **texto para construir/inicializar los vectores** (resuelve el cold start: hasta una
empresa con un solo contrato tiene una descripción que la ubica) y después dejamos que **las
etiquetas de quién ganó los corrijan**. Texto para ubicar, victorias para afinar.

## Por qué le gana al TF-IDF que van a usar los demás

- **TF-IDF** compara **palabras crudas**: solo matchea si los textos comparten términos.
- **Factores latentes** comparan **conceptos**: matchean aunque usen palabras distintas para la
  misma capacidad. Capturan sinónimos y jerga equivalente, que es exactamente donde el
  problema real vive (las empresas describen su trabajo cada una a su manera).

## Las perillas (todas se tunean en dev, no en el test)

1. **k = número de factores.** Pocos (k=10) → el modelo es tosco, mezcla capacidades distintas.
   Muchos (k=1000) → memoriza ruido y sobreajusta. Suele andar bien entre 100 y 300.
2. **Qué texto de la empresa:** solo `texto_awards`, o también las descripciones del
   `historial` (respetando `fecha < t`).
3. **Cuánto pesa este match vs. el B0** en la fusión final.

## La "dificultad" de la licitación como dial

No todas las licitaciones son igual de predecibles:

- **Fácil / predecible:** una oficina que siempre le compra a la misma empresa → el B0
  (incumbencia) ya acierta, el texto sobra.
- **Difícil / abierta:** una oficina con historial disperso, sin dueño claro → el B0 flota y el
  match de capacidad tiene que hacer el trabajo.

Medimos esa "dificultad" (por ejemplo, con la *entropía* del historial de ganadores de ese
PSC+oficina: baja = un dueño claro, alta = repartido) y la usamos como dial de fusión:

```
puntaje_final = (1 − dificultad) · incumbencia_B0  +  dificultad · match_de_capacidad
```

En las fáciles manda el B0; en las difíciles manda el modelo de capacidad. Esto además
**protege los casos fáciles** (no les mete ruido), que es justo el problema que tuvo el v2.

## Por qué nos distingue

- Casi todos van a medir "cuántas palabras comparten" (TF-IDF).
- Nosotros medimos "qué tan alineada está la **capacidad** de la empresa con el
  **requerimiento** de la licitación, en un espacio **aprendido**, modulado por cuán
  **predecible** es la licitación".
- Tiene teoría atrás (factores latentes, sistemas de recomendación, IRT), es incremental
  (arranca barato con LSA), ataca justo donde el B0 muere (los S_new), y es **interpretable**:
  podemos mirar qué palabras cargan cada factor y *explicar* por qué rankeamos a una empresa.

## Qué construimos primero

El primer ladrillo es la **Ruta A (LSA)**: construir los vectores de capacidad y requerimiento
con SVD, rankear por su alineación, fusionarlo con el B0/v2 y medir el **Δ MRR en dev
(2023S2)**, dejando 2024S1 sellado. Si mueve el MRR, pasamos al híbrido (afinar con las
etiquetas) y al dial de dificultad.

## Glosario rápido

- **Vector:** una lista de números que ubica algo en varios ejes a la vez.
- **Factor latente:** un eje de capacidad que emerge de los datos (no se define a mano).
- **Producto punto:** multiplicar dos vectores eje por eje y sumar; mide alineación.
- **SVD / LSA:** la técnica que comprime miles de palabras en pocos factores agrupando las que
  aparecen juntas.
- **Cold start:** el problema de ubicar a una empresa (o ítem) con poco o ningún historial.
