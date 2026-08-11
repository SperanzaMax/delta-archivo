# Pre-registro — correcciones ELÍPTICAS y el punto de cruce de la hidratación

**Se congela con hash ANTES de generar un solo dato.** Fecha: 2026-08-11. Deriva de
`REVISION_20260811.md`, cuyo smoke (`smoke_eliptica.py`) es lo único ya medido y está declarado en §2.

---

## 1. Qué se pregunta, y por qué no está contestado

`INFORME_GEMACION_ACOTADA.md` (2026-08-10) cerró la gemación como mecanismo de indexación con este
argumento: `emb(v_r)` ya está óptimamente colocado **porque contiene la entidad que la consulta
menciona**. Eso es una propiedad del generador (`generar_revisiones.py:25` interpola la entidad en cada
revisión), no de las correcciones reales, que son **elípticas**: «no, es Beto».

**Esto no reabre §5 del prereg anterior**, que se comprometió a no probar una **tercera geometría**. No
hay geometría nueva: se usa `g_orbita` tal como está implementada en `correr_acotada.py`, sin tocar ε.
Lo que cambia es la **distribución de datos**. El cierre anterior queda intacto **para su régimen**.

**Y la pregunta no es «¿gana la gemación?».** Si la hidratación por co-referencia es perfecta,
reconstruye el texto auto-contenido y estamos exactamente en el régimen ya medido, donde la gemación
pierde por el peaje. Eso está decidido de antemano y no se pregunta. Lo que se pregunta es:

> **¿A partir de qué tasa de error de la resolución de co-referencia conviene anclar geométricamente
> en vez de hidratar el texto?**

Es una pregunta con respuesta numérica, con valor de ingeniería directo, y que no puede no informar:
cualquier resultado —incluido que no exista tal punto— dice algo.

## 2. Lo único ya medido antes de congelar (declarado)

`smoke_eliptica.py`, N = 60, `nomic-embed-text` en minúscula. Es un chequeo de **rango**, no un dato
del experimento, y se declara para que no cuente como conocimiento oculto:

| | coseno con la consulta |
|---|---|
| v2 auto-contenida | 0,8497 |
| v2 elíptica | 0,4237 |
| elíptica de otra entidad | 0,4064 |
| **top-1 de la elíptica correcta entre 60** | **0,0167 = azar (1/60)** |

## 3. Diseño

**Datos.** N = 3000 entidades, `gen_hechos(rng(0))` — el mismo corpus, la misma semilla que
`correr_hechos.py`, sin re-generar. Por entidad: un **ancla** auto-contenida (v0, la plantilla
completa) y **8 revisiones**. Cada revisión existe en dos textos:

- **elíptico** — una de cuatro formas fijas, sin nombrar la entidad: `no, it's {v}.` ·
  `actually, {v}.` · `sorry, i meant {v}.` · `correction: {v}.`
- **hidratado** — la plantilla completa con la entidad, idéntico en forma a lo que genera
  `generar_revisiones.py`. Es la co-referencia resuelta **perfectamente**.

**Condiciones.** El barrido de τ **no cuesta embeddings**: se construye eligiendo por entrada, con
probabilidad τ, el texto elíptico en lugar del hidratado. Los dos conjuntos se codifican una vez.

| condición | dirección de la entrada de la revisión r |
|---|---|
| `hidratada_τ` | `emb(hidratado_r)` con prob. 1−τ; `emb(elíptico_r)` con prob. τ |
| `g_orbita` | `E0 + ε·t`, `t` tangente aleatoria al **ancla** — no depende del texto de la revisión |

τ ∈ **{0, 0,05, 0,10, 0,20, 0,40, 1,00}**. Los dos extremos tienen nombre: **τ = 0** es la hidratación
perfecta (y reproduce el `duplicados` del 10-ago); **τ = 1** es la corrección cruda, sin resolver.

**Parámetros, todos heredados sin re-ajustar:** ε = 0,30 · k = 5 (top-k) · margen = 0,02 ·
10 semillas × submuestreo de 1000 · IC95 por t de Student con 9 gl · K ∈ {1, 2, 4, 8} revisiones.

**Métrica principal: VIGENTE** — la entrada de la última revisión está en el top-k de la consulta.
Difiere del prereg anterior, que usaba COBERTURA, **y el cambio se declara acá con su razón**: en
régimen elíptico la pregunta del objetivo es si la corrección se recupera, no si conviven las dos
versiones. COBERTURA se reporta como secundaria.

## 4. Predicciones

**P-E0 — control BLOQUEANTE de comparabilidad.** A τ = 0 y K = 8, `hidratada_0` ≥ `g_orbita` en
VIGENTE. Es el régimen del 10-ago y tiene que reproducirlo. **Si `g_orbita` ganara acá, el harness
cambió respecto de la campaña anterior y nada de lo demás se lee.**

**P-E1 — PRINCIPAL.** Existe τ* ≤ 0,25 tal que para todo τ > τ*, `g_orbita` − `hidratada_τ` en VIGENTE
a K = 8 es ≥ +0,02 sin cruzar cero.
**Predicción puntual, del modelo lineal de cosenos del smoke:** el coseno de `hidratada_τ` cae como
0,8497 − 0,4260·τ y `g_orbita` es plano en 0,8497 − 0,036 → **τ* ≈ 0,075**. Se registra el número
esperado para que el acierto o el fallo sean legibles, no sólo el signo.

**P-E2 — control de régimen.** A τ = 1 (corrección cruda) y K = 8, VIGENTE de `hidratada_1` < 0,10.
Si la corrección elíptica cruda **se recuperara igual**, el régimen no es el que el smoke sugiere y
P-E1 pierde sentido aunque confirme.

**P-E3 — mecanicista.** El coseno de la entrada vigente contra la consulta decae linealmente en τ para
`hidratada_τ` (pendiente ≤ −0,30 por unidad de τ) y es **plano** en `g_orbita` (|pendiente| ≤ 0,01).
Es lo que hace que el cruce sea un cruce y no una coincidencia.

## 5. Falsación, comprometida por adelantado

- **Si τ* no existe** —`g_orbita` no supera a `hidratada_τ` en **ningún** τ, ni siquiera en τ = 1—
  entonces la geometría no sirve ni siquiera donde el texto **no lleva la clave de recuperación**.
  En ese caso la gemación queda descartada **con generalidad**, no sólo en el régimen auto-contenido,
  y esta línea se cierra definitivamente. No se prueba otra variante.
- **Si τ* > 0,25** el mecanismo sólo paga cuando la co-referencia es muy mala, y el resultado se
  reporta como negativo práctico: en cualquier sistema con hidratación razonable, no vale la pena.
- **Si P-E0 falla**, se reporta el fallo del control y no se emite ningún otro veredicto.

## 6. Lo que este experimento NO puede decir

- Sigue siendo **índice no paramétrico sobre encoder congelado**. No dice nada del índice co-entrenado.
- Las cuatro formas elípticas son fijas y sintéticas; no son una muestra de correcciones humanas.
- La hidratación se modela como **acierto o crudo**, con probabilidad τ. Una co-referencia real puede
  fallar de otras maneras (hidratar con la entidad **equivocada**), que sería peor que no hidratar y
  no se modela acá.
- τ es un parámetro impuesto, no medido: el experimento da el punto de cruce, **no** dice a qué tasa
  falla la co-referencia en un sistema real.
