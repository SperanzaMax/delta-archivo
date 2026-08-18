# PREREG — La abstención, ¿consulta el archivo o la forma de la pregunta?

**Fecha de congelamiento:** 2026-08-18, antes de correr una sola predicción.
**Script:** `micro_lm/sonda_inyeccion.py` (hash aparte, en `PREREG_INYECCION_HASH.txt`).
**Costo:** CPU, sobre checkpoints ya entrenados. Cero GPU.

## §1 · Por qué existe

El 17-ago la campaña `x` dio **4 de 4 pasan / 5 de 5 fallan** la compuerta de abstención, separadas
por el margen sobre el atajo. El informe declaró como salvedad que faltaba «el control de permutación
de etiquetas», y esa era la única prueba pendiente capaz de tumbar el positivo.

**Ese control queda descartado por análisis, antes de gastarlo** (§6). En su lugar va éste, que ataca
la hipótesis alternativa que sigue viva.

**Hipótesis alternativa a matar:** que el modelo no detecte la ausencia *comparando la consulta con el
archivo*, sino por una **firma marginal de la consulta misma**. El generador la hace plausible: en una
pregunta con respuesta el par (rel, ent) sale del generador de hechos, mientras que en una consulta
`NOSE` la relación se sortea uniforme sobre `RELACIONES` y la entidad sobre las no dichas
(`idioma.py:205-222`). Si esas dos marginales difieren, existe una regla que acierta `NOSE` **sin leer
el archivo**, y `nose` alto con `falsa_abst` bajo se conseguiría sin ninguna consulta a memoria.

Si esa alternativa es cierta, el resultado del 17-ago no es «el modelo sabe que no lo tiene» sino «el
modelo reconoce las preguntas que en su distribución de entrenamiento no suelen tener respuesta» — que
es precisamente lo que [[objetivo-memoria-persistente-llm]] no quiere.

## §2 · El diseño: contraste pareado por inyección

Para cada consulta de tipo `NOSE` con par (rel_q, ent_q) se arman **dos** tensorizaciones del **mismo
episodio** y la **misma consulta**, palabra por palabra:

- **A (ausente):** el episodio tal cual. Respuesta correcta = `NOSE`.
- **B (inyectada):** el mismo episodio más **un enunciado** que dice el hecho preguntado,
  `formas(rel_q, ent_q, v, nivel)`, con `v` sorteado del pool que corresponde a la relación.
  Respuesta correcta = `v`.

Todo lo demás es idéntico: mismas sesiones, mismo orden, misma consulta, mismos turnos. La única
variable es si el hecho está en el archivo. Por eso el contraste es pareado y no entre grupos: no
compara preguntas distintas, compara **la misma pregunta con y sin la respuesta disponible**.

El enunciado inyectado va en la sesión con menos enunciados, y la corrida **aborta** si el
truncamiento deja de ser 0,0000 (la compuerta que ya existe desde el 14-ago).

## §3 · Predicciones (congeladas)

Sobre los checkpoints que **pasan** la compuerta (`x1_s0`, `x2_s0`, `x2_s2`):

- **P-1 (principal).** La tasa de `NOSE` se derrumba al inyectar el hecho:
  `NOSE` en B **≤ 0,20**, con `NOSE` en A **≥ 0,60**, y caída A−B **≥ 0,40**.
- **P-2.** No sólo deja de abstenerse, sino que **recupera el valor recién inyectado**:
  acierto de `v` en B **≥ 0,50**.
- **P-3 (control de sanidad, PUEDE fallar).** La tasa de `NOSE` en A reproduce el `nose` reportado
  para ese checkpoint el 17-ago **±0,10**. Si no lo reproduce, el instrumento no está midiendo lo
  mismo que la campaña y las otras dos predicciones no se leen.

Todo se reporta **desagregado por `nose_ent` y `nose_rel`**. `nose_rel` es el caso difícil (la entidad
sí está, la relación no) y es donde la firma marginal tendría menos para agarrarse.

**Contraste declarado:** se corre además sobre un checkpoint que **falla** la compuerta (`x4_s0`). No
tiene predicción asociada — es exploratorio y se reporta como tal.

## §4 · Qué resultado mata qué

- **P-1 y P-2 cumplen** → la abstención consulta el archivo. La firma marginal queda descartada y el
  resultado del 17-ago se sostiene con evidencia directa, no por descarte.
- **P-1 falla** (sigue diciendo `NOSE` con el hecho delante) → la abstención está enganchada a la
  forma de la pregunta. El «4 de 4» del 17-ago mide reconocimiento de distribución, no consulta a
  memoria, y hay que reescribir el informe.
- **P-1 cumple y P-2 falla** → mira el archivo pero no lo usa para responder: deja de abstenerse sin
  recuperar el valor. Sería un resultado intermedio y se reporta como tal, sin redondear a favor.

## §5 · Compromiso por adelantado

Si P-1 falla en alguno de los tres checkpoints que pasan la compuerta, **el informe
`INFORME_ABSTENCION_20260817.md` se corrige antes de cualquier paso nuevo**, y la cabeza de abstención
separada no se construye sobre él. No se prueba una segunda variante de la sonda para rescatar el
resultado.

## §6 · Por qué NO se corre el gemelo de permutación de etiquetas

Registrado para no repetirlo. El control planteado el 17-ago era entrenar un gemelo asignando `NOSE`
al azar al 40 % de las preguntas, conservando la frecuencia marginal. **No discrimina, y se puede ver
sin correrlo:**

Con la marginal conservada, la etiqueta miente de forma **simétrica y minoritaria dentro de cada
condición**. Una pregunta sin respuesta recibe `NOSE` con 0,4 y un valor arbitrario con 0,6 repartido
entre ~100 valores (≈0,006 cada uno): el argmax sigue siendo `NOSE` → `nose` ≈ 1. Una pregunta con
respuesta recibe `NOSE` con 0,4 contra su valor real con 0,6: el argmax es el valor real →
`falsa_abst` ≈ 0. **El gemelo permutado pasaría la compuerta**, no por señal sino porque la
información sigue estando en la entrada y el ruido de etiqueta se marginaliza.

Además, la hipótesis que ese gemelo iba a matar —«`nose` alto se consigue disparando `NOSE` por
frecuencia»— **ya está descartada por los datos existentes**: `x2_s0` tiene `nose` 0,8635 con
`falsa_abst` **0,0000 exacto**, y un modelo que dispara por frecuencia no puede tener `falsa_abst`
cero.

Un control de permutación es válido cuando la etiqueta permutada es la única fuente de la asociación
medida. Acá no lo es: la asociación es recuperable de la entrada. Lo que hay que permutar entonces no
es la etiqueta sino **la disponibilidad del hecho**, que es exactamente lo que hace §2.

---

## §7 · ENMIENDA E-1 (2026-08-18, después de la primera corrida, declarada antes de correr la nueva condición)

**Qué la motiva.** La corrida sobre `x1_s0` dio el resultado partido en dos por tipo: `nose_ent`
0,959 → 0,073 (caída 0,886, acierto 0,927) y `nose_rel` 0,880 → 0,719 (caída 0,162). Antes de leer el
segundo número como «la abstención de `nose_rel` no consulta el archivo», hay una explicación
alternativa que lo produce igual:

**La condición B saca a `nose_rel` de distribución.** `idioma.py:161` sortea las entidades del
episodio con `replace=False`, así que **en todo el entrenamiento cada entidad aparece con exactamente
una relación**. En `nose_rel` la entidad preguntada YA está en el archivo con otra relación, de modo
que inyectar el hecho le deja **dos relaciones a la misma entidad: una configuración que el modelo
nunca vio**. En `nose_ent` no pasa — la entidad es nueva y el episodio queda estructuralmente igual.

El diseño §2 es por lo tanto válido para `nose_ent` y **confundido para `nose_rel`**, y el número
agregado de la §3 promedia dos cosas distintas. Se declara antes de seguir.

**Condición C (reemplazo), sólo para `nose_rel`:** en vez de AGREGAR el hecho preguntado, se
**reemplazan** los enunciados del hecho que esa entidad ya tenía por un único enunciado
`formas(rel_q, ent_q, v, nivel)`. La entidad conserva una sola relación, la preguntada, y el episodio
queda dentro de la distribución de entrenamiento. Sigue siendo pareado: misma consulta, mismas
sesiones salvo el hecho de esa entidad.

**Predicciones de C (congeladas antes de correrla):**

- **P-4.** Si el 0,719 era artefacto de estar fuera de distribución: `C_nose` **≤ 0,20** y
  `C_acierto` **≥ 0,50` en `nose_rel`.
- **P-5.** Si C sigue alto (`C_nose` ≥ 0,50), entonces la abstención de `nose_rel` **no consulta el
  archivo por relación**: el modelo indexa por entidad y `NOSE` en `nose_rel` sale de algo más grueso
  que «no tengo ese hecho». Sería el hallazgo, no un fallo del instrumento.

**Diagnóstico extra, sin predicción asociada (exploratorio, se reporta como tal):** entre las
respuestas de B/C que no son `NOSE` en `nose_rel`, qué fracción es **el valor del otro hecho de esa
misma entidad** — que separa interferencia de identidad de simple ruido.

**Lo ya corrido no se descarta ni se re-interpreta a favor:** `nose_ent` queda como está (P-1 falla en
el agregado, P-2 cumple, P-3 cumple), y el veredicto agregado de la §3 se declara **no interpretable**
por promediar dos regímenes con signo opuesto.
