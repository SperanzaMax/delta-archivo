# PRE-REGISTRO · CAMINO LATERAL CON CONV PROPIA (`lat2`)

Escrito el 2026-08-24 por la mañana, **antes** de lanzar y con el pool sin GPU. Se congela y se
hashea antes de correr. Sale del §7 de `INFORME_CAMINO_LATERAL_20260824.md` y del diagnóstico
`DIAGNOSTICO_CONV_COMPARTIDA_20260822.md`.

## 1. Por qué existe

`lat` cerró bien y cerró grande: `err_identidad` **0,0000 en las tres semillas**, la colisión de
clave disuelta (`ident_rep` 0,0564 / 0,4683 / 0,2529 → 0,0000 / 0,0000 / 0,0069) y la bimodalidad
entre semillas desaparecida. W-0 a W-3 cumplen 3/3.

Pero paga en dos lugares, y los dos tienen la misma sospecha de causa:

- **`anterior`**, que se desploma en una semilla (0,8125 → 0,3798) mientras en las otras dos va al
  techo. Es lo que Maxi destapó el 22 preguntando por qué no parar a los 4000.
- **`nose_rel`**, que baja en dos de tres (0,9235 → 0,5842 · 0,5893 → 0,7194 · 0,7755 → 0,5816) y es
  lo que hace fallar W-4 vía `nose` en s0.

El diagnóstico del 22 apunta a un defecto de diseño **propio**: en `lat`, la query de la lectura y
el mixer usan la **misma** `blk["conv"]`. Lo escribí en el prereg como virtud —«no estrena
parámetros»— y acopla dos cosas con balances opuestos. El mixer quiere el mix que le sirve a la
regla delta; la query quiere contexto para formar entidad × relación (distancia 2) y **poco**
contexto para no diluir el marcador temporal. En `cual era antes el precio de banco ?` el token
`antes` cae fuera de la ventana de la conv, y encima en `lat` la query en su posición pasa a ser
`conv3(cual, era, antes)` en vez de `antes` puro. **La conv da la query conjunta y cobra el marcador
de orden.**

## 2. La condición `lat2`

Idéntica a `lat` salvo que la query se forma sobre **`blk["convq"]`, propia**, inicializada en
**`[1, 0, 0]`**.

| | punto de inyección | la query se forma sobre |
|---|---|---|
| `pre` (control) | antes del mixer | `ln1(h)` — función pura del token |
| `lat` | el mismo | `conv3(blk["conv"], ln1(h))` — conv **compartida** con el mixer |
| `lat2` | el mismo | `conv3(blk["convq"], ln1(h))` — conv **propia**, arranca en `[1,0,0]` |

**La propiedad que ninguna condición anterior tuvo: `lat2` contiene a `pre` como caso particular.**
Con `convq = [1,0,0]` la condición es idénticamente `pre` (verificado, K-1 abajo), así que
estructuralmente no puede ser peor, y cualquier contexto que aparezca es contexto que el modelo fue
**a buscar** por gradiente, no que le vino impuesto por el diseño.

**Contabilidad de parámetros, exacta.** El compromiso del 22-ago decía «384 params = 0,044 %» y el
número completo es otro, así que se declara acá: `convq` se instancia en los **cuatro** bloques para
que el árbol no cambie de forma y los checkpoints sigan siendo intercambiables (+1.536 params,
863.859 → 865.395), pero la lectura entra en el bloque 0 y **sólo esa se usa**. **384 params
efectivos (0,044 %), 1.536 en el árbol (0,178 %).**

### 2.1 El weight decay, y el control que sale gratis

Los `convq` de los bloques 1-3 **no** quedan clavados en `[1,0,0]`, y esto estaba escrito mal en la
primera versión de este documento. El optimizador es `adamw(weight_decay=0.01)`, que decae **todo**
parámetro tenga gradiente o no. Medido en el smoke de 60 pasos: el bloque 0 se movió **0,014011**
(gradiente, en los taps `p−1` y `p−2`) y los bloques 1-3 exactamente **0,000235** cada uno, sólo en
el tap 0, que es decay puro.

Esto **cambia cómo se lee el riesgo del §7**, y para bien. El atractor de un `convq` sin gradiente no
es `[1,0,0]` sino `[0,0,0]`, así que un `convq` final atenuado podría leerse como «el modelo aprendió
a atenuar la query» cuando en realidad es decay. Pero los bloques 1-3 son exactamente esa
trayectoria, con gradiente **cero garantizado por construcción**:

> **Cualquier diferencia entre el `convq` del bloque 0 y los de los bloques 1-3 es gradiente y no
> decay.** Sin simular nada, sin suponer una tasa y sin correr un control aparte.

Se declara acá como parte del instrumento, no como observación posterior.

## 3. Chequeo de instrumento, CORRIDO antes de escribir las predicciones

`chequeo_lat2.py`, pesos al azar, CPU, sin entrenar nada. Todo con `maxabs`, no con tolerancias:

| | qué afirma | medido | |
|---|---|---|---|
| **K-1** | `lat2[convq=[1,0,0]]` **es** `pre` | query `0,000e+00` · tronco `0,000e+00` | **CUMPLE** |
| **K-2** | `lat2[convq:=conv]` **es** `lat` | query `0,000e+00` | **CUMPLE** |
| **K-3** | `convq` no afecta al mixer | tronco sin lectura `0,000e+00` | **CUMPLE** |
| **K-4** | con `convq` perturbada, ve al vecino y no al lejano | `p−1` **0,6611** · `p−5` **0,0000** | **CUMPLE** |

K-1 y K-2 juntas dicen que el espacio que `lat2` puede explorar **contiene a las dos condiciones ya
corridas**. K-3 es la afirmación que justifica el cambio entero —el desacoplamiento— y hasta hoy
estaba leída del código y nunca medida. K-4 es la misma separación limpia entre contexto local y
global que `lat` cumplió el 22 (0,7533 contra 0,0000 exacto); si se perdiera, `lat2` estaría
reintroduciendo la dependencia global que rompió a `post`.

**K-5 · el código nuevo no mueve nada de lo ya corrido.** Es lo que protege el control reusado, y se
verificó por los dos caminos:

- **evaluación** — `ser.py` sobre `p3_s0` y `w3_s0` con el código nuevo: las 14 métricas idénticas
  hasta el último dígito a las medidas antes de agregar `convq`.
- **entrenamiento** — 40 pasos frescos de `pre` (semilla 0, misma config) con el código nuevo contra
  el código anterior: **las 64 hojas comunes del árbol de params, `maxabs = 0,0`**, y todas las
  métricas de evaluación iguales. La única diferencia en el JSON es el conteo de params y las rutas
  de salida.

## 4. El control se REUSA, otra vez, y por la misma razón

Control: las tres unidades `pre` (`p3_s0/s1/s2`, 26000 pasos), las mismas del camino lateral. Con
K-5 verificado, el contraste sigue siendo pareado de verdad: mismo generador, mismo `entrenar.py`,
mismo presupuesto, semillas apareadas por construcción. Y el segundo brazo de comparación —`lat`
(`w3_s0/s1/s2`)— también está completo, así que **esta campaña cuesta sólo 3 unidades** y se lee
contra dos condiciones ya medidas.

Config de `lat2`, idéntica a la de `pre` y a la de `lat`: nivel 3, `d=128`, `capas=4`, `batch=64`,
`lr=1e-3`, `p_vieja=0.35`, `p_nose=0.4`, `--abst cabeza`, `idioma=2`, **26000 pasos con horizonte
26000**, sin siembra, semillas 0/1/2. Familia `v3_s*`.

## 5. Predicciones

Instrumentos, los mismos de siempre: `ser.py` (n=2048, semilla 54321) y `diag_relacion.py` (2048
muestras), los dos leyendo `donde` y la regla de decisión **del checkpoint**.

- **V-0 · BLOQUEANTE.** `lat2` aprende la tarea: acierto ≥ 0,70 en al menos 2 de 3 semillas. Va
  primero por la lección de `post`. Con K-1 el riesgo es bajísimo —arranca siendo `pre`— y
  justamente por eso, si fallara, el que se cae es algo más grande que esta condición.

- **V-1 · CONSERVACIÓN, y es la principal.** `lat2` **conserva** lo que `lat` ganó: `ident_rep`
  ≤ 0,05 en al menos 2 de 3 semillas, pareado. Es la pregunta de si el efecto sobrevive al
  desacoplar la conv, o si el modelo, libre de elegir, elige no usar contexto y vuelve a `pre`.

- **V-2 · REPARACIÓN de `anterior`, y es lo que motiva la campaña.** `anterior` en `lat2` ≥ 0,70 en
  las **tres** semillas. `lat` da 1,0000 / 1,0000 / **0,3798**; `pre` da 0,9471 / 0,8317 / 0,8125.
  La compuerta la falla `lat` y la pasa `pre`, así que discrimina.

- **V-3 · REPARACIÓN de `nose_rel`.** `nose_rel` en `lat2` no cae más de **0,05** respecto de su
  gemela `pre`, en al menos 2 de 3 semillas. `lat` cae 0,3393 en s0 y 0,1939 en s2.

- **V-4 · NO-INTERCAMBIO, ahora completo.** `falsa_abst` ≤ 0,10 en las tres y `nose` no cae más de
  0,05 respecto de su gemela `pre`. **Es la W-4 de la campaña anterior, con `anterior` y `nose_rel`
  sacados afuera como V-2 y V-3 en vez de quedar fuera del prereg.** El hueco de W-4 fue declarado
  el 22 y confirmado el 24; no se repite dos veces.

## 6. Regla de decisión, comprometida por adelantado

- **V-1 falla** → el desacoplamiento devuelve el modelo a `pre` y la ganancia de `lat` **venía del
  acoplamiento mismo**, no de la query conjunta. Sería un resultado incómodo y fuerte: querría decir
  que lo que disuelve la colisión de clave es que el mixer y la query compartan filtro, y la lectura
  del 24-ago habría que reescribirla. Se reporta así, sin buscar una tercera conv.
- **V-1 pasa y V-2 falla** → el daño en `anterior` **no** viene de la conv compartida. El
  diagnóstico del 22 queda refutado por su propia corrección, y el pago pasa a atribuirse a la query
  conjunta en sí. La línea se cierra ahí: no se prueba una cuarta forma de query.
- **V-1 y V-2 pasan y V-3 falla** → la caída de `nose_rel` es **intrínseca a la query conjunta** y no
  un efecto del acoplamiento. Es la lectura que ya sugiere el §4 del informe del 24 (con la relación
  ausente, media query sigue coincidiendo con una entrada real y el modelo se ancla ahí), y quedaría
  confirmada con una intervención en vez de por interpretación.
- **V-1, V-2 y V-3 pasan** → `lat2` es la condición a adoptar como base del proyecto, y hay que
  correrla sobre `--idioma 3` antes de escribir nada más fuerte.

## 7. Riesgo declarado

`lat2` puede quedarse en `pre`. La inicialización en `[1,0,0]` es deliberadamente conservadora y el
gradiente tiene que **querer** mover esos 384 params contra el resto de la red. Si eso pasa, V-1
falla por una razón trivial —el modelo no exploró— y no por la razón interesante. **Se declara por
adelantado la evidencia que separa las dos**: se reporta `convq` aprendida al cierre, coeficiente por
coeficiente, **contra los `convq` de los bloques 1-3 como línea de base de decay puro** (§2.1). Si el
bloque 0 no se separa de ellos, el resultado es «no exploró» y **no** es evidencia contra la query
conjunta.

Un riesgo hermano, y es de comparabilidad: en `pre` la query pasa por **un** factor con weight decay
(`qr`), y en `lat2` por **dos** en cascada (`convq` y `qr`), lo que en el softmax de la lectura actúa
como temperatura. No pone a `lat2` en peor posición que a `lat` —que usa `blk["conv"]`, igual de
decaída, y aun así disolvió la colisión entera— pero se deja dicho porque si V-1 falla por poco es lo
primero que hay que mirar.

Y sigue en pie el riesgo del §7 del prereg anterior: tres semillas, bimodalidad medida. Todo se
reporta pareado por semilla, nunca por media.

