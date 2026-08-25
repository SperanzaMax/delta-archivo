# INFORME · el slot nulo — la abstención dentro de la memoria

Evalúa `PREREG_SLOT_NULO.md` (SHA `f95b6e9d`). Unidades `y3_s0/s1/s2` a 26000 pasos (la familia se
renombró de `n3_*` a `y3_*` al cazar la colisión de prefijo antes de gastar una unidad). Control
reusado `p3_s0/s1/s2` = `pre` + `cabeza`, mismo generador, mismo presupuesto, semillas apareadas.

**Antes de nada: el primer análisis de estas unidades midió con un instrumento roto.** Ver el §4 y
`NOTA_INSTRUMENTO_SLOT_20260825.md`. Todo lo que sigue está medido con el arreglo, y con la regresión
verificada contra las unidades ya publicadas.

## 1. Resultado, pareado por semilla

|  | s0 `cabeza` → `slot` | s1 `cabeza` → `slot` | s2 `cabeza` → `slot` |
|---|---|---|---|
| acierto | 0,9705 → 0,8991 | 0,7769 → 0,7859 | 0,8351 → 0,8999 |
| `nose` | 0,9119 → **0,0000** | 0,5416 → **0,0000** | 0,7298 → **0,0000** |
| `falsa_abst` | 0,0082 → 0,0000 | 0,0041 → 0,0000 | 0,0353 → 0,0000 |
| `anterior` | 0,9471 → 0,9231 | 0,8317 → 0,8269 | 0,8125 → 0,9231 |
| AUC(`s_max`) | — | — | 0,5369 · 0,5078 · 0,5235 |

| predicción | criterio | medido | |
|---|---|---|---|
| **S-0** bloqueante | acierto ≥ 0,70 en ≥ 2/3 | 0,8991 · 0,7859 · 0,8999 | **CUMPLE 3/3** |
| **S-1** principal | `nose` ≥ 0,50 y `falsa_abst` ≤ 0,10 en ≥ 2/3 | `nose` 0,0000 en las tres | **NO CUMPLE 0/3** |
| **S-2** mecanicista | AUC(`s_max`) ≥ 0,60 en ≥ 2/3 | 0,5369 · 0,5078 · 0,5235 | **NO CUMPLE 0/3** |
| **S-3** pareada vs `cabeza` | `nose` no cae > 0,05 | −0,9119 · −0,5416 · −0,7298 | **NO CUMPLE 0/3** |
| **S-4** no-intercambio | `vigente` no cae > 0,02; `anterior` y `nose_rel` reportados | `anterior` −0,024 · −0,005 · **+0,111** | sin ruptura |

## 2. Por qué `nose` da CERO EXACTO, que es lo único que había que explicar

Un cero limpio en las tres semillas es la firma de un artefacto, y en este proyecto ya lo fue siete
veces. La primera vez lo era (§4). Con el instrumento arreglado **sigue dando cero**, y el
diagnóstico post-hoc (`diag_slot.py`, declarado como post-hoc y sin estatus de prereg) dice por qué.

La decisión de abstenerse sale de la **masa** del slot, con umbral 0,5 —el mismo que hereda de
`cabeza`, donde el logit es libre—. Medida sobre 2048 muestras:

| | s0 | s1 | s2 |
|---|---|---|---|
| masa del slot, preguntas **con** respuesta | 0,4074 | 0,4046 | 0,4020 |
| masa del slot, preguntas **sin** respuesta | 0,4086 | 0,4065 | 0,4037 |
| masa máxima observada | 0,4679 | 0,4597 | 0,4754 |
| fracción con masa > 0,5 | **0,0000** | **0,0000** | **0,0000** |
| AUC (sin vs con respuesta) | 0,5190 | 0,5313 | 0,5182 |

Desagregado por tipo, en s0: `vigente` 0,4090 · `anterior` 0,3996 · `nose_ent` 0,4087 · `nose_rel`
0,4084. **La masa es la misma pase lo que pase.**

**Y el número que lo cierra: la tasa base empírica de preguntas sin respuesta es 829/2048 = 0,4048.**
La masa media del slot es 0,4074 / 0,4046 / 0,4020. **El slot convergió al prior**, que es exactamente
el óptimo de la BCE para un predictor sin señal utilizable. Y como el prior está por debajo de 0,5,
que nunca se abstenga no es un umbral mal elegido: es la **consecuencia necesaria** de haber aprendido
la tasa base en vez de la pertenencia.

Las dos lecturas que el diagnóstico existía para separar quedan resueltas del lado (a): el slot no
aprendió nada sobre pertenencia. No es que la señal esté y no cruce el umbral —AUC 0,52 es azar—.
S-2 lo confirma por la otra vía y sobre otra señal: el score del archivo tampoco separa (0,5369 /
0,5078 / 0,5235 contra la basal 0,4984 del 16-ago), o sea la memoria no aprendió pertenencia ni
siquiera de forma que no llegue a la salida.

**Esto es coherente con el A-3b del propio prereg**, el control que dio vuelta un falso positivo antes
de escribir las predicciones: en un modelo que no lo entrenó, el slot no detecta ausencia. Ahora
sabemos que entrenarlo 26000 pasos con supervisión densa tampoco alcanza.

## 2.bis El control del umbral: la objeción se levantó, se probó y NO sobrevivió

Al revisar contra `DISENO_ATRIBUCION.md` apareció una contradicción entre el diseño y la
implementación. El §3 del diseño dice, textual y por adelantado:

> *«"Que gane el slot nulo" NO puede ser el criterio de abstención... El nulo tiene que competir por
> masa relativa, no por victoria.»*

Y la implementación hace exactamente eso: el logit binario es `log(m/(1−m))` (`modelo.py:306`) y la
decisión es `a > 0` (`entrenar.py:123`), o sea **masa > 0,5** — con 41 entradas, que el nulo se lleve
más atención que las otras cuarenta juntas. Más exigente todavía que «ganar». Eso abría una objeción
seria al veredicto: `cabeza` decide con un escalar libre y `slot` con un número atado a un softmax
que suma 1, así que podrían no haber competido en igualdad de condiciones.

**Se probó** (`umbral_slot.py`, post-hoc y sin estatus de prereg), barriendo 400 umbrales sobre la
masa:

| | AUC de la masa | umbral 0,5 heredado | mejor umbral posible (Youden) |
|---|---|---|---|
| `y3_s0` | 0,5190 | `nose` 0,0000 · `falsa_abst` 0,0000 | th 0,392 → `nose` 0,7961 · **`falsa_abst` 0,7498** |
| `y3_s1` | 0,5313 | 0,0000 · 0,0000 | th 0,397 → 0,7153 · **0,6374** |
| `y3_s2` | 0,5182 | 0,0000 · 0,0000 | th 0,383 → 0,8215 · **0,7834** |

**Ningún umbral pasa la compuerta S-1 en ninguna semilla.** El mejor punto alcanzable abstiene ~80 %
de las veces cuando no hay respuesta y ~75 % de las veces cuando **sí** la hay: J entre +0,038 y
+0,078, que es ruido. La masa no separa las dos poblaciones, y por eso ningún corte puede.

**Lectura, declarada antes de correr:** es la (a). La restricción `masa > 0,5` contradice al §3 del
diseño y hay que arreglarla si esta condición se vuelve a tocar, pero **no explica el resultado**. El
negativo es del mecanismo. La objeción queda como nota al pie y el trípode cierra limpio.

Vale registrar el orden: la contradicción se encontró leyendo el diseño **después** de haber
reportado el veredicto, se declaró como objeción propia al resultado que ya se había dado por bueno,
y se corrió el control que podía darla vuelta. No la dio vuelta.

## 3. El §5 decide, y estaba comprometido por adelantado

> **S-1 falla con S-0 pasando** → la memoria **no** es mejor lugar que la cabeza. El trípode queda
> cerrado con `cabeza` ganando, que ya es publicable, y **esta línea no se reintenta con una cuarta
> forma de slot**.

Es la celda que salió, y con S-0 pasando 3/3 sin ambigüedad. **El trípode `token` / `cabeza` / `slot`
queda cerrado**: `cabeza` pasa la compuerta en 4 de 5 unidades, `token` y `escala` fallan en 5 de 5, y
`slot` falla en 3 de 3 con el mecanismo identificado.

**Un riesgo declarado que NO se materializó, y vale anotarlo:** el §6 temía que el slot le robara masa
a la integración y rompiera la lectura. El slot se lleva el **40 % de la masa en todas las consultas**
y aun así el acierto es 0,90 / 0,79 / 0,90 y `anterior` no se cae (incluso sube 0,111 en s2). O sea la
lectura tolera perder cuatro décimos de su masa a un vector constante. Es observacional y no estaba
preregistrado, pero acota el hallazgo del 16-ago sobre integración.

**`v_nulo`, el reporte del §6 que no es criterio:** partía de cero exacto y quedó en norma 1,3382 /
1,5701 / 1,7106. O sea el modelo **no** lo usó como compuerta pura —restarle masa a la lectura— sino
que además le aprendió un valor. Lo cual, dado que la masa es constante, significa que aprendió a
inyectar un vector fijo en toda respuesta: funciona como un sesgo aprendido de la lectura, no como una
señal de ausencia.

## 4. El instrumento estaba roto, y era el mismo error que ya se había arreglado una vez

El cierre automático del 24 evaluó estas unidades con `ser.py` tal como estaba, y dio `nose = 0,0000`
**y** `falsa_abst = 0,0000` exactos. `ser.py:89` preguntaba `abst == "cabeza"` y nunca fijaba
`E._ABST`. Con `--abst slot` el entrenamiento usa el camino binario de `cabeza`
(`entrenar.py:337`, `_bin = a.abst in ("cabeza", "slot")`) y pone `NOSE` en −1e9 dentro del softmax de
valores, así que medir con el argmax plano **no puede** emitir abstención: el cero era del instrumento.

Es literalmente el desfase de fechas que el comentario de `ser.py:83-87` documenta haber arreglado
para `cabeza` el 18-ago, repetido con `slot` porque la condición se escribió con `==` en vez de `in`.
Con el arreglo, el acierto de las tres unidades sube (0,7572 → 0,8991 · 0,4307 → 0,7859 · 0,7744 →
0,8999), o sea **el veredicto de S-0 cambia de dudoso a cumple 3/3** — el arreglo favorece al slot y
aun así S-1 y S-2 fallan.

**Regresión verificada antes de usar el arreglo para nada:** `ser.py` sobre `p3_s0` con el código
nuevo da acierto 0,9705 · `err_identidad` 0,0122 · `nose` 0,9119 · `falsa_abst` 0,0082 ·
`nose_rel` 0,9235 · `nose_ent` 0,9016 — **idéntico hasta el último dígito** a lo medido el 22 y el 24.
Ninguna unidad `pre`, `token`, `escala` o `cabeza` cambia.
