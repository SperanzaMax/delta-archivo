# HALLAZGO · el modelo SÍ se calla, en 2 de 3 — y la que falla no falla por la abstención

**2026-08-28.** Post-hoc **declarado como tal**, sobre `ser_cob_s*_20260827.json`, que ya estaban en
disco. No se corrió ningún modelo, no se re-entrenó nada y **no decide ningún veredicto**: los de A5
ya se juzgaron el 27 y quedan como están.

---

## 0. Una hipótesis propia que se cayó, y hay que anotarla primero

La sospecha con la que se empezó era que el piso de la métrica —a cobertura 0,70 con `p_nose` 0,4 hay
que contestar 400 preguntas que no tienen respuesta— **escondía** el efecto de la condición.

**Es falsa.** El piso se le suma **idéntico** a las dos condiciones, así que se cancela en la resta y
el Δ de E-1 estaba bien medido. Verificado punto por punto:

| | Δ SER (E-1) | Δ excedente sobre el piso |
|---|---:|---:|
| s0 | −0,0137 | −0,01375 |
| s1 | −0,1708 | −0,17075 |
| s2 | +0,0653 | +0,06525 |

Son el mismo número. **El §5 del informe de A5 ya lo decía bien** («ese piso se le suma igual a las
dos condiciones») y acá se leyó de más. E-1 sigue en 1/3 y la vía sigue cerrada.

Lo que sí quedó de mirar el piso es un detalle que vale: **el excedente del tratamiento es constante
en las tres coberturas** (0,00575 en `s0` y 0,00625 en `s1`, iguales a 0,60, 0,70 y 0,80), mientras el
del control **decrece** con la cobertura. Todo lo que crece con la cobertura es piso obligatorio.

---

## 1. Lo que sí apareció, y no es un artefacto de la métrica

La métrica de cobertura igualada existe para comparar dos condiciones sacando del medio el punto de
operación. Pero **el punto de operación propio también es un dato**, y estaba calculado en los mismos
archivos sin que nadie lo leyera.

Es lo que el modelo hace **cuando decide solo**:

| unidad | `vigente` | cobertura | SER | `nose` | `falsa_abst` | `err_identidad` |
|---|---:|---:|---:|---:|---:|---:|
| **b3_s0** | 0,9996 | 0,5948 | **0,00050** | **0,9994** | **0,0000** | 0,0000 |
| p3_s0 | 0,9689 | 0,6278 | 0,05175 | 0,9069 | 0,0076 | 0,0132 |
| **b3_s1** | 0,9979 | 0,5940 | **0,00075** | **1,0000** | 0,0008 | 0,0000 |
| p3_s1 | 0,7918 | 0,7798 | 0,30900 | 0,5382 | 0,0034 | 0,1210 |
| b3_s2 | **0,6762** | 0,6058 | 0,20375 | 0,6097 | **0,2473** | 0,0447 |
| p3_s2 | 0,8406 | 0,6875 | 0,18775 | 0,7195 | 0,0349 | 0,0735 |

**`b3_s1` detecta el 100 % de las preguntas sin respuesta con una falsa abstención de 0,0008 y un SER
de 0,00075.** `b3_s0` queda en 0,9994 con falsa abstención **exactamente 0,0000**.

Y eligen bien **cuánto** contestar: 0,5948 y 0,5940 contra una fracción respondible de 0,60. El
control se pasa —0,6278 y 0,7798— y paga contestando lo que no puede.

## 2. Por qué esto importa para la frase del cierre del 27

El cierre del 27 dice:

> «El modelo sabe cuándo no sabe. Lo que falta es convertir eso en la decisión de callarse.»

**Esa frase no describe a `b3_s0` ni a `b3_s1`.** En su punto de operación las dos convierten el saber
en decisión, y la decisión es casi perfecta. Describe a `b3_s2`, y a las unidades de las campañas
anteriores.

La frase correcta sobre este banco es más incómoda y más útil: **el modelo lo convierte en decisión en
2 de 3 semillas, y en la tercera se rompe.** Que es, exactamente, el resultado sobre **varianza** que
el §3 del informe de A5 puso como lo más importante de la campaña. La conclusión del día se escribió
más pesimista que los propios datos.

## 3. Y la que se rompe no se rompe por la abstención

`b3_s2` es la única que falla, y su firma no es de un detector malo:

- **`vigente` 0,6762.** No aprendió la tarea. E-0, que era **bloqueante**, pide ≥ 0,70 y ella no llega.
- **`err_identidad` 0,0447**, que `lat2` había llevado a 0,0000 y que en `s0` y `s1` sigue en 0,0000.
- **`falsa_abst` 0,2473**, o sea se calla en preguntas que **sí** tenían respuesta.

Eso no es un modelo que no sabe callarse. Es un modelo que **no encuentra lo que está archivado**, y
que en consecuencia se calla de más. La abstención se degrada **detrás** de la recuperación.

> **Cuando la indexación anda, la abstención sale casi gratis. Cuando la indexación se rompe, la
> abstención se rompe con ella.**

Es la confirmación cuantitativa del §2 del `PLAN_FOCO_20260824.md`, que lo había afirmado por otra
vía: *los tres frentes son tres nombres para preguntar cómo indexa y consulta el archivo*.

## 4. Lo que NO autoriza

- **No revive A5.** E-1 da 1/3 con la cuenta corregida y el criterio de abandono del §7 se aplica
  igual. La vía del blanco sigue cerrada.
- **No dice «el modelo sabe cuándo no sabe».** Todo esto es supervisado: la cabeza se entrenó con
  etiquetas. Sigue vigente el §8 del `PLAN_FOCO` y su cierre de seis meses.
- **Son 2 unidades buenas de 3, con n=4000 y una sola semilla de datos (54321).** No se promedian y no
  se reporta una media. Con la varianza de este banco, `s2` no es un outlier que se descarta: es un
  tercio del resultado.
- **`b3_s2` no pasó E-0**, que era bloqueante. Que se la haya contado igual dentro de E-1 no cambia el
  veredicto —sin ella E-1 queda 1 de 2— pero conviene anotarlo para el próximo pre-registro: **una
  guarda bloqueante por unidad tiene que decir qué se hace con la unidad que la falla.**

## 5. La dirección que esto sugiere

No es «cómo hacer que el modelo se calle». En 2 de 3 ya se calla, y muy bien.

Es **por qué `s2` no aprende la tarea**, que es un problema de estabilidad de entrenamiento y de
indexación, no de mecanismo de abstención. Y esa pregunta no está bloqueada por ningún cierre.
