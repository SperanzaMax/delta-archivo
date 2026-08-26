# DESVIACIONES · `PREREG_DOS_DETECTORES.md` (SHA `91494aa0`)

---

## D-D1 · Las unidades principales y la réplica se INTERCAMBIAN

**Qué decía el prereg.** §3 fijaba `v3_s0/s1/s2` (`lat2`) como unidades principales y `p3_*` (`pre`)
como réplica, con este argumento: *«`lat2` es la única condición donde la query es conjunta, así que
es la única donde la pregunta "¿lo recuperado matchea las dos componentes?" tiene sentido»*.

**Qué se hizo.** Las principales pasan a ser **`p3_s0/s1/s2`** y `v3_*` pasa a réplica.

**Por qué.** El censo de D-0 —que el prereg autoriza correr antes que nada— muestra que en `lat2`
**el blanco de D-2 no tiene casos**:

| unidad | `err_identidad` | `invento` | casos de `err_identidad` en 512 |
|---|---:|---:|---:|
| `v3_s0` (`lat2`) | **0,0000** | 0,1898 | **0** |
| `p3_s0` (`pre`) | 0,0101 | 0,0370 | 3 |
| `p3_s1` (`pre`) | 0,2361 | 0,5121 | 72 |
| `p3_s2` (`pre`) | 0,1406 | 0,3073 | 45 |

Es consecuencia directa de un resultado ya publicado y que yo tenía delante cuando escribí el
prereg. `lat2` cerró bien el 25-ago justamente **eliminando la colisión de clave**, con
`err_identidad` en 0,0000 en las tres semillas. **Pedirle a una sonda que prediga un fallo que la
condición eliminó es pedirle que prediga la nada.** El error es mío y es de no haber cruzado dos
documentos propios.

**Por qué esto NO es elegir la unidad por el resultado.** El criterio es **poder estadístico**, no
desenlace: se elige dónde el fallo *ocurre*, y eso se sabe del censo de tasas base sin haber
ajustado ni evaluado una sola sonda. Ninguna sonda se corrió antes de tomar esta decisión, y el
criterio de D-2 (AUC ≥ 0,70 y ventaja ≥ 0,05) **no se toca**.

**Consecuencia sobre la lectura, declarada acá.** En `v3_*` la pregunta de D-2 queda **no evaluable
por construcción**, y eso hay que reportarlo como tal —no como fallo— igual que se hizo con `post` en
la sonda de abstención del 22-ago. Lo que sí se puede leer en `v3_*` es D-1 reducido a un solo
fallo, y eso es informativo por su cuenta.

---

## D-D2 · El censo preliminar se corrió con n=512, no con la n del prereg

**Qué se hizo.** Para decidir D-D1 se corrió el censo con `--n 512` en cuatro unidades, en vez de
los 6000 del §3.

**Por qué.** Era la pregunta «¿hay casos?», que se contesta con dos órdenes de magnitud menos de
muestras y cuesta minutos en vez de una hora. Las tasas base no necesitan precisión de tercer decimal
para decidir si un blanco tiene 0 casos o 72.

**Consecuencia.** **Los números de ese censo no entran en ningún veredicto.** D-0 se juzga con la
muestra completa de 6000, y ahí ya se ve una diferencia que hay que vigilar: con n=512, `p3_s0` da
`nose` 0,963 contra 0,9119 publicado, que excede el ±0,02 del criterio. Si con n=6000 sigue fuera de
banda, **D-0 es bloqueante y no se reporta nada más hasta entender por qué**.

---

## D-D3 · EL BLANCO DE D-1 ESTABA CONTAMINADO — el número −0,3174 no es un resultado

**Qué decía el prereg.** §4, D-1 define el blanco compuesto como «el modelo se equivocó», o sea
`invento` **o** `err_identidad` **o** `err_version` contra `acierto` y `acierto_nose`.

**Qué salió, y por qué no se puede leer.** En `p3_s1` el compuesto da AUC 0,4385 contra 0,7559 del
único, o sea $\Delta = -0,3174$. Antes de anotarlo como «componer es peor», el número tiene una
contradicción interna que lo invalida:

| | |
|---|---:|
| fracción «sin respuesta» entre los **errores** | 0,6063 |
| fracción «sin respuesta» entre los **aciertos** | 0,3123 |
| → «sin respuesta» debería predecir error con AUC **> 0,5** | |
| AUC de la sonda de ausencia sobre el blanco de error | **0,3453** |
| AUC de la cabeza sola sobre el blanco de error | **0,3412** |

Las dos **invertidas**, y las dos predicen ausencia bien (0,8403 y 0,8532). No pueden ser las dos
cosas a la vez, así que el problema es el blanco.

**La causa, y es un defecto de diseño mío.** `clasificar` asigna `acierto_nose` cuando el modelo se
abstiene y no había respuesta, y `abstencion` cuando se abstiene y sí la había. **Ninguna de las dos
cuenta como error.** Entonces `invento` sólo existe donde **la cabeza ya decidió no abstenerse**, y
dentro de las preguntas sin respuesta el logit alto va siempre a `acierto_nose` y el bajo a
`invento`.

> **El blanco está condicionado a la decisión del detector que se quiere evaluar. Es circular.** Un
> detector no puede juzgarse contra un blanco que él mismo ya modificó.

**El arreglo, y es también la formulación correcta de la pregunta.** El blanco pasa a ser
**«si el modelo contestara un valor, ¿estaría mal?»**, o sea `argmax` sobre los logits con `NOSE`
excluido, comparado con el target, **sin mirar la cabeza**. Para `tgt == NOSE` cualquier valor está
mal por definición. Eso es exactamente lo que un detector tiene que anticipar y no depende de lo que
el detector decidió.

**Alcance del arreglo.** Se rehace **sólo D-1**, en una corrida aparte y con el instrumento nuevo
declarado. **D-2 y D-3 no se tocan** y siguen siendo válidos: D-2 usa `err_identidad` restringido a
preguntas con respuesta, donde la abstención es 0,06 % en `p3_s1` y 2,8 % en `p3_s2`, o sea
prácticamente no hay condicionamiento. Se reporta esa tasa junto al resultado en vez de suponerla
despreciable.

**Lo que NO se hace:** no se toca el criterio de D-1 (ventaja ≥ 0,05 en ≥ 2/3), ni las semillas de
generación, ni las featuras. Sólo cambia la definición del blanco, y el motivo está escrito acá antes
de volver a mirarlo.

---

## D-D4 · D-0 se juzga contra el instrumento oficial EN LA MISMA MUESTRA, no contra un número publicado

**Qué decía el prereg.** §4, D-0 pedía reproducir las métricas publicadas «dentro de ±0,02».

**Qué pasó.** Con n=6000 las tres unidades reproducen `nose` y `falsa_abst` sin problema, pero
`acierto` queda fuera de banda en dos: `p3_s1` da 0,8004 contra 0,7769 publicado (+0,0235) y `p3_s2`
0,8555 contra 0,8351 (+0,0204).

**Por qué el criterio estaba mal puesto, y es un defecto del prereg.** El número publicado se midió
con **n=2000 y semilla 54321**; el mío con **n=6000 y otra semilla**. Son dos muestras
independientes, y el error estándar combinado de la diferencia es ≈ 0,014, así que **±0,02 exige una
concordancia más fina que el propio ruido de muestreo**. El criterio no podía cumplirse de forma
confiable ni con un instrumento perfecto.

**El control que lo resuelve, y es exacto en vez de argumentativo.** Se corrió `ser.py` —el
instrumento oficial, que fija `_DONDE` y `_ABST` desde el checkpoint— con **exactamente la semilla y
la n de esta corrida** (77001, 6000) sobre `p3_s1`:

| | `ser.py` oficial | esta sonda |
|---|---:|---:|
| `acierto` | **0,8004** | **0,8004** |
| `nose` | **0,5426** | **0,5426** |
| `falsa_abst` | **0,0006** | **0,0006** |
| n con respuesta | **3593** | **3593** |

**Coinciden dígito a dígito en las cuatro cifras.** La sonda reconstruye la predicción desde
`logits` y `ab` en vez de llamar a `E.predecir_cabeza`, y aun así da lo mismo, que era justamente lo
que había que demostrar.

**Consecuencia.** **D-0 PASA.** La discrepancia con el número publicado es de muestreo y no de
instrumento, y queda demostrado con una identidad exacta, no con una estimación de error. Para
próximos preregs, el criterio correcto es *«reproduce el instrumento oficial sobre la misma muestra»*
y no *«se parece a un número medido en otra muestra»*.
