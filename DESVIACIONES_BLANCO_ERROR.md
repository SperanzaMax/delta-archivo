# DESVIACIONES · `PREREG_BLANCO_ERROR.md` (SHA `d065838f`) + enmienda (`07191f7f`)

---

## D-B1 · La campaña corre SIN SIEMBRA, igual que su control

**Qué se hizo primero, y estaba mal.** El primer lanzamiento (26-ago 13:38) usó `SEMBRAR=1`, que
copia la base `n3_sX.pkl` y arranca la fase de abstención desde ahí. **Abortó en el primer tramo**,
antes de entrenar un solo paso, con:

> `ABORTA: el checkpoint se entreno con horizonte de lr 20000 y se pidio 26000.`

**Por qué, y el diagnóstico es el que corrige el diseño.** Las configs en disco lo dicen:

| checkpoint | horizonte | pasos |
|---|---:|---:|
| `n3_s0` (la base) | **20000** | 12000 |
| `p3_s0` (el control de este experimento) | **26000** | 26000 |
| `v3_s0` | **26000** | 26000 |

**El control `p3_*` nunca se sembró:** se entrenó desde cero a 26000 pasos, igual que `v3_*`. Es lo
que `ENMIENDA_E1_QUERY_CONJUNTA.md` fijó para esa campaña — las ramas que se comparan tienen que
partir del mismo lugar, porque heredar una base entrenada le daría a una de ellas un encoder que la
otra tiene que desaprender.

**Entonces sembrar no era sólo incompatible con la guarda: rompía el pareo.** Una unidad sembrada
contra un control entrenado desde cero no es una comparación pareada, y el número que saliera no
mediría el blanco sino el punto de partida.

**Corrección:** `SEMBRAR=0`, 26000 pasos, horizonte 26000, desde cero, exactamente como `p3_*`. Las
tres copias sembradas se borraron; **no habían entrenado un solo paso**, así que no hay nada que
descartar de los datos.

**Costo real del error:** una sesión de Colab de la cuenta H, abierta y cerrada sin entrenar. Cero
pasos de cómputo perdidos, porque la guarda de horizonte hizo exactamente lo que existe para hacer.

**Lo que esto dice a favor de las guardas, y vale anotarlo:** dos guardas distintas se activaron hoy
sobre el mismo lanzamiento. La de `blanco`, que yo había escrito mal, habría abortado la campaña por
un motivo espurio y la arreglé antes de lanzar. La de `horizonte`, que ya existía, abortó por un
motivo REAL que yo no había visto — que el control no se sembraba. **La segunda me salvó de correr
26000 pasos × 3 semillas de una comparación que no era pareada.**

---

## D-B2 · El instrumento de la métrica principal se escribió con la campaña ya corriendo, y se declara acá

**27-ago.** `ser.py` no puede medir lo que el §4 pide. Mide en **el** punto de operación del modelo
(`a > 0`, el umbral con el que se entrenó), y dos unidades con distinto blanco operan a coberturas
distintas, así que sus SER no son comparables tal cual. Eso es justamente lo que la métrica del §4
existe para sacar del medio.

Se escribió `ser_cobertura.py`: corre las dos unidades sobre el **mismo** lote (misma semilla de
datos), guarda el logit de la cabeza sin decidir nada, y toma como umbral el cuantil que da la
cobertura pedida, de modo que la cobertura sale exacta salvo empates.

**Lo que NO se movió, y es lo que importa:** las coberturas siguen siendo las tres del §4 (0,60 ·
0,70 · 0,80) y están escritas como constante en el módulo, no como flag. Poder elegirlas por línea
de comandos sería poder elegir el punto de comparación después de ver los datos, que es exactamente
lo que el pre-registro existe para impedir.

**Prueba de humo, declarada como tal.** Para verificar que el script corre se lo pasó por `b3_s0`
(paso 8000) contra `p3_s0` (paso 26000) con **n=256**. Dio el tratamiento mejor en las tres
coberturas (Δ SER −0,0195 / −0,0078 / −0,0078). **Ese número no es un resultado y no se va a
informar como tal:** n=256 es ruido, y sobre todo los pasos no están igualados. La comparación del
§3 es a **presupuesto igualado** y todavía no existe — hay que esperar a que `b3_*` llegue a 26000.

Se anota igual, en vez de borrarlo, porque el orden de los hechos es parte del dato: el instrumento
quedó fijo **antes** de que existieran los checkpoints que va a medir, y si más adelante da lo mismo
que esta prueba de humo, tiene que poder verse que nadie lo ajustó en el medio.

**Un detalle del presupuesto que juega en contra del tratamiento y conviene tener presente:** en esa
prueba el control tenía **más del triple** de pasos. Si el signo se sostuviera a presupuesto
igualado, no sería por ventaja de entrenamiento.
