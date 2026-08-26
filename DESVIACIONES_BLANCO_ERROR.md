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
