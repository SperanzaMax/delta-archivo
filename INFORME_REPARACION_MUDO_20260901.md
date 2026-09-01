# El atractor mudo NO es reparable: «absorbente» queda bien puesta

**2026-09-01.** Pedido de Maxi. `PREREG_REPARACION_MUDO.md` (SHA `a410d47a`), congelado antes de
sembrar. Seis unidades a 6000 pasos, compuerta pasada (los seis sembrados arrancan en abstención
1,0000 exacta, heredando el atractor).

## 1. El resultado

| unidad | brazo | paso | vigente | `nose` | falsa_abst | abstención |
|---|---|---:|---:|---:|---:|---:|
| `rp3_s3` | ranking | 6000 | 0,0000 | 1,0000 | 1,0000 | **1,0000** |
| `rp3_s6` | ranking | 6000 | 0,0000 | 1,0000 | 1,0000 | **1,0000** |
| `rp3_s7` | ranking | 6000 | 0,0000 | 1,0000 | 1,0000 | **1,0000** |
| `rc3_s3` | bce (control) | 6000 | 0,0000 | 1,0000 | 1,0000 | **1,0000** |
| `rc3_s6` | bce (control) | 6000 | 0,0000 | 1,0000 | 1,0000 | **1,0000** |
| `rc3_s7` | bce (control) | 6000 | 0,0000 | 1,0000 | 1,0000 | **1,0000** |

**E-1 NO CUMPLE, 0 de 3.** Ninguna baja de 0,90; todas quedan en 1,0000 exacto. **E-2 da 0**: el
control tampoco sale, así que ni siquiera hay un efecto de reinicio que atribuir.

> **El atractor mudo es ABSORBENTE en el sentido fuerte: una unidad que cayó no sale, ni cambiándole
> la función de pérdida, ni reiniciando Adam, ni con warmup nuevo.** La palabra que la línea usó desde
> el 29-ago queda bien puesta, y ahora probada en la dirección que podía refutarla.

## 2. El contraste que le da valor, y es el hallazgo práctico

La **misma** pérdida `ranking`, sobre las **mismas** semillas:

| | qué hace | resultado |
|---|---|---|
| **PREVENCIÓN** — entrenar desde cero con `ranking` | evita que caiga | **4 de 4 mudas emiten** (29-ago) |
| **REPARACIÓN** — aplicarla a una unidad ya muda | sacarla del atractor | **0 de 3** |

> **La intervención funciona sólo antes.** No es que `ranking` sea débil: es que **hay una ventana
> temporal**, y una vez cerrada el estado no se revierte con la herramienta que lo habría evitado.

Encaja con el predictor validado hoy sobre 76 unidades del banco —quien emite **0 de 512** respuestas
en el paso ~2500 termina mudo 4 de 4; quien emite ≥1 termina mudo 0 de 72—: **el desenlace se decide
temprano y después ya no se negocia.**

## 3. Por qué el negativo es fuerte y no un artefacto de presupuesto

Es la objeción obvia («6000 pasos son pocos») y hay tres razones para descartarla:

1. **La comparación es contra sí misma.** Las `rk3` entrenadas desde cero salieron del silencio en
   **1000 pasos** (medido en el smoke de esta mañana: 1,0000 → 0,6387). Acá 6000 pasos no movieron
   nada, en ninguna de las seis.
2. **No hay tendencia.** No es que bajen despacio: la abstención es **1,0000 exacta** en todos los
   hitos de las seis unidades. Un presupuesto corto produce una pendiente floja, no un cero exacto.
3. **El control da lo mismo**, así que no hay ninguna señal incipiente que un presupuesto mayor
   pudiera amplificar.

**Aun así se declara el límite:** esto dice que no se repara **en 6000 pasos con estas dos pérdidas**.
No prueba que sea imposible con otra intervención —una sacudida de `lr`, reinicializar sólo la cabeza,
o volver a `p_nose`=0 un tramo— y ésas quedan abiertas.

## 4. Consecuencia para la línea

**Todo el esfuerzo de abstención tiene que ir a la PREVENCIÓN**, y el momento es el arranque. Las
unidades sembradas desde `b3_s3`/`b3_s6` —que son 22 de las 31 corridas posteriores al 29-ago— parten
de un estado del que **está probado que no se sale**. Eso no invalida lo que midieron, pero acota qué
puede salir de ahí: **cualquier intervención evaluada sobre una unidad muda mide su capacidad de
resucitar, no su capacidad de enseñar.**
