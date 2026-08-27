# INFORME · Fase 0 de la ausencia de la RELACIÓN — pasa R-1, y el control la desarma

Evalúa el §3 de `PREREG_AUSENCIA_RELACION.md` (SHA `86870655…`), congelado antes de escribir el
instrumento. CPU, checkpoints ya entrenados, cero GPU.

## 1. Resultado formal

Positivos `tipo == 0` (entidad presente, relación presente) contra negativos `tipo == 3` (entidad
presente, relación **ausente**), con `nose_ent` excluido. 6215 casos por unidad.

| unidad | bloque (R-1) | permutada (R-0) | ciega | R-1 |
|---|---:|---:|---:|---|
| `v3_s0` | 0,8054 | 0,4790 | 0,5165 | CUMPLE |
| `v3_s1` | 0,8130 | 0,4893 | 0,5165 | CUMPLE |
| `v3_s2` | 0,8972 | 0,4537 | 0,5165 | CUMPLE |

**R-1 cumple 3/3**, con R-0 pasando y sin fuga por longitud.

**Nota de instrumento.** Con n chico (807 casos) R-0 **falló** (permutada 0,5613): el bloque tiene 259
dimensiones y la sonda sobreajustaba. Se corrigió con más muestras, no bajando el umbral. La guarda
hizo exactamente lo que existe para hacer.

## 2. Dónde vive la señal, y por qué eso obligó a un control

| señal | s0 | s1 | s2 |
|---|---:|---:|---:|
| `estado` | 0,8071 | 0,8167 | 0,8964 |
| `s_ent` | 0,5564 | 0,5683 | 0,5728 |
| `s_margen` | 0,5595 | 0,5667 | 0,5713 |
| `leido` | 0,5127 | 0,5284 | 0,5329 |
| `s_max` | 0,5030 | 0,5100 | 0,4906 |

**Todo lo que viene del archivo está en azar.** `s_max` y `leido` no separan nada. La señal está
entera en `estado`, que es el vector del que sale el logit de respuesta — y por eso una sonda ahí
puede estar leyendo **la decisión que el modelo ya tomó** en vez de información sobre la ausencia.

## 3. El control post-hoc, y lo que dice

Declarado como post-hoc: se agregó **después** de ver dónde vivía la señal. Es el mismo criterio que
el P-2 del informe del score del archivo (16-ago), que exigía superar a la confianza de salida por
0,03.

| unidad | `estado` | confianza de salida | ganancia |
|---|---:|---:|---:|
| `v3_s0` | 0,8071 | 0,8074 | **−0,0003** |
| `v3_s1` | 0,8167 | 0,8097 | +0,0070 |
| `v3_s2` | 0,8964 | 0,8663 | +0,0302 |

**Dos de tres no aportan nada, y el tercero pasa el umbral por 0,0002.**

> El 0,80-0,89 de R-1 **no es información sobre la ausencia de la relación**. Es la confianza de
> salida del modelo, leída de otra forma.

## 4. Qué se hace con esto

**R-1 cumple formalmente**, así que el §6 habilitaría la condición. **No se lanza igual**, y el motivo
se escribe acá para que la decisión quede trazable: si `estado` no aporta sobre la confianza de
salida, una cabeza que lea `estado` no va a encontrar lo que la salida no tenga ya. Serían 26000 × 3
pasos para redescubrir la calibración.

Queda a criterio de Maxi. La recomendación es **no correrla** y, si se corriera, hacerlo sabiendo que
el techo esperado es el de calibración y no el de capacidad.

## 5. Lo que sí queda establecido, y vale

**El negativo del 16-ago se confirma y se precisa.** Aquel eje era grueso y mezclaba `nose_ent` con
`nose_rel`; la sospecha era que el 0,4984 fuera un artefacto de promediar dos poblaciones. **No lo
era**: controlando por entidad, las señales del archivo siguen en azar (`s_max` 0,4906-0,5100). En la
memoria co-entrenada no hay representación de la ausencia, y ahora está medido en el eje fino además
del grueso.

**Y la pared es la misma que la de A5.** El mismo día, dos líneas independientes —el blanco `error` y
la ausencia de la relación— dieron el mismo diagnóstico: la información existe y no se convierte en
decisión. Eso es más fuerte que cualquiera de las dos por separado.
