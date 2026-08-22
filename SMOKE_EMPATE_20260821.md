# Smoke del EMPATE DE CLAVE (2026-08-21) — la señal existe, pero no donde la fui a buscar

Chequeo de instrumento previo a cualquier pre-registro, según la regla que dejó el monitor v1 el
20-ago: la señal se verifica que EXISTA antes de escribir predicciones sobre ella. Script
`micro_lm/smoke_empate.py`. Checkpoints `ckpts/rt_congelados/` (regla D-1: nada que se esté
entrenando). 512 muestras por unidad, sólo preguntas con respuesta, generador de prueba 77000+semilla.

Mientras esto corría, `t4_s2` se extendía en Colab. Son cosas distintas y sobre checkpoints distintos,
así que no se repite la D-1 del 20-ago.

## De dónde sale la hipótesis

`INFORME_ROUNDTRIP_20260820.md`: `err_identidad` es **colisión de clave**. Con relación única el
error es 0,005-0,014; con relación repetida, 0,38-0,54 = el azar entre las dos que empatan. La vía
propuesta era buscar eso en la atención sobre el archivo — dos pesos altos y parecidos —, que sería
la primera señal en la **entrada** después de tres vías cerradas que miraban la salida.

## Primera pasada: NEGATIVA, y su diagnóstico es lo que sirve

Métrica `r21 = p2/p1` sobre la distribución de lectura en la posición de la respuesta.

| | `c3_s0` | `c4_s0` |
|---|---:|---:|
| AUC(`r21`; repetida vs única) | 0,4895 | 0,5100 |

Azar. Y el motivo aparece en la tabla, no en el AUC: **`r21 ≈ 0,92` y `gap ≈ 0,016` en las ocho
celdas por igual**, con ~6 entradas válidas (uniforme = 0,167; el top-1 mide ~0,20). La lectura es
**casi plana**: no hay «un top-1 y un top-2», hay seis pesos parecidos siempre, y `r21` está saturado
contra 1 por construcción. Mismo defecto de forma que el monitor v1, en otra parte del instrumento.

Segunda métrica declarada en el acto y con su motivo: `z12 = (s1-s2)/std`, el margen entre los dos
primeros scores **crudos** en unidades de la dispersión del episodio. También azar (0,4857 / 0,5092).

## El diagnóstico que da vuelta el resultado: el instrumento apuntaba a la posición equivocada

En `modelo.tronco` la lectura se inyecta en el bloque 0 **antes** de la conv y del mixer, sobre
`h = emb[x]`. Por lo tanto la query que consulta el archivo es `ln(emb[token]) @ qr`: **función pura
del token de esa posición**, sin una sola operación de contexto delante. La posición de la respuesta
—donde miran `scores_archivo` y las dos primeras pasadas— es el último token de la pregunta, y ahí no
hay nada que matchear. Por eso salía plana en todas las celdas.

**Consecuencia mecánica, y es la que explica el hallazgo de ayer:** el modelo **no puede formar una
query conjunta entidad × relación** en el bloque 0. Consulta el archivo token por token y la
conjunción la resuelve aguas abajo, integrando. Eso *deriva* el atajo de la relación del
`INFORME_ROUNDTRIP` en vez de sólo constatarlo: en la posición del token de la relación la query
matchea a todas las entradas que la comparten, y el empate es entre ellas.

Encaja además con el hallazgo del 16-ago (`INFORME_RANK_HECHO`): el modelo acierta sin que la entrada
correcta gane, porque integra en vez de seleccionar. Una lectura casi uniforme es esa misma cosa
medida por otra cara.

## Tercera pasada: la señal aparece, débil y en la dirección predicha

`z_foco` = margen en la posición de máximo matcheo · `z_min` = el menor margen entre las posiciones
de la consulta (¿existe *alguna* posición con dos entradas empatadas?).

| AUC (repetida vs única) | `r21` | `z12` | `z_foco` | `z_min` |
|---|---:|---:|---:|---:|
| `c3_s0` | 0,4895 | 0,4857 | **0,6367** | **0,6204** |
| `c3_s0` · sólo NO revisados | 0,4545 | 0,4441 | **0,6347** | **0,6498** |
| `c4_s0` | 0,5100 | 0,5092 | 0,5852 | 0,5832 |
| `c4_s0` · sólo NO revisados | 0,4851 | 0,4920 | **0,6426** | 0,6021 |

Signo correcto: la relación repetida tiene **menos** margen (`c3_s0`: `z_foco` 0,5737 repetida contra
0,8018 única).

**El confound de versiones queda descartado, y por el lado incómodo para mí:** un hecho revisado tiene
sus dos versiones en el archivo con la misma entidad y relación, así que empatan y podrían estar
produciendo todo el efecto. Sacándolos, el efecto **sube** en tres de las cuatro medidas (`c4_s0`
pasa de 0,5852 a 0,6426). No es el confound disfrazado.

## Lo que NO autoriza este smoke

- **La magnitud es baja.** AUC 0,58-0,65 detectando la *condición* de colisión. Y sobre el eje que
  haría falta para abstenerse —`err_identidad` vs acierto— da **0,53-0,58**, que es poco: detectar que
  dos entradas empatan no es todavía saber que la respuesta va a salir mal.
- Dos unidades, una semilla cada una, 512 muestras. Los AUC tienen error de muestreo apreciable.
- No se probó ningún nulo. El corte sin etiquetas del 20-ago enseñó que **el nulo es lo que da el
  veredicto**: ahí U-1 «pasaba» 2/8 y eran exactamente las 2 donde el nulo también pasaba.

Sin nulo, esto es una señal candidata, no un resultado. El pre-registro tiene que declararlo antes de
correr y tiene que separar las dos preguntas que acá están mezcladas: *detectar la colisión* y
*convertirla en abstención*.
