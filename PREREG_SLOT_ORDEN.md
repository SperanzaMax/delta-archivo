# PRE-REGISTRO · el slot nulo con gradiente de ORDEN · «enseñarle a buscar distinto»

**2026-08-31, noche.** Se congela **antes de lanzar**. Idea de Maxi, de dos pasos en la misma
conversación: *«¿y si le ponemos una cabeza que se ocupe sólo de buscar y devolver la respuesta o no
tenemos esa información?»* y *«¿y si le enseñamos a buscar diferente, que no todo sea lo mismo?»*

> **Enmendado la misma noche, ANTES de lanzar y DESPUÉS de correr la compuerta W-0, que es CPU y no
> mira ningún resultado del tratamiento.** La compuerta abrió en lo que decidía —(c): todo el
> gradiente va a la búsqueda y **0,0 exacto** a la salida— y obligó a corregir tres cosas: el peso
> pasa a ser **por semilla** (§3.1), la **saturación de la siembra** se declara porque cambia lo que
> el experimento puede medir (§4.1), y **W-3 se reescribe** porque su versión original se cumplía sola
> en la siembra. Todo en `INFORME_COMPUERTA_SLOT_20260831.md`.

---

## 1. La pregunta, y por qué no es repetir el 25-ago

> El slot nulo le da al softmax del archivo la opción «ninguna», que es lo único que puede
> representar el vacío. **Fracasó convergiendo al prior.** ¿Fracasó porque la ausencia no es
> representable, o porque **nada le exigía discriminar**?

**Los tres fracasos documentados, y cada pieza de este diseño ataca uno:**

| fecha | qué se probó | cómo falló | qué faltaba |
|---|---|---|---|
| 24-25 ago | **slot nulo** + BCE | convergió al **prior** (0,4074 / 0,4046 / 0,4020 contra tasa base 0,4048) | la BCE premia acertar el prior; con señal débil se queda ahí |
| 29 ago | `balance` y `ranking` | se fueron a **inventar** (invento hasta 0,1966) | tocaban sólo al vigilante, nada decía que errar fuera peor que callarse |
| 31 ago | `recompensa` + **orden**, interfaz `token` | ordenó (AUC 0,51→0,66) y **no alcanzó** | el logit de `NOSE` sale del vocabulario, al final del circuito |

**Nunca se corrieron juntas.** Y la combinación no es una cuarta variante: cada componente tapa el
agujero por el que se cayó una de las tres.

## 2. El argumento mecánico, que es lo que hace distinto a este intento

Con `--abst slot` (`modelo.py:303-306`) el logit de abstención **es la masa del slot dentro del
softmax del archivo**, no una proyección del estado final:

```python
m = clip(masa_nulo["p"], 1e-6, 1-1e-6)
return logits, log(m / (1 - m))
```

**Consecuencia:** el gradiente del término de orden llega **directamente a `k_nulo`, a `qr` y a
`kw`** — al mecanismo de búsqueda— en vez de a la matriz de salida. Hoy, con `token`, el término de
orden empujaba el logit de `NOSE` en `head`, que está después de que el softmax ya aplastó la
evidencia.

> **Eso es «enseñarle a buscar distinto» en el sentido literal: el gradiente que pide discriminar
> ausencia de presencia entra en la búsqueda, no en la salida.**

**Y explica por qué esto puede funcionar donde lo de hoy se quedó en 0,66.** Medido esta tarde: la
ausencia **no está** en los scores del archivo (`s_max` AUC **0,5115**, sonda sobre el vector
completo **0,5065**, nulo 0,4803 — azar por cuatro vías). El término de orden sobre `slot` no
**lee** esa señal: le pone gradiente al mecanismo para que la **cree**. Precedente medido del propio
proyecto: el blanco `error` da **0,65 post-hoc** y **1,0000 entrenado**.

## 3. Diseño

| | `--abst` | `--perdida-cabeza` | `--rec-rank` | prefijo |
|---|---|---|---:|---|
| **CONTROL A**, ya en disco | `token` | recompensa | 0,008 | `r03_s3` · `r03_s6` |
| **CONTROL B** | `slot` | recompensa | **0,0** | `k03_s3` · `k03_s6` |
| **TRATAMIENTO** | `slot` | recompensa | **1,56** (s3) · **5,45** (s6) | `w03_s3` · `w03_s6` |

**El CONTROL B es el que hace válido el resultado** y por eso se corre aunque cueste GPU: separa
«el slot ayuda» de «el orden ayuda». Sin él, un tratamiento que cumpla no dice cuál de las dos piezas
fue.

Todo lo demás heredado e idéntico: sembradas desde `b3_s3`/`b3_s6` (que traen `k_nulo` y `v_nulo`
**sin un solo gradiente encima**, verificado en W-0(a)), `L=0`, `M=0,5`, `F=0,2`, `CE=1,0`,
`p_nose=0,4`, nivel 3, lr 1e-3, horizonte 12000, **3000 pasos**. **Cuatro unidades.**

### 3.1 · EL PESO VA POR SEMILLA, y el motivo se declara acá

El criterio quedó escrito antes de medir —igualar el `|g|` que el término de orden pone en `k_nulo`
con el `|g|` que la pérdida base pone en `kw`, en el **checkpoint de siembra**—. Al aplicarlo dio
**1,5617** en `b3_s3` y **5,4459** en `b3_s6`: **dispersión 3,49×**, contra **1,12×** que dio el mismo
criterio con `token` esta mañana sobre estos mismos dos checkpoints.

**Por eso no se promedia.** Promediar daría 3,5, un número que no describe a ninguna de las dos
unidades, y encima coincide por casualidad aritmética con el ratio de gradientes del 30-ago, que es
un cociente de otras dos cosas. **Cada unidad corre con el peso derivado en SU propia siembra**, que
es el criterio ya declarado aplicado sin el promedio, no un criterio nuevo.

**Y la dispersión no es ruido: es el hallazgo 1 del informe de la compuerta.** El `|g|` en `k_nulo`
depende de cuántas muestras zafan de la saturación del slot, que es 16 % en `s3` y 37 % en `s6`.
**Se declara antes: si el resultado se parte entre las dos semillas, el peso es sospechoso primero.**

## 4. Predicciones, fijadas ANTES

**W-0 · COMPUERTA, YA CORRIDA · `INFORME_COMPUERTA_SLOT_20260831.md`.** Antes de gastar GPU:
(a) `k_nulo` y `v_nulo` sin un solo gradiente encima **CUMPLE** —coseno 1,0000 exacto con su propio
`init_params`, razón de normas 0,866972 en las dos: lo que se movió fue **weight decay**, no
gradiente—; (b) la masa del slot no es constante **CUMPLE**, con la salvedad del §4.1; (c) el término
de orden recibe gradiente en `k_nulo`, en `qr` y en `kw`, y **0,0000e+00 exacto en `head`**:
**CUMPLE**. **Todo el gradiente entra en la búsqueda y nada en la salida**, que es lo que el diseño
prometía y lo que lo distingue de la corrida de hoy con `token`.

### 4.1 · LA SIEMBRA ARRANCA SATURADA, y eso cambia lo que el experimento puede medir

Se declara **antes** porque lo cambia: la masa del slot en la siembra vale 0 o 1 y casi nada en el
medio —**84 % y 63 % de las muestras pegadas al clip** de `modelo.py:306`, y el logit toma **10 y 8
valores distintos sobre 64**—. El log-odds real, sin clip, va de **−834 a +395**: el slot compite
contra un `logsumexp` cuyo rango es de cientos de nats, porque entra a un softmax **ya entrenado sin
él**. El término de orden arranca en **3,4056** y **5,6468** contra `log 2 = 0,6931`, o sea **5 a 8
veces peor que cualquier constante** (con `token` arrancaba en 0,83 y 1,03).

> **La pregunta que se contesta deja de ser «¿el gradiente de orden crea la señal de ausencia?» y
> pasa a ser «¿puede primero DESATURAR el slot y después crear la señal?».** Es más duro, y es la
> tarea real: el término tiene gradiente sobre `kw` y `qr`, así que **puede** comprimir la escala de
> los scores del archivo. Que pueda no quiere decir que le alcancen 3000 pasos, y eso ya está
> cubierto por W-7.

**W-1 · PRINCIPAL.** El **tratamiento** supera **0,65** de AUC del logit de abstención contra la
ausencia, en las dos semillas. Hoy `token` con orden da 0,6620 y 0,6681, así que el umbral **no es
regalado**: pide igualar lo mejor que hay y hacerlo desde la búsqueda.

**W-2 · MECANICISTA, y es la que decide.** El tratamiento supera al **CONTROL B** en AUC por
**≥ 0,05** en las dos semillas. **Es lo único que atribuye el efecto al ORDEN y no al slot.**

**W-3 · REESCRITO POR LA COMPUERTA, y el motivo va escrito.** La versión original pedía que la masa
del slot «dejara de estar clavada en el prior», con desvío > 0,10. **Ese criterio ya se cumple en la
siembra —0,364 y 0,462— y por SATURACIÓN, no por graduación**, así que tal como estaba no podía
reabrir ni cerrar nada: un slot que vale 0 o 1 tiene desvío enorme y no gradúa. Se reemplaza por lo
que aquel criterio quería decir:

> **La fracción de muestras pegadas al clip BAJA de la siembra (0,8438 en s3 y 0,6250 en s6) a menos
> de 0,50 en las dos semillas, y el logit toma más de 32 valores distintos sobre 64.** O sea: el
> término desatura el slot y lo vuelve una variable graduada.

**Si sigue saturado, la masa del slot no puede representar grados de ausencia**, y eso cierra la vía
de la búsqueda con evidencia más fuerte que la del 25-ago, porque acá el gradiente entró **directo**
al mecanismo y aun así no la desaturó.

**Y se declara la relación con los otros criterios, que es la lección de O-6 de hoy: W-3 NO es
precondición de W-1 ni de W-2.** Un logit de dos valores puede tener AUC alto si parte bien la
muestra, así que el tratamiento podría cumplir W-1 y W-2 con W-3 fallando. En ese caso la lectura es
que **el slot funciona como una bandera binaria y no como una medida**, y así hay que informarlo.

**W-4 · CONTROL DEL 29-AGO.** `invento` no supera al del **CONTROL B** por más de 0,02. Si el orden
vuelve a desacoplar la decisión del valor, se ve acá.

**W-5 · NULO.** RECUP no cae más de 0,05 respecto del origen (0,3654 / 0,3835).

**W-6 · PRECONDICIÓN, y se declara como tal por la lección de hoy.** La abstención tiene que quedar
**estrictamente entre 0,05 y 0,95**. **Si se va a un extremo, W-1, W-2 y W-4 son NO EVALUABLES**, no
«fallan» — es el octavo defecto de pre-registro del mes y no se repite: **O-6 era la precondición de
otros tres criterios y sólo uno lo declaraba.**

**W-7 · RIESGO.** 3000 pasos pueden no alcanzar. Si W-1 falla pero el término de orden bajó de
`log 2 = 0,6931`, es **presupuesto y no un negativo**, igual que O-7 hoy.

## 5. Cómo se lee cada desenlace, escrito ANTES

| desenlace | lectura | qué se hace |
|---|---|---|
| **W-1, W-2 y W-3 cumplen** | la ausencia SÍ es representable en la búsqueda, y lo que faltaba era el gradiente de orden | es el resultado de la línea; reabre el slot nulo y explica el 25-ago |
| **W-1 sí, W-2 no** | el slot alcanzaba solo | el mérito es del slot, no del orden; se informa así |
| **W-1 y W-2 sí, W-3 falla** (sigue saturado) | el slot sirve de **bandera binaria**, no de medida de grado | se informa así, sin venderlo como representación de la ausencia |
| **W-3 falla y W-1 también** | la ausencia **no** es representable ahí, ni siquiera con el gradiente entrando directo a la búsqueda | **cierra la vía de la búsqueda**, con evidencia más fuerte que la del 25-ago |
| **el resultado se parte entre s3 y s6** | el peso, que difiere 3,49× entre las dos, es el primer sospechoso | no se adjudica por semillas: se revisa el §3.1 antes que nada |
| **W-4 se dispara** | el orden desacopla del valor, tercera vez | se cierra el orden como mecanismo |
| **W-6 se rompe** | extremo | **NO EVALUABLE**, se extiende o se ajusta el nivel; nada se adjudica |

## 6. Lo que NO contesta

- **No hace que el modelo sepa cuándo no sabe.** Sigue supervisado.
- **No dice que escale.** El slot nulo es una entrada más en un archivo de 40; con archivos grandes
  su masa relativa cambia y eso no se mide acá.
- **No arregla la recuperación**, que el informe de hoy dejó como el techo real (ausencia
  decodificable a 0,70 del estado final).
- **3000 pasos**, semillas sin base, comparables sólo contra sí mismas.
