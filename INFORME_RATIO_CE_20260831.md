# El ratio de gradientes no es una constante, y el gradiente chico era EQUILIBRIO

**2026-08-31.** Sobre checkpoints ya en disco, **cero GPU**. Instrumentos nuevos:
`micro_lm/medir_ratio_ce.py`, `micro_lm/medir_equilibrio_q.py` y `micro_lm/medir_politica_q.py`.

Esto **no** es una campaña. Es el chequeo previo que el §4 de `PRECISION_RECOMPENSA_L_CE.md`
(SHA `4b61894e`) dejó pedido antes de escribir el pre-registro de `--rec-ce`, y que su §5 convirtió
en regla general:

> antes de contrastar dos valores de un peso, medir cuánto gradiente mueve ese peso contra el resto
> de la pérdida.

El chequeo se hizo, y **el resultado cancela la campaña que iba a habilitar.**

---

## 1. El 3,5 no replica, y no es una constante

Medido con la misma métrica del 30-ago (gradiente respecto de los **logits**, `|g|` medio en la
columna de `NOSE` contra `|g|` medio de las otras 241), `n=1024`, semilla 54321 pareada, `p_nose=0,4`:

| unidad | paso | interfaz | `|g|` NOSE | `|g|` resto | **ratio** | ratio con **CE=0** |
|---|---:|---|---:|---:|---:|---:|
| `b3_s3` origen sembrado | 26000 | cabeza | 1,739e−08 | 6,774e−06 | **389,5** | 58,9 |
| `b3_s6` origen sembrado | 26000 | cabeza | 1,217e−08 | 6,679e−06 | **549,0** | 88,3 |
| `t03_s3` (L=0) | 3000 | token | 4,494e−06 | 6,322e−06 | **1,41** | 0,11 |
| `t53_s3` (L=0,5) | 3000 | token | 5,694e−07 | 6,606e−06 | **11,60** | 0,76 |
| `t03_s6` (L=0) | 3000 | token | 4,974e−07 | 6,412e−06 | **12,89** | 0,92 |
| `t53_s6` (L=0,5) | 3000 | token | 1,872e−07 | 6,431e−06 | **34,35** | 2,61 |

**Tres cosas, y las tres van contra la intervención tal como estaba escrita.**

1. **El ratio varía 390× entre checkpoints del mismo experimento** y **9× entre semillas de la misma
   celda** (`t03_s3` 1,41 contra `t03_s6` 12,89). El 3,5 del 30-ago era una foto de la mitad de una
   corrida —el `PRECISION` se escribió con la campaña en el paso 2000 de 3000— y la trayectoria
   389 → 3,5 → 1,41 es monótona y coherente. **No es una cantidad de diseño: es el estado de los
   pesos en un instante.**
2. **Desde el punto de siembra real el objetivo es inalcanzable.** Las unidades nuevas volverían a
   salir de `b3_s3`/`b3_s6`, donde el ratio es **389-549**, y bajando `--rec-ce` **hasta cero** sólo
   se llega a **59-88**. El §4 pedía igualarlo a 1,0. Por este camino no se puede.
3. **La CE no aporta gradiente a la columna de `NOSE`.** Verificado con dispersión relativa
   **0,00e+00 exacta** en las 6 unidades × 2 valores de `L`: `_recompensa` hace
   `lg_v = lg.at[:, NOSE].set(-1e9)` y el `.set()` corta el camino, así que la CE se calcula sobre
   `lg_v` y su gradiente hacia esa columna es cero por construcción. **Bajar la CE no sube la señal
   que decide callarse; sólo baja la que compite con ella.** La intervención era RELATIVA y eso no
   estaba dicho.

**El único número estable de la tabla es el denominador**, `|g|` del resto: 6,3-6,8e−06 en las seis
unidades. Toda la variación del ratio viene de la columna de `NOSE`.

## 2. El control temporal, que sale gratis de la historia del checkpoint

`t03_s3` guarda 12 hitos. `abstencion` por paso:

| paso | 250 | 500 | 750 | 1000 | 1500 | 2000 | 2500 | 3000 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `abstencion` | 0,0000 | 0,4355 | 0,4688 | 0,4746 | 0,4766 | 0,4805 | 0,4688 | 0,4785 |

**`q` se clava entre el paso 250 y el 500 y no se mueve más.** En esos mismos 2500 pasos el ratio
bajó de ~389 (siembra) a 1,41 (final). **Mejoró dos órdenes de magnitud y la abstención no se movió
ni una centésima.** Si el bloqueo fuera de magnitud, ése era el momento de aflojarse.

## 3. La explicación alternativa, y es la que gana

El instrumento del 30-ago promedió **`|g|`**, y el valor absoluto borra el signo. Con el signo puesto,
las dos hipótesis se separan sobre el mismo lote:

- **falta de magnitud** → `|media|` ≈ `media(|.|)`: cada muestra aporta poco.
- **equilibrio** → `|media|` ≪ `media(|.|)`: cada muestra aporta mucho y la suma se cancela, con las
  dos poblaciones tirando en sentidos opuestos.

Los signos estaban derivados antes de mirar, de `_recompensa` con L=0, M=0,5, F=0,2: con respuesta
`d(rec)/dq = 0,3 − 1,5c` (negativo si `c > c*`), sin respuesta `d(rec)/dq = +0,5` siempre.

`n=2048`, gradiente **con signo** en la columna de `NOSE`:

| unidad | con respuesta | sin respuesta | NETA | BRUTA | **cancelación** |
|---|---:|---:|---:|---:|---:|
| `t03_s3` | +1,036e−06 | −2,482e−06 | −3,882e−07 | 2,128e−06 | **0,8176** |
| `t53_s3` | +5,044e−08 | −3,503e−07 | −1,118e−07 | 2,546e−07 | **0,5610** |
| `t03_s6` | +1,491e−07 | −2,555e−07 | −1,467e−08 | 2,051e−07 | **0,9285** |
| `t53_s6` | +5,508e−08 | −2,259e−07 | −5,866e−08 | 1,385e−07 | **0,5765** |

**Las dos poblaciones tiran en sentidos opuestos en 4 de 4, y entre el 56 % y el 93 % de la magnitud
se cancela.** El gradiente chico en `NOSE` **no es falta de señal: es la firma de que `q` ya está en
el óptimo de la pérdida.** No hay una colina que subir; está en el fondo.

> **Se invirtió la causalidad.** El 30-ago leyó el gradiente chico como la CAUSA del bloqueo. Es su
> CONSECUENCIA. Y explica de una sola vez por qué `L` no movía nada, por qué el resultado era robusto
> a semilla y a origen, y por qué las cuatro unidades caían en el mismo número.

**Precisión, para no decir de más:** la fuerza neta no es cero, es **chica**. En `t03_s3` queda
−3,882e−07 contra 2,128e−06 de magnitud bruta, y el signo empuja hacia **más** silencio (coherente
con que el `c` medio, 0,2805, esté por debajo del 0,4283 que equilibraría un `q` global). No es un
punto crítico exacto: es un residuo cinco veces menor que la señal que se canceló, y ese residuo no
alcanzó para mover `q` en 2500 pasos.

## 3.bis · `q` NO es una constante: es un VOLADO por muestra

Al medir el equilibrio apareció un número que no encaja con el diagnóstico de ayer:

> `q` media **0,4902** · `q` desvío **0,4906**

Un desvío de 0,49 sobre una variable acotada en [0,1] con media 0,49 no puede ser una constante
(desvío ~0) ni una uniforme (0,289): es la firma de una **Bernoulli(0,5)**. `medir_politica_q.py`,
`n=2048`:

| unidad | masa en los extremos | masa en el centro [0,3-0,7) | veredicto |
|---|---:|---:|---|
| `t03_s3` | 0,8560 | 0,0098 | **BIMODAL** |
| `t53_s3` | 0,9790 | 0,0015 | **BIMODAL** |
| `t03_s6` | 0,9844 | 0,0000 | **BIMODAL** |
| `t53_s6` | 0,9888 | 0,0005 | **BIMODAL** |

**El modelo no se calla «a medias» en todas las preguntas: se calla del todo en la mitad de las
preguntas y contesta del todo en la otra mitad.** La «constante ≈0,50» del 30-ago era la MEDIA de una
moneda, no el valor por muestra. **Octava vez en el proyecto que una media esconde su distribución**
(D-012 en E3, la meseta falsa de E1, `hidratada_τ` el 11-ago, E-I3c y E-I3d, entre otras).

**Y contra qué está correlacionado ese volado: contra nada de lo medible.**

| unidad | acuerdo con «no hay respuesta» | azar | acuerdo con «c < c*» | azar |
|---|---:|---:|---:|---:|
| `t03_s3` | 0,4985 | 0,5004 | 0,5156 | 0,4992 |
| `t53_s3` | 0,4995 | 0,4998 | 0,5166 | 0,5005 |
| `t03_s6` | 0,4985 | 0,4994 | 0,5044 | 0,5013 |
| `t53_s6` | 0,4985 | 0,4993 | 0,5093 | 0,5016 |

**Las dos dan el azar exacto, en 4 de 4.**

> **★ Y ACÁ SE CAYÓ UNA HIPÓTESIS MÍA, que es la razón de que el control existiera.** Al ver que
> `frac c > c* = 0,4856` coincidía con `abstencion = 0,4902` escribí que el modelo podía estar
> implementando la política óptima por muestra, y que entonces el cuello sería la recuperación. **El
> acuerdo pareado lo desmiente: 0,5156 contra 0,4992 de azar.** La coincidencia era **marginal y
> falsa**: yo comparaba `frac c > c*` calculada **sólo sobre las que tienen respuesta** contra una
> abstención calculada **sobre todas**. Con la población bien pareada, «debería callarse» tiene tasa
> 0,71-0,75 y «se calla» 0,50: ni siquiera coinciden los márgenes.
> **Regla que ya estaba y volvió a pagar: dos tasas marginales iguales no son un acuerdo. Se mide
> pareado, muestra por muestra.**

## 4. Consecuencia para el orden derivado del `DICTAMEN_GEMINI`

El orden era **(1) bajar `--rec-ce` · (2) castigo superlineal en la confianza · (3) el schedule de
`F`**.

**El (1) se cae, y con el mejor motivo posible: no porque falle, sino porque su premisa no existe.**
No hay señal ahogada que destapar. Cualquier reponderación de términos —`rec-ce`, `L`, `F`, o un
schedule— mueve **dónde está el equilibrio**, y el 30-ago ya midió que mover el equilibrio sólo
cambia el valor de la constante.

**El (2) queda como la única candidata viva, y ahora con una razón mecánica que antes no tenía.** Un
castigo superlineal en `c` es lo único de la lista que **no** es una reponderación global: cambia la
forma de la fuerza **por muestra**, que es la coordenada donde el equilibrio se arma.

**Pero con una advertencia nueva que sale del §3.bis y que no estaba en el dictamen:** un castigo
superlineal en `c` sólo puede ayudar **si la decisión se apoya en `c`**, y hoy no se apoya (acuerdo
0,50-0,52 contra azar). Endurecer la penalidad de una variable que la decisión no está mirando puede
no mover nada. **Antes de escribir ese pre-registro hay que medir contra qué está correlacionado el
volado**, que es lo que hace `sonda_volado.py` y es exploratorio.

## 3.ter · ★★ EL VOLADO NO ES UN VOLADO: LO GOBIERNA LA RELACIÓN PREGUNTADA

`sonda_volado.py`, **declarado exploratorio antes de correr**, `n=4096`. Se agrupa por cada variable
del input y se mide la pureza de la decisión dentro de cada grupo, **contra un nulo con la misma
cantidad de grupos y el mismo reparto de tamaños** — sin ese nulo, agrupar por una variable de muchos
valores infla la pureza sola, que es el defecto del `m=1` del 12-ago.

| variable | grupos | pureza | nulo | **dif** `t03_s3` | **dif** `t03_s6` |
|---|---:|---:|---:|---:|---:|
| tipo de pregunta | 4 | 0,5127 | 0,5130 | −0,0003 | +0,0000 |
| consulta, pos. 3 (artículo) | 3 | 0,7981 | 0,5120 | **+0,2860** | +0,2920 |
| **consulta, pos. 4 (sustantivo de la RELACIÓN)** | 8 | **0,9771** | 0,5178 | **+0,4592** | **+0,4654** |
| consulta, pos. 5 | 7 | 0,5640 | 0,5132 | +0,0508 | +0,0502 |
| consulta, pos. 6 (**entidad**) | 30 | 0,5376 | 0,5325 | +0,0051 | +0,0051 |
| consulta, pos. 7 (**entidad**) | 30 | 0,5217 | 0,5198 | +0,0019 | +0,0014 |

La pregunta es `BOS cual es <art> <sust> de <ent> ?` o `BOS cual era antes <art> <sust> de <ent> ?`,
así que la posición 4 es el **sustantivo de la relación** en las preguntas por la vigente, y la 3 su
artículo, que es su proxy (`RELACIONES[rel] = (sust, _, art)`).

> **★★ Agrupando por la relación preguntada, la decisión de callarse es UNÁNIME dentro del grupo
> (pureza 0,977 y 0,982 contra un nulo de 0,517), y la ENTIDAD no aporta NADA (+0,005 y +0,002).**
> El modelo no tira una moneda: **se calla en unas relaciones y contesta en otras, y no mira de quién
> le están preguntando.**

**Esto explica de un solo tiro todo lo que se venía midiendo por separado:**

| observación | por qué |
|---|---|
| `q` bimodal 0/1 | dentro de cada relación la decisión es unánime |
| media ≈0,50 | se calla en aproximadamente la mitad de las relaciones |
| independiente de la ausencia | la ausencia depende de entidad **y** relación; el atajo sólo lee la relación |
| robusta a semilla, origen y `L` | es un atajo **estructural** de la tarea, no un accidente de la optimización |
| insensible al gradiente | no es cuestión de fuerza sino de **qué variable** mira la decisión |

**Y no es un hallazgo huérfano: es el ATAJO DE LA RELACIÓN, ya probado por intervención el 22-ago**
(`INFORME_BIMODALIDAD_20260822.md`, `sonda_atajo_relacion.py`), donde sustituir la entidad no cambiaba
la respuesta y una sola variable —cuánto mira la entidad— explicaba el acierto de las tres semillas
con error de 1-2 puntos. **Lo nuevo son dos cosas:**

1. **el atajo también gobierna la ABSTENCIÓN**, no sólo la respuesta, y se llega por otra vía
   (pureza por token con nulo, en vez de sustitución de entidad) → convergencia independiente;
2. **en la abstención el atajo está tomado SIEMPRE**: 0,977 y 0,982 en las dos semillas, mientras que
   en la respuesta variaba muchísimo (0,098 · 0,974 · 0,543). **La decisión de callarse cae en el
   atajo con mucha más consistencia que la de responder.**

**★ Y la intervención ya está escrita y nunca se corrió:** el §propuesta del 22-ago pide un
**`--p-colision`** (hoy 0,42 por accidente) subido a 0,9, que baja el techo del atajo de 0,79 a 0,55
«con lo cual el gradiente deja de tener dónde estacionarse». Se propuso para el acierto. **Hoy hay una
segunda razón, independiente, para correrlo.**

## 4.bis · La pregunta que queda abierta, y es mejor que la que había

El forward es determinista, así que el volado **es** una función del input. Sabemos tres cosas de esa
función y las tres son negativas: no es la ausencia de respuesta, no es la confianza, y no es una
constante. **Eso es un objeto mucho más preciso que «`q` es una constante»**, porque un volado
determinista y balanceado tiene que estar leyendo *algo* del input, y ese algo es identificable.

Y reordena la prioridad: mientras la decisión sea independiente de la evidencia, **ninguna función de
pérdida que reponga pesos sobre esa decisión puede calibrarla**. Es el mismo dictamen que R4 le puso
a la geometría, en otro plano: *hay que arreglar de qué depende la decisión, no cuánto cuesta cada
desenlace.*

## 5. Lo que NO dice

- **No mide el gradiente sobre los parámetros**, sino sobre los logits. Se eligió así para que los
  números fueran comparables contra la tabla del 30-ago. Un `|g|` chico en el logit no implica
  mecánicamente un paso chico en los pesos.
- **No dice que `L` no importe.** L-2 sigue **NO DECIDIBLE**, por la razón del `PRECISION`.
- **No toca la fase H.** Las cuatro unidades de `cabeza` siguen **NO EVALUABLES por presupuesto**
  (`NOTA_LECTURA_FASE_H_20260830.md`, SHA `4a0900bf`), y este informe no cambia eso: el criterio de
  abandono del §7 exige las dos interfaces y sigue sin poder aplicarse.
