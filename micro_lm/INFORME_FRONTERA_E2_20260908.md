# Escalón 2 · las proyecciones propias, y el atractor del que no se vuelve

2026-09-08. Evalúa `PREREG_FRONTERA_E2.md` (SHA `688260f2`). Tres unidades `ap3_s0/s1/s2`,
`--donde attnp`, 26.000 pasos, desde cero, idénticas a `at3` salvo en la única variable.

**Las tres hipótesis se contestaron. Una se confirmó y dos se refutaron, y las tres refuerzan lo
mismo.**

## 1. El resultado que ordena todo · la bimodalidad

`nose_rel` por semilla, sin promediar, porque la media no describe a ninguna.

Las tres unidades cerraron los 26.000 pasos.

| unidad | posiciones efectivas de la atención | `vigente` | `anterior` | `nose_rel` |
|---|---|---|---|---|
| `ap3_s0` | **1,41 de 24** | 0,3141 | 0,1904 | 0,4752 |
| `ap3_s1` | **2,08 de 24** | 0,2868 | **0,0712** | 0,4181 |
| `ap3_s2` | **20,69 de 24** | **1,0000** | **1,0000** | **0,9875** |
| `at3` (sin proyecciones, 3 de 3) | 9,6 a 11,3 | 1,0000 | 1,0000 | 0,9977 |

**Con proyecciones libres la atención de lectura tiene dos atractores y la semilla decide.** La
correlación con el rendimiento es perfecta y no hay casos intermedios.

**Y el colapso es permanente, con el presupuesto entero gastado.** No es aprendizaje lento. Se
detectó en el paso 6.000 y las dos unidades llegaron al 26.000 sin moverse de ahí, con `vigente` en
0,3141 y 0,2868 contra 1,0000 de su hermana, y `anterior` en 0,1904 y 0,0712 contra 1,0000. **Veinte
mil pasos más no revierten nada**, así que no había que esperar, había que no caer.

## 2. H3 · CONFIRMADA, y era la que había que escribir antes

El pre-registro dice, textual, que el techo **no** tiene que subir, y que si `attnp` superara el
0,9977 de `at3` **sería evidencia contra la ley enunciada como cobertura y no un éxito**.

`ap3_s2` cerró en **0,9875**. Mismo techo, no más alto. La ley aguanta por una tercera vía
independiente, y esta vez con la condición que **contiene** a la anterior como caso particular, así
que el resultado se puede leer sin la salvedad que el escalón 1 tuvo que declarar.

## 3. H1 · REFUTADA, y ahora sí de forma concluyente

La razón de selectividad de `ap3_s2` con N = 600 da **1,01**, contra 0,94 a 1,05 de seis controles
y un umbral pre-registrado de 1,5.

En el escalón 1 esa refutación tenía una salida. `attn` no puede aprender a discriminar porque no
hay un solo peso entre `x` y el softmax, así que el resultado no distinguía «la tarea no premia
discriminar» de «al modelo le faltan parámetros». **Acá los parámetros están, son 49.152, el
gradiente los movió, y la selectividad no se movió.** La salida se cerró.

**La tarea no premia discriminar. Punto.**

## 4. Lo que el modelo hizo con la libertad, y es el hallazgo

`ap3_s2` no usó las proyecciones para enfocar. Las usó para **abrir**.

| | peso en la propia posición | posiciones efectivas |
|---|---|---|
| `attn` | 0,42 a 0,47 | 9,6 a 11,3 de 24 |
| **`ap3_s2`** | **0,1057** | **20,69 de 24** |

Bajó el sesgo estructural a la propia posición de 0,45 a 0,11 y llevó la atención a casi uniforme
sobre las 24 posiciones. **Cuando el modelo pudo elegir dónde mirar, eligió mirar en todos lados.**

Es la confirmación más fuerte que podía aparecer de que el requisito es de **cobertura y no de
foco**, porque no sale de comparar arquitecturas fijas sino de ver hacia dónde va el gradiente
cuando lo dejan elegir.

## 5. H2 · REFUTADA

Se predijo que `attnp` cruzaría a los dos brazos **antes** del paso 6.000, contra el 6.000 de `at3`.
No sólo no cruzó antes, dos de las tres **no cruzaron nunca**. La observación post-hoc del escalón 1
sobre el arranque lento no se sostiene como se la había formulado.

## 6. El control del §6 hizo exactamente lo que estaba escrito

Sin él, el colapso se confundía con «el weight decay se comió las proyecciones», que era el
diagnóstico obvio y equivocado.

| | `\|wqr\|` | distancia a la identidad |
|---|---|---|
| bloque 0, con gradiente | 11,0616 | **3,2350** |
| bloques 1 a 3, decay puro | 10,7014 | 0,6123 |

El bloque 0 está **cinco veces más lejos** de la identidad que los que sólo reciben decay, y difiere
de ellos en 0,1395. **Hubo gradiente. El diseño no está roto, el modelo eligió esto.**

## 7. La instrucción de diseño que sale, y es contraintuitiva

**Para una memoria consultada desde una capa temprana conviene la atención SIN proyecciones
propias**, no porque sea más potente sino porque **no puede aprender a mirar mal**. La cobertura
queda garantizada por construcción y llega al techo 3 de 3, mientras que darle al modelo la
capacidad de elegir abre un atractor del que no vuelve y en el que cae 2 de 3.

Dicho de otro modo. **La capacidad de aprender a atender es también la capacidad de aprender a no
atender**, y en esta tarea el segundo resultado es más probable que el primero.

Queda abierto si el colapso se puede prevenir. Los candidatos naturales son calentar la atención con
temperatura, congelar las proyecciones los primeros pasos, o penalizar la entropía baja. Ninguno se
probó y ninguno está pre-registrado.

## 8. Lo que este escalón NO resuelve

Sigue sin ser el Micro LM de Frontera. Arregla la diferencia 1, la query de lectura. **El tronco
sigue siendo recurrente y el vocabulario sigue cerrado**, que son las dos que deciden.

## 9. Archivos

`corridas_20260908/ap3_s*.json`, `salud_attnp.py`, `control_attn_ap3_s2.json`,
`control_attn_ruido_n600.json`, `curvas_frontera.py`, `selectividad_attn.py`.
