# La frontera del margen — dónde está el corte, y cuánto lo corre la cabeza

**Prereg:** `PREREG_FRONTERA.md`, congelado 2026-08-18 a las 23:54 UTC (SHA `0acfe89b…`), antes de
tocar la infraestructura y antes de correr un paso.
**Campaña:** 3 corridas base desde cero + **18 fases** previstas, de las cuales corrieron **12**
(las 6 de la semilla 1 quedaron sin correr: ver D-F1).
**Desviaciones:** `DESVIACIONES_FRONTERA.md`.

---

## §1 · La pregunta, y por qué bloqueaba todo lo demás

El 17-ago la separación entre los modelos que aprendían a abstenerse y los que no era **perfecta**, y
la explicaba el «margen sobre el atajo»: `vigente` al cerrar la base menos **0,5906**, que es lo que
vale la política de *no abstenerse nunca* con `p_nose` 0,4. Los de margen alto pasaban; los de margen
bajo fallaban. Pero **entre +0,2358 y +0,4071 no había ni un punto medido**, así que no se sabía si
el margen era un umbral o una pendiente, ni dónde caía el corte.

Importa porque **si hay que entrenar el modelo hasta casi la perfección antes de poder enseñarle a
decir «no sé», el método no sirve para nada real**: ningún modelo útil llega a `vigente` 1,0 en su
dominio.

Y la campaña del 18-ago reformuló la pregunta: con `token` el margen predecía perfecto (0 de 5), y
**4 de esos mismos 5 modelos pasaban con `cabeza`**. Entonces la pregunta dejó de ser «¿dónde está la
frontera?» y pasó a ser **«¿la arquitectura la mueve, y cuánto?»**.

---

## §2 · Cómo se muestreó el hueco

La base (nivel 2, 3 semillas, desde cero) se cortó **por valor de `vigente`**, no por número de paso:
es el margen lo que se quería controlar, y fijar el paso lo habría dejado al azar de la semilla. Se
guardó un checkpoint al cruzar por primera vez 0,85 · 0,90 · 0,95, y desde cada corte se entrenaron
**2000 pasos** con `p_nose` 0,4, en las dos condiciones (`token` y `cabeza`), con Adam reinicializado.

`f2_s0` y `f2_s2` cruzaron los tres umbrales y dieron **6 cortes**; `f2_s1` no (ver D-F1).

---

## §3 · El resultado: hay un corte, y cae dentro del hueco

### `token` — la separación es perfecta, y el salto está localizado

| margen | unidad | vía | `falsa_abst` | `nose` | |
|---:|---|---|---:|---:|---|
| +0,1489 | t3_s0 | completo | 0,2246 | 0,6378 | falla |
| +0,1672 | t4_s0 | completo | 0,1342 | 0,7106 | falla |
| +0,1787 | t4_s1 | completo | 0,1713 | 0,7000 | falla |
| +0,1870 | t3_s1 | completo | 0,1757 | 0,5765 | falla |
| +0,2358 | t3_s2 | completo | 0,2237 | 0,6379 | falla |
| **↑ el corte cae en este intervalo ↓** | | | | | |
| +0,2826 | k85t2_s0 | sub-entr | **0,0855** | 0,9068 | **pasa** |
| +0,3087 | k85t2_s2 | sub-entr | 0,0477 | 0,8334 | pasa |
| +0,3271 | k90t2_s2 | sub-entr | 0,0462 | 0,8745 | pasa |
| +0,3370 | k90t2_s0 | sub-entr | 0,0271 | 0,8379 | pasa |
| +0,3695 | k95t2_s0 | sub-entr | 0,0596 | 0,8149 | pasa |
| +0,3808 | k95t2_s2 | sub-entr | 0,0423 | 0,8986 | pasa |
| +0,4094 | t2_s0 | completo | 0,0205 | 0,8919 | pasa |
| +0,4094 | t1_s0 | completo | 0,0667 | 0,9046 | pasa |

**13 unidades ordenadas por margen y ni una inversión.** Entre +0,2358 y +0,2826 —dos puntos
separados por 0,047— `falsa_abst` cae de **0,2237 a 0,0855**, un factor de **2,6×**. No es una
pendiente suave: es un salto, y **cae dentro del hueco que la campaña fue a muestrear**.

### `cabeza` — el mismo orden, corrido hacia abajo

| margen | unidad | vía | `falsa_abst` | `nose` | |
|---:|---|---|---:|---:|---|
| +0,1489 | c3_s0 | completo | 0,1189 | 0,5826 | falla |
| **↑ el corte cae acá ↓** | | | | | |
| +0,1672 | c4_s0 | completo | 0,0898 | 0,6150 | **pasa** |
| +0,1787 | c4_s1 | completo | 0,0746 | 0,6189 | pasa |
| +0,1870 | c3_s1 | completo | 0,0970 | 0,5406 | pasa |
| +0,2358 | c3_s2 | completo | 0,0841 | 0,5962 | pasa |
| +0,2826 | k85c2_s0 | sub-entr | 0,0667 | 0,8972 | pasa |
| +0,3087 | k85c2_s2 | sub-entr | 0,0264 | 0,7563 | pasa |
| +0,3271 | k90c2_s2 | sub-entr | 0,0328 | 0,7963 | pasa |
| +0,3370 | k90c2_s0 | sub-entr | 0,0462 | 0,8040 | pasa |
| +0,3695 | k95c2_s0 | sub-entr | 0,0315 | 0,8330 | pasa |
| +0,3808 | k95c2_s2 | sub-entr | 0,0395 | 0,8581 | pasa |
| +0,4094 | c2_s0 | completo | 0,0029 | 0,8449 | pasa |
| +0,4094 | c1_s0 | completo | 0,0118 | 0,9064 | pasa |

**`cabeza` pasa en 12 de 13.** La única que falla es c3_s0, la de margen más bajo de toda la serie.

### **El número que responde la pregunta del prereg**

| | corte | |
|---|---|---|
| `token` | entre **+0,2358 y +0,2826** | |
| `cabeza` | entre **+0,1489 y +0,1672** | |
| **la cabeza corre la frontera** | **≈ 0,10 de margen hacia abajo** | (0,1154 en el peor caso, 0,1337 en el mejor) |

Con 129 parámetros —el **0,015 %** del modelo— se le puede enseñar a callarse a un modelo que
recupera **10 puntos peor**. Ésa es la respuesta a «¿la arquitectura la mueve, y cuánto?».

---

## §4 · Los cuatro veredictos

**F-1 (forma de la frontera) CUMPLE, en sus dos mitades.**
Spearman(margen, `falsa_abst`) = **−0,8033** en `token` y **−0,8886** en `cabeza`, ambos con n=13 y
contra el ≤ −0,70 exigido. Y la segunda mitad también: **hay salto y no pendiente, y el corte cae
dentro del hueco medido**.

> **Nota de cálculo, porque el número cambió mientras se escribía este informe.** El primer cálculo
> daba −0,819 y −0,909; el segundo, con los mismos datos, −0,786 y −0,885. La causa es que `t1_s0` y
> `t2_s0` (y sus pares `c1_s0`/`c2_s0`) **empatan en margen** (+0,4094 los dos, porque las dos bases
> cerraron en `vigente` 1,0000), y la primera implementación rompía el empate según el orden en que
> `glob` devolvía los archivos. Los valores que se reportan usan **rangos promediados**, que es el
> tratamiento correcto de los empates y no depende del orden de lectura. La conclusión no cambia en
> ninguna de las tres versiones —las tres superan holgadamente el umbral—, pero un coeficiente que se
> mueve entre corridas con los mismos datos es un número que no se puede publicar.

**F-2 — la primera mitad cumple, la segunda NO.**
`cabeza` pasa la compuerta en los tres márgenes del hueco, en las dos semillas disponibles (6 de 6).
Pero el prereg pedía además que **en el margen más bajo (0,85) `cabeza` le ganara a `token` en
`falsa_abst` por ≥ 0,05**, y gana por **0,0188** (s0: 0,0667 vs 0,0855) y **0,0213** (s2: 0,0264 vs
0,0477). **No cumple, y se registra así.**
El motivo es transparente y no rescata la predicción: en el margen 0,85 **`token` ya pasa
holgadamente**, así que no queda distancia que recuperar. La ventaja de la cabeza no está donde el
prereg la fue a buscar; está **por debajo** del hueco.

**F-3 (control de sanidad, podía fallar) CUMPLE.** En el margen más alto las dos condiciones pasan.
Si `token` hubiera fallado también ahí, el punto de introducción no sería el eje que gobierna y la
campaña del 17-ago necesitaría otra explicación. No fue el caso.

**F-4 (lo que haría útil el método) NO CUMPLE en su forma literal.** Pedía que existiera **algún
margen del hueco** donde `cabeza` pasara y `token` fallara en las tres semillas. No existe: `token`
pasa en los seis puntos del hueco.
**Pero lo que F-4 buscaba ocurre, un escalón más abajo:** en el rango +0,1672 a +0,2358, `token`
falla en **4 de 4** y `cabeza` pasa en **4 de 4** (c4_s0, c4_s1, c3_s1, c3_s2). El hueco estaba mal
ubicado: la frontera de `token` no estaba *dentro* de él sino en su borde inferior, y la separación
entre las dos condiciones vive por debajo. **La recomendación operativa que F-4 quería producir se
obtiene igual, con otro número:** con cabeza propia alcanza con entrenar hasta un margen de
**≈ +0,17**; con `token` hay que llegar a **≈ +0,26**.

---

## §5 · El confound del §4 del prereg, evaluado

El prereg declaró antes de correr que el eje A **confunde margen con grado de entrenamiento**: un
modelo detenido en el paso 3000 con `vigente` 0,90 no es equivalente a uno que llega a 0,90 a 12000
pasos por dificultad de la tarea. Y fijó el criterio: *«Si el margen predice igual en ambas [vías], es
el margen. Si no, el margen era un proxy»*.

**Lo medido: el margen predice igual en ambas vías, sin una sola inversión.** Las 13 unidades de
`token` se ordenan por margen mezclando las dos vías (5 «completo» que fallan, 6 «sub-entrenado» que
pasan, 2 «completo» que pasan), y las dos vías coinciden en el sentido.

**Lo que igual no se puede afirmar, y se dice explícitamente:** las dos vías siguen confundidas en el
rango muestreado, porque **todos** los puntos entre +0,2826 y +0,3808 son sub-entrenados y **todos**
los de margen ≤ +0,2358 son de entrenamiento completo. La consistencia es evidencia *a favor* de que
el margen es la variable, no una demostración. **Lo que faltaría es la celda cruzada**: un modelo de
nivel 3 o 4 detenido en un margen alto, o un nivel 2 entrenado a fondo pero trabado en margen bajo
—que es, justamente, lo que `f2_s1` iba a dar y no dio.

---

## §6 · Qué queda

- **`f2_s1` no cruzó nunca** (D-F1): la campaña cierra con 2 semillas y no 3.
- **La celda cruzada del §5** es el experimento que separaría margen de grado de entrenamiento, y es
  barato: alcanza con cortar por valor de `vigente` una corrida de nivel 3.
- **c3_s0 sigue siendo la única unidad que no pasa con cabeza** en toda la serie, y `sonda_umbral.py`
  sugiere —post-hoc— que su techo es de calibración y no de capacidad.
