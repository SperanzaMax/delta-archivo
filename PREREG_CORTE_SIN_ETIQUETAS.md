# PREREG · El corte sin etiquetas

Congelado el 2026-08-20 antes de mirar un solo logit de las unidades nuevas y antes de escribir la
sonda. Sale del §1 de `PLAN_20260820.md`.

---

## 1 · La pregunta, y por qué bloquea la línea

`INFORME_UMBRAL_PROSPECTIVO_20260819.md` dejó demostrado que la información para decidir cuándo
callarse **está** en el logit de la cabeza de abstención, que el corte `a*` transfiere dentro de un
nivel y que el nulo por permutación está limpio (0/20). Pero **`a*` se elige con etiquetas**: hay que
saber de antemano qué preguntas tenían respuesta en el archivo. En uso real esa información no
existe, y si existiera no haría falta el modelo.

**Esto es lo único que hoy separa «la información está en el logit» de «el modelo sabe cuándo no
sabe»** (ver el objetivo declarado en la memoria del proyecto). Mientras el corte necesite etiquetas,
la abstención medida es una propiedad del analista, no del modelo.

## 2 · De dónde sale la hipótesis (de un dato, no de una intuición)

El control C-A del 19-ago midió el corte en unidades del propio logit, `z* = (a* − μ)/σ`:

| c1_s0 | c2_s0 | c3_s0 | c3_s1 | c3_s2 |
|---:|---:|---:|---:|---:|
| −0,368 | −0,255 | +0,379 | +0,280 | +0,342 |

**Consistente en magnitud (≈0,3 σ) y separado por dificultad sólo en el signo.** Si el corte vive
siempre a una distancia fija de la media del logit, se puede fijar mirando nada más que la
distribución de `a`.

**El modo de falla, declarado antes de correr:** el signo depende de la dificultad, y la dificultad
tampoco se conoce en producción. Si el signo no se puede inferir de la propia distribución, la idea
no cierra. **Ése es el test real, no la magnitud.**

## 3 · Unidades

**Familia `c` (condición `cabeza`), las 8 que tienen `params["abst"]` entrenado:**
`c1_s0 · c2_s0 · c3_s0 · c3_s1 · c3_s2 · c4_s0 · c4_s1 · c4_s2`, todas a 14000 pasos.

Las tres de **nivel 4 son conjunto de validación genuino**: la tabla del §2 se derivó de las otras
cinco y `c4_*` nunca se miró con esta lente. No se las excluye ni se las pondera aparte, pero el
informe **debe reportar el resultado por unidad**, nunca la media (regla del proyecto desde la
campaña base).

Las familias `t` (token) y `s` (escala) **quedan fuera**: su `params["abst"]` está en la
inicialización en cero porque nunca lo entrenaron, así que su logit no es una medición.

Muestreo idéntico al del 19-ago: **rng 90000+semilla para AJUSTE, 77000+semilla para PRUEBA**
(lotes independientes, no dos vistas del mismo lote), 32 lotes de 64 = **2048 muestras por unidad y
por muestra**, `p_nose = 0,4`, `p_vieja = 0,35`.

## 4 · Los tres estimadores, ordenados por cuánta información externa usan

Los tres estiman el corte **sobre la muestra de AJUSTE** y se juzgan **sobre la de PRUEBA**.

**U-1 · VALLE DE LA MEZCLA — cero etiquetas, es el principal.**
EM de una mezcla de dos gaussianas 1-D sobre `a`, inicialización determinista (medias en los
cuantiles 0,25 y 0,75, varianzas iguales a la muestral, pesos 0,5/0,5), 200 iteraciones o tolerancia
1e-6. El corte `â` es el punto **entre las dos medias** donde las densidades ponderadas se igualan.
Motivación mecánica, no ajuste: hay dos poblaciones —con respuesta y sin respuesta— y ya está medido
que el logit las separa (AUC 0,825 en la unidad más difícil, ≥0,99 en las fáciles). **No usa ninguna
etiqueta, ni de esta unidad ni de ninguna otra.**
Si EM no converge, si un componente colapsa (peso < 0,02 o desvío < 1e-6) o si las medias quedan a
menos de 0,05 σ, se registra **«sin corte»** y **cuenta como fallo** de la unidad.

**U-2 · CONSTANTE TRANSFERIDA, leave-one-out.**
`â = μ + s · z̄ · σ`, con `z̄` = mediana de |z*| de las **otras siete** unidades y `s` el signo
inferido **sin etiquetas de la unidad evaluada**, mediante el estadístico declarado acá y no otro:
**la asimetría de Fisher de `a`**. El mapeo asimetría→signo se ajusta también leave-one-out (regla de
umbral en 0 sobre la asimetría, orientación elegida por mayoría en las otras siete). Usa etiquetas de
otras unidades, ninguna de la evaluada.

**U-3 · CUANTIL DE LA TASA BASE.** `â` = cuantil `1 − p_nose` de `a`. Es la línea de base obvia y
usa un dato de diseño (`p_nose = 0,4`) que en producción se conocería sólo de forma aproximada. Se
declara acá para que no pueda presentarse después como hallazgo: **si U-3 iguala a U-1, U-1 no aporta
nada** y hay que decirlo.

**Referencias de contraste, no estimadores:** el oráculo `a*` con etiquetas (techo, ya medido) y el
criterio sin calibrar σ>0,5 (piso, ya medido: falla en las cinco unidades del 19-ago).

## 5 · Criterio de juicio

El de siempre, sin tocar: una unidad **pasa** si en la muestra de prueba `falsa_abst ≤ 0,10` **y**
`nose ≥ 0,50`.

- **S-1 (principal).** U-1 pasa en **≥ 6 de 8** unidades.
- **S-2 (costo contra el oráculo).** En las unidades donde el oráculo `a*` existe, la caída media de
  `nose` de U-1 respecto de `a*` es **≤ 0,10**, sin que `falsa_abst` supere 0,10.
- **S-3 (el nulo, y es el control que hace falta acá).** Permutar etiquetas **no sirve** como nulo:
  U-1 y U-3 no las miran, así que el corte no cambiaría. El nulo correcto es **destruir la estructura
  conservando μ y σ**: se reemplaza `a` por una gaussiana de la misma media y desvío y se corre el
  mismo estimador. **U-1 debe pasar en ≤ 1 de 8.** 100 repeticiones, se reporta la tasa.
- **S-4 (necesidad).** σ>0,5 falla en **≥ 6 de 8**. Si no falla, el problema no existía.
- **S-5 (el signo, que es el modo de falla del §2).** La regla de asimetría de U-2 acierta el signo
  de `z*` en **≥ 7 de 8** unidades.

**Qué significa cada desenlace, comprometido por adelantado:**
- S-1 y S-3 cumplen → **el corte se puede fijar sin etiquetas**, y es el resultado que desbloquea la
  mitad de abstención del objetivo.
- S-1 falla pero S-5 cumple → la dirección se infiere y lo que falla es la magnitud: sigue vivo por
  la vía de U-2, con el costo declarado de necesitar etiquetas de otras unidades.
- S-1 y S-5 fallan → **la idea no cierra**, se reporta el negativo y el corte sin etiquetas queda
  como problema abierto. **No se prueba un cuarto estimador en este experimento.**

## 6 · Lo que este experimento NO puede decir

- Vale para **un idioma cerrado de 242 tokens** y un modelo de 863.730 parámetros. No dice nada sobre
  texto natural (vara 3 del objetivo, §4 del plan).
- Ocho unidades del mismo modelo y la misma tarea: es un test de **mecanismo**, no una calibración
  lista para producción.
- `p_nose = 0,4` es una tasa base alta y fija. La sensibilidad del corte a la tasa base **no se mide
  acá** y queda anotada como la continuación natural.

## 7 · Análisis, y lo que está prohibido

Se reportan las 8 unidades con `â`, `falsa_abst`, `nose`, `vigente` y pasa/no pasa, más el oráculo y
σ>0,5 al lado. **Está prohibido** elegir el estimador después de ver los resultados: U-1 es el
principal por declaración, y U-2/U-3 se reportan pase lo que pase. Cualquier análisis que no esté en
este documento se marca **post-hoc** en el informe, como se hizo con `verificar_umbral_estable.py`.
