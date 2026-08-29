# HALLAZGO · la campaña ya está decidida, y lo que la decidió pasó en el paso 2500

**2026-08-29.** **Post-hoc, declarado como tal.** Todo lo que sigue se midió sobre checkpoints y
`corridas_*/*.json` que ya estaban en disco al cierre del 28. No se entrenó nada, no se corrió ningún
modelo nuevo y **no se juzga `PREREG_TASA_REGIMEN.md`**, que sigue congelado y se juzga a 26000 pasos
como dice.

Instrumentos de un solo uso en el scratchpad de la sesión (`sonda_cabeza.py`, `sonda2.py`), los dos
apoyados en `ser_cobertura.sondear()`, que es lo que ya existía: corre la unidad **sin decidir nada**
y devuelve el logit crudo de la cabeza y el argmax de valores por separado. n=2000, semilla de datos
54321, la misma del 27 y del 28.

---

## 1. Lo primero, porque cambia el resto: T-0 y T-1 ya son inalcanzables

`PREREG_TASA_REGIMEN.md` fija **T-0 bloqueante: `vigente` ≥ 0,70 en ≥ 4 de 6**, y **T-1 principal:
≥ 3 de 6 en el régimen** (`nose` ≥ 0,99 **y** `falsa_abst` ≤ 0,01).

Cuatro de las seis unidades (`s3`, `s6`, `s7`, `s8`) están en abstención total. Su `falsa_abst` es
**1,0000**, así que **fallan T-1 por el segundo término aunque su `nose` valga 1,0000** — el
pre-registro previó exactamente esto al pedir las dos cosas juntas. Y su `vigente` es 0,0000.

Quedan dos unidades disponibles para cuatro plazas de T-0 y tres de T-1.

> **Los dos criterios están fallados por aritmética, no por pronóstico.** Terminar los 63000 pasos que
> faltan no puede cambiarlos.

Y no es que las dos que quedan estén cómodas. `b3_s4` lleva desde el paso 17000 oscilando alrededor de
0,66 (0,63 · 0,67 · 0,66 · 0,71 · 0,64 · 0,66 · 0,60 · 0,68), que es el mismo perfil con el que `b3_s2`
terminó en **0,6503** a 26000. El 0,7060 del paso 18500 es un máximo puntual y
`NOTA_LECTURA_CURVAS_20260824.md` ya dice qué hacer con esos.

---

## 2. Un predictor perfecto, y lo que predice es cero contra no-cero

La fase de abstención total al principio **no es la anomalía**: la tienen las nueve unidades con blanco
`error`, y estaba anotada el 26 («se abstiene del 100 % durante ~3000 pasos y afloja SOLA»). La
anomalía es que cuatro no salen.

Mirando **un solo hito**, el del paso 2500, con el lote de evaluación que el propio entrenamiento ya
corre (n=8 lotes de B=64, o sea 512 muestras):

| unidad | `abstencion` @2500 | respuestas emitidas | desenlace |
|---|---:|---:|---|
| b3_s1 | 0,9961 | **2** de 512 | vive |
| b3_s0 | 0,9922 | **4** de 512 | vive |
| b3_s5 | 0,9922 | **4** de 512 | vive |
| b3_s2 | 0,9844 | **8** de 512 | vive |
| b3_s4 | 0,9980 | **1** de 512 | vive |
| b3_s3 | 1,0000 | **0** | muda a 22000 |
| b3_s6 | 1,0000 | **0** | muda a 25000 |
| b3_s7 | 1,0000 | **0** | muda a 13500 |
| b3_s8 | 1,0000 | **0** | muda a 8000 |

**La separación es cero contra no-cero.** `b3_s4` vive con **una sola respuesta de 512** y llegó a
`vigente` 0,68.

### Verificado contra todo el banco, no sólo contra estas nueve

Se aplicó la misma regla a **las 40 corridas del repo** con cabeza binaria que llegaron a ≥6000 pasos
y tienen hito en 2500: doce familias, dos reglas de decisión (`cabeza` y `slot`), los dos blancos.

> **40 de 40 sin un error.** 36 predichas vivas y vivas, 4 predichas mudas y mudas.

Y aparece un segundo hecho en la misma tabla, que no se fue a buscar:

> **La fase muda temprana es exclusiva del blanco `error`.** Las 31 corridas con blanco `ausencia`
> están en `abstencion` 0,02–0,32 en el paso 2500. Las 9 con blanco `error` están en 0,98–1,00, las
> nueve. El control pareado lo dice solo: `p3_s0/s1/s2` dan 0,0859 · 0,2012 · 0,0625 contra
> 0,9922 · 0,9961 · 0,9844 de sus `b3` hermanas.

---

## 3. Qué es exactamente una unidad muda

Aquí se cayó la hipótesis con la que empecé, y hay que anotarla primero.

**Lo que supuse y es FALSO:** que la recuperación estuviera sana y la cabeza la tapara. `vigente` es
una métrica **compuesta** —`pred = NOSE si a > 0, si no el argmax`— así que con la cabeza clavada daría
0,0000 aunque el argmax fuera perfecto. Era una explicación limpia y no se sostiene.

Midiendo el argmax de valores **ignorando la cabeza**, sobre las preguntas que sí tienen respuesta:

| unidad | paso | **RECUP** (argmax sin cabeza) | AUC de `a` sobre «me equivoco» | rango de `a` (p1–p99) | `a` > 0 |
|---|---:|---:|---:|---:|---:|
| b3_s0 | 26000 | **1,0000** | **1,0000** | 28,85 | 0,406 |
| b3_s1 | 26000 | 0,9992 | 0,9999 | 29,81 | 0,406 |
| b3_s4 | 19000 | 0,7970 | 0,8050 | 14,02 | 0,425 |
| b3_s2 | 26000 | 0,7911 | 0,8073 | 14,18 | 0,400 |
| b3_s5 | 6000 | 0,7414 | 0,7555 | 5,58 | 0,541 |
| **b3_s6** | 25000 | **0,3960** | **0,5707** | **3,24** | **1,000** |
| **b3_s3** | 22000 | **0,3665** | **0,5572** | **2,77** | **1,000** |
| **b3_s7** | 13500 | **0,3218** | **0,5387** | **1,06** | **1,000** |
| **b3_s8** | 8000 | **0,3050** | **0,5251** | **0,85** | **1,000** |
| p3_s0 | 26000 | 0,9789 | 0,9581 | 24,39 | 0,377 |
| p3_s1 | 26000 | 0,7784 | 0,6994 | 18,68 | 0,224 |
| p3_s2 | 26000 | 0,8719 | 0,8019 | 19,20 | 0,319 |

**Tres cosas, y las tres importan:**

**1. La recuperación TAMBIÉN está rota, pero no está en cero.** 0,31–0,40 contra 0,74–1,00 de las
vivas. Muy por encima del azar (1/242), o sea que la unidad muda **aprendió una parte real de la
tarea** y no llega. Ninguna de las tres `p3` de control cae en esa banda.

**2. La cabeza no discrimina: es una constante.** AUC 0,525–0,571 sobre su propio blanco, o sea azar,
contra 1,0000 en `b3_s0`. Y el rango entero de sus logits es **0,85 a 3,24**, contra 28,85. No está
saturada en +∞, está **apretada apenas por encima del umbral**.

**3. Y esa constante es exactamente el prior.** Si la cabeza no mira la entrada, su valor óptimo es el
logit de la tasa base de error, que se calcula sin ajustar nada: `0,4065 + (1−0,4065)·(1−RECUP)`.

| unidad | RECUP | tasa base de error | logit predicho | `a` mediano medido | dif |
|---|---:|---:|---:|---:|---:|
| b3_s8 | 0,3050 | 0,8190 | 1,509 | 1,508 | **0,001** |
| b3_s7 | 0,3218 | 0,8090 | 1,444 | 1,250 | 0,194 |
| b3_s3 | 0,3665 | 0,7825 | 1,280 | 1,007 | 0,273 |
| b3_s6 | 0,3960 | 0,7650 | 1,180 | 0,985 | 0,195 |

**El orden coincide en las cuatro y la más joven cae a 0,001 del valor predicho.** Es colapso al prior,
medido. Es el riesgo que el comentario del §142 de `entrenar.py` declaró por adelantado el 26 —«si
colapsa al prior va a ser por esto»— y que **E-4 de A5 no encontró en `s0/s1/s2`**. Estaba en las
semillas que no se habían corrido.

---

## 4. El mecanismo, que es una carrera entre dos relojes

El blanco `error` es **autorreferencial**: la etiqueta de la cabeza es el error del propio modelo,
`(argmax != tgt)` con `stop_gradient`.

Al empezar, el modelo se equivoca en todo, así que la etiqueta es la constante 1 y la cabeza aprende a
decir «me voy a equivocar» siempre. **Y tiene razón.** Para salir de ahí, la recuperación tiene que
empezar a acertar mientras la cabeza todavía no terminó de clavarse en la constante, porque sólo
cuando algunas muestras piden `a` bajo y otras `a` alto nace la discriminación que la cabeza necesita.

Los dos relojes corren juntos y la semilla decide cuál llega primero. Hay dos puntos fijos y los dos
son autoconsistentes:

- **el bueno** — recupero bien y sé cuándo voy a fallar;
- **el degenerado** — no recupero casi nada, digo que me equivoco siempre, y **es verdad**.

Esto explica por qué la familia `ausencia` nunca cae: su blanco es `es_nose`, **fijo y observable desde
el paso 1**, así que su cabeza nunca pasa por la fase constante. No es una diferencia de dificultad,
es que un blanco no depende del estado del modelo y el otro sí.

### Lo que esto le hace a la frase del 27

> «El modelo sabe cuándo no sabe. Lo que falta es convertir eso en la decisión de callarse.»

**Estas cuatro unidades convierten el saber en decisión perfectamente, y son inútiles.** Su `nose` vale
1,0000, su `invento` vale 0,0000 y su exactitud global es **0,4065 clavado**, que es el piso trivial de
`metrica-exactitud-global`. La abstención perfecta y el conocimiento nulo son **el mismo estado**.

La hipótesis que el §3 del `ESTADO_20260828_NOCHE.md` dejó escrita —«callarse es trivial y lo difícil
es salir del silencio sin empezar a inventar»— queda **con mecanismo y con número**.

---

## 5. Lo que esto NO autoriza

- **No juzga `PREREG_TASA_REGIMEN`.** Ese pre-registro se juzga a 26000 pasos con `ser_cobertura.py`,
  campo `propio`, y nada de acá lo reemplaza. Lo que sí hace es mostrar que dos de sus criterios ya
  están decididos.
- **El predictor del paso 2500 se eligió DESPUÉS de ver los desenlaces.** Con n=40 y regla elegida
  post-hoc, es una **hipótesis con evidencia fuerte, no un resultado validado**. Para usarlo como
  criterio hay que pre-registrarlo y probarlo en unidades nuevas.
- **Y el margen es de una muestra en 512.** `b3_s4` se salvó con una. Como criterio operativo, la tasa
  de abstención sobre 512 muestras es demasiado frágil; la versión robusta es el **mínimo del logit
  `a`** sobre un lote grande, que es continua y no depende del tamaño del lote.
- **La deriva de RECUP con los pasos está CONFUNDIDA.** Los cuatro puntos (0,3050 @8000 · 0,3218
  @13500 · 0,3665 @22000 · 0,3960 @25000) dan r = 0,9836 y una pendiente de +0,0052 cada 1000 pasos,
  pero **son cuatro unidades distintas en cuatro pasos distintos, no una trayectoria**. No alcanza para
  decir que una unidad muda mejora sola, y los checkpoints intermedios no se guardan, así que con lo
  que hay en disco **no se puede desconfundir**.
- **No dice nada sobre escala.** 863.730 parámetros, idioma de 242 tokens, `p_nose` 0,4, un nivel.
- **Sigue siendo supervisado.** El §8 del `PLAN_FOCO_20260824.md` y su cierre de seis meses no se tocan.

---

## 6. ~~La pregunta que queda abierta~~ · CONTESTADA EL MISMO DÍA, y este §6 se RETIRA

> **⚠ Lo que sigue en letra tachada se escribió a la mañana y es INCORRECTO.** Se corrigió el mismo
> día con `INFORME_ATRACTOR_MUDO_FASE1_20260829.md`, que evalúa `PREREG_ATRACTOR_MUDO.md`
> (SHA `2be4a610`). Se deja escrito en vez de borrarlo para que la corrección sea auditable.

~~Si la pendiente del §5 fuera real, una unidad muda necesitaría **~84000 pasos** para llegar a RECUP
0,70.~~ **La extrapolación era un artefacto de ajustar una RECTA a una curva CÓNCAVA.** Las pendientes
por tramo decrecen monótonamente (+0,0071 → +0,0048 → +0,0014 por 1000 pasos) y un ajuste de
saturación da SSE **12,6 veces menor** que el lineal.

**Y el experimento que este §6 proponía se corrió, en su versión posible.** No a 80000 —`entrenar.py:443`
aborta si se cambia el horizonte, porque `HOR` es el `decay_steps` del cosine— sino llevando `b3_s3`
de 22000 a 26000 con el horizonte propio y **archivando cada checkpoint parcial** para medir RECUP
como trayectoria. Resultado, con n=8000 y diseño pareado:

> **RECUP no se mueve: −0,0021 en 4000 pasos, o sea 0,4 σ.** Dentro de una sola unidad, con las mismas
> preguntas, la recuperación de una muda está **plana**.

**El atractor es ABSORBENTE**, no un cuello de botella lento. La Fase 2 (70000 pasos) no se lanza.

**Dos correcciones más que salen de ahí, y afectan al §5 y al §3 de arriba:**

1. **La pendiente entre unidades NO era ruido de medición** —se re-midieron las cuatro con n=8000 y
   r pasa de 0,9836 a 0,9864, con el rango a 12,5 σ—. Era real; lo que estaba mal era leerla como una
   recta extrapolable. La sospecha de que fuera ruido se probó y **se cayó**.
2. **Los RECUP del §3 se midieron con n=2000, cuyo σ es 0,0135.** Los valores buenos, con n=8000, son
   `s8` 0,3040 · `s7` 0,3432 · `s3` 0,3841 · `s6` 0,3884. La banda «0,31–0,40» del §3 se lee mejor
   como **0,30–0,39**, y la fila de `b3_s3` en la tabla de colapso al prior se recalcula con 0,3841.
