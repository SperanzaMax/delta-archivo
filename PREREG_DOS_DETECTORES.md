# PRE-REGISTRO · DOS DETECTORES, NO UNO — y dónde vive la señal

**2026-08-26.** Se congela antes de implementar y antes de mirar un solo número. Sale del pedido de
Maxi de buscar una mejor manera de detectar la alucinación, y de `DISENO_BANDERA_20260826.md`.

---

## 1. La observación que lo motiva (todo ya medido, nada nuevo acá)

**(a) En este banco hay DOS fallos distintos, y `ser.py` los separa desde el 15-ago.**

| categoría | qué pasó | mecanismo |
|---|---|---|
| `invento` | la respuesta no estaba y el modelo contestó un valor | ausencia no detectada |
| `err_identidad` | la respuesta estaba y trajo la de otra entidad | mala atribución |

**(b) El detector de confianza funciona bien en uno y se rompe al mezclar.**
`INFORME_MITIGACION_20260815.md`, textual: la confianza de salida separa aciertos de errores con
**AUC 0,8631**; evaluado **con preguntas sin respuesta**, el AUC cae a **0,7397** y la ventaja sobre
el azar baja de 1,68× a 1,16×.

**(c) Las siete vías cerradas buscaron UN detector para el conjunto.** Todas midieron «¿hay
respuesta?» o «¿esto vino del archivo?», y todas cayeron en AUC 0,50-0,67.

**(d) El trípode ya probó, en otro plano, que dos decisiones distintas no deben compartir un canal.**
`cabeza` gana a `token` justamente por separarlas.

## 2. La hipótesis

> **El detector único falla porque está resolviendo dos problemas distintos con un solo número.
> Especializar un detector por fallo, y componerlos, gana al mejor detector único.**

Y su corolario espacial, que es la idea de la bandera de Maxi.

> **La señal de mala atribución vive en la posición de máximo foco de lectura, y para cuando llega a
> `pos_q` —que es donde hoy se decide— ya se diluyó.**

El número que lo motiva está en `INFORME_FOCO_LECTURA_20260816.md`. En `pos_q` la entropía de lectura
es **1,7118 / 1,7660** contra un techo de ln(6) ≈ 1,79, o sea casi uniforme, mientras el foco real
(entropía **1,0482 / 1,0197**, masa 0,65) vive en posiciones intermedias. Y verificado en el código
el 26-ago, `entrenar.py:113` lee el logit de la cabeza **sólo en `pos_q`**.

## 3. Unidades, datos y protocolo

**Unidades principales:** `v3_s0`, `v3_s1`, `v3_s2` — nivel 3, `lat2`, `abst cabeza`, 26000 pasos,
`p_nose` 0,4. Se eligen porque `lat2` es la única condición donde la query es conjunta, así que es la
única donde la pregunta «¿lo recuperado matchea las dos componentes?» tiene sentido.

**Réplica declarada:** `p3_s0`, `p3_s1`, `p3_s2` — idénticas salvo `donde=pre`. Si el efecto aparece
igual en `pre`, entonces **no depende de la query conjunta** y hay que decirlo.

**Muestreo.** `p_nose` 0,4 y `p_vieja` 0,35, o sea la distribución de entrenamiento. Dos muestras
independientes con semillas de generación distintas, **fijadas acá**:

- **ajuste** `rng = 90000 + semilla`, n = 6000
- **prueba** `rng = 77000 + semilla`, n = 6000

Todo lo que se elige —pesos de la sonda, umbrales, cualquier hiperparámetro— se elige **sólo** con
la muestra de ajuste. La de prueba se mira una vez.

**Sondas.** Regresión logística sin regularizar salvo un `l2` fijo de 1,0, sobre features
estandarizadas con la media y el desvío **de la muestra de ajuste**. Nada de búsqueda de
hiperparámetros.

**Featuras, declaradas por adelantado y sin ampliar después:**

| nombre | qué es | de dónde sale |
|---|---|---|
| `est_q` | el estado final `hn` en `pos_q`, 128 dims | lo que la cabeza ya ve hoy |
| `est_foco` | el estado `hn` en la posición de mínima entropía de lectura, 128 dims | lo que la bandera transportaría |
| `lect_q` | 4 escalares de la lectura en `pos_q` (entropía, masa top-1, margen top1−top2, logsumexp) | — |
| `lect_foco` | los mismos 4 escalares en la posición de foco | — |
| `salida` | 3 escalares de la salida (max softmax, margen top1−top2, entropía del vocabulario) | el detector del 15-ago |
| `cab` | el logit de la cabeza de abstención | el detector que ya gana |

## 4. Las predicciones

**D-0 · SANIDAD, BLOQUEANTE.** Las seis unidades reproducen sus métricas ya publicadas
(`vigente`, `nose`, `falsa_abst`) dentro de ±0,02 con el instrumento de este script. Si no, no se
mide nada más y se arregla el instrumento primero. *Esto está acá porque en este proyecto un
instrumento roto ya dio ceros limpios ocho veces.*

**D-1 · PRINCIPAL — la separación gana.**
Blanco compuesto = «el modelo se equivocó», o sea `invento` **o** `err_identidad` **o**
`err_version`, contra `acierto` y `acierto_nose`.

- **único**: la mejor sonda entrenada sobre ese blanco compuesto, con todas las featuras.
- **compuesto**: dos sondas con blancos separados —una sobre `tgt == NOSE` y otra sobre
  `err_identidad` restringida a las preguntas **con** respuesta— combinadas por la regla
  `p(error) = p_A + (1 − p_A) · p_B`.

> **Cumple si el AUC del compuesto supera al del único por ≥ 0,05 en ≥ 2 de 3 semillas.**

**D-2 · LA BANDERA — hay dilución y el foco la conserva.**
Sonda sobre `est_foco + lect_foco` contra sonda sobre `est_q + lect_q`, las dos con el blanco de
mala atribución (`err_identidad` entre las preguntas con respuesta).

> **Cumple si el foco alcanza AUC ≥ 0,70 en ≥ 2 de 3 semillas Y le gana a `pos_q` por ≥ 0,05 en
> ≥ 2 de 3.**

Las dos mitades juntas. Si la señal existe pero `pos_q` la tiene igual, **no hay dilución** y la
pieza nueva de la bandera —el transporte— no aporta nada, aunque el detector sirva.

**D-3 · NULO, tiene que fallar.** Las mismas sondas con las etiquetas permutadas al azar en la
muestra de ajuste, 20 repeticiones.

> **OK si el AUC medio queda en 0,50 ± 0,03.** Si el nulo pasa, la sonda está sobreajustando y
> D-1 y D-2 quedan **inválidas**, no discutibles.

**D-4 · RÉPLICA EN `pre`, sin criterio de éxito.** Se corre todo sobre `p3_*` y se reporta. Si D-2
cumple igual en `pre`, el efecto **no** depende de la query conjunta y se dice así.

**D-5 · CONTRASTE HONESTO, sin criterio.** Se reporta el AUC de la cabeza sola (`cab`) sobre cada
blanco. Es el detector que ya existe y ya gana, y cualquier cosa nueva tiene que compararse contra
él, no contra el azar.

## 5. Criterio de abandono, comprometido por adelantado

Esto es lo que faltó las cinco veces que la línea de detección falló, y lo que sí se hizo bien con el
test de k.

> **Si D-1 falla y D-2 falla, la línea de la bandera se cierra. No se prueba una segunda forma de
> emitir la señal, ni una tercera featura, ni otro clasificador.**

Y si **D-3 pasa** (el nulo separa), no hay veredicto de ninguna clase, sólo un instrumento a
arreglar.

## 6. Cómo se lee cada desenlace, escrito ANTES

| celda | lectura |
|---|---|
| D-1 y D-2 cumplen | la separación es el mecanismo y el transporte aporta. Se construye la bandera con prereg propio |
| D-1 cumple, D-2 no | **la separación es el hallazgo y la bandera no hace falta**. Se puede implementar con dos cabezas en `pos_q`, que es mucho más barato |
| D-1 no, D-2 sí | hay señal en el foco pero componer no ayuda. Sospechar que las dos sondas están midiendo lo mismo |
| ninguna | se cierra la línea, y el paper del trípode queda como estaba |

**Lo que este experimento no contesta, dicho ahora:** nada de esto es sin etiquetas. Las sondas se
entrenan con supervisión, igual que `cabeza` y `slot`. **No habilita la frase «el modelo sabe cuándo
no sabe»** y no se va a escribir aunque los dos criterios cumplan.

## 7. Desviaciones

Cualquier apartamiento de este documento se registra en `DESVIACIONES_DOS_DETECTORES.md` con su
motivo, antes de reportar el resultado.
