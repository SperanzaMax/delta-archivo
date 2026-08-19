# Desviaciones del pre-registro — la frontera del margen (`PREREG_FRONTERA.md`)

Prereg congelado el 2026-08-18 a las 23:54 UTC (SHA `0acfe89b…`), **antes de tocar la infraestructura
y antes de correr un solo paso**. Acá va todo lo que se apartó de él.

---

## D-F1 — **`f2_s1` nunca cruzó 0,85: la campaña cierra con 2 semillas y no 3**

**Lo que pedía el prereg (§3):** nivel 2, semillas 0/1/2 → 3 márgenes × 3 semillas × 2 condiciones =
**18 fases**.

**Lo que pasó:** las bases `f2_s0` y `f2_s2` cruzaron los tres umbrales y dieron 6 cortes. `f2_s1`
**se estancó y nunca cruzó 0,85**:

| presupuesto | `vigente` máximo |
|---|---|
| 6000 pasos (lo que pedía el diseño) | 0,7777 |
| 14000 pasos (primera extensión) | 0,8234 |
| 20000 pasos (segunda extensión, hasta el horizonte de lr) | **0,8048 al cierre; nunca superó 0,8234** |

Se extendió **dos veces** antes de darla por perdida, precisamente porque en este proyecto ya hubo
cuatro negativos que resultaron ser impaciencia (E-I3b, E-I3c y el corte prematuro del 13-ago). Acá
no lo es: la curva llega a una meseta y **baja**, y 20000 pasos es el horizonte sobre el que está
calculado el decaimiento de la tasa de aprendizaje, así que seguir cambiaría la curva de aprendizaje
y dejaría de ser la misma corrida.

**Consecuencia:** corrieron **12 de las 18 fases**. Los tres márgenes quedan con **2 semillas cada
uno** en lugar de 3.

**Qué se pierde, dicho explícitamente:** F-2 pedía «al menos 2 de 3 semillas» y se evalúa sobre
2 de 2, que es un criterio más exigente por unidad pero con menos réplicas. Y **la bimodalidad entre
semillas —medida en este proyecto desde E-I3c— queda sin muestrear en el hueco**: con dos semillas no
se puede distinguir «las dos condiciones convergen siempre» de «tuvimos suerte con estas dos».

**Lo que NO se pierde:** `f2_s1` no es una semilla perdida al azar, es **información**. Una base de
nivel 2 que se traba en 0,80 es exactamente el tipo de modelo para el que la cabeza importa, y el
hecho de que exista refuerza el planteo del §1 del prereg: los modelos reales no llegan a `vigente`
1,0.

---

## D-F2 — **F-4 no se cumple en su forma literal, y el motivo es que el hueco estaba mal ubicado**

F-4 pedía que existiera **algún margen del hueco** donde `cabeza` pasara y `token` fallara en las
tres semillas. **No existe: `token` pasa en los seis puntos del hueco.**

Se registra como incumplida. **No se reescribe la predicción después de ver los datos.**

Lo que corresponde decir, separando el hecho de la interpretación:

- **Hecho:** el corte de `token` no está *dentro* del hueco sino en su **borde inferior** (entre
  +0,2358 y +0,2826). El hueco se definió el 17-ago con nueve modelos, y esos nueve dejaban el corte
  sin acotar por abajo.
- **Hecho:** en el rango +0,1672 a +0,2358 —**por debajo** del hueco, con datos que ya existían del
  18-ago— `token` falla en 4 de 4 y `cabeza` pasa en 4 de 4.
- **Interpretación:** la comparación que F-4 quería hacer es válida y da el resultado que buscaba,
  pero **en otro rango del eje**. La recomendación operativa se obtiene igual, con otros números
  (≈ +0,17 con cabeza contra ≈ +0,26 con `token`).

**Por qué esto no es reescribir la predicción a conveniencia:** las cuatro unidades que sostienen la
comparación **no se corrieron para esta campaña**; son de la campaña de la cabeza del 18-ago, cerrada
y publicada antes de que existiera este prereg. Lo que la campaña de la frontera aportó fue **acotar
el corte de `token` por arriba**, y eso es lo que permite afirmar que las dos fronteras son
distintas.

---

## D-F3 — **La segunda mitad de F-2 no se cumple**

F-2 pedía además que en el margen más bajo (0,85) `cabeza` le ganara a `token` en `falsa_abst` por
**≥ 0,05**. Gana por **0,0188** (s0: 0,0667 vs 0,0855) y **0,0213** (s2: 0,0264 vs 0,0477).

**No cumple.** El motivo es visible en la tabla y no rescata la predicción: en el margen 0,85
**`token` ya pasa holgadamente** (0,0855 contra un techo de 0,10), así que no queda distancia que
recuperar. Un criterio de ventaja mínima absoluta no tiene sentido donde el control ya está cerca del
piso; **eso es un defecto del criterio, escrito por mí, y se anota como tal sin tocarlo**.

---

## D-F4 — El coeficiente de Spearman cambió entre corridas con los mismos datos

Registrado con los números a la vista, porque es un error de método detectado al escribir el informe.

El primer cálculo dio −0,819 (`token`) y −0,909 (`cabeza`); el segundo, **sobre datos idénticos**,
−0,786 y −0,885. Causa: `t1_s0` y `t2_s0` (y sus pares `c1_s0`/`c2_s0`) **empatan en margen**
(+0,4094 ambos, porque las dos bases cerraron con `vigente` 1,0000) y la primera implementación
rompía el empate según el orden en que `glob` devolvía los archivos.

**Corregido con rangos promediados**, que es el tratamiento correcto de los empates y no depende del
orden de lectura: **−0,8033** y **−0,8886**. Las tres versiones superan holgadamente el −0,70
exigido, así que **el veredicto de F-1 no depende de esto**; se documenta porque un coeficiente que
se mueve entre corridas con los mismos datos no es publicable.

---

## D-F5 — El presupuesto de fase y el corte por valor: decisiones de implementación

Ninguna de las dos es desviación —el prereg fija las dos en §3 y §7— pero conviene dejar asentado
cómo se resolvieron, porque no era obvio:

El corte se hizo **por valor de `vigente`**, así que cada uno cayó en un paso distinto (3000, 3250,
3500 en la semilla 0; 3000, 3250, 3750 en la 2), mientras el presupuesto de fase es **2000 pasos para
todas**. Como `--pasos` es absoluto y el checkpoint trae su propio contador, cada fase corre hasta
*corte + 2000*. Eso quedó escrito en `fases.tsv` (lo genera `preparar_fases.py`) en lugar de
recalcularse en cada script, para que no puedan desincronizarse.

Las 12 fases usan **Adam reinicializado**, igual que la campaña de la cabeza, y el horizonte de la
tasa de aprendizaje se mantuvo en 20000 en todas.

---

## D-F6 — El confound del §4 sigue sin resolverse (declarado antes de correr)

El prereg declaró que el eje A confunde **margen** con **grado de entrenamiento**, y fijó el criterio:
*«Si el margen predice igual en ambas [vías], es el margen. Si no, el margen era un proxy»*.

**Lo medido: predice igual, sin una sola inversión en 13 unidades.** Pero las dos vías siguen
confundidas en el rango muestreado — todos los puntos entre +0,2826 y +0,3808 son sub-entrenados y
todos los de margen ≤ +0,2358 son de entrenamiento completo. **Es evidencia a favor, no una
demostración.** La celda que faltaría es un nivel 3 o 4 cortado en margen alto (o un nivel 2 entrenado
a fondo pero trabado en margen bajo, que es lo que `f2_s1` iba a dar y no dio).
