# Desviaciones del pre-registro — cabeza de abstención (`PREREG_CABEZA_ABSTENCION.md`)

Prereg congelado el 2026-08-18 (SHA `7d7eebfb…`, 12:12 UTC; corrección del conteo de params
193→129 a las 12:15) **antes de implementar y antes de correr un solo paso**. Este archivo registra
todo lo que se apartó de él, incluido lo que se apartó *a favor* del resultado.

---

## D-C1 — **P-4 falla en su forma literal, y falla hacia el lado bueno**

**Lo que pedía el prereg:** P-4 era el control de sanidad sobre `vigente`: la cabeza no tiene que
comprar su ganancia en abstención a costa de la precisión, así que `vigente` de `cabeza` debía
quedar **dentro de ±0,05** de `token`.

**Lo que salió,** con las 7 unidades cerradas (`cabeza` − `token`):

| unidad | Δ `vigente` | veredicto |
|---|---:|---|
| 1_s0 | **+0,0633** | FALLA, hacia arriba |
| 2_s0 | −0,0075 | cumple |
| 3_s0 | **+0,0668** | FALLA, hacia arriba |
| 3_s1 | −0,0004 | cumple |
| 3_s2 | **+0,0928** | FALLA, hacia arriba |
| 4_s0 | −0,0019 | cumple |
| 4_s1 | +0,0171 | cumple |

**Falla en 3 de 7, y las tres veces por exceso: la cabeza recupera MEJOR.** Ninguna unidad se pasó
hacia abajo. El criterio se escribió para detectar que la ganancia estuviera comprada con precisión
y pasó exactamente lo contrario.

**Qué se hace:** se registra el fallo **como está**. No se toca el umbral ni se reescribe P-4 como
prueba de una cola después de ver los números. El resultado que se reporta es «P-4 falla en su forma
literal», no «P-4 cumple si se lo lee bien».

**Lectura mecánica, y es coherente con el diseño:** en `cabeza`, `NOSE` sale del softmax de valores,
que entonces no gasta masa de probabilidad en una opción que no es un valor. Predice que la ganancia
sea mayor donde `token` se abstiene de más — y **Spearman(`falsa_abst` de `token`, Δ`vigente`) =
+0,714** sobre las 7 unidades. **Con n = 7 eso es sugerente y nada más** (p ≈ 0,07 bilateral), y
además 1_s0 lo rompe en parte: tiene `falsa_abst` bajo (0,0667) y aun así gana +0,0633. Queda como
**observación exploratoria**, no como mecanismo establecido.

---

## D-C2 — El presupuesto de fase (2000 pasos) lo elegí yo, no el prereg

El prereg fija las tres condiciones, el ckpt base común y el reinicio de Adam, pero **no dice cuántos
pasos dura la fase**. Se usaron **2000** para las 21 unidades, elegidos por mí antes de lanzar y
sostenidos sin cambio en toda la campaña.

**Qué se pierde, dicho explícitamente:** si `token` necesitara más presupuesto que `cabeza` para
llegar al mismo lugar, esta campaña lo leería como que `cabeza` es mejor cuando en realidad sería
más rápida. **No es una posibilidad teórica en este proyecto:** E-I3b y E-I3c ya mostraron dos veces
que un negativo intermedio era impaciencia. Lo que acota el riesgo acá es que el fallo de `token` no
es de nivel sino de **forma**: su `nose` es alto (0,58-0,71) y lo que no baja es `falsa_abst`, o sea
no está a mitad de camino de aprender, está en otro punto de operación.

---

## D-C3 — `c4_s1` se cerró un día después que las otras 20

Las 20 primeras unidades corrieron el 2026-08-18. `c4_s1` quedó sin verificar esa noche y se
confirmó el **2026-08-19** (14000 pasos, `nose` 0,6189 · `falsa_abst` 0,0746 → **pasa la
compuerta**). Sube P-1 de «3 de 5» a **4 de 5**.

**Por qué no contamina:** el criterio de la compuerta estaba fijado desde el 15-ago, la unidad se
corrió con el mismo script y el mismo presupuesto que las otras, y **sólo podía sumar a P-1** — que
es lo que se dijo por adelantado al dejarla pendiente, no después de ver el número.

---

## D-C4 — `sonda_umbral.py` es EXPLORATORIA post-hoc

No está en el prereg. Se escribió **después** de ver que c3_s0 fallaba, para separar «la cabeza no
puede» de «el umbral σ>0,5 está mal puesto». Todo lo que salga de ahí —incluido que el techo sea de
calibración y no de capacidad— **es generación de hipótesis, no confirmación**, y así se reporta.

Vale igual una lección de método que salió de ella y que no depende de la sonda: **el óptimo pegado
al borde del criterio no generaliza.** Eligiendo el umbral con el criterio real (`f_abst ≤ 0,10`)
pasaban 2 de 6 unidades; pidiendo **margen** al elegirlo (0,07) y juzgando con 0,10, pasan 5 de 6.

---

## D-C5 — Un artefacto propio, cazado por md5 (la 8ª vez en el proyecto)

`c4_s1` entró en la primera tabla de normas **como si fuera un resultado y era copia bit a bit del
checkpoint base**: `tramo_abst.sh` siembra con `cp`, así que el `.pkl` existe desde antes de
entrenar un solo paso. Un checkpoint sembrado y uno entrenado se ven igual desde afuera —mismo
nombre, mismo tamaño aproximado, fecha reciente—.

**Control agregado:** la sonda compara el md5 contra su base y descarta la unidad si coinciden.
Verificado hoy para la corrida buena: `c4_s1` md5 `d27712f5…` contra `x4_s1` `2df80359…` — distintos,
la unidad entrenó de verdad.
