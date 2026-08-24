# DESVIACIONES · campaña del slot nulo

`PREREG_SLOT_NULO.md` quedó congelado con SHA `f95b6e9d`. Este archivo registra todo lo que se apartó
de él, con la razón, sin tocar el documento congelado.

---

## D-1 · la familia pasa de `n3_s*` a `y3_s*` (2026-08-24, antes de lanzar)

**Lo que decía el prereg (§3):** «Familia `n3_s*`».

**El problema:** `n` es el prefijo de la **campaña base** y `ckpts/n3_s0.pkl`, `n3_s1.pkl` y
`n3_s2.pkl` **ya existen** desde el 15-17 de agosto. Lanzar con ese prefijo habría hecho que el
rotador tomara esos checkpoints como el estado desde el cual continuar.

Las consecuencias posibles, en orden de gravedad:

1. La guarda de identidad del checkpoint aborta y se pierde el tramo — el mejor caso.
2. `paso_de()` lee `corridas_*/n3_s*.json` de agosto y cree que la corrida ya está avanzada, con lo
   cual el horizonte de la curva de lr queda mal.
3. Un modelo entrenado como `token` se reanuda como `slot`, y la campaña mide una mezcla de dos
   condiciones sin que nada lo avise.

**La corrección:** familia **`y3_s*`**, verificada libre en `ckpts/` y en todos los `corridas_*/`
(0 archivos). Nada más cambia: misma configuración, mismas semillas, mismo control `p3_s*`.

**Por qué se declara y no se corrige en silencio:** el prefijo es parte de la identidad de la
corrida, y el proyecto ya tiene un antecedente caro de dos cosas con el mismo nombre. Cambiarlo sin
registro dejaría el informe hablando de una familia que no existe en disco.

## D-2 · se lanza a las 18:25, en la ventana mala del pool

Cuatro días seguidos con la misma forma: **de mañana hay T4, de tarde se seca**. Se lanza igual
porque los rotadores esperan con vueltas y descanso, y llegan despiertos a la ventana de la mañana.

No afecta el resultado, sólo el calendario. Se registra para que el tiempo hasta el cierre no se lea
como un problema de la campaña.
