# Enmienda al pre-registro de la ley del reloj

`PREREG_LEY_DEL_RELOJ.md`, SHA `c3b5f1af`, commit `022be60`.

Escrita el 2026-09-09, **después de aplicar el criterio a los dos puntos que ya estaban en disco (8 y
60 entidades) y ANTES de que terminen los dos puntos nuevos (15 y 30)**. Eso importa. La enmienda no
puede estar elegida para que la hipótesis salga bien, porque los puntos que deciden la hipótesis
todavía no existen.

## Qué está mal

La tercera condición del criterio dice que ningún hito posterior puede bajar de `0,80 × meseta`.
**Es un umbral imposible de sostener con batch 4.**

Con 4 muestras por hito, `acierto` sólo puede valer 0 · 0,25 · 0,50 · 0,75 · 1. Sobre una meseta de
0,9838 el umbral queda en **0,7871**, así que **un hito de 3 aciertos sobre 4 ya lo perfora**.

Medido sobre `arch4e60largo_s0`, que termina con acierto 0,9838:

    hitos por debajo del umbral bajo   246 de 801   (31 %)
    el ultimo, en el paso              19.300
    t_techo que devuelve el criterio   19.125   de 20.000 pasos
    donde cruza de verdad la media movil   4.675

El criterio no estaba midiendo cuándo llega al techo. Estaba midiendo **cuándo ocurrió el último
lote con mala suerte**, que con ruido de muestreo tiende al final de la corrida por construcción.

## La corrección

La cláusula de permanencia pasa a aplicarse **sobre la media móvil de 8 hitos**, no sobre hitos
sueltos. Mismo espíritu, que es descartar un pico suelto, sin heredar el ruido de `n = 4`.

> **`t_techo`** = el paso del primer hito en el que la media móvil de 8 hitos alcanza
> `0,90 × meseta` **y** la media móvil no vuelve a bajar de `0,80 × meseta` en ningún punto
> posterior.

Todo lo demás del pre-registro queda **intacto**. La definición de `meseta`, el corte de 0,80 para
declarar que una corrida no llegó al techo, las hipótesis H1 y H2, la condición de refutación, la
prohibición de ajustar un exponente con cuatro puntos y el riesgo declarado de una sola semilla.

## Lo que se reporta igual

El criterio original queda implementado en `t_techo.py` bajo `--regla original`, y sus números van
en el informe al lado de los corregidos. **La enmienda se muestra, no se esconde.**
