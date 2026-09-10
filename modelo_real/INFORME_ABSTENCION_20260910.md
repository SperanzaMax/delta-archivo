# La abstención, doce unidades y tres semillas por brazo

2026-09-10. Implementa `PREREG_ABSTENCION.md` y `ENMIENDA_ABSTENCION_20260910.md` (`f89071b6`).
Cada número sale de **512 muestras frescas** por semilla, con los cuatro brazos de control corridos
sobre las mismas muestras.

## El resultado en una línea

**H1 se cumple, y sólo con el presupuesto largo.** `abstlargo` (19.000 pasos después del
calentamiento) llega a `GLOBAL` **0,7227 ± 0,0531** con `acierto` **0,8055 ± 0,1171**, o sea supera el
piso del mudo y alcanza el 0,80 que pedía el pre-registro. Los otros tres brazos no.

**Y hay que decir de qué tamaño es el margen: el 0,80 se cumple por 0,0055 con una sd de 0,1171.**
El criterio pre-registrado se evalúa sobre la media y la media pasa, pero una campaña con otras tres
semillas podría dar lo contrario.

## Los cuatro brazos

| brazo | calentamiento | post | `GLOBAL` | `acierto` | `nose` | H1 |
|---|---|---:|---|---|---|---|
| `abst3s` | ninguno | 16.000 | 0,6764 ± **0,0069** | 0,6483 ± 0,0536 | 0,7104 ± 0,0654 | no |
| `abstcal3s` | 3.000 | 13.000 | 0,6960 ± 0,0608 | 0,7733 ± 0,1254 | 0,5902 ± 0,2928 | no |
| `abstcal6p` | 6.000 | 13.000 | 0,6764 ± **0,0069** | 0,6730 ± 0,1158 | 0,6715 ± 0,1789 | no |
| `abstlargo` | 3.000 | 19.000 | **0,7227** ± 0,0531 | **0,8055** ± 0,1171 | 0,5979 ± 0,2763 | **sí** |

## Ningún contraste sale significativo, y la razón está medida

| contraste | `GLOBAL` | `acierto` |
|---|---|---|
| H3 · calentar 3.000 contra no calentar | +0,0195 ± 0,0676 · t = +0,50 | +0,1250 ± 0,1338 · t = +1,62 |
| par A · calentar 6.000 contra 3.000, post igualado | −0,0195 ± 0,0566 · t = −0,60 | −0,1003 ± 0,2256 · t = −0,77 |
| par D · 19.000 post contra 13.000 post | +0,0267 ± 0,0413 · t = +1,12 | +0,0322 ± 0,0979 · t = +0,57 |

Los tres pareados por semilla, los tres con |t| < 1,7. **Ayer, con una semilla por brazo, el mismo
contraste de H3 se leyó como +0,0792 y +2,99 σ.** Esa σ era la de la curva dentro de una corrida, no
la de entre semillas, y la de entre semillas es seis veces más grande.

**La celda que faltaba tiene respuesta:** con el presupuesto post-calentamiento igualado, calentar
6.000 en vez de 3.000 no deja al modelo en mejor lugar (−0,0195, t = −0,60).

## Por qué no hay potencia, y no es falta de prolijidad

**`r(acierto, nose) = −0,8842` sobre las doce unidades, t = −5,99 con 10 grados de libertad.**

El modelo no se mueve hacia arriba, se mueve **a lo largo de una curva de operación**. Entre las doce
unidades `acierto` recorre 0,5734 a 0,9181 y `nose` recorre 0,2648 a 0,8326, en direcciones opuestas.
Qué semilla toca decide **dónde te parás en la curva**, y eso mete en `acierto` y en `nose` una
varianza que no tiene nada que ver con lo que el brazo hace.

**La vara de Maxi es lo único que atraviesa eso, y ahora está medido.** `GLOBAL` es una suma de las
dos componentes anticorrelacionadas, así que el movimiento a lo largo de la curva se cancela:

| | rango entre las 12 unidades |
|---|---|
| `acierto` | 0,3447 |
| `nose` | 0,5678 |
| **`GLOBAL`** | **0,1425** |

Y en los dos brazos donde el punto de operación no se dispersó, `GLOBAL` tiene **sd 0,0069**, entre
ocho y veintiséis veces menor que la de sus propias componentes.

> «no queremos un modelo callado, queremos uno que responda bien o diga no sé»

La métrica que salió de esa frase resultó ser la única de las tres que mide capacidad en vez de
punto de operación.

## Los controles, que ahora sí son controles

Agregados sobre las doce unidades, 6.144 muestras cada uno:

| control | `GLOBAL` | `acierto` |
|---|---|---|
| `BARAJADO` | 0,3052 | 0,0348 a 0,0592 |
| `VACIO (ceros)` | **0,000000** | 0,0000 |
| `lectura APAGADA` | **0,000000** | 0,0000 |

**El cero de `VACIO` y de `lectura APAGADA` es ahora un cero sobre 6.144 muestras**, no sobre cuatro.
Sin archivo el modelo no contesta nada, y eso queda cerrado.

`BARAJADO` tira el acierto de 0,65-0,81 a 0,03-0,06: el modelo lee el archivo para responder. Su
`GLOBAL` de 0,3052 **no es una falla del control**, es que barajar no cambia la respuesta correcta en
las clases de abstención (ver la nota E-6 de la enmienda).

## H2 queda sin evidencia

`nose_rel < nose_aus` se cumple en **7 de 12** unidades. Por brazo: 3 de 3 en `abstcal6p`, 2 de 3 en
`abst3s`, 1 de 3 en `abstcal3s` y 1 de 3 en `abstlargo`. La misma dispersión del punto de operación
se lo come.

## Dos observaciones que quedan anotadas

1. **El modelo sigue aprendiendo cuando se le acaba el presupuesto.** La evaluación final está por
   encima de la cola de la curva en los **cuatro** brazos: +0,0176 · +0,0945 · +0,0150 · +0,0677. No
   hay meseta al final en ninguno.
2. **Ninguna de las doce llega al techo** por el criterio de desestabilización (`acierto` del tramo
   medio ≥ 0,80). La más alta es `abstlargo_s0` con 0,6924. El criterio no aplica y ninguna se
   marca, que es lo correcto y no un resultado.

## Lo que sigue, y sale del resultado y no de las ganas

El cuello no es el presupuesto ni el calentamiento, es que **el punto de operación lo elige la
semilla**. Mientras eso siga así, ningún contraste sobre `acierto` o `nose` va a tener potencia con
un número razonable de semillas: con sd 0,05 en `GLOBAL` y un efecto de 0,027 harían falta unas
55 semillas por brazo.

La salida es **fijar el punto de operación en vez de sortearlo**, y para eso hace falta medir
`mudez`, el simétrico de `invento`, que se agregó hoy y que las doce unidades de esta campaña **no
traen**.
