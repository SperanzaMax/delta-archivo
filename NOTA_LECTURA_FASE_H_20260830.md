# NOTA DE LECTURA · la fase H no parte del mismo lugar que la T

**2026-08-30, con las cuatro unidades de `cabeza` CORRIENDO y antes de ver ningún resultado final.**
Se escribe ahora, no después, porque condiciona cómo se lee el desenlace.

---

## El hecho, ya medido en los primeros hitos

Las cuatro unidades de `cabeza` arrancan con **`abstencion` = 1,0000 exacto** (mudas), porque heredan
de `b3_s3`/`b3_s6` la cabeza **colapsada al prior** que el informe de la bifurcación del 29 midió como
una constante (AUC 0,52-0,57, el logit de la tasa base).

Las cuatro de `token` arrancaron en el extremo **opuesto**: `abstencion` = 0,0000, porque bajo
`cabeza` el token `NOSE` estaba fuera del softmax de valores y su logit nunca se entrenó.

**El prereg declaró esta simetría antes de correr y es lo que hace valiosa la comparación** (L-4: si
las dos convergen al mismo intermedio, el intermedio es el óptimo de la pérdida y no un resto del
arranque).

## El problema, y es de presupuesto, no de mecanismo

> **La Etapa 1 son 3000 pasos para las dos interfaces, y las dos NO tienen la misma distancia que
> recorrer.**

`token` arrancaba locuaz y llegó al intermedio (0,49) dentro del presupuesto. `cabeza` arranca muda,
y el antecedente directo dice que salir de ahí lleva tiempo: **el aviso del 26-ago está escrito
textual** —«las dos semillas se abstienen del 100 % durante ~3000 pasos y después aflojan SOLAS; NO
ES COLAPSO, no matarla en el minuto 2000»—.

**O sea: 3000 pasos es exactamente el orden de magnitud en el que aquellas unidades todavía no habían
aflojado.**

## Cómo se lee cada desenlace, comprometido ANTES

| desenlace de `cabeza` a 3000 pasos | lectura |
|---|---|
| sale del silencio y supera el piso | **L-1 cumple en H**, y con L-4 se puede juzgar la convergencia |
| sale del silencio y NO supera el piso | mismo resultado que `token`: refuerza que `q` es constante |
| **queda muda (`abstencion` ≈ 1,0000)** | **NO se lee como fracaso de la interfaz.** Es indistinguible de «no le alcanzó el presupuesto para salir del atractor», y este proyecto ya tiene **cuatro** negativos que eran impaciencia |

**En la tercera celda, el criterio de abandono del §6 NO se puede aplicar**, porque pide las dos
interfaces y una de las dos no habría sido medida en condiciones comparables. Lo que corresponde es
declarar el desenlace como **no evaluable por presupuesto** y decir qué haría falta: extender esas
unidades, con el horizonte de lr ya fijado en 12000 en su config, así que extender no toca la curva.

## Por qué esto no es aflojar el criterio

No se cambia ningún umbral ni se mueve el arco. **L-1, L-3 y L-4 siguen exactamente como estaban.**
Lo único que se declara por adelantado es que **una celda concreta del desenlace no es interpretable**,
y eso es lo contrario de aflojar: es impedir que un negativo sin potencia se cuente como negativo.

Es la quinta vez este mes que la potencia decide la lectura, y la primera en que se anota **con la
campaña ya corriendo y antes del dato**.
