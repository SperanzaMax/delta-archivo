# El modelo sí enfoca — y el foco no sabe si el hecho está

**2026-08-16** · `foco_lectura.py`, `foco_posiciones.py`, `score_pos_foco.py` · CPU, cero GPU

> **⚠ Este informe reemplaza una versión anterior del mismo día que concluía «el modelo no enfoca el
> archivo: lo promedia».** Esa conclusión era **falsa y fue un artefacto de la posición donde miré**.
> La versión anterior declaraba ese riesgo como límite; la verificación se corrió y lo confirmó. Se
> deja constancia porque el error y su detección son parte del resultado.

## Lo que pasó

**Primera medición**, en `pos_q` —la posición desde la que se decide la respuesta—: entropía de la
lectura 1,7242 contra un techo de ln(6) ≈ 1,80, y masa del top-1 de 0,27 contra 0,167 del reparto
uniforme. Leído solo, eso decía que la lectura era casi uniforme y que el modelo no seleccionaba nada.

**Verificación** (`foco_posiciones.py`), recorriendo **todas** las posiciones de la consulta:

| | `n4_s0` | `n3_s2` |
|---|---:|---:|
| entropía en `pos_q` | 1,7118 | 1,7660 |
| **entropía mínima sobre todas** | **1,0482** | **1,0197** |
| **masa top-1 máxima sobre todas** | **0,6492** | **0,6346** |

**El modelo enfoca, y con fuerza**: concentra hasta el 65 % de la masa en una sola entrada del
archivo, en posiciones **intermedias** de la consulta. Para cuando llega a `pos_q` la distribución ya
está difusa, porque el estado recurrente ya integró lo leído. Medir el foco en la posición de
respuesta era medirlo después de que ocurrió.

## Y el resultado principal de la Fase 0 sobrevive al control

Si el foco vive en otra posición, la señal de ausencia podría vivir ahí también — y entonces el
AUC 0,4984 sería un artefacto de la sonda. Se re-midió en cuatro lugares (`score_pos_foco.py`,
n = 4000):

| dónde se toma el score | `n4_s0` | `n3_s2` |
|---|---:|---:|
| `pos_q` *(réplica de la Fase 0)* | **0,4984** | **0,5022** |
| **posición de máximo foco** | **0,5007** | **0,5077** |
| máximo sobre todas las posiciones | 0,5293 | 0,5429 |
| margen en la posición de foco | 0,5081 | 0,5377 |

La réplica de `pos_q` reproduce **exactamente** los valores de la Fase 0. Y donde el modelo más
concentra —con 0,65 de masa en una entrada— **el score sigue en el azar**.

## El resultado, ahora bien enunciado

**El modelo selecciona una entrada del archivo con fuerza, y la fuerza de esa selección no codifica
si lo que buscaba está.** Enfoca igual de fuerte cuando el hecho existe y cuando no existe.

> **Siempre encuentra algo, y encuentra con la misma convicción cuando no hay nada que encontrar.**

Esa es la forma precisa del hallazgo de la Fase 0, y es más fuerte que la versión anterior: no es que
falte selección —la hay— sino que **la selección no tiene un estado de «vacío»**. Coherente con el
mecanismo candidato: el softmax de lectura suma 1 siempre, así que la mejor entrada gana con la misma
masa relativa haya o no haya un buen candidato.

## Lo que corrige del plan

**El slot nulo vuelve a ser viable, y la objeción que le puse hace unas horas era mía y estaba mal.**
Había argumentado que una clave nula competiría contra ~6 entradas y se llevaría 1/7 de la masa; eso
se calculó sobre la distribución **difusa de `pos_q`**. Donde la lectura ocurre de verdad, la masa
**se concentra** (0,65 en una entrada), así que un slot nulo puede efectivamente ganar la competencia
cuando nada matchea. La reparación sigue en pie.

## Método

Dos correcciones el mismo día sobre la misma medición, las dos detectadas por controles escritos
**antes** de mirar: el límite «no está descartado que concentre en posiciones intermedias» quedó
declarado en la primera versión, y correrlo cambió la conclusión. Es la octava vez en el programa que
un número limpio escondía dónde estaba puesta la sonda — y la primera en que el límite declarado por
adelantado fue exactamente el que falló.
