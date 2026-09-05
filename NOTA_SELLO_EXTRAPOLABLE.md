# NOTA · el sello de orden no extrapola, y como se arregla · 2026-09-05

Sale de la ingenieria inversa de hoy (`INGENIERIA_INVERSA_20260905.md`). **No es un pre-registro y no
propone correr nada todavia**, es el diseño escrito para cuando toque.

## El problema, medido

`arch["ord"]` tiene **64 filas** y se indexa por el numero de turno. La indexacion de JAX **clampea
en silencio**: verificado hoy, los turnos 63, 64, 65 y 200 devuelven todos la fila 63. Pasado el
turno 64 el sello colapsa a una constante y **todos los hechos parecen igual de nuevos**, que es
exactamente el fallo que el sello vino a resolver.

Hoy se puso una guarda (`modelo.sello`) que lo vuelve **NaN** en vez de clampear, verificada
identica bit a bit con indices validos. Eso convierte un fallo silencioso en uno ruidoso, **pero no
levanta el tope**.

## Por que el tope existe

Porque el sello se penso como **identidad de turno**: una fila aprendida por numero de turno. Con 40
entradas por episodio, 64 filas sobran. Con memoria larga no hay numero que alcance.

## Lo que el sello tiene que preservar, y es menos de lo que parece

El hallazgo del 13-ago (0,4570 → 0,9956, DOI `rs-10896018`) es que **la clave lleva orden y por eso
el modelo puede comparar turnos**. Lo que la tarea necesita es **comparabilidad** —cual de dos
versiones es mas nueva—, no la identidad del turno. Eso abre tres salidas.

| opcion | extrapola | co-entrenado | el sello de un hecho viejo cambia al llegar hechos nuevos |
|---|---|---|---|
| **(a) sinusoidal** sobre el turno | si | **no** | no |
| **(b) `ord` indexado por turno NORMALIZADO**, con interpolacion | si | si | **si** |
| **(c) un vector aprendido `u` por una funcion monotona del turno**, p. ej. `u * log(1+t)` | si | si | no |

**(b) queda descartada para este objetivo, y la razon es la que importa:** si el sello depende del
turno relativo, cada hecho nuevo **reescribe el sello de todos los viejos**. El archivo dejaria de
ser inmutable, y con eso se cae la escritura incremental (R1) y la persistencia (R2), que son
justamente los requisitos que el objetivo pide y que el banco todavia no cumple.

**(c) es la que preserva las dos cosas**: mantiene el sello co-entrenado —que es el hallazgo— y deja
cada entrada sellada de una vez y para siempre. El costo es que pierde la capacidad de distinguir
turnos por identidad; hay que medir si la tarea la usaba.

## El control que hay que correr antes de creerse nada

Reentrenar con (c) y comparar contra el `ord` de tabla **a igual presupuesto y en el banco chico**,
donde el tope de 64 **no muerde**. Si (c) empata ahi, el cambio es gratis y habilita el archivo
largo. Si pierde, el sello estaba usando identidad de turno y eso es un resultado en si mismo.
