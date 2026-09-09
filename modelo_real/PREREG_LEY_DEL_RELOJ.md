# Pre-registro · LA LEY DEL RELOJ

Congelado el 2026-09-09, **antes de mirar ningún tiempo hasta el techo**.

## De dónde sale

El 9-sep quedó medido que agrandar el problema **no baja el techo**. Tres tareas que a 2.000 pasos
estaban en el azar llegan a 0,9366 · 0,9900 · 0,9838 con 20.000. Lo que cambia no es a dónde llega
sino cuánto tarda.

Si ese "cuánto tarda" es una función ordenada del tamaño del problema, entonces hay una **ley de
escalado para instalar memoria episódica en un modelo congelado**, y eso no existe en la literatura.

## La definición de `t_techo`, fijada acá y no después

Sobre la serie de `acierto` en los hitos de una corrida.

1. **`meseta`** = media de `acierto` sobre el último 25 % de los hitos.
2. Si `meseta < 0,80`, la corrida **no llegó al techo** y `t_techo` queda indefinido. No se
   extrapola ni se estima.
3. **`t_techo`** = el paso del primer hito en el que la media móvil de **8 hitos consecutivos**
   alcanza `0,90 × meseta` **y** ningún hito posterior baja de `0,80 × meseta`. La segunda condición
   descarta un pico suelto.

## Condición del barrido

Se varía **una sola cosa**, la cantidad de entidades. Todo lo demás fijo, y es la configuración del
6-sep que está medida funcionando.

    8 valores · 1 relación · batch 4 · archivo 4 · sin versiones · sin abstención
    20.000 pasos · semilla 0 · TinyLlama congelado · escritura capa 11 · lectura capa 2

    entidades   8   15   30   60

Los puntos de 8 y 60 ya están corridos (`ancla` y `arch4e60largo`). Se agregan 15 y 30.

## Qué contaría como resultado, y qué no

- **H1.** `t_techo` es **monótono creciente** en la cantidad de entidades, 4 de 4.
- **H2.** La relación es **sublineal o lineal**, o sea `t_techo(60)/t_techo(8) <= 60/8 = 7,5`.
  Si es superlineal la ley existe igual pero la conclusión práctica se invierte.
- **Refutación.** Si `t_techo` no es monótono, o si algún punto no llega al techo con 20.000 pasos,
  **no hay ley** y se dice así.

Con cuatro puntos **no se ajusta un exponente**. Cuatro puntos alcanzan para decir si hay orden y en
qué dirección, nada más. Un exponente pide más puntos y más semillas, y sería otro pre-registro.

## Riesgo declarado

Todo esto es **una semilla por punto**. La variabilidad entre semillas de `t_techo` no está medida,
así que una no-monotonía chica podría ser ruido. Si H1 falla por poco, lo que corresponde es medir
la variabilidad, no declarar que no hay ley.
