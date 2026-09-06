# ENMIENDA · `PREREG_RELOJ_O_BANDERA.md` (SHA `aabb4b20`) · 2026-09-06

Se escribe **después de un smoke** de 1 lote × 4 muestras sobre `lg3_s0`, y antes de la corrida
completa. Se declara así en vez de meterla en el pre-registro para que quede claro qué se decidió
sin mirar y qué se decidió mirando.

## E-1. Una condición más: `extra_corridas`

`corrido` mueve **dos cosas a la vez**: la frontera (de 24 a 12) y el rango de las entradas ajenas
(de U(0,23) a U(0,11)). Si rompe, el pre-registro lo atribuye al valor absoluto, pero no separa
**cuál de los dos bloques** es el que hay que dejar quieto.

Se agrega **`extra_corridas`**: episodio en 24..63 **intacto**, extra en U(0,11). Cambia el rango de
las ajenas y **no** toca la frontera ni al episodio.

- Si `extra_corridas` ≈ `real` (|Δíndice| ≤ 0,05, |ΔRECUP| ≤ 0,03) y `corrido` rompe, entonces lo que
  importa es **en qué filas de `ord` cae el episodio**, no dónde caen las ajenas.
- Si `extra_corridas` también rompe, el modelo depende de las filas específicas de los dos bloques y
  la lectura es la misma pero más amplia.

Ningún criterio C-0..C-4 cambia. `extra_corridas` no puede salvar ni hundir una hipótesis por sí
sola: desagrega.

## E-2. C-1 puede quedar no evaluable, y ya se sabe por qué

En el smoke, bajo `corrido` las 4 entradas correctas cayeron **todas** por debajo de 24 (`n` bajo/alto
= 4/0): el hecho que se pregunta tiende a estar en los primeros turnos del episodio, así que con el
corrimiento de 12 casi todo el material útil se va abajo del umbral y la brecha interna de C-1 se
queda sin la mitad alta.

Si en la corrida completa `n_alto` es 0 o muy chico (< 10), **C-1 no se evalúa como brecha interna** y
se reporta en su lugar la comparación **pareada** `real` → `corrido` sobre las mismas muestras: mismo
contenido, misma entrada correcta, mismo orden relativo, y lo único distinto es el valor absoluto del
turno. Esa comparación contesta la misma pregunta y es más limpia; lo que se pierde es la
desagregación interna, que era un lujo, no el criterio.

Se declara acá para no elegir la lectura después.

## E-3. El barrido de corrimiento, y por qué rescata C-1

Se agrega `--corrimientos D`: el episodio se mueve a `24−D .. 63−D` y las ajenas quedan **fijas** en
U(0,11) — que por E-1 no cambia nada, así que el barrido aísla al episodio. Con `D` chico, sólo las
primeras entradas del episodio cruzan por debajo de 24 y **las demás siguen arriba**, que es
exactamente lo que a `corrido` (D = 12) le faltaba para poder desagregar C-1: ahí hay muestras de los
dos lados del umbral y la brecha se puede medir dentro de la misma condición.

Barrido declarado antes de correrlo: `D ∈ {0, 2, 4, 6, 8, 10, 12}`, mismas semillas, mismos lotes.

- Si la degradación es un **escalón** que sigue al cruce del umbral —los que quedan en turno ≥ 24
  siguen en 1,0 y los que caen por debajo se desploman— la marca de orden es **absoluta**.
- Si la degradación es **suave y pareja en los dos lados**, lo que importa es la distancia recorrida
  y no la frontera, y el criterio C-1 no adjudica.
