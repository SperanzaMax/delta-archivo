# DISEÑO · LA POLITICA DE ESCRITURA (2026-08-22, sin correr)

Es lo unico grande que queda del brazo interno, y es el frente de «que no olvide nunca lo que le
dije». Se escribe hoy mientras la campania de la query conjunta ocupa la GPU. **No esta corrido y no
tiene pre-registro todavia**: lo que sigue es el diseño y, sobre todo, la razon por la que el
experimento obvio no se puede correr tal cual.

## 1. El estado de hoy, y por que la pregunta todavia no existe

`modelo.escribir` archiva **un vector por enunciado**, tomado en la posicion de su ultimo token. «Un
hecho dicho = una entrada». Esta declarado en el docstring como la politica mas simple que existe, y
deja para despues la pregunta de QUE conviene guardar.

En el regimen medido hasta hoy el modelo **no olvida nada**, y no por virtud: el archivo no tiene
cota, entra todo. La pregunta de la eviction sorpresa-gated de [[vigia03-capacity-scheduling]] —no
gastar rango en lo predecible, echar primero lo de baja sorpresa— **no puede ni formularse** mientras
no haya presion de capacidad.

## 2. El experimento obvio, y por que sale vacio

Lo obvio es acotar el archivo a `N` entradas y comparar politicas a igual presupuesto: `fifo`
(retener las ultimas `N`), `azar`, y `sorpresa` (retener las `N` de mayor residuo comprometido
`beta * ||v - S k||`, que es la señal de CENTINELA-01).

Ese experimento **no puede dar señal en la tarea actual**, y conviene verlo antes de gastar la GPU y
no despues. En el generador de hoy cada enunciado es un hecho distinto y todos son igual de
informativos; cual se pregunta despues se sortea uniformemente. Entonces:

- la sorpresa no tiene nada que discriminar, porque no hay enunciados predecibles;
- y ninguna politica puede vencer al azar, porque no hay forma de saber cual hace falta.

Seria un negativo por diseño, de la familia del control vacio `m=1` del 12-ago: **un control tiene
que poder fallar, y un experimento tiene que poder ganar.** Con este generador, la unica politica
optima posible es «guardar todo», que es la de hoy.

## 3. Lo que falta construir: relleno

Para que la politica sea una pregunta hace falta que el episodio tenga **enunciados que no valga la
pena archivar**. Es la situacion real y por eso importa: una conversacion es mayormente relleno con
unos pocos hechos adentro. Tres tipos, en orden de cuanto exigen:

  · **repeticion** — un hecho ya dicho, dicho otra vez igual. Redundante de verdad: el archivo ya lo
    tiene. Es el caso mas facil y sirve de control de sanidad de la sorpresa.
  · **parafraseo** — el mismo hecho con otras palabras. Redundante en contenido, no en forma.
  · **charla** — enunciados bien formados sin contenido factual, que nunca se preguntan.

La prediccion se vuelve entonces nitida y falsable: con el archivo acotado a menos entradas que
enunciados, **`fifo` retiene lo ultimo (mezcla de relleno y hechos) y `sorpresa` retiene los hechos**.

## 4. La compuerta que va antes de cualquier entrenamiento

Antes de entrenar una sola unidad hay que medir, sobre un checkpoint que ya existe, que **la sorpresa
separa hechos de relleno**: AUC del residuo comprometido, hecho contra relleno, con su nulo. Si no
separa, la politica sorpresa-gated no tiene de donde agarrarse y el experimento se cae ahi, gratis.

Esa compuerta tiene que poder fallar, y puede: nada garantiza que un enunciado de charla tenga
residuo bajo. Un token raro puede dar residuo alto por ser raro, no por ser informativo, y ese seria
justamente el modo en que la idea se rompe.

## 5. Riesgo declarado

La sorpresa se mide **al escribir**, y lo que importa es si el hecho se va a **preguntar despues**.
Son dos cosas distintas y el experimento no debe confundirlas: la sorpresa es una apuesta sobre la
utilidad futura, no una medicion de ella. Si `sorpresa` gana, lo que se habra mostrado es que la
informatividad local predice la utilidad futura **en esta tarea**, donde lo que se pregunta son
hechos. Es un resultado real y es mas chico que «el modelo sabe que guardar».

## 6. Orden

1. relleno en el generador (`idioma.py` / `datos.py`), con su chequeo de padding;
2. compuerta del §4 sobre checkpoints existentes, sin entrenar;
3. recien ahi pre-registro y campania.

Hoy queda en el paso 0. La campania de la query conjunta tiene prioridad porque ataca el error
dominante y esta corriendo.
