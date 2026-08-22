# DESVIACIONES · campania de la QUERY CONJUNTA (2026-08-22)

## D-1 · el horizonte de la lr no viajaba, y la campania arranco con la curva equivocada

**Que paso.** La enmienda E-1 fijo 26000 pasos con horizonte 26000. El rotador recibe `PASOS` como
argumento y se lo pasa al tramo, pero **`HORIZONTE` no se exportaba**: `tramo_abst.sh` lo toma del
entorno con un default propio de `20000`, heredado de la campania de abstencion. Las seis unidades
arrancaron con `pasos 26000` y `horizonte 20000`, o sea con la curva de lr decayendo hasta 20000 y
los ultimos 6000 pasos planchados en la lr minima. No es lo declarado, y ademas vaciaba buena parte
del sentido de la enmienda, que era **dar presupuesto efectivo**, no pasos nominales.

**Como se caza.** Leyendo la **config del primer checkpoint**, no el log del rotador. El log imprime
lo que el rotador cree que esta corriendo; el checkpoint guarda lo que `entrenar.py` recibio de
verdad. Es la misma leccion que la D-1 de la replica del 20-ago —«la espera correcta no es que el
JSON diga 20000 sino que el checkpoint diga 20000»— aplicada a un parametro en vez de a un paso.

**Que se hizo.** Se detuvieron los seis rotadores en el paso 1000 de 26000, se borraron los
checkpoints y los JSON parciales, se agrego `export HORIZONTE="${HORIZONTE:-$PASOS}"` al rotador y se
relanzo todo desde cero. Se perdieron unos 15 minutos de GPU. **No se aprovecho nada de lo corrido**,
para que no quede ninguna unidad con dos regimenes de lr pegados en el mismo JSON.

**Por que no se dejo correr igual.** El error era simetrico —les pegaba a `pre` y a `post` por
igual—, asi que el contraste habria seguido siendo valido y la tentacion de seguir era real. Se
relanzo lo mismo, por dos razones: el numero declarado en un prereg hasheado tiene que ser el que
corrio, y sobre todo la enmienda existia **precisamente** para que el presupuesto no fuera el problema
si el resultado salia negativo. Correr con la curva corta era volver a dejar abierta la puerta que la
enmienda venia a cerrar.

**Regla que queda.** Un parametro que define la identidad de la corrida tiene que llegar por una via
que se verifique en el checkpoint, y se verifica **en el primer checkpoint, no al final**. El chequeo
de identidad al reanudar (que ya existe para `nivel`, `semilla`, `lr`, `idioma`, `d`, `capas`,
`horizonte` y desde hoy `donde`) protege contra que un tramo cambie de config a mitad de camino, pero
**no** contra que la campania entera arranque con la config equivocada: para eso hay que mirar, una
vez, lo que el ckpt declara.
