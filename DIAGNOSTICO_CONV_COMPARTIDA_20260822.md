# DIAGNOSTICO · LA CONV COMPARTIDA, Y POR QUE `lat` PAGA EN «ANTERIOR» (2026-08-22)

Escrito con la campania `lat` todavia corriendo (paso ~7500 de 26000). **El veredicto formal es a
26000 con el instrumento del prereg**; esto es diagnostico de operacion, sobre la historia de las
corridas y no sobre checkpoints medidos con la sonda declarada.

Sale de una pregunta de Maxi —«si a los 4000 alcanza el valor ideal, ¿por que no lo paramos ahi?»—
que obligo a mirar las metricas que yo no estaba mirando.

## El hecho

| `anterior` (recuperar la version vieja) | 1k | 2k | 4k | 6k | 8k | final |
|---|---|---|---|---|---|---|
| `lat` s0 | 0,333 | 0,379 | 0,268 | 0,251 | **0,243** | — |
| `pre` s0 | 0,197 | 0,327 | 0,617 | 0,783 | 0,738 | **0,986** |

En las tres semillas de `lat`, `anterior` oscila entre 0,12 y 0,41 **sin subir**, mientras `pre` a la
misma altura ya va por 0,74-0,82 y termina en 0,91-0,99.

**`lat` gana en `vigente` y pierde en `anterior`.** Es un intercambio, y hasta que Maxi pregunto no
estaba a la vista porque yo miraba `vigente` y el detector de atajo, las dos metricas donde `lat` va
bien.

## Hueco propio en el pre-registro, declarado

`PREREG_CAMINO_LATERAL.md` §5 pone en W-4 (no-intercambio) a `falsa_abst` y a `nose`, y **no pone
`anterior`**. Debio estar: es una de las tres capacidades que la tarea mide y es justo la que se
rompio. El prereg de la mañana tampoco lo tenia. **No se cambia el criterio ahora** —eso seria mover
el arco—; se deja asentado que W-4 mira menos de lo que deberia, y el informe reportara `anterior`
como observacion aunque no estuviera pre-registrado.

## La causa, y esta en el codigo

`modelo.tronco`, lineas 133-134:

```python
h = h + lectura(conv3(blk["conv"], ln(blk["ln1"], h)))          # la query de `lat`
h = h + jax.vmap(delta_mixer, ...)(blk, conv3(blk["conv"], ...)) # el mixer
```

**Las dos usan la MISMA `blk["conv"]`.** En el pre-registro yo lo escribi como una virtud —«la conv es
la misma del bloque, asi que no estrena parametros; las tres condiciones tienen los mismos 863.859»—
y esa simetria es real, pero **acopla dos cosas que necesitan balances opuestos**:

- el **mixer** necesita el mix de contexto local que le sirva a la regla delta;
- la **query** necesita **mucho** contexto para juntar entidad x relacion (que caen a distancia 2)
  y **poco** para no diluir el marcador temporal.

Y el marcador es el punto: la pregunta por la version vieja es `cual era antes el precio de banco ?`.
El token `antes` esta a distancia 4-5 de la entidad, **fuera de la ventana de la conv**, y ademas en
`lat` la query de *cada* posicion es una mezcla de tres tokens, con lo cual la query en la posicion de
`antes` es `conv3(cual, era, antes)` en vez de `antes` puro. La conv da la query conjunta y **cobra el
marcador de orden**.

Con una sola conv compartida, el modelo tiene que elegir un balance y paga en el otro lado. El patron
observado —gana en `vigente`, pierde en `anterior`— es exactamente eso.

## La correccion, y una propiedad que la hace fuerte

**Una conv PROPIA para la query**: `blk["convq"]`, kernel 3, `3 × D = 384` parametros sobre 863.859,
o sea **0,044 %**. Asi el modelo elige cuanto contexto quiere en la query sin tocar lo que el mixer
necesita.

Y tiene una propiedad que ninguna de las condiciones anteriores tenia: **si `convq` aprende
`[1, 0, 0]`, la query vuelve a ser `ln1(h)` exacto, o sea `lat2` degenera en `pre`.** La condicion
nueva **contiene al control como caso particular**, con lo cual no puede ser estructuralmente peor
—solo peor por optimizacion—. Eso convierte el experimento en una pregunta mas limpia: no es «¿esta
forma de query es mejor?», es «¿el modelo ELIGE usar contexto en la query cuando puede elegir?», y la
respuesta se lee en los pesos aprendidos de `convq`.

Inicializacion propuesta, y hay que declararla porque no es neutral: `convq = [1, 0, 0]`, es decir
**arrancar exactamente en `pre`**. Cualquier contexto que aparezca sera algo que el modelo fue a
buscar, no algo con lo que lo largamos.

## Estado

**No se toca `modelo.py` todavia.** La campania `lat` esta rotando entre cuentas y `tramo_abst.sh`
re-sube ese archivo en cada tramo; agregar una rama `lat2` no cambiaria el computo de `lat` —seria un
`elif` nuevo— pero la regla que este proyecto se dio es no editar los cinco archivos del generador
mientras haya campania viva, justamente para no tener que razonar sobre eso. Se aplica al cerrar.

## Lo que esto le agrega al informe de la mañana

El informe de hoy dejo un trade-off: *una query conjunta necesita contexto ya computado y la lectura
util necesita entrar antes del computo*. Este diagnostico sugiere un **segundo** trade-off, hermano
del primero y de la misma familia: *el contexto que le sirve a la query no es el mismo que le sirve al
mixer, y compartir la conv obliga a elegir*. Los dos se resuelven igual: **desacoplando**. El primero
desacoplando el punto de inyeccion de la formacion de la query (que es lo que `lat` hizo); el segundo,
desacoplando la conv de la query de la conv del mixer.
