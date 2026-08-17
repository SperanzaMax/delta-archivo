# El hecho SIEMPRE está escrito — lo que pierde es el ranking

**2026-08-16** · `rank_hecho.py` · CPU, cero GPU

## El límite que cierra

Las dos sondas dejaron el hecho propio **inalcanzable en todas sus versiones**, y desde afuera «nunca
se escribió» y «se escribió y la lectura no lo alcanza» producen exactamente el mismo síntoma. Para
separarlas hacía falta mirar el archivo por dentro.

Se agregó un mapeo **enunciado → hecho** (`idioma.episodio(con_origen=True)` y
`datos.lote(con_origen=True)`), puramente contable: **no consume una sola llamada al RNG**, y se
verificó que sesiones, consultas y targets son idénticos desde la misma semilla. Con eso se puede
preguntar en qué puesto del ranking de lectura quedó la entrada del hecho preguntado, **medido en la
posición de máximo foco** (no en la de respuesta, donde ya se sabe que la distribución está difusa).

## Resultado, n = 4000 por checkpoint

| | `n4_s0` err / acierto | `n3_s2` err / acierto |
|---|---|---|
| **entrada AUSENTE del archivo** | **0,0000 / 0,0000** | **0,0000 / 0,0000** |
| rank 0 (gana la lectura) | 0,1368 / **0,5047** | 0,1771 / **0,4652** |
| **rank mediano** | **2,0 / 0,0** | **2,0 / 1,0** |
| rank medio | 2,28 / 1,32 | 1,96 / 1,47 |

## Tres lecturas

**1. El hecho está escrito SIEMPRE. `ausente = 0,0000` en los cuatro grupos.** No hay un solo caso en
8000 muestras en que el hecho preguntado no tenga su entrada en el archivo. **La hipótesis de escritura
queda cerrada definitivamente**, y con ella las tres candidatas que fuimos probando hoy: no se
corrompe al vecino, no se pierde la corrección, y no se omite el hecho.

**2. Lo que pierde es la competencia.** Cuando el modelo falla, la entrada correcta está ahí pero
queda en **el puesto 2** de la mediana y sólo gana el 14-18 % de las veces; cuando acierta, la mediana
es **0-1** y gana el 47-50 %. El error de identidad es **enteramente de lectura**, y por lo tanto
**convertible en abstención** — que es lo que la campaña necesitaba saber, ahora por evidencia directa
y no por descarte.

**3. El dato que no esperaba, y es el más interesante: el modelo acierta sin que la entrada correcta
gane.** Incluso entre los **aciertos**, la entrada del hecho preguntado encabeza la lectura sólo la
mitad de las veces y su rank mediano no es 0 en los dos checkpoints. Es decir: **acertar no requiere
que la clave correcta gane** — el modelo integra varias entradas y resuelve aguas abajo.

Eso es coherente con todo lo medido hoy y lo explica junto: si la respuesta se arma integrando y no
seleccionando, entonces (a) el score máximo no tiene por qué codificar presencia/ausencia —§4.1—, y
(b) el margen de la lectura tampoco. **El archivo funciona como un banco de evidencia parcialmente
ordenado, no como un índice que devuelve un registro.**

## Consecuencia para el mecanismo de abstención

Refuerza el slot nulo **y le cambia el criterio de éxito**. Si el modelo puede acertar con la entrada
correcta en el puesto 2, entonces «que gane el slot nulo» no puede ser la única condición de
abstención: haría abstenerse en casos que hoy se responden bien. La forma correcta es que el nulo
compita **por masa relativa**, no por victoria — y eso hay que medirlo, no suponerlo.

## Límites

- Dos checkpoints, una semilla por nivel, `p_nose = 0`.
- El rank se mide en la posición de máximo foco de la **capa 0**, que es donde se inyecta la lectura.
  No dice qué hacen las capas siguientes con lo leído.
- «Ausente» sólo puede darse por truncamiento de sesión, y el truncamiento está medido en 0,0000
  desde el 14-ago; el resultado confirma esa medición por otra vía.
