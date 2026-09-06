# INFORME · el sello de orden es una BANDERA, no un reloj · 2026-09-06

Pre-registro `PREREG_RELOJ_O_BANDERA.md` (SHA **`aabb4b20`**) con `ENMIENDA_RELOJ_O_BANDERA.md`
(E-1 control de aislamiento, E-2 C-1 podía quedar sin muestras, E-3 barrido de corrimiento).
Instrumento `micro_lm/reloj_o_bandera.py`. Tres checkpoints de la campaña del archivo largo
(`lg3_s0/s1/s2`), 64 muestras por celda, **en CPU, sin entrenar y sin gastar Colab**. Máxima
temperatura 43 °C.

## 1. Qué se preguntó

El 5-sep la campaña cerró con que **el sello de orden aprende a descartar lo viejo**. El §3 de aquel
informe declaró lo que el control barajado no separaba: entender la recencia de **usar el sello como
dirección de búsqueda**. Y había una razón práctica para resolverlo antes que nada: el punto (a) de
lo que quedó abierto era agrandar `ord`, que tiene 64 filas — **y agrandar la tabla sólo sirve si el
sello es un reloj**.

Cinco condiciones sobre los **mismos lotes**, cambiando únicamente el vector de turnos. El archivo se
escribe una sola vez (`modelo.escribir` no depende de los turnos), así que las celdas comparten
contenido, pregunta y entrada correcta: diseño pareado exacto.

## 2. Los números

| unidad | condición | índice | masa correcta | RECUP | acierto |
|---|---|---:|---:|---:|---:|
| `lg3_s0` | `real` | 0,1668 | 0,7973 | 1,0000 | 1,0000 |
| | `barajado` | 0,7978 | 0,2186 | 0,4531 | 0,0938 |
| | `corrido` | 0,7509 | 0,2634 | 0,7812 | **0,0781** |
| | `extra_corridas` | 0,1675 | 0,7966 | 1,0000 | 1,0000 |
| | `comprimido` | 0,1580 | 0,8052 | 1,0000 | 1,0000 |
| | `invertido` | 0,9408 | 0,0897 | 0,1406 | 0,0000 |
| `lg3_s1` | `real` | 0,2263 | 0,7371 | 1,0000 | 1,0000 |
| | `barajado` | 0,8295 | 0,1878 | 0,4062 | 0,0312 |
| | `corrido` | 0,7986 | 0,2181 | 0,6406 | **0,0156** |
| | `extra_corridas` | 0,2262 | 0,7373 | 1,0000 | 1,0000 |
| | `comprimido` | 0,2264 | 0,7372 | 1,0000 | 1,0000 |
| | `invertido` | 0,9513 | 0,0793 | 0,1875 | 0,0000 |
| `lg3_s2` | `real` | 0,2367 | 0,7309 | 1,0000 | 0,9688 |
| | `barajado` | 0,7863 | 0,2288 | 0,5312 | 0,0312 |
| | `corrido` | 0,7589 | 0,2547 | 0,7344 | **0,0156** |
| | `extra_corridas` | 0,2391 | 0,7287 | 1,0000 | 0,9688 |
| | `comprimido` | 0,2356 | 0,7319 | 1,0000 | 0,9688 |
| | `invertido` | 0,9230 | 0,1052 | 0,3125 | 0,0000 |

**`real` reproduce `masa_turnos.py` con los mismos parámetros al cuarto decimal** (índice 0,1908 y
RECUP 1,0000 con un lote de 4, las dos herramientas): el instrumento está validado contra el que
adjudicó la campaña.

### El barrido (E-3), que es donde se ve el mecanismo

Episodio movido a `24−D .. 63−D`, ajenas **fijas** en U(0,11). `corr_0` es el ancla.

| D | índice (s0/s1/s2) | acierto (s0/s1/s2) | **RECUP con la correcta en turno < 24** | **RECUP con la correcta en turno ≥ 24** |
|---:|---|---|---|---|
| 0 | 0,17 / 0,23 / 0,24 | 1,00 / 1,00 / 0,97 | — (n=0) | **1,0000 / 1,0000 / 1,0000** |
| 2 | 0,37 / 0,42 / 0,41 | 0,66 / 0,64 / 0,61 | **0,5714 / 0,5000 / 0,5357** | **1,0000 / 1,0000 / 1,0000** |
| 4 | 0,59 / 0,63 / 0,61 | 0,30 / 0,27 / 0,25 | 0,7308 / 0,6538 / 0,6154 | **1,0000 / 1,0000 / 1,0000** |
| 6 | 0,73 / 0,76 / 0,73 | 0,11 / 0,08 / 0,06 | 0,7937 / 0,6667 / 0,7143 | 1,0000 (n=1) |
| 8-12 | 0,75-0,80 | 0,00-0,08 | 0,64-0,81 | — (n=0) |

En `corr_2` la misma condición, el mismo archivo y las mismas ajenas dan **1,0000 para toda muestra
cuya entrada correcta quedó en turno ≥ 24** y **0,50-0,57 para las que cruzaron por debajo**. La
brecha de C-1 es **0,4286 / 0,5000 / 0,4643**, contra el 0,15 que el pre-registro pedía.

## 3. El veredicto

- **C-0 (sanidad, bloqueante): CUMPLE.** Índice 0,17-0,24 dentro de [0,12; 0,31] y RECUP 1,0000.
- **C-1: CUMPLE del lado BANDERA.** Brecha 0,43-0,50 en las tres semillas. No es una pendiente: es un
  **escalón que sigue al umbral 24**, con el lado alto clavado en 1,0000 exacto.
- **C-2: CUMPLE.** `comprimido` —todas las ajenas apiladas en el mismo turno— es indistinguible de
  `real`: |Δíndice| ≤ 0,009 y ΔRECUP = 0,0000 en las tres. **El orden ENTRE las entradas viejas no se
  usa.** Alcanza con que estén del lado bajo.
- **C-3: CUMPLE.** Bajo `invertido` el índice sube a 0,92-0,95 en las tres: la masa se va a las
  entradas de turno alto, que ahí son las ajenas. La política es **«preferir turno alto»**.
- **C-4 (el que daría vuelta la lectura del 5-sep): NO se activa.** RECUP bajo `invertido` es
  0,14-0,31, lejísimos del 0,90.
- **E-1: `extra_corridas` es idéntico a `real`** (Δíndice ≤ 0,0024, ΔRECUP = 0,0000). Mover las ajenas
  no cambia nada; **lo que importa es en qué filas de `ord` cae el episodio**.

> **El sello de orden no es un reloj: es una bandera.** El modelo aprendió qué **filas** de `ord`
> significan «esto es de la conversación en curso», no una relación de antes-y-después. Con el orden
> relativo intacto y sólo el valor absoluto corrido dos lugares, la respuesta correcta pasa de 1,00 a
> 0,64 y la respuesta del modelo de 1,00 a **0,02-0,08**.

## 4. La explicación alternativa, y por qué no se sostiene

**La objeción buena:** en `corr_2` las muestras que caen por debajo de 24 son justo aquellas cuyo
hecho está en los **primeros turnos del episodio**, y podrían ser más difíciles por sí mismas — la
caída sería de la pregunta, no del sello.

**Lo que la descarta, con los datos que ya están:** en `real` la RECUP global es **1,0000 exacto**, y
un promedio de 1,0000 obliga a que todo subgrupo valga 1,0000, incluidas esas mismas muestras con el
hecho en los turnos 24-25. Son las mismas preguntas, el mismo archivo y el mismo modelo: lo único que
cambió es que su turno pasó de 24 a 22.

**La segunda objeción:** que al correr el episodio hacia abajo sus turnos se mezclen con los de las
ajenas. No pasa: con D = 2 el episodio ocupa 22..61 y las ajenas 0..11, sin solape.

## 5. Qué cambia — y qué NO cambia

**No se cae nada de lo publicado el 5-sep.** El modelo sí aprende a descartar lo viejo, el índice de
masa sí se mueve de 0,79 a 0,17 con el entrenamiento, y R11 sigue contestado. Lo que se acota es
**cómo**: no por antigüedad, por pertenencia a un bloque de filas fijo.

**Lo que sí cambia es el próximo paso.** Agrandar `ord` de 64 a N filas **no arregla el archivo
largo**: las filas nuevas llegan sin significado, y este resultado muestra que el modelo no transporta
la política a filas que no vio en ese rol. El punto (a) de la lista de anoche —«el cuello identificado
y barato»— era barato pero no era el cuello.

**Y toca el objetivo de fondo.** En memoria persistente de verdad, los turnos crecen sin cota: en la
conversación 200 nada va a estar en 24..63. Una marca absoluta funciona exactamente mientras el
archivo se parezca al del entrenamiento.

**Un tercer hallazgo, colateral y consistente con la película larga del 5-sep:** en el barrido la
RECUP se sostiene alta (0,64-0,81) mientras el acierto se desploma a 0,02-0,08, con la masa correcta
cayendo de 0,80 a 0,25. **Encuentra la entrada y contesta mal.** Recuperar y contestar vuelven a
aparecer como dos capacidades separadas, y el sello no sólo dirige la búsqueda: sostiene la
concentración de la lectura.

## 6. Lo que sale de acá

1. **Una marca de orden relativa a la consulta** (distancia al turno actual, no el turno), o
   **entrenar con la frontera sorteada por muestra** para que ninguna fila pueda significar «ajeno».
   Es un cambio de arquitectura y va con su propio pre-registro.
2. El régimen de 3280 entradas entrenado, que sigue abierto y ahora tiene una predicción: si la marca
   sigue siendo absoluta, va a fallar por la misma razón.
3. Medir lo mismo sobre un checkpoint entrenado con la frontera sorteada — es el control que
   convierte esta lectura en una causa.

## 7. Límites declarados

- Tres checkpoints de una sola campaña, 2000 pasos, 300 casilleros. Nada sobre otras arquitecturas ni
  sobre 3280 entradas.
- No mide turnos > 63: desde el 5-sep eso es NaN por `mode="fill"` y es otro experimento.
- `corrido` mueve la frontera hacia abajo; hacia arriba no hay lugar dentro de 64 filas, así que la
  simetría del efecto queda sin medir.
