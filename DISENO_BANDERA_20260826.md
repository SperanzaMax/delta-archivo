# DISEÑO · LA BANDERA DE RECUPERACIÓN — idea de Maxi, 2026-08-26

> *«cuando encuentra el dato levanta una bandera y continúa, y el modelo en otra parte busca si hay
> alguna bandera levantada para saber si el dato es existente o creado sintéticamente»*

Esto **no es un pre-registro**. Es el análisis de la idea contra todo lo que ya está medido, y
termina en una prueba barata que puede cerrarla sin gastar una sola unidad del pool.

**Veredicto corto:** la versión literal de la idea ya está refutada tres veces, por tres caminos
distintos. Pero **la idea tiene un componente que ninguno de esos tres tenía**, y ese componente
apunta a un hueco mecánico que está medido y verificado en el código de hoy. Vale, reformulada.

---

## 1. Lo que la idea propone, descompuesto

| pieza | qué hace |
|---|---|
| **A · emisión** | en el momento en que el modelo lee el archivo, se emite una señal |
| **B · transporte** | la señal viaja con el cómputo, sin diluirse |
| **C · consumo** | otra parte del modelo la lee y decide si el dato es real o no |

Las tres piezas importan por separado, y la que salva la idea es la **B**, que es la única que el
proyecto nunca probó.

---

## 2. La versión literal ya está refutada, y hay que decirlo primero

### 2.1 Si la bandera se emite y se lee en la interfaz de memoria, es el SLOT NULO

Corrido el 25-ago, tres semillas, 26000 pasos. La masa del slot dio **0,4074 / 0,4046 / 0,4020**
contra una tasa base de preguntas sin respuesta de **0,4048**. Convergió al prior, que es el óptimo
exacto de la BCE para un predictor sin señal utilizable. AUC 0,5190 / 0,5313 / 0,5182. Un barrido de
400 umbrales no rescata nada.

### 2.2 Si la bandera se lee al final del cómputo, es la CABEZA

Verificado hoy en el código, no recordado. `modelo.py:302` calcula el logit de abstención desde
`hn`, que es exactamente el mismo estado final que alimenta el softmax de vocabulario, y
`entrenar.py:113` lo lee **sólo en la posición de respuesta**:

```python
a = jnp.take_along_axis(a, pos[:, None], axis=1)[:, 0]
```

O sea que la pieza C —«otra parte del modelo consulta la señal»— ya existe, ya está medida, y **es la
condición que gana el trípode**.

### 2.3 Y la pregunta que la bandera contestaría ya se hizo, con el monitor de desacuerdo

El monitor del 20-ago preguntaba, sin etiquetas, **«¿esta respuesta viene del archivo?»**. Falló 0/8
con AUC 0,502-0,669, y el motivo está escrito y es el que mata la versión literal de la bandera:

> **el desacuerdo mide si la respuesta viene del archivo, no si viene de la entrada CORRECTA.**

Con el control M-3 pasando 8/8 (0,977-0,997), o sea con el instrumento funcionando perfecto.

### 2.4 El problema de fondo, en un número

El puntaje de matcheo del archivo separa «la respuesta está» de «la respuesta no está» con
**AUC 0,4984 y 0,5022**. Azar exacto. Y re-medido en la posición de **máximo foco** sigue en el azar
(0,5007 y 0,5077).

> **En la interfaz de memoria no hay información sobre presencia, en ninguna posición.** Una bandera
> emitida ahí no tendría qué levantar.

---

## 3. La parte de tu idea que NO está probada, y por qué puede valer

Hay una diferencia entre la bandera y las tres condiciones ya corridas, y no es cosmética.

**`slot` y `cabeza` leen la señal en un solo punto. Tu bandera la ESCRIBE en un punto y la LEE en
otro.** Eso es la pieza B, el transporte, y el proyecto nunca la probó.

Importa por un número propio que está en `INFORME_FOCO_LECTURA_20260816.md` y que nadie volvió a
mirar desde entonces:

> Recorriendo **todas** las posiciones de la consulta, la lectura concentra hasta el **65 %** de la
> masa en una sola entrada del archivo (entropía mínima 1,02-1,05 contra un techo de ln(6) ≈ 1,79),
> **en posiciones intermedias**. **En la posición de respuesta la distribución ya está difusa**
> porque el estado recurrente integró lo leído.

Cruzado con lo que verifiqué hoy en el código, eso dice algo concreto:

| | dónde está la evidencia | dónde se decide |
|---|---|---|
| lectura del archivo | posiciones **intermedias**, foco 0,65, entropía **1,0482 / 1,0197** | — |
| cabeza de abstención | — | `pos_q`, entropía **1,7118 / 1,7660** contra un techo de 1,79 |

Ese segundo número es lo que hace que valga la pena. **En la posición donde se decide, la lectura
está a un pelo de ser uniforme** (1,71 y 1,77 contra un máximo de 1,79). No es que la evidencia esté
un poco borrosa ahí: prácticamente no está.

**La decisión se toma en el lugar donde la evidencia ya se diluyó.** Tu bandera es exactamente el
vehículo que la llevaría intacta de un lugar al otro. Eso no es un rodeo de algo ya probado: es una
pieza que falta.

Segunda diferencia, menor pero real: la masa del slot está atada a un softmax que suma 1, y una
bandera como canal libre no tiene esa restricción. No es la causa del fallo del slot —la causa fue
que no había señal— pero le saca una atadura de encima.

---

## 4. La corrección que hay que hacerle a la idea, y es la que decide todo

Vos escribiste *«para saber si el dato es existente o creado sintéticamente»*. En este banco esa
distinción **no tiene casos**, y está medido:

> `err_fuera = 0,0000` en los cuatro niveles. **El modelo nunca inventa contenido.** Toda respuesta
> errada es un valor **real del archivo** puesto en la entidad equivocada.

Con vocabulario cerrado la fabricación libre está excluida por construcción, así que una bandera que
pregunte «¿este dato existe?» va a decir **sí siempre**, y va a tener razón siempre. No separa nada.

**La distinción que sí existe en este banco, y que es la que importa:**

| ~~pregunta que no sirve~~ | pregunta que sí |
|---|---|
| ~~¿el dato existe o lo inventé?~~ | **¿lo que recuperé corresponde a lo que me preguntaron?** |

Y esa es, palabra por palabra, la distinción que el informe del monitor dejó anotada como lo único
que faltaba medir:

> *«algo que separe anclado en la entrada correcta de anclado en cualquier entrada, que es la
> distinción que ni el logit ni el desacuerdo hacen.»*

### Por qué recién ahora se puede

Hasta el 22-ago el modelo **no podía formar una query conjunta**: la lectura entra en el bloque 0
sobre `emb[x]`, así que la query era función pura del token y el modelo consultaba token por token,
sin poder preguntar por entidad y relación a la vez.

`lat2` cambió eso, y cerró bien el 25-ago (V-0, V-1 y V-2 cumplen 3/3, `ident_rep` 0,0000 en las tres
semillas, `anterior` 1,0000 en las tres). **Con query conjunta, por primera vez se puede preguntar si
la entrada recuperada matchea las DOS componentes de la consulta en vez de una.** Esa señal no
existía cuando se corrieron las siete vías de detección.

---

## 5. La bandera, reformulada

> **No marca «encontré algo» —el modelo siempre encuentra algo—. Marca «lo que encontré coincide con
> las dos componentes de lo que me preguntaron».**
>
> Se emite donde el foco de lectura es máximo, no en la posición de respuesta.
>
> Es un canal libre, no una masa de softmax normalizada.

Las tres cosas a la vez. Cambiar sólo una la devuelve a una condición ya corrida.

---

## 6. Antes de construir nada, EL TEST BARATO

Esto es lo que faltó las cinco veces que la línea de detección falló, y lo que sí se hizo bien con el
test de k del 24-ago. **La prueba se corre sobre checkpoints que ya están en disco, en CPU, sin
tocar el pool.**

**Pregunta:** en un checkpoint `lat2` ya entrenado, ¿la coincidencia entidad × relación de la entrada
top-1 de la lectura separa los aciertos de los errores?

El mapeo enunciado → hecho ya existe y ya se usó (§4.6 del paper del 16-ago, con reproducibilidad
verificada y sin consumir llamadas al RNG), así que la etiqueta se puede construir sin entrenar nada.

| resultado | qué significa | qué se hace |
|---|---|---|
| **AUC ≈ 0,50** | no hay nada que transportar. La bandera no tiene qué levantar | **se cierra, sin gastar una unidad** |
| **AUC ≥ 0,70** | la señal existe y hoy se diluye antes de llegar a la decisión | se construye la bandera, con prereg propio |
| entre medio | ambiguo | se decide con la magnitud, declarada por adelantado |

**Y hay que declarar el criterio de abandono antes de correr**, como en el test de k. Propongo el
mismo formato: si el AUC no llega a 0,70 en al menos 2 de 3 semillas, la bandera se cierra y no se
prueba una segunda forma de emitirla.

**El control que puede fallar y tiene que estar:** medir la misma señal en la **posición de
respuesta**. Si ahí da lo mismo que en el foco máximo, entonces no hay dilución y la pieza B —el
transporte, que es lo único nuevo de la idea— no aporta nada, aunque la señal exista.

---

## 7. Una aclaración que hay que dejar escrita para no usarla de comodín

El `PLAN_FOCO_20260824.md` comprometió cerrar la línea por seis meses si el test de k daba negativo.
Dio negativo el mismo día y **la línea se cerró**.

Lo que se cerró es **la detección SIN ETIQUETAS**, y concretamente la hipótesis de que la colisión de
clave era la causa de que las siete vías fracasaran. La bandera **es supervisada**, igual que
`cabeza` y `slot`, así que no cae bajo ese cierre. Pero esto se escribe acá, por adelantado, para que
no sirva de excusa más adelante: si el test del §6 da negativo, **se cierra igual**, y el cierre no se
negocia con una octava forma de emitir la señal.

---

## 8. Lo que la bandera NO va a contestar, aunque salga bien

- **No dice que el modelo sabe cuándo no sabe.** Sigue siendo supervisada. El techo medido es de
  calibración, no de capacidad.
- **No dice nada sobre escala.** 863.730 parámetros, idioma sintético de 242 tokens.
- **No detecta invención**, porque en este banco la invención no ocurre. Detectaría **mala
  atribución**, que es el modo de fallo que sí existe acá y que es más difícil de cazar, porque
  cualquier verificación del tipo «¿este dato existe?» lo da por bueno.
