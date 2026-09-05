# INGENIERIA INVERSA DEL MICRO-LM · que hace, como lo hace, por que lo hace · 2026-09-05

Pedido de Maxi esta mañana: **antes de gastar otra T4, desarmar lo que hay, entender por que cada
pieza esta donde esta, escribir que TIENE que hacer el sistema para cumplir el objetivo, y recien
entonces mirar que falta.** Este documento es eso. Todo lo que afirma esta leido del codigo o medido
hoy, y lo que es interpretacion mia va marcado.

---

## PARTE 1 · QUE HACE

En una frase: **es un modelo de lenguaje de 3,5 MB entrenado desde cero que aprende hechos dichos en
una conversacion partida en sesiones, los guarda en un archivo que sobrevive al reseteo de su estado,
y despues contesta preguntas sobre ellos sabiendo cual version rige y cuando no sabe.**

Seis piezas, y ninguna es accesoria.

| pieza | donde vive | que hace |
|---|---|---|
| **el idioma** | `idioma.py` | 242 tokens cerrados, hechos `<rel> de <ent> es <val>` en 4 parafrasis, correcciones, 4 niveles |
| **el tronco** | `modelo.py:tronco` | 6 bloques de conv causal k=3 + regla delta + MLP |
| **la escritura** | `modelo.py:escribir` | un vector por enunciado, tomado en su ultimo token |
| **la lectura** | `modelo.py:responder` | softmax sobre las claves del archivo, inyectado en el bloque 0 |
| **el sello de orden** | `arch["ord"]` | una tabla de 64 vectores sumada a la clave, para saber cual version es mas nueva |
| **la abstencion** | `NOSE` / `abst` / `k_nulo` | cuatro interfaces distintas para decir «no se» |

**Lo que hace que la prueba sea de memoria y no de contexto largo, y es la decision de diseño
central:** el estado recurrente `S` **se resetea entre sesiones** y el archivo **no**. Lo unico que
cruza de la sesion 1 a la sesion 5 es lo que se archivo. Si el modelo contesta bien, es porque leyo
el archivo.

---

## PARTE 2 · COMO LO HACE · el camino de un dato, de punta a punta

Sigo un hecho desde que se dice hasta que se contesta.

**(1) Se dice.** `idioma.episodio` sortea 4 hechos con entidades distintas y los reparte en 4
sesiones. Con probabilidad 0,5 cada hecho recibe una **correccion** — que es literalmente «te dije
Ana y ahora te digo Beto» — y esa correccion puede ir pegada al hecho o en una sesion posterior.

**(2) Se tokeniza.** `datos.lote` arma `ses (B, S, 96)` y guarda en `cortes` el indice del **ultimo
token de cada enunciado**, mas `turnos`, que es el numero de orden global del enunciado dentro del
episodio.

**(3) Se archiva.** `modelo.escribir` pasa las S sesiones por el tronco **apiladas como batch**
—son independientes por construccion, asi que el scan secuencial se paga una vez— y con
`take_along_axis` saca el estado en las posiciones de `cortes`. Sale `(B, S*E, D)`. Una entrada por
enunciado. Politica declarada: **un hecho dicho, una entrada.**

**(4) Se pregunta.** `datos.lote` arma la consulta de 12 tokens. En la forma `directa`
—`cual es el director de norte ?`— la **entidad** queda a distancia 1 del `?` y la **relacion** a
distancia 3.

**(5) Se lee.** `modelo.responder` proyecta el archivo a claves `ak = archivo @ kw + ord[turnos]` y a
valores `av = archivo @ vw`. La query sale del estado en el bloque 0, `q = convk(convq, ln1(h)) @ qr`,
y la lectura es `softmax(q·ak/sqrt(D)) @ av @ wo`, **sumada al residuo antes del mixer**.

**(6) Se contesta.** El logit del ultimo token de la pregunta. La respuesta es **un solo token**, un
valor o `NOSE`.

**Tres cosas de este camino deciden todo lo que el proyecto midio en cinco semanas:**

- la lectura entra **en el bloque 0**, donde `h` todavia es la embedding pura, y por eso puede haber
  ceros exactos;
- la query se forma con una **conv de alcance corto**, y por eso hay una ventana;
- el softmax de lectura **suma 1 siempre**, y por eso la ausencia no tiene donde vivir y hubo que
  inventarle cuatro interfaces distintas.

---

## PARTE 3 · POR QUE · cada decision, trazada a la medicion que la puso ahi

Esto es lo que hace que el sistema no sea arbitrario. Ninguna de estas piezas se eligio por gusto.

| decision | la midio | y dijo |
|---|---|---|
| inyectar en el **bloque 0** y no al final | E-I2 (brazo interno) | temprano 0,7275 contra tardio 0,3827 |
| **sello de orden** co-entrenado en la clave | E-I3, 13-ago | 0,4570 → 0,9956. Sin el, el modelo no compara turnos |
| **barajar** el archivo | diseño | que la posicion en el tensor no codifique el rol |
| **respuesta de un token** | 12-ago | 10 de 11 «abstenciones» eran el parser, no el modelo |
| `T_SES=96`, `E_MAX=10` | 14-ago | con 40 y 4 se truncaba el 34 % y la meseta de 0,6707 **era el padding** |
| **24 relaciones** (idioma v3) | 20-23 ago | con 6, el 72,4 % de los episodios tenia colision de clave, y ahi el error saltaba de 0,01 a 0,5 |
| query **lateral** (`lat`) y no `post` | 22-ago | `post` mueve la inyeccion y rompe el modelo (0,97 → 0,39) |
| `convq` **propia** (`lat2`) | 24-ago | `lat` comparte la conv con el mixer y **cobra el marcador de orden** |
| `convq` arranca en `[1,0,0]` | diseño | asi `lat2` **contiene a `pre`** y no puede ser peor |
| **kernel 5** | 1-sep | con kernel 3 la relacion cae afuera de la ventana: sensibilidad **0,000000 exacto** |
| **`F < M`** en la recompensa | 29-ago | con F=1,5 nunca convenia callarse. Error de derivacion propio |
| medir **exactitud global**, no `nose` | 28-ago | `nose` le da 1,0000 a un modelo que contesta NOSE a todo |

**La lectura que sale de la tabla, y vale mas que cualquier fila:** de las doce decisiones, **nueve
salieron de un fallo medido, no de una idea previa.** El sistema es la sedimentacion de sus propios
errores. Esa es tambien la razon por la que las piezas no son sueltas y por la que romper una
—cambiar `E_MAX`, mover la inyeccion— rompe mediciones que ya estan publicadas.

---

## PARTE 4 · QUE TIENE QUE HACER · el objetivo, traducido a requisitos

El objetivo, en las palabras de Maxi:

> «que un LLM no olvide nunca **lo que le dije**»

y la vara final, del 13-ago:

> «un modelo de cero con esto incorporado en su ADN [...] y que no olvide lo que lee o lo que
> hablamos»

Traducido a cosas que el sistema tiene que **poder hacer**, y cada una con su estado real hoy:

| # | requisito | estado |
|---|---|---|
| **R1** | escribir lo dicho, **incrementalmente**, sin recomputar la conversacion entera | ❌ `escribir()` recomputa TODO el episodio en cada forward |
| **R2** | que lo escrito **sobreviva al cierre del proceso**, no solo al reseteo del estado | ❌ el archivo es un tensor intermedio, no existe fuera del forward |
| **R3** | que el archivo **crezca** sin retocar un peso | ⚠️ parcial: `kw/vw/qr/wo` son (D,D) y no dependen de N, **pero `ord` tiene 64 filas** |
| **R4** | que la consulta **vea la pregunta entera** | ✅ resuelto (kernel 5), y medido con `attn` |
| **R5** | saber **cual version rige** | ✅ hasta el turno 63. ❌ desde el 64 |
| **R6** | saber **decir que no sabe** | ✅ 0,988-0,993 en este banco |
| **R7** | que la lectura **no se diluya** al crecer el archivo | ❓ **nunca se midio** — se mide hoy |
| **R8** | **texto natural**, no 242 tokens | ❌ |
| **R9** | responder **texto**, no un token | ❌ |
| **R10** | decidir **que guardar** | ❌ VIGIA-03, sin correr |

---

## PARTE 5 · LA RESTA · lo que falta, y en que orden

**Lo primero, y es un hallazgo de hoy: R5 se rompe en silencio.**
`ord` tiene 64 filas y la indexacion de JAX **clampea sin avisar**. Verificado hoy: los turnos 63,
64, 65 y 200 reciben **todos el sello del turno 63**. O sea que pasado el turno 64 el mecanismo que
resuelve el conflicto de versiones —el que subio de 0,4570 a 0,9956 y tiene DOI— colapsa a una
constante, y **el modelo no tiene forma de saberlo**. Hoy no se ve porque el banco archiva 40
entradas como maximo. Con memoria larga se ve siempre.

**Lo segundo, y es la premisa que ordena todo lo demas: el banco nunca probo un archivo grande.**
Un episodio archiva a lo sumo `4 sesiones × 10 enunciados = 40` entradas. Todo lo que el proyecto
sabe —el sello, la ventana, la abstencion, el atractor— esta medido con un archivo de a lo sumo 40
entradas. El objetivo pide un archivo que crece conversacion tras conversacion. **La distancia entre
las dos cosas no es un detalle de escala, es la pregunta entera.**

Y de ahi el orden, que es por **cual bloquea al siguiente** y no por cual es mas lindo:

1. **Medir la dilucion** (`PREREG_DILUCION.md`, SHA `f4d91c12`). Cuesta minutos de CPU y decide si
   el filtrado previo merece una T4 o no. **Corriendo hoy.**
2. **Arreglar `ord`** para que extrapole, o al menos que **falle ruidosamente** en vez de clampear.
   Es una guarda de una linea y evita que el proximo experimento de memoria larga mida basura.
3. **Exponer `attn` en `--donde`.** El modelo ya lo tiene y el CLI **no lo ofrece**
   (`choices=("pre","post","lat","lat2")`), asi que hoy el acceso global se puede **medir** pero no
   **entrenar**. Es una linea, mas la guarda de identidad.
4. **Escritura incremental y persistente** (R1, R2). Es el salto de verdad, y es lo que convierte
   «archivo» en «memoria».
5. Recien despues, R8 y R9: texto natural y respuesta generada.

---

## PARTE 6 · QUE SE QUEMA EN T4, Y QUE NO

El pedido era evitar pruebas sin sentido. Con lo de arriba, esto es lo que queda:

**NO va a T4 todavia:**
- `PREREG_FILTRADO_PREVIO` (`3b7032b0`) — es la respuesta correcta a la pregunta correcta, **pero su
  premisa (que la dilucion rompe la precision) no estaba medida.** Sale de la fila hasta que la curva
  de hoy la confirme o la tire. Si la tira, se ahorra la campaña entera.
- `PREREG_MAGNITUD_Q` (`44c550e2`) — la constante `q` es un problema real y viejo, pero es de
  **abstencion**, y la abstencion ya esta en 0,99 con kernel 5. Es optimizar una pieza que no
  bloquea a ninguna otra.

**SI va a T4, y en este orden:**
1. **Entrenar con `attn`**, una vez expuesto. Ayer se midio que la atencion completa **mata el corte**;
   lo que no se sabe es si un modelo **entrenado** asi es mejor, igual o peor. Es la unica pregunta
   abierta donde ya existe el codigo y falta el gasto.
2. **Entrenar con archivo grande**, con el `ord` arreglado. Es el primer experimento del proyecto que
   ataca el objetivo de frente en vez de por un flanco.

**El criterio, dicho corto:** una campaña se lanza cuando lo que devuelve cambia lo que se hace
despues. Las dos preregistradas devuelven un numero que **no cambia el orden de nada**; las dos de
arriba deciden si el archivo sirve para lo que Maxi quiere que sirva.

---

## CIERRE DEL DIA · lo que la medicion cambio de este documento

`INFORME_DILUCION_20260905.md`, prereg `f4d91c12`. R7 pasa de **❓ nunca se midio** a **medido**, y
el resultado **reordena la Parte 6**.

- **R7 queda contestado, y al reves de como se venia suponiendo.** El softmax **no se diluye por
  numero**: con 3280 competidores de ruido la entrada correcta sigue ganando el 78,5 % de las veces.
  Lo que rompe la busqueda es el **contenido**, en dos capas —interferencia (RECUP 0,46) y colision
  (RECUP 0,012)—. La dilucion pura existe pero es **del valor leido**, no del ranking.
- **Aparece R11, que no estaba en la lista:** *descartar lo que ya no viene al caso.* Medido hoy:
  marcar el archivo largo como anterior **no cambia nada** (0,0059 contra 0,0039). El sello ordena
  versiones de un mismo hecho y **no** filtra por antiguedad.
- **La Parte 6 cambia.** `PREREG_FILTRADO_PREVIO` sale de la lista de espera y pasa a **enmendar y
  correr**, porque su hipotesis sobrevivio y ahora tiene mecanismo propio; lo que hay que reescribir
  es F-2 y el puntaje del filtro. Y lo primero en T4 sigue siendo **entrenar con archivo largo**,
  ahora con una razon medida en vez de una intuicion: hoy sabemos **cuanto** se rompe y **por que**.

**Lo que el dia deja como metodo, y es lo que Maxi pidio esta mañana:** desarmar el banco antes de
gastar la GPU encontro, en una mañana de CPU, que **la premisa de una de las dos campañas listas para
lanzar era falsa**. La campaña no era mala; la pregunta que decia responder no era la que el banco
tenia. Eso no se veia desde adentro de la campaña, se vio al mirar el sistema entero contra el
objetivo.
