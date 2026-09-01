# Alternativas laterales, escritas contra los hechos duros del 1-sep

No son continuaciones de lo que está corriendo: son ángulos distintos al mismo objetivo —que el modelo
diga «eso no lo tengo» en vez de inventar—. Cada una dice **qué hecho medido la motiva** y **qué la
falsaría**, porque una idea sin eso no se puede priorizar.

## Los hechos con los que hay que razonar

1. La búsqueda **no ve la relación** (ventana de la conv); la señal de ausencia vive **aguas abajo**
   (AUC 0,48 en la búsqueda contra 0,7003 en el estado final).
2. **Recuperar mejor compra detección** (+0,535 de AUC por punto de RECUP) **y no la agota**: con
   RECUP 1,0000 exacto el techo se queda en 0,93-0,96.
3. El **atractor mudo es absorbente**: 0 de 6 se reparan. Hay que **prevenir**.
4. El desenlace **se decide en el paso ~2500** (predictor perfecto sobre 76 unidades).
5. Decir NOSE es **barato de emitir y caro de aprender**: su logit recibe 389-549× menos gradiente en
   las unidades mudas.
6. `blanco=error` **entrenado** llega a techo 1,0000, y 2 de 9 unidades a exactitud global 1,0000.

---

## A · Introducir la ausencia DESPUÉS, y ahora se sabe por qué (barata, lista para correr)

El criterio operativo del 15-ago —«introducir `NOSE` sólo cuando `vigente` supere la tasa de preguntas
sin respuesta»— se probó una vez, **en el punto equivocado** (con `vigente`=0,13, donde abstenerse
seguía siendo lo mejor) y colapsó. **Los hechos 3 y 4 lo reviven con fundamento nuevo:** si el
desenlace se decide a 2500 pasos y después no se repara, entonces **el único momento que importa es
el arranque**, y en el arranque el modelo no recupera nada — o sea que `blanco=error` le está pidiendo
«¿te vas a equivocar?» cuando la respuesta honesta es «siempre», y la constante «callate» es el óptimo
real de lo que se le pide.

**Intervención:** `p_nose`=0 hasta que `vigente` cruce un umbral, y recién ahí introducir las preguntas
sin respuesta. **Falsable:** la fracción de semillas que termina útil tiene que subir sobre el 2 de 9
del control. **Qué la mata:** que las unidades caigan igual, o que al introducir la ausencia tarde el
modelo ya no la aprenda.

## B · Que la ausencia sea contenido, no falta de contenido

Hoy «no está» es la **ausencia de una entrada**, y por eso no tiene firma: no hay nada que leer. La
vuelta: **un segundo archivo de consultas fallidas**. Cuando se pregunta algo que no está, se escribe
esa consulta con marca de ausencia. La segunda vez, «no lo tengo» es un **hit**, no un miss.
**Por qué puede funcionar:** convierte la detección de ausencia en un problema de **recuperación**, que
es lo único que este modelo hace bien (RECUP 1,0000 en varias familias). **Qué la mata:** que no
generalice a consultas nuevas — sólo ayudaría con las repetidas. Es una prueba barata y decide rápido.

## C · Presupuesto de escritura: saber que no lo guardaste

Hoy el archivo se escribe **siempre y entero**. Si el modelo tuviera que **elegir qué escribir** con
cupo limitado, lo que no escribió es algo sobre lo que **tiene un registro propio de no tenerlo**. La
ausencia deja de ser «no lo encuentro» y pasa a ser «sé que no lo guardé». Conecta con la *eviction
sorpresa-gated* de [[vigia03-capacity-scheduling]], que está diseñada y sin correr. **Qué la mata:**
que el cupo degrade la recuperación más de lo que compra en detección — medible directo con la
pendiente del hecho 2.

## D · La verificación como PASO, no como clase (la más alineada con el objetivo)

Hoy `NOSE` es una **clase a predecir**. La alternativa: un paso obligatorio que compara **lo pedido
contra lo recuperado**, y cuya salida *es* la decisión. «No sé» deja de ser una etiqueta y pasa a ser
**el resultado de un cómputo**, que es exactamente la condición innegociable de
[[plan-modelo-que-sabe-que-no-sabe]]: la corrección adentro, no al lado.
**Lo que lo hace viable hoy y no antes:** el hecho 1 dice que la señal se construye aguas abajo, y el
hecho 6 que entrenar la detección llega a 1,0000. Con el kernel 5, la query contendría entidad **y**
relación, así que la comparación tiene las dos componentes para verificar.
**Qué lo mata:** que el paso de verificación aprenda a devolver una constante — el mismo colapso al
prior que ya mató a la cabeza en 4 de 9 unidades. Se ataja con el control que ya existe (`ranking`).

## E · Cortar el cómputo, no vetarlo (literal, la frase de Maxi)

*«que avise cuando no encuentra la información ANTES de dar una información equivocada»*. Hoy
`predecir_cabeza` computa el valor **y** la cabeza, y después vetea con `where(a>0, NOSE, argmax)`.
Las dos rutas se calculan siempre. Si la cabeza **cortara** el cómputo del valor, el gradiente
cambiaría de forma: el valor dejaría de recibir señal en las muestras abstenidas, y la cabeza pasaría
a ser una compuerta real y no un filtro de salida. **Qué lo mata:** que sin gradiente en las
abstenidas el modelo se refugie en callarse — es el atractor mudo otra vez, y habría que combinarlo
con A.

## F · Dos pasadas con latencias distintas (la pieza original que nunca se construyó)

El plan del 9-ago pedía **tres piezas**: dos vías con velocidades distintas, un disparador y un
monitor. **Ninguna se construyó**; lo que hubo fueron sondas post-hoc. Y hoy la versión más barata de
eso —dos búsquedas con ruido— ya da **72 % de precisión en ausencia con el modelo confiado**, que es
señal en el régimen exacto donde la confianza es ciega. La versión entrenada (dos queries `qr1`/`qr2`
con el desacuerdo en la pérdida) sigue sin probarse, y el precedente del proyecto dice que entrenar
lo que post-hoc daba 0,65 lo llevó a **1,0000**.

---

## Orden sugerido

**A primero** (barata, lista, y ataca el único momento que importa según el hecho 4). **D después**,
que es la más alineada con el objetivo y la que el kernel 5 habilita. **B y C** son pruebas rápidas
que pueden morir en una tarde. **E** sólo junto con A. **F** es la más ambiciosa y la que más se
parece al plan original.
