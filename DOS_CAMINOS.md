# Los dos caminos

**2026-08-10.** Documento de encuadre. No es un pre-registro y no fija predicciones: define **qué se
está investigando**, para que el programa entero pueda leerse contra una vara.

---

## El objetivo

> **«Que un LLM no olvide nunca lo que le dije.»** — Maximiliano Speranza

El énfasis está en **«lo que le dije»**. Ahí se abre la bifurcación, y las dos ramas son problemas
distintos con literatura distinta, métricas distintas y competencia distinta.

---

## Camino A — «que no olvide lo que LEYÓ»

**Qué es.** Que el modelo retenga y sepa usar material que se le presentó: documentos, contexto largo,
una base de conocimiento.

**Cómo lo ataca el mundo.** Ventanas de contexto cada vez más largas, RAG sobre corpus, caches KV,
atención lineal e híbrida, compresión de contexto. Es un campo enorme, con recursos enormes.

**Qué tenemos acá.** Todo Ligamento vive en este camino:

| | qué aporta |
|---|---|
| **E1** | El techo de capacidad es real, y la hibridación con softmax lo restituye (+0,0792 IC95 [+0,0747, +0,0838]). E-004: **una sola** cabeza softmax alcanza. |
| **E2** | El contexto que llega tarde no sirve, y la maquinaria de cross-attention no hace falta en este régimen. |
| **E2-b** | **El contexto es precondición del cómputo, no corrección.** Un acceso en el primer bloque: 0,9998. El mismo acceso en el último: 0,4990. |
| **E3** | La compuerta de cómputo se estrangula y **impide usar contexto que sí está llegando** (costo +0,4860 vs +0,0896). Converge con E2-b por otra vía. |
| **E4 / E-006** | Dónde poner la frontera entre especializar y compartir: **unificar temprano, especializar tarde** (replicado, 3 de 3 predicciones). |

**Su límite, dicho de frente.** Todo esto ocurre **dentro de una corrida**. Cuando la sesión termina,
no queda nada. Y es el camino donde competimos contra laboratorios con presupuestos que no tenemos.

**Para qué sirve entonces.** Es **infraestructura**: dice dónde hay que poner el mecanismo de memoria
y cuánto cómputo hace falta detrás del acceso para que sirva de algo. E2-b y E3 dan el primer número
de esa serie, y son medibles porque el modelo es chico y se puede mirar entero — que es justamente lo
que no se puede hacer en un modelo de frontera.

---

## Camino B — «que no olvide lo que LE DIJE»

**Qué es.** Memoria **episódica de la interacción con una persona**, que **sobrevive al cierre de la
sesión**, y que soporta que la persona **corrija** algo dicho antes y el modelo sepa cuál versión rige.

**Por qué es otro problema, no una variante del A.** Tres diferencias estructurales:

1. **Persiste entre sesiones.** No nace y muere en el forward. Crece sin cota.
2. **Es de una persona, no de un corpus.** Lo que importa no es cuánta información entra, sino que
   *esta* información — la que me dijo este usuario — siga estando dentro de un año.
3. **Tiene versiones.** «El director es Ana», y después «no, es Beto». Un corpus no se contradice a sí
   mismo; una conversación sí, todo el tiempo.

**La tarea canónica del objetivo.** «El director de X es Ana» → «es Beto» → «¿quién dirige X?» es
literalmente *lo que le dije, y después lo corregí*. Por eso la tarea de hechos versionados no es un
experimento lateral: es **el** experimento.

**Qué tenemos acá.**

- **La gemación** como idea propia: al revisar un recuerdo no se sobrescribe, se deposita la versión
  nueva **cerca**, porque la cercanía codifica la correlación. Linaje: la memoria dispersa de Kanerva
  y el DNC; el aporte propio es que **la geometría reemplaza la matriz de enlaces** — O(1) por
  escritura en vez de O(N²).
- **R1–R13 medidos**, con un dictamen de diseño claro: **geometría para agrupar, metadato para
  ordenar**; el eje va **por recuerdo**; la deriva del encoder es el obstáculo real pero **sobre un
  modelo preentrenado que se afina, la memoria persistente es viable**; y la fusión de mediciones
  independientes es lo que hace escalar el mecanismo.
- **Los cuatro chequeos de admisión de encoders**, que salieron de dos noches perdidas.

**Estado al 2026-08-10.** Primera tanda con datos válidos: P2 (control) **OK**, P3 **cumple**, y
**P1 no confirma** — con las dos condiciones en el techo (`duplicados` 0,9988 · `gemacion` 0,9928),
lo que hace de este un negativo **sin potencia**, no una equivalencia. P4 —la predicción con margen
para discriminar— se está corriendo con textos reales tras invalidar un primer intento por error de
implementación (D2).

---

## Por qué el camino B es el que decide

Contra la vara del objetivo:

- El camino **A** no puede cumplirlo ni en principio: todo lo que mide se evapora al cerrar la sesión.
- El camino **B** es el único que ataca la persistencia de frente.

De ahí la regla operativa: **cuando haya que elegir entre cerrar algo de Ligamento y avanzar la
gemación, gana la gemación.** No porque Ligamento esté mal —está bien hecho y anclado— sino porque
mide otra cosa.

**Y donde los dos caminos se tocan** hay una pregunta que no está contestada por nadie y que sale de
juntar los dos: **E2-b dice que el contexto tiene que llegar *antes* del cómputo para servir.** Si eso
vale también para un archivo persistente, entonces la respuesta recuperada del índice **no puede
inyectarse al final** — que es exactamente lo que hace todo pipeline RAG. Esa es la hipótesis que sólo
se puede formular teniendo los dos caminos medidos con el mismo instrumento.

---

## Lo que falta en cada uno

**Camino A** — redactar E3 + E4 (tres hallazgos publicables), y destrabar TMLR.

**Camino B** — cerrar P4; decidir si el negativo de P1 se re-testea en un régimen con potencia (más
revisiones, índice más poblado, o k más chico); y el salto grande: pasar del índice no paramétrico a
uno **co-entrenado dentro de la red**, que es donde están los dos obstáculos reales (el gradiente no
fluye por la selección top-k, y el *stale index*).
