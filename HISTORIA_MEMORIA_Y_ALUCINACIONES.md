# Memoria persistente y alucinaciones en modelos de lenguaje

## Historia completa de mi investigación, del 7 de agosto al 8 de septiembre de 2026

**Maximiliano Speranza** · investigador independiente · ORCID 0009-0005-0413-8554

Éste es mi documento de síntesis. Reconstruye qué hice, en qué orden, y qué quedó establecido.
Todo lo que figura acá está medido y tiene informe, pre-registro con hash y commit en este repo,
salvo donde digo explícitamente que es lectura, hipótesis o dato post-hoc.

---

## 1. Lo que busco, y es lo que ordena todo lo demás

Lo formulé así el 10 de agosto de 2026, y lo dejo escrito con mis palabras porque la formulación
exacta importa.

> que un LLM no olvide nunca **lo que le dije**

No es «lo que leyó». Esa distinción me cambió el encuadre entero. «Lo que leyó» lleva a contexto
largo, RAG sobre documentos y ventanas grandes, que es lo que hace medio campo. «Lo que le dije»
lleva a memoria episódica de la interacción con una persona, que sobrevive al cierre de la sesión, e
incluye que el usuario **corrija** algo dicho antes y que el modelo sepa cuál versión rige.

De ahí sale mi tarea canónica, los **hechos versionados**. «El director de X es Ana», y después «no,
es Beto». Es literalmente lo que le dije y después lo corregí.

Y de ahí sale mi vara final, que fijé el 13 de agosto.

> yo voy a terminar de creer que esto funciona cuando creemos un modelo de cero con esto incorporado
> en su ADN, no algo que está adosado al LLM, y lo probemos y no olvide lo que lee o lo que hablamos

Tres exigencias. Modelo entrenado desde cero, con el índice adentro de la arquitectura y
co-entrenado, y probado sobre lo que lee y sobre lo que se habló.

---

## 2. Cómo arranqué, y el problema técnico del que salgo

El 7 de agosto planteé la pregunta inicial. ¿Qué pasa si acoplo la regla delta a un RAG con
embeddings para que no se pierdan de la memoria las cosas viejas importantes?

El problema al que apunto ya lo tenía medido en mi proyecto Ligamento. Un estado recurrente de tamaño
fijo tiene techo de capacidad, y la regla delta **sobrescribe**, así que al llenarse lo viejo se pisa.
La hibridación con softmax restituye el techo pero paga costo cuadrático. Mi idea era la vía
alternativa, en vez de agrandar el estado, archivar afuera lo que va a ser desalojado.

El 8 de agosto hice la precisión que definió el diseño experimental de todo el mes. Lo que quiero
**no** es un pipeline RAG externo enchufado a un modelo. Deltas y retrieval tienen que vivir dentro
de la misma arquitectura, no ser artefactos separados que se unen para un fin. El índice como parte
del modelo, no un retriever congelado que concatena texto al contexto.

Ese mismo día hice el barrido de literatura y apareció un **scoop parcial**. Cuatro papers de 2026
(HOLA, HAM, Tensor Cache, Memory Caching) ocupan la idea genérica. En la segunda pasada rescaté la
novedad, y es una diferencia real y no cosmética. Los cuatro competidores son **intra-secuencia**, su
memoria nace y muere en el forward. Un RAG es persistente **entre** secuencias, sesiones y
documentos, crece sin cota y se consulta por vecinos más cercanos. Ese cruce, índice persistente entre
secuencias más regla delta, co-entrenado, no está hecho.

Los dos obstáculos que probablemente explican por qué nadie lo hizo, y los escribí antes de empezar.
Primero, el gradiente no fluye por la selección de los k mejores. Segundo, el índice queda viejo,
porque los embeddings cambian durante el entrenamiento mientras el índice está congelado.

Advertencia que medí y no especulé, también del 8 de agosto. A la escala de mi banco el argumento
económico «evita el costo cuadrático» **no se sostiene**, así que el experimento sólo puede medir
capacidad y mecanismo, jamás eficiencia.

---

## 3. Primera etapa. La gemación, y por qué la cerré

Es idea mía, del 8 de agosto. Al revisar un recuerdo no se sobrescribe, se deposita una versión nueva
en un lugar **cercano**, porque la cercanía codifica la correlación. Linaje honorable y no scoop, la
memoria distribuida esparcida de Kanerva y el DNC, que resuelve el enlace temporal con una matriz de
N por N y por eso no escaló. Mi aporte propio, **la geometría reemplaza la matriz de enlaces**, con
costo constante por escritura.

### Lo que medí en geometría pura, sin red

- **R1.** La geometría **agrupa perfecto pero no ordena**. Identifica el clúster con 1,000 en toda la
  grilla y recupera la versión **más vieja**.
- **R2.** Con un eje temporal la recencia se vuelve legible, y el óptimo es intermedio.
- **R4.** El eje **por recuerdo** es estructuralmente superior al global. Dictamen de diseño que me
  sobrevivió todo el mes, **geometría para agrupar, metadato para ordenar**.
- **R5.** La memoria persistente funciona mientras el coseno entre el marco de hoy y el de escritura
  esté por arriba de 0,7. Y la deriva real que medí entrenando desde cero consume ese presupuesto en
  **decenas de pasos** mientras el entrenamiento dura miles. Aprender **es** mover las coordenadas.
- **R6.** El preentrenado me da vuelta el veredicto. Sobre un modelo ya entrenado que se afina, el
  coseno queda en 0,882 contra 0,207 desde cero. La deriva catastrófica es un fenómeno del
  **aprendizaje inicial**, no de la vida útil del modelo.
- **R7 y R8.** Con claves reales el panorama es mucho más duro (acierto directo 0,148, no 1,000). Mi
  hipótesis del barrio funciona, la métrica correcta es el **rango** y no el coseno, y la **fusión de
  las cuatro cabezas** fija la posición aunque ninguna alcance sola.
- **R10 y R11.** El colapso al crecer el índice era artefacto de mi banco (dimensión 16 por cabeza).
  En un LLM real, con espacio muy anisotrópico, escala perfecto con una sola medición. **La capacidad
  la da la dimensión ambiente, no la de la señal.**

### Los tres negativos pre-registrados que cerraron la línea

Entre el 10 y el 11 de agosto, con pre-registro hasheado antes de mirar el dato.

1. La gemación con paso fijo **se aleja acumulativamente** del ancla.
2. Con la órbita, la reparación funciona y **el mecanismo pierde igual**. Si la deriva hubiera sido el
   problema, acotarla alcanzaba. El problema es el desplazamiento en sí, porque el vector del texto ya
   está óptimamente colocado y moverlo sólo puede alejarlo.
3. En el régimen elíptico, que diseñé a propósito para favorecerla, el umbral de utilidad queda en
   0,45 de tasa de error de la co-referencia, o sea nunca en un sistema real.

**El hallazgo independiente más útil de esa etapa**, y no era lo que iba a medir. Archivar turnos
conversacionales **sin resolver las co-referencias** pierde el **100 %** de las correcciones, con cero
exacto en diez semillas de diez, y **en silencio**.

Manuscrito publicado, DOI `10.21203/rs.3.rs-10669947/v1`.

**Límite que declaro.** Todo eso vale para índice no paramétrico sobre encoder congelado. No dice nada
sobre un índice co-entrenado dentro de la red, donde el argumento «el encoder ya lo puso donde
corresponde» deja de aplicar por construcción. Ése siguió siendo mi hueco.

---

## 4. Segunda etapa. El brazo interno, índice co-entrenado dentro de la red

Lo arranqué el 12 de agosto. Tarea cross-secuencia, una sesión escribe, otra revisa, otra consulta,
con el estado **reseteado** entre sesiones, así que el piso es el azar por construcción.

- **E-I0**, control bloqueante, pasa. Las dos mitades, y el control podía fallar.
- **E-I2**, mi pregunta central, y la respuesta es que sí. El archivo guarda vectores producidos por
  el propio modelo, así que acertar exige alinear una consulta formada leyendo una pregunta con una
  clave formada leyendo una afirmación, en contextos distintos. Resultado 0,7275. **El gradiente que
  no fluye por la selección de los k mejores no impide aprender a consultar.**
- **El hallazgo de esa jornada.** Con **una** versión archivada, 0,9974. Con **dos** compitiendo,
  0,4576, que es el azar entre la vieja y la nueva. Encuentra el hecho y **no sabe cuál versión rige**.
  Es exactamente el modo de falla de R1 y R4, reproducido por un mecanismo que no tiene nada en común
  con aquél.

### El sello de orden, que es mi primer resultado grande

**E-I3, 13 de agosto.** Le agrego un sello de orden co-entrenado a la clave archivada.

| condición | resultado |
|---|---|
| sin sello | 0,4570 |
| **con sello** | **0,9956** |
| sello barajado (control) | 0,4768 |

Lo que hace válido el resultado es la celda que **no** ganó. El barajado tiene exactamente los mismos
parámetros y el sello sin relación con el turno real, así que descarta que la ganancia sea capacidad
extra del lector. Quince corridas ordenadas perfecto por condición, sin un solo solape entre semillas.

Simetría del número. El conflicto costaba 0,5398 y el sello devuelve 0,5385. Repone lo que la
geometría perdía y nada más.

La serie posterior acota el cómo.

- **E-I3b.** Preferir lo último y usar el orden son **dos capacidades distintas** y se aprenden en
  momentos distintos. La vigente satura a los 3000 pasos, la anterior recién despega a 4000. Un
  negativo intermedio era impaciencia mía.
- **E-I3c.** Con la fuga tapada el sello aporta el orden, y un **sello mentiroso es peor que ninguno**.
  El lector se apoya en el sello, así que un marcador de tiempo corrupto no es información faltante,
  es información dañina.
- **E-I3d.** Sí compara turnos, pero sólo sabe buscar el máximo. La frase que lo resume, **el lector
  aprende a leer el reloj para saber qué hora es ahora, no para reconstruir la secuencia de lo que
  pasó**.
- **E-I4, E-I4b y E-I4c.** El archivo envejece y el daño es gradual, pero tres formas distintas de
  forzar la deriva me quedaron siempre por arriba del umbral de 0,70. La lectura que ordena las tres,
  R6 lo predijo desde afuera, un modelo convergido no mueve su marco.

---

## 5. Tercera etapa. El micro-LM, mi modelo desde cero

Lo arranqué el 13 de agosto, y es mi vara. Idioma cerrado pero legible de 242 tokens, 863.730
parámetros, 3,5 MB.

Resultado inicial del nivel más difícil, paráfrasis más corrección elíptica más sesiones separadas con
el estado reseteado. Vigente 0,9881 y anterior 0,9922. O sea, **contesta bien sobre algo dicho en una
sesión anterior cuyo estado ya no existe, incluso corregido después**.

Y ahí me apareció el primer artefacto grande. Los niveles fáciles daban peor que el difícil, y me lo
delató justamente eso, cuando el orden de dificultad sale al revés el problema es del instrumento. Era
**relleno de padding** truncando el 34 % de los enunciados. Corregido, los niveles 1 y 2 saturan en
1,0000.

### Lo que encontré el 15 de agosto, y es de método

**Todas mis corridas hasta ese día usaron probabilidad de pregunta sin respuesta igual a cero.** O
sea, la abstención nunca había sido una opción que el modelo pudiera tomar, y el «abstención 0,0000»
que venía reportando no era un resultado, era una métrica que no existía. Ocho días de campañas
midiendo la mitad de mi objetivo.

Lo puse como prioridad absoluta. El error silencioso no cuesta una respuesta, cuesta la confianza en
todas las demás.

### El SER, y mi primera radiografía de la alucinación

Lo medí el 15 de agosto sobre checkpoints ya entrenados.

- El **versionado está resuelto**, el error de versión queda por debajo de 0,0078 en los cuatro
  niveles.
- Lo que rompe es **identificar la entidad**, con 0,2227 y 0,2324 en los niveles difíciles.
- **El error fuera de dominio es 0,0000 en los cuatro niveles.** El modelo **nunca inventa contenido**.
  Toda respuesta errada es un valor **real** del archivo puesto en la entidad equivocada.

Eso me reformula el problema. La alucinación acá no es fabricar un dato, es **atribuir mal uno
verdadero**, que es peor de detectar, porque preguntar «¿este dato existe?» la da por buena.

---

## 6. La campaña de la abstención, y el trípode

### Mitigación sin reentrenar

El modelo **sabe cuándo se equivoca**, con área bajo la curva de 0,86 separando aciertos de errores
por confianza de salida. Con umbral calibrado en una mitad y medido en la otra, contestar el 78 % y
callarse el 22 % borra el **53,6 %** de los errores silenciosos sin tocar el modelo. Pero **se quiebra
en el caso central**, cuando la pregunta no tiene respuesta el área cae a 0,74 y la ventaja sobre el
azar de 1,68 veces a 1,16. El modelo confía casi igual inventando que acertando.

### Dónde vive la abstención, el trípode

Tres formas de emitir el «no sé», pareadas desde el mismo checkpoint base.

| interfaz | resultado |
|---|---|
| `token`, el «no sé» como una palabra más del vocabulario | falla en las unidades difíciles |
| **`cabeza`**, salida binaria separada, 129 parámetros extra | **gana, 2,0 y 1,7 y 2,8 veces mejor a presupuesto igualado** |
| `escala`, sólo renormalizar el vector del token | falla, y refuta la explicación barata |
| `slot` nulo dentro del archivo | **convergió al prior**, cero señal |

Con **129 parámetros**, el 0,015 % del modelo, le puedo enseñar a callarse a un modelo que recupera
diez puntos peor. Ésa fue la respuesta a la pregunta que me bloqueaba la línea, **no** hace falta
entrenar hasta casi la perfección antes de introducir el «no sé».

Y el mecanismo, que es limpio. **No era la norma del vector, era que dos decisiones de naturaleza
distinta competían por la misma masa de probabilidad.** El slot convergió al prior exactamente,
0,4074 contra una tasa base de 0,4048, que es el óptimo de la pérdida para un predictor sin señal.

Paper `preprint/tripode/`, «Where Abstention Lives», diez páginas.

### Las cinco vías que cerré buscando la señal de ausencia

1. **El score del archivo.** 0,4984 y 0,5022, **el azar exacto**, con tres controles que descartan
   artefacto. La diferencia entre recuperar e inventar **no vive en la interfaz de memoria**.
2. **La mezcla de gaussianas sin etiquetas.** Peor que no hacer nada. La información está pero no en
   forma de valle, las poblaciones están separadas 1,2 desviaciones y una mezcla necesita 2 para tener
   dos modas.
3. **El monitor de desacuerdo interno.** Cero de ocho. Y la razón encaja con todo lo demás, el modelo
   **nunca inventa en el vacío**, se ancla en otra entrada real y queda igual de estable. El desacuerdo
   mide si la respuesta viene del archivo, no si viene de la entrada **correcta**.
4. **El empate de clave.** El positivo es real y tiene el mejor control de todo mi programa, pero con
   dos entradas empatadas el modelo acierta la mitad de las veces, así que el empate predice el
   **riesgo**, no el **error**.
5. **La clave discreta.** Con símbolos el evento «ninguna coincidencia» existe y es observable, y
   **sigue sin separar**. Hallazgo positivo no buscado, la clave de 4096 bits se reemplaza por **64
   bits** con recuperación 1,0000 exacta en tres semillas, compresión de 64 veces sin perder un punto.

**Las cinco fallan en el mismo punto, separan estados del modelo y no aciertos de errores.**

### El atractor mudo, y la abstención perfecta como estado degenerado

Con el blanco de entrenamiento «¿me voy a equivocar?», cuatro de nueve unidades terminaron abstiniéndose
del 100 %. Mecanismo identificado, es una **carrera entre dos relojes**. El blanco es autorreferencial,
al empezar el modelo se equivoca en todo, la etiqueta es la constante uno, y la cabeza aprende «me
equivoco siempre» **y tiene razón**. Dos puntos fijos autoconsistentes, el bueno y el degenerado.

Encontré el predictor, cero respuestas emitidas en el paso 2500, acertado en **40 de 40** corridas del
repo.

Y la frase que le pone precio a todo esto. Esas unidades convierten el conocimiento en decisión
**perfectamente** y son inútiles. Exactitud global clavada en **0,4065**, el piso trivial.

> **La abstención perfecta y el conocimiento nulo son el mismo estado.**

Probé cinco funciones de pérdida distintas (balance, ranking, recompensa, orden, subsidio al
silencio). Todas mueven el **valor** de la constante y ninguna la vuelve una función de la pregunta.
Mudo, locuaz y medio son la misma patología.

**Y el techo, que medí con cinco lectores distintos.** La ausencia es decodificable del estado con
área bajo la curva de **0,70** y ninguno pasa de 0,7003. El cuello no era la pérdida ni la interfaz,
**es que la información de ausencia en el estado sólo da 0,70**. Para saber que algo no está, primero
hay que buscarlo bien.

---

## 7. El giro. Lo que yo leía como alucinar era un error de indexación

**20 de agosto.** Mi experimento de ida y vuelta encontró la causa del error de identidad, y no era
la que yo suponía. Es **colisión de clave**. El modelo usa la **relación** para encontrar el hecho.

| condición | acierto | error de identidad |
|---|---|---|
| relación única en el episodio | 0,94 a 0,99 | 0,005 a 0,014 |
| **relación repetida** | 0,45 a 0,58 | **0,38 a 0,54, el azar entre las dos que empatan** |

Y la cuenta cierra hacia atrás. La probabilidad de colisión por el vocabulario chico multiplicada por
esa tasa reconstruye exactamente el 0,19 a 0,21 global que venía midiendo desde el 15 de agosto.

**23 de agosto.** La colisión resultó ser una propiedad del **vocabulario**, no del modelo. Con seis
relaciones y cuatro hechos, el 72,1 % de los episodios tiene dos hechos que comparten relación.

> Lo que yo leía como «el modelo alucina» estaba dominado por «dos entradas empatan y elige una».

**Esto me reordenó el programa entero.** Memoria persistente, alucinación y abstención colapsaron en
una sola pregunta, cómo se indexa y se consulta el archivo.

### Por qué el modelo no podía usar la entidad

**21 y 22 de agosto.** La consulta se formaba en el bloque cero, sobre la embedding cruda del token,
así que la query era **función pura del token de su posición**. Medido, no inferido, intervenir todo
el contexto mueve la query **0,00000000 exacto**.

> El modelo no puede formar una query conjunta de entidad por relación, consulta token por token y
> resuelve la conjunción aguas abajo.

Mi primer intento de arreglo fue mover la lectura después del mezclador. **Falla en todo**, y me deja
un hallazgo mayor. La ventana de inyección útil no es «temprano», es **antes del primer mezclador**.
Media capa más tarde el acierto cae de 0,97 a 0,39. Y con eso me queda un trade-off que parecía
estructural, una query conjunta necesita contexto ya computado y la lectura útil necesita entrar antes
de que el cómputo ocurra.

Segundo intento, el **camino lateral**, formar la query con una convolución corta sin mover el punto
de inyección. Es la tercera vía, y funciona.

**24 y 25 de agosto, resultado.** El error de identidad se va a **0,0000 en las tres semillas**, y la
bimodalidad entre semillas, que era lo que me impedía afirmar nada, **desaparece**. Con la convolución
propia (`lat2`) la versión anterior vuelve a 1,0000 en las tres.

Eso acota hacia atrás **todas** mis lecturas anteriores del error de identidad como alucinación.

---

## 8. La ley de la ventana, lo que encontré el 1 y 2 de septiembre

El diagnóstico es aritmético. En la pregunta de mi banco la **entidad** queda a distancia 1 de la
posición de lectura y la **relación** a distancia 3. La convolución que forma la query tenía kernel 3,
o sea alcance 2, así que **la relación cae un token afuera de la ventana en el 100 % de las consultas**.

No es que el modelo la ignore. **No la puede ver.** Medido, sensibilidad de la búsqueda a la relación
igual a **0,0000 exacto**.

El arreglo es una línea, kernel 5, que cuesta 1.280 parámetros sobre 865.395.

| métrica | kernel 3 | kernel 5 |
|---|---|---|
| caso difícil, falta la relación | 0,5850 · 0,6090 · 0,7349 | **0,9931 · 1,0000 · 1,0000** |
| exactitud global (piso trivial 0,4065) | 0,91 a 0,93 | **0,988 a 0,993** |

Tres semillas **sin un solo solape**.

**La ley, y es exacta.** La sensibilidad de la búsqueda a un token de la consulta es cero exacto en
cuanto ese token pasa el alcance, y el escalón **se mueve con el kernel**. Sesenta celdas de sesenta.

**El cruce que lo generaliza.** Reordenando la pregunta, la ceguera cambia de componente en el mismo
modelo. Lo que decide el fallo es **dónde está escrito** un componente, no qué componente es.

**Validación externa**, sobre `mamba-130m-hf` real, sin entrenar. Cambiar un token a distancia 5 a 8
deja la salida de la convolución en **0,0 exacto**, ochenta celdas de ochenta. Lo que demuestro es la
**disociación**, el estado ve toda la secuencia y la query que lo lee ve una ventana.

**Por qué importa afuera.** El kernel 4 es el default de los modelos desplegados, Mamba-2 y Gated
DeltaNet en Qwen3-Next. Alcance 3. Cualquier parte de una pregunta a más de tres tokens del final no
entra en la query de esa capa.

**El diagnóstico accionable, y no hace falta entrenar nada.** Cambiar un token de la consulta y mirar
si la búsqueda se mueve. Si no se mueve, ese token es invisible para la búsqueda, y la confianza que
el modelo muestre sobre él no está ganada.

### Las tres precisiones que le siguieron

**3 de septiembre, sobre el modelo real.** En un modelo profundo la ventana **no bloquea, atenúa**.
Corte exacto en la capa cero, y desde la capa uno la recurrencia restituye la señal atenuada, a 1,077
por token en 24 capas y 1,028 en 48. Tres veces los parámetros y el doble de capas no cambian la tasa,
o sea es propiedad de la arquitectura.

> Donde no quedan capas con que pagar, la ventana pone un techo. Donde quedan, pone un peaje.

El efecto conductual **existe, es grande y es transitorio**, más de 0,14 en el paso 100 y cero desde
el paso 400.

**El resultado que me salió de dos fotos de una clase de probabilidad.** Sobre **33.585 preguntas
reales de cuatro corpus**, la probabilidad de que la distancia supere 2 da 0,9625 en SQuAD, 0,9978 en
Natural Questions (búsquedas reales de Google), 0,9943 en TriviaQA y **1,0000** en HotpotQA. Empeora
con la dificultad. La limitación principal del paper pasó a ser su resultado más aplicable.

**4 de septiembre, el control de acceso global.** Con atención causal completa el corte **desaparece**,
120 de 120 en las seis distancias, mientras la convolución corta da cero exacto más allá de su alcance.
Y una lección, la ventana **no es puramente arquitectónica**, existe sólo si el entrenamiento abrió
los taps.

La instrucción de diseño que sale. Si un modelo consulta una memoria desde una capa temprana, esa
capa necesita **acceso global**. Una convolución corta vuelve invisible parte de la consulta, el fallo
es **silencioso**, y se corrige poniendo atención completa ahí.

### La cuarta precisión, del 8 de septiembre, y es la que cambia el enunciado

El 8 de septiembre entrené por primera vez un modelo **con** acceso global desde el principio, y el
resultado me obligó a corregir cómo yo venía explicando la ley.

Mi predicción era que la atención completa iba a ganar por **pesar mejor** la relación. Escribí el
criterio antes de correr nada, la razón entre la sensibilidad a la posición de la relación y la de una
posición sin señal, y pedí que fuera al menos 1,5. **Dio 1,00, 1,01 y 0,94 en las tres semillas**,
exactamente lo mismo que dan cuatro controles que nunca entrenaron con acceso global. Mi predicción
quedó refutada por mi propio criterio.

Y sin embargo el modelo con atención completa **rinde mejor** que el de kernel 5. Como el
pre-registro me obligaba a buscar la causa alternativa antes de escribir nada, la busqué, y está
medida.

| | pico de sensibilidad | masa acumulada en 20 distancias | distancias con señal |
|---|---|---|---|
| kernel 5 | **0,0447** | 0,1432 | 4 de 20 |
| atención completa | 0,0110 | **0,2021** | 20 de 20 |

**La atención tiene un pico cuatro veces menor y una masa 1,41 veces mayor. Gana por área, no por
altura.** Mi razón de selectividad comparaba dos distancias y por eso no podía ver lo que había
cambiado, porque el kernel concentra mucho en cuatro posiciones y da cero exacto en el resto,
mientras la atención reparte poco en todas.

Lo que esto refina, y es lo que vale. **El corte del kernel 3 era binario.** Lo que arregla el acceso
global no es pesar mejor la relación, es que **nada quede en cero**. Es un requisito de
**cobertura**, no de foco. Enunciada así la ley es más fuerte y más fácil de aplicar, porque para
cumplirla no hace falta que la arquitectura aprenda a mirar el lugar correcto, alcanza con que no
tenga ningún lugar ciego.

---

## 9. El archivo largo, la última pared que encontré

**5 de septiembre.** Decidí desarmar lo que tenía antes de gastar otra GPU.

**La premisa que no había mirado nadie, yo incluido.** Mi banco nunca probó un archivo grande. Todo lo
que la línea sabía estaba medido con 40 entradas como techo, y mi objetivo pide lo contrario.

Exactitud contra tamaño del archivo, con el mejor checkpoint del momento.

| entradas | exactitud |
|---|---|
| 40 | **1,0000** |
| 160 | 0,3008 |
| 3280 | **0,0039** |

Con cuatro conversaciones guardadas cae **debajo del piso trivial**.

**Y el control me da vuelta la causa.** Con 3280 competidores, cambiando sólo de qué están hechos los
distractores, recuperación 0,7852 con ruido, 0,4590 con textos reales de entidades ajenas y 0,0117 con
textos reales.

> **El softmax no se diluye por número. Aguanta 3280 competidores sin contenido. Lo que rompe la
> búsqueda no es cuántos son, es qué dicen.**

Dos capas separables, **interferencia** y **colisión**. De dilución pura queda otra cosa, se ensucia
el **valor** y no el ranking, porque la lectura es un promedio ponderado.

### El colapso era transferencia, no un techo

La campaña de esa noche cerró con los cinco criterios. Entrenadas con archivo largo, las tratadas dan
**1,0000 · 0,9872 · 0,9933** con 161 entradas, contra controles entrenados en corto que dan
**0,0104 · 0,0333 · 0,0000**. Y casi no se paga en el régimen corto.

**Y el sello de orden aprende a descartar lo viejo.** Índice de masa con sello real contra barajado,
0,17 a 0,26 contra 0,78 a 0,81, brechas de más de 0,52, con recuperación 1,0000. A la mañana el mismo
control me daba 0,8886 contra 0,8885. Misma tabla, misma arquitectura, lo único que cambió es que vio
archivos largos durante el entrenamiento.

---

## 10. El sello de orden es una bandera, no un reloj

**6 de septiembre.** Ataqué primero el control de recencia genuina, porque **decide** si agrandar la
tabla de turnos sirve. Y la respuesta es que no.

Con el orden relativo **intacto** y el episodio corrido dos lugares, toda muestra cuya entrada correcta
quedó por arriba del umbral da recuperación **1,0000 exacto**, y las que cruzan por debajo caen a 0,50
a 0,57. Brecha de 0,43 a 0,50 en las tres semillas. Es un **escalón**, no una pendiente.

El modelo aprendió qué **filas** significan «esto es de la conversación en curso», no una relación de
antes y después.

**La contraparte de peso, la misma tarde.** El modelo no aprendió 64 marcas de turno, aprendió **una
marca de ajeno copiada en las 24 filas bajas**, con coseno 0,86 a 0,90 entre ellas, más del doble de
norma y apuntando en dirección opuesta. Es literalmente un bit, no un reloj. Los controles dan todo en
cero, así que **la partición se formó al ver archivos largos**.

**Y al descartar la alternativa apareció lo más útil.** La tabla de turnos sostiene **dos funciones
entrelazadas**, un **bit de pertenencia** (el escalón, correlación 0,86 a 0,88) y un **gradiente de
recencia** intra episodio (correlación negativa de 0,69), que es el sello de orden con DOI. Por eso el
tope de 64 filas no rompe una cosa, rompe las dos a la vez.

**El diseño que sale, y su predicción falsable.** Separar las dos funciones. Bit de pertenencia
calculado y no memorizado, y **sello relativo a la consulta**, que convierte el tope de 64 en una
**ventana de recencia** en vez de un techo del archivo.

Lo implementé el 6 a la noche con su compuerta de cuatro comprobaciones que podían fallar, todas
pasan. La campaña la corrí el 7 de septiembre, seis unidades completas en 2000 pasos.

### La medición definitiva, el 8 de septiembre, y el bit compra más de lo que yo esperaba

**La pregunta del sello se contesta en las tres unidades sin bit**, y cierra. Mover la frontera
manteniendo el orden relativo da recuperación **1,0000 exacto igual que el régimen real** en las tres
semillas, mientras barajar los turnos la baja a 0,31 a 0,45 e invertir la pertenencia a 0,36 a 0,52.
Es un **escalón**, no una pendiente, y confirma con el sello relativo ya entrenado lo que el 6 había
medido sobre el absoluto. La consecuencia práctica es la misma. **Agrandar la tabla no sirve.**

**Y las tres unidades con bit contestan otra pregunta.** Con el bit de pertenencia el modelo queda
**inmune a las cinco perturbaciones**, con recuperación de 0,97 a 1,00 en todas y la condición
invertida en **1,0000** contra 0,36 a 0,52 sin bit. Desacopla de quién es una entrada de cuándo fue,
que era exactamente lo que yo había diseñado que hiciera.

Tengo que declarar el corolario porque me cambia cómo se lee este banco. **Sobre las unidades con bit
las condiciones de esta prueba dejan de ser un test del sello.** No es que la bandera desaparezca, es
que el modelo tiene un tercer camino y ya no necesita la tabla para esto.

**La contraparte de peso da lo mismo por otro lado.** Con el bit, las filas de la tabla pierden un
30 % de norma, de 2,17 a 2,37 contra 3,14 a 3,49, y la varianza deja de concentrarse en una sola
dirección, 0,59 a 0,66 contra 0,78 a 0,83. **El bit descarga la tabla.** Es la contraparte geométrica
exacta de lo conductual, y son dos instrumentos independientes diciendo lo mismo.

---

## 11. El archivo co-entrenado sobre un transformer real

**6 de septiembre a la noche.** TinyLlama de 1,1B **congelado**, 22 capas, vocabulario natural de
32.000 tokens, preentrenado por otros. Entrenables sólo las matrices del archivo, **2,11 M sobre
1,10 B, o sea el 0,192 %**. Seiscientos pasos, treinta minutos, en CPU sin GPU.

Tarea cross-secuencia, el hecho se dice en un forward y se pregunta en otro independiente.

| condición | acierto |
|---|---|
| con lectura | **1,0000** |
| lectura apagada | **0,0000** |
| archivo barajado | **0,0000** |

Con el archivo barajado las predicciones salen **rotadas exactamente como el barajado**. No falla, lee
lo que le puse.

> **El mecanismo no era una propiedad de mi banco.** Un transformer real preentrenado y congelado
> aprende a formar la query y a leer su propio archivo persistente con el 0,192 % de sus parámetros.

Límite honesto, el diagnóstico corre sobre pocas muestras y para publicar eso es poco.

---

## 12. Estado de la alucinación, medido

Éste es el resumen que contesta la pregunta del título.

**Las tres clases de error, y su destino.**

1. **Invento**, contestar un valor cuando no había respuesta. **Eliminable** hasta el límite de
   detección.
2. **Confusión al leer**, atribuir mal un dato verdadero. **Convertible** en abstención, y lo probé
   por evidencia **directa** y no por descarte. La entrada del hecho preguntado está escrita siempre,
   cero casos ausentes en 8000 muestras, así que el error es **enteramente de lectura**.
3. **Memoria falsa por escritura corrupta.** La hipótesis externa decía que era indetectable por
   diseño. **La refuté con las dos sondas**, el vecino queda intacto y el hecho entero se degrada.

**El SER medido con kernel 5 y con archivo largo, el 6 de septiembre.**

| régimen | SER | error de versión | error de identidad | error fuera de dominio |
|---|---|---|---|---|
| 15 de agosto, kernel 3 | 0,7754 | ≤0,0078 | 0,2227 | 0,0000 |
| **kernel 5, archivo corto** | **0,0060 a 0,0140** | 0,0000 | **0,0000** | 0,0000 |
| **kernel 5, archivo largo** | **0,0000 a 0,0156** | ≤0,0031 | **0,0000 a 0,0031** | 0,0000 |

La colisión de clave, que el 20 de agosto identifiqué como la causa del error de identidad, **está
resuelta**, y lo que la resolvió fue el kernel 5 de la ley de la ventana. No hizo falta atacarla de
frente.

**Lo único que me queda es el invento**, entre 0,0149 y 0,0746 sobre las preguntas sin respuesta. Es la
alucinación pura, y tiene dos ataques conocidos y sin combinar.

**El límite, y es el que importa para leer todo esto.** El cero en error fuera de dominio es en parte
una propiedad de mi banco. El idioma es cerrado y todo lo que el modelo puede decir está en el archivo.
Un modelo con vocabulario abierto tiene un grado de libertad que acá no existe, así que «no inventa»
**no transfiere sin más**.

**Y el titular corregido, que es el honesto.** No es «un modelo sin alucinaciones». Es **un modelo
cuyo error o viene avisado, o está acotado y localizado**.

---

## 13. Lo que quedó establecido, en una lista

1. Un índice **persistente entre sesiones y co-entrenado dentro de la red** funciona, y el gradiente
   que no fluye por la selección de los k mejores no lo impide.
2. La geometría **agrupa y no ordena**. Es propiedad del enfoque y no del método, porque dos
   mecanismos sin nada en común fallan igual.
3. El **sello de orden** resuelve el conflicto de versiones, de 0,4570 a 0,9956, con su control
   barajado en 0,4768. DOI publicado.
4. El sello es una **bandera y no un reloj**, y sostiene dos funciones entrelazadas en una sola tabla.
5. El **contexto es precondición del cómputo**, no corrección. Un acceso temprano da 0,9998 y el mismo
   acceso tardío da 0,4990. Es dicotomía, no umbral gradual.
6. La **ley de la ventana**. Lo que cae fuera del alcance de la convolución que forma la query es
   invisible, con cero exacto, y en un modelo profundo atenúa en vez de bloquear.
7. La **alucinación en un modelo con archivo es un error de recuperación, no de generación**.
8. La **abstención vive mejor en una cabeza separada**, con 129 parámetros, y no en la memoria.
9. La **abstención perfecta y el conocimiento nulo son el mismo estado**, con exactitud global clavada
   en el piso trivial.
10. El techo de la detección de ausencia desde el estado es **0,70**, con cinco lectores distintos, y
    depende de la recuperación.
11. El **archivo no comprime ni desaloja**, puede crecer sin tocar un peso, y el cuello no es la
    velocidad sino la **precisión**.
12. Lo que rompe el archivo largo **no es cuántos competidores hay sino qué dicen**, y el colapso era
    transferencia y no un techo.
13. El mecanismo **transfiere a un transformer real congelado** con el 0,192 % de sus parámetros.

---

## 14. Lo que cerré en negativo, y por qué vale

Cada uno con criterio de abandono comprometido por adelantado.

- La **gemación** como mecanismo de indexación, en dos regímenes, incluido el que diseñé a propósito
  para favorecerla.
- Mi hipótesis **GPS** en forma fuerte. En forma débil, dar el barrio y no la casa, funciona.
- Las **cinco vías** para detectar la ausencia desde una señal interna.
- El **slot nulo** como lugar donde vive la abstención, con 400 umbrales probados.
- La **bandera de recuperación** en su versión literal, por su propio criterio.
- El **envejecimiento del archivo** como vía, tres formas de empujar y tres veces por arriba del
  umbral.
- El **escriba**, verificar antes de guardar, cerrado en fase cero con cero GPU.
- Cinco **funciones de pérdida** para romper la constante de abstención.

Uno de ellos, el test de k, me cerró una línea entera en horas en vez de meses, porque el criterio
estaba escrito antes.

---

## 15. Lecciones de método, que son parte del resultado

- **Nueve veces un número limpio me escondió un artefacto**, y varias veces el artefacto era de mi
  propio instrumento. Un cero exacto se investiga antes de reportarse.
- **Un negativo sin barrido de presupuesto no es un negativo.** Cinco negativos de mi programa eran
  impaciencia.
- **Un control de sanidad tiene que poder fallar.** Mi control con una sola candidata estaba vacío,
  acertar no requería leer.
- **Lo que da el veredicto es cruzarlo con el nulo**, y elegir el nulo correcto es la mitad del
  trabajo. Permutar etiquetas sólo vale si la etiqueta es la única fuente de la asociación medida.
- **Un control puede tener un confound propio.**
- **Correr el control antes del veredicto** me dio vuelta lecturas ya escritas, más de una vez el
  mismo día.
- **Un hallazgo de arquitectura vale más que uno de entrenamiento y hay que buscarlo primero.** Doce
  intentos por la vía del entrenamiento, todos negativos o parciales, y el primer diagnóstico
  mecanicista lo resolvió en un día.
- **Antes de congelar un pre-registro**, preguntarme si la métrica se movería en caso de que la
  intervención funcione perfecto. Siete criterios míos no se pudieron leer como estaban escritos.
- **No promediar entre semillas** sin dar la distribución. Con bimodalidad, la media miente.
- **Todo instrumento que carga un checkpoint tiene que leer de su configuración todo lo que decide la
  arquitectura**, no sólo lo que el autor recordó ese día.
- **Un script de campaña no está listo hasta que corrió su primer tramo de verdad.**
- **Una condición de fin que sólo puede decir «sigue vivo» no es una condición de fin.**
- **Los veredictos automáticos se verifican.** Seis veces el juez me imprimió una conclusión que sus
  propios números desmentían.

### Las cuatro que agregué el 8 de septiembre, todas del mismo día

Fue el día en que más aprendí sobre mis propios instrumentos, y las cuatro salieron de resultados que
yo ya estaba a punto de contar como hallazgos.

- **El signo de la primera componente principal de un SVD es arbitrario.** Mi medición de la
  geometría daba rampa positiva en las tres unidades de una rama y negativa en las tres de la otra,
  un tres contra tres perfecto que se leía como que el bit invierte el gradiente de recencia. En
  valor absoluto las seis dan lo mismo. **Un tres contra tres perfecto en una cantidad cuyo signo no
  está fijado no es un resultado, es una convención de la biblioteca.**
- **Una función que reimplementa a otra no hereda sus defaults.** Mi instrumento del sello copiaba la
  función de respuesta a mano para poder guardar la distribución de lectura, y al copiarla se quedó
  sin el bit de pertenencia. Medí tres unidades con el bit apagado, o sea con una arquitectura que no
  era la suya. La firma del error vale la pena reconocerla, **la recuperación aguantaba y la respuesta
  se caía**, que es lo que pasa cuando a un modelo le sacás una entrada de la que aprendió a depender.
  Mi regla de que todo instrumento lea su configuración quedó más ancha. **La configuración decide
  variables globales y también argumentos de llamada.**
- **Un instrumento cuyo resultado es el registro no puede escribir siempre al mismo archivo.**
  Correrlo sobre una unidad nueva me pisó en silencio la medición de cuatro días antes. La recuperé
  del historial de versiones.
- **Un umbral pre-registrado necesita que su ruido esté medido.** Fijé el umbral de una predicción en
  1,5 tomando como referencia un 1,12. Al correr el juez sobre los controles, uno dio **1,44**, o sea
  un modelo que nunca vio la intervención quedaba a 0,06 de confirmar. Subí la muestra de 120 a 600
  casos y ese 1,44 se volvió **1,05**, con los cuatro controles entre 0,94 y 1,05. Era ruido. **Lo
  escribí como enmienda antes de mirar el resultado del experimento, que es la única forma en que
  vale.**

---

## 16. Publicación

- **Gemación**, DOI `10.21203/rs.3.rs-10669947/v1`, publicado.
- **TRÍPODE**, «Where Abstention Lives», escrito y con metadatos listos.
- **Sello de orden**, DOI `rs-10896018`.
- **Ventana**, mi cuarto preprint, en inglés y en español, en prescreening, y paper 11988 en TMLR.
- **El quinto**, sobre el atractor y la tabla del área bajo la curva, verificado nueve de nueve, al que
  sólo le falta la versión en castellano.

---

## 17. Lo que me queda abierto, en orden

1. **El invento**, lo único que queda de la alucinación en mi banco. Dos ataques conocidos y sin
   combinar, subir la proporción de preguntas sin respuesta y cambiar el blanco a error.
2. **Fase 2 del sello relativo**, archivo de más de 64 turnos. Quedó habilitada, porque las dos
   predicciones cerraron el 8 de septiembre.
3. **Una perturbación que el bit de pertenencia no cubra**, si quiero seguir midiendo el sello. Con
   el bit puesto, las cinco que tengo dejaron de morder.
4. **El régimen de 3280 entradas entrenado**, abierto desde el 5 de septiembre.
5. **Escalar el banco sobre el transformer real**, meter versiones y abstención, y medir el error en
   **vocabulario abierto**, que es donde el «no inventa» deja de estar garantizado por el banco.
6. **La política de escritura**, la expulsión gateada por sorpresa, que nunca corrí.
7. **Los 32 instrumentos** que mi auditoría todavía marca por no leer la arquitectura de su
   configuración.

---

## 18. La línea que abrí, y que a esta altura ya está corriendo

Le puse el nombre el 7 de septiembre.

> lo que busco es crear un micro modelo de igual condición que lo hacen los modelos grandes de
> frontera, para de esta forma poder descubrir cómo hacer que tengan memoria persistente incorporada
> y eliminar las alucinaciones de este tipo de modelos

### Lo primero, contestarme bien la pregunta de la que salió

Me pregunté si los modelos de frontera usan kernel 3 o 5. **No.** Un transformer denso hace atención
completa en cada capa y no hay ninguna convolución formando la query.

Pero el matiz es justo donde vive mi hallazgo. En la **capa cero** de un denso el estado de un token
es su embedding más su posición, o sea la query es **función pura del token de su posición**, que es
literalmente mi condición más pobre. Lo que le da a un denso una query compuesta no es un kernel más
grande, es la **profundidad**, porque después de una sola capa de atención el estado ya mezcló todo el
prefijo. Así que el análogo de mi ventana en un denso no es el alcance del kernel sino **cuántas capas
de atención hay antes del punto donde se forma la query que consulta la memoria**, y el corte de cero
exacto **no existe** en un transformer denso.

Por eso mi ley se enuncia como condicional, y su alcance real es grande igual. Muerde en todo modelo
que use atención **local** o **recurrencia**, y ahí es adonde se está moviendo frontera por costo,
ventanas deslizantes, capas locales alternadas con globales, y los híbridos desplegados. Vale para esa
mitad del campo, no para la otra, y prefiero decirlo así antes que estirarlo.

### Y lo segundo, dónde está de verdad la modificación que busco

No está en la ventana. El cuello de un modelo denso son tres cosas distintas que conviene no mezclar.
La ventana de contexto es finita y su corte es duro, lo que sale no se degrada, no está. Dentro de la
ventana la atención se diluye. Y sobre todo, **los pesos no cambian**, o sea lo que el modelo aprendió
en la conversación vive en las activaciones y se tira al terminar.

El tercero es el que nadie tiene resuelto y es el que yo ataco. La modificación es **darle al
transformer un archivo direccionable con clave sellada, co-entrenado con el modelo** y no adosado
encima. Lo que ya tengo medido y lo sostiene, que el archivo **no comprime ni desaloja** y puede
crecer sin tocar un peso, que el sello de orden resuelve el conflicto de versiones, que el bit de
pertenencia desacopla de quién es una entrada de cuándo fue, y que el cuello no es la velocidad sino
la **precisión de la búsqueda**, donde lo que la rompe no es cuántos competidores hay sino qué dicen.

Y las alucinaciones no van en paralelo, **son el mismo problema visto de dos lados**. Lo que yo leía
como inventar era colisión de clave, o sea un error de indexación, o sea un problema de acceso de la
consulta. Si el modelo recupera exactamente el hecho correcto no tiene que inventarlo, y si sabe que
no lo tiene puede abstenerse.

### El escalón 1, lanzado y medido el 8 de septiembre

Nunca había entrenado una unidad con acceso global. Censo de mis 157 checkpoints, cero. Mi control
del 4 de septiembre midió atención completa sobre pesos entrenados con convolución, que es otra cosa.

Antes de gastar una GPU me hice una sospecha y la medí. Pensé que mi atención de lectura, que usa el
mismo vector como consulta, clave y valor, iba a colapsar en la propia posición y ser una identidad
disfrazada. **Me equivoqué.** Atiende de verdad, la propia posición se lleva 0,42 a 0,47 y quedan
entre 9,6 y 11,3 posiciones efectivas de 24, el perfil es plano de la distancia 1 a la 23 y depende
del contenido. Lo que sí queda en pie es que **sin proyecciones propias no puede aprender a qué
atender**, sólo hereda la geometría de las embeddings.

La comparación me salió gratis porque los dos brazos ya estaban corridos a 26.000 pasos con los mismos
hiperparámetros. Esto es lo que va, sobre el caso difícil donde la respuesta no está.

| paso | kernel 3 | kernel 5 | atención completa |
|---|---|---|---|
| 2.000 | 0,4223 | 0,3819 | 0,2724 |
| 4.000 | 0,5737 | 0,6411 | 0,5331 |
| 6.000 | 0,6045 | 0,7548 | **0,7642** |
| 8.000 | 0,6360 | 0,7858 | **0,8833** |
| 10.000 | 0,6427 | 0,8793 | **0,9435** |
| 12.000 | 0,6544 | 0,9432 | **0,9841** |
| 16.000 | 0,6506 | 0,9613 | **0,9910** |
| 20.000 | 0,6471 | 0,9907 | **0,9930** |

Falta el tramo final para el veredicto formal, pero la dirección ya no está en discusión. Las tres
semillas van en 0,9886, 0,9904 y 1,0000, muy por encima del 0,80 con el que yo había escrito la
refutación.

**Y hay algo en la forma de estas curvas que confirma el refinamiento mejor que el resultado.** Mi
atención completa y mi kernel 5 **convergen al mismo techo**, con la diferencia bajando de 0,0975 en
el paso 8.000 a 0,0023 en el 20.000. La atención llega antes, no llega más alto. El kernel 3, en
cambio, se queda en 0,6471 y no llega nunca.

Es exactamente lo que predice la ley enunciada como cobertura. El kernel 5 tiene alcance 4 y **ya
cubre** la relación, que cae a distancia 3, así que alcanza el mismo techo. La atención cubre todo, o
sea cubre de más, **y cubrir de más no compra techo**. El kernel 3 tiene alcance 2, deja la relación
afuera, y ninguna cantidad de pasos lo arregla. **El techo depende de si la posición relevante está
cubierta o no, y no de cuánto más se cubra.** Eso es lo que hace que la ley sea una condición binaria
y no una escala. **La atención completa no sólo alcanza al kernel 5, lo pasa desde el paso 8.000**, y deja
al kernel 3 estancado en 0,64.

Hay algo que no había previsto y que me parece el dato más honesto de la corrida. **Arranca más
lento.** En el paso 2.000 va por debajo de los dos brazos, y recién los cruza entre el 4.000 y el
6.000. Es consistente con lo que medí a la mañana, porque sin proyecciones propias el modelo tiene que
aprender a atender moviendo la geometría de sus embeddings, que es un camino más largo. Si eso se
confirma, es un argumento a favor del escalón 2 que no estaba en mi pre-registro.

Y lo que me refutó mi propia predicción del mecanismo está contado arriba, en el §8. Gana por área y
no por altura, y el requisito es de cobertura y no de foco.

### El escalón 2, diseñado y esperando

El escalón 1 no es el Micro LM de Frontera y no lo voy a llamar así. Es cerrar mi ley de la ventana
por el otro lado.

El escalón 2 sí lo es, y ya está diseñado con el parche exacto. Proyecciones propias para la atención
de lectura, inicializadas **en la identidad y no al azar**, porque así la condición nueva contiene a
la vieja como caso particular y un resultado peor se puede leer. Es la propiedad que hoy le falta a mi
atención completa y es el defecto exacto que me hundió un experimento el 22 de agosto. Cuesta 196.608
parámetros en el árbol y 49.152 efectivos, contados del árbol y no citados de un comentario viejo.

Y lo que ese cambio me regala es lo que más me importa. **Con tronco de atención el archivo deja de
poder estar compensando** el techo de capacidad de la regla delta, porque la atención resuelve todo lo
que está dentro de la secuencia y al archivo sólo le queda aportar lo que está **fuera**, o sea las
sesiones anteriores. Cualquier ganancia que quede ahí es memoria persistente genuina.

Después queda el vocabulario, que es la diferencia que decide si el nombre se sostiene. Medí que de
mis 242 tokens **158 son respuestas legales** contra un archivo de a lo sumo 40 entradas, así que
margen para inventar hay, unos 120 valores que el modelo podría decir y no dice. Lo que me falta no es
el margen sino la **forma**, porque mi respuesta es un solo token y la alucinación que importa en los
grandes es composicional.

**Nada que conserve el tronco recurrente o el vocabulario cerrado se llama de Frontera.**

---

## 19. Cierre

En un mes y un día pasé de una pregunta sin protocolo a un modelo entrenado desde cero que contesta
sobre lo que se le dijo en una sesión anterior, sabe cuál versión rige después de una corrección,
distingue entre archivo propio y ajeno, avisa cuando el dato no está, y resiste que le muevan los
turnos del archivo, con exactitud global de 0,988 a 0,993 contra un piso trivial de 0,4065. El
mecanismo ya lo vi funcionando sobre un transformer real congelado, y hoy tengo corriendo la primera
versión de mi modelo con el mismo acceso global que usan los grandes.

Lo que más cambió no fue el resultado sino el diagnóstico. Entré creyendo que el problema era que el
modelo inventa. Salgo sabiendo que **no inventa**, que atribuye mal, que atribuir mal era colisión de
clave, que la colisión de clave era un problema de **acceso** de la consulta, y que el acceso se
arregla con 1.280 parámetros.

Y de los últimos dos días me llevo una corrección más, que es sobre mí. Tres veces tuve un resultado
listo para contar y las tres se cayeron cuando fui a mirar el instrumento en vez del número. Un tres
contra tres perfecto que era el signo arbitrario de una biblioteca, una caída de acierto que era yo
midiendo un modelo sin la entrada de la que aprendió a depender, y un umbral que un control mío casi
cruza por ruido de muestreo. **Ninguna de las tres la habría visto mirando el resultado.** Las vi
mirando de dónde salía.

Lo que busco sigue siendo lo mismo, y es la vara.

> que un LLM no olvide nunca lo que le dije
