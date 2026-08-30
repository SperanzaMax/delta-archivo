# Post de LinkedIn — la abstención perfecta

**⚠ NO PUBLICAR TODAVIA.** Dos razones. (1) El preprint no está enviado y el post necesita el DOI.
(2) La primera versión de este texto decía «nueve modelos exactamente iguales, cambiando sólo el
número de arranque», y eso **es falso**: el confound de la siembra encontrado el 29 mostró que las
semillas 0 a 2 arrancaron con 12000 pasos de entrenamiento previo y las 3 a 8 desde cero. Ya está
corregido acá, pero **si el texto se copió antes de las 18 h del 29, hay que volver a copiarlo**.

**Estado.** El preprint todavía no está enviado, así que el texto está escrito para funcionar **sin
link**. Cuando salga el DOI se agrega al final, donde está la marca.

Imagen sugerida, la tabla de las seis unidades entrenadas desde cero con las cuatro degeneradas
resaltadas, o la curva de abstención al paso 2500.

**Estilo verificado**, sin raya larga y sin dos puntos, según la regla de escritura de Maxi.

---

## Texto

Entrené seis modelos desde cero con la misma receta. Cuatro terminaron siendo detectores de
alucinación perfectos. Y son completamente inútiles.

Quiero contar por qué, porque me parece que dice algo incómodo sobre cómo medimos esto.

La idea era simple. Le puse dos partes, una que busca la respuesta y otra que decide si conviene
contestar o callarse. Y para entrenar a la segunda usé lo que parece la mejor señal posible, que es
preguntarle si el modelo se va a equivocar en caso de contestar.

El problema es que esa etiqueta depende del propio modelo, y entonces hay más de una solución
consistente.

Pensalo como un estudiante al que le pedís que antes de cada examen prediga si va a aprobar. Hay dos
maneras de volverse un predictor perfecto. La primera es estudiar mucho, aprobar casi siempre y
aprender a reconocer los temas flojos. La segunda es no estudiar nada y predecir que va a reprobar
todos los exámenes. Acierta el cien por ciento de las veces.

El segundo estudiante no está mintiendo ni fallando. Tiene razón. Es un predictor perfecto de su
propio rendimiento, y su rendimiento es cero.

Eso es lo que hicieron cuatro de esos seis modelos. Se abstienen de todo. Su tasa de acierto al
callarse es 1,0000 y su tasa de invención es 0,0000. Perfectos en las dos únicas cifras que este
campo suele reportar, y mejores que los modelos que sí funcionan.

Su exactitud real es 0,4065. Que es exactamente el número que sacás abstenéndote de todo, sin modelo
y sin entrenar nada.

Y no es que el detector esté roto. Calculé qué valor debería tener si en vez de mirar cada pregunta
hubiera aprendido solamente la estadística general. La fórmula predice 1,513 y el modelo tiene 1,507.
Está haciendo exactamente lo óptimo para un buscador que no encuentra nada.

Lo que me parece que hay que llevarse es esto. Que un modelo diga no lo sé en vez de alucinar suena
como el objetivo correcto, y lo es, pero tomado solo tiene una solución tramposa que gana todas las
mediciones. Nadie tiene que hacer trampa a propósito. El entrenamiento llega ahí solo, en una
minoría importante de los casos, y las dos métricas que mirarías para darte cuenta te dicen que salió
perfecto.

La corrección es simple y hay que aplicarla. Medir la exactitud global contra su piso trivial,
siempre, al lado de cualquier métrica de abstención. Un modelo que no supere ese piso no aprendió
nada, por impecable que se vea callándose.

Y da vuelta la pregunta. Callarse bien es fácil, tan fácil que un modelo que no sabe nada lo hace
perfecto. Lo difícil es lo otro, salir del silencio sin empezar a inventar. En estos experimentos eso
se decide en los primeros 2500 pasos, y uno de los modelos que se salvó lo hizo por una sola
respuesta entre 512.

Los pre-registros están congelados con su hash antes de correr, y los datos crudos por semilla están
acá.

https://github.com/SperanzaMax/delta-archivo

<<ACÁ VA EL DOI CUANDO SALGA>>

#MachineLearning #AI #Research #LLM #Hallucination

---

## Texto alternativo de la imagen

Tabla de seis modelos entrenados desde cero con la misma receta. Dos alcanzan una recuperación de
0,73 y 0,80. Los otros cuatro quedan entre 0,30 y 0,40 y se abstienen en el cien por ciento de las
preguntas, con una exactitud global de 0,4065 que coincide con el piso trivial.

---

## Chequeo de estilo

Sin raya larga y sin dos puntos en el cuerpo del post. Los únicos dos puntos del archivo están en
esta sección de notas y en el encabezado, que no se publican.
