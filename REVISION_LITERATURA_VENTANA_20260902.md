# ¿Está ocupado el hallazgo de la ventana? · revisión hecha el 2-sep, antes de anunciar nada

Cumple la condición **3** de `CRITERIO_DESCUBRIMIENTO.md`, la única que faltaba: *verificado contra la
literatura el mismo día, leyendo los trabajos más cercanos y preguntándoles explícitamente por
nuestras piezas*.

## Qué hay que buscar, dicho con precisión

El hallazgo tiene **tres piezas** y sólo cuenta como ocupado si alguien tiene **las tres**:

1. **La ventana con la que se forma la query limita qué parte de la consulta condiciona la
   recuperación.** No es la distancia entre la clave y el valor en el texto: es la distancia entre
   cada componente de **la pregunta** y la posición donde se forma la query.
2. **Y cuando una parte queda afuera, el modelo no falla silenciosamente: recupera por la parte que
   sí ve y responde con confianza.** O sea, es una causa mecánica de **no abstenerse**.
3. **Y se diagnostica sin entrenar nada:** se cambia un token de la consulta y se mira si la búsqueda
   se mueve.

## Lo más cercano, y por qué no lo cubre

### CAT · *On the Power of Convolution Augmented Transformer* (arXiv 2407.05591)

**Es el trabajo más cercano que existe y hay que citarlo de frente.** Define *N-gram associative
recall* y su Teorema 1 usa un filtro causal de largo exactamente **N**, el tamaño del n-grama. La
familia de la idea —el filtro tiene que abarcar lo que la consulta necesita— **está ahí**.

Tres cosas lo separan de esto, y las tres son verificables en el propio paper:

- **Su consulta es un n-grama CONTIGUO**, los últimos N tokens, sin nada irrelevante en el medio. Acá
  lo que manda es la **distancia** de un componente, no el largo de un bloque: en «cual es `<art>`
  `<sust>` de `<ent>` ?» los dos componentes que importan están a distancia **1** y **3** con basura
  gramatical entre ellos, y un filtro de largo 3 no los cubre aunque los dos «quepan» en tres tokens.
- **Teorema 1 es un resultado de EXISTENCIA**, una construcción de pesos que resuelve la tarea. No hay
  ninguna medición de qué pasa cuando el filtro es **demasiado corto**. Sus experimentos usan ancho
  fijo W=3 y no aíslan el efecto del ancho.
- **Y reportan lo contrario en la práctica:** *«While K/Q convolution helps in theoretical
  constructions for N-gram AR, in real experiments, they don't provide noticeable performance
  benefits.»* Nosotros medimos el efecto opuesto y grande —`nose_rel` de 0,59-0,73 a 0,99-1,00— en un
  régimen que ellos no probaron: memoria **persistente entre secuencias** y consulta **compuesta**.

**No mencionan abstención, alucinación, ni respuesta confiada con parte de la consulta invisible.**

### Zoology · *Measuring and Improving Recall in Efficient Language Models* (arXiv 2312.04927)

Es el paper que fija la vara del campo, y **atribuye el 82 % del hueco de perplejidad al recall
asociativo**. Su explicación es que la convolución aplica un **filtro fijo que no puede adaptar, según
la entrada, qué tokens mezclar**, mientras la atención localiza el token que hace juego a cualquier
distancia. Es una afirmación sobre **adaptabilidad** e **input-dependence**, no sobre el
**alcance de la ventana con la que se forma la query**. Y en **MQAR, la consulta es UN TOKEN**: por
construcción no puede tener una parte adentro y otra afuera de la ventana.

**Ese es, probablemente, el motivo por el que nadie lo vio.** La tarea canónica del campo no tiene
consultas compuestas.

### La familia delta y los modelos de producción

Acá está la implicancia práctica, y es concreta: **el kernel de la short conv es 4 en todos**, o sea
el token actual y los **tres** anteriores.

- **Mamba-2:** kernel 4 por defecto, y la implementación de `causal_conv_1d` admite una ventana
  **máxima de 4**. No es una elección de hiperparámetro: es un techo del kernel usado.
- **Gated DeltaNet / Qwen3-Next:** la conv de kernel 4 se aplica **por separado a Q, K y V** antes de
  la recurrencia.
- **Kimi Linear** e **Inkling:** kernel 4.

Con alcance 3, cualquier componente de la pregunta que quede a más de tres tokens de la posición de
lectura **no entra en la query de esa capa**. En una pregunta tan común como *«¿cuál era el anterior
X de Y?»* el token que marca el tiempo cae afuera. Es exactamente el defecto que este proyecto ya
había medido por otra vía en `lat`, con `anterior` en 0,3798 contra 0,8125 de su gemela
(`INFORME_CAMINO_LATERAL_20260824.md`).

### Lo que se buscó y no apareció

`query formation window`, `query receptive field`, `partial query conditioning`, `local convolution
query`, `short convolution` + recall, y sensibilidad de la recuperación a tokens individuales de la
consulta por distancia. Nada mide el escalón. Lo que aparece del lado de la atención plena
—**Multi-Token Attention**, de Meta— va en la dirección opuesta y lo confirma de rebote: es una
convolución **sobre los pesos de atención** para que la atención pueda condicionarse en varios
tokens a la vez, y su motivación es que un solo token no alcanza para localizar. Nadie lo plantea
como **qué parte de la consulta queda ciega**, ni lo conecta con abstenerse.

## Veredicto

**Las tres piezas juntas están libres.** La pieza 1 tiene un precursor teórico real y hay que citarlo
sin adornos (CAT, Teorema 1: el filtro debe abarcar el n-grama). Las piezas **2** y **3** —que esto
es una causa mecánica de responder con confianza sobre lo que no se dijo, y que se diagnostica
ablando un token— **no las tiene nadie**.

**Límite de esta revisión, que va pegado:** es una búsqueda por términos y lectura de los trabajos más
cercanos, no una revisión sistemática. Y en un modelo profundo el efecto se atenúa: acá la lectura
ocurre en el bloque 0 **antes** del mixer, así que la ventana es literalmente todo lo que la query
puede ver y el cero es **exacto**; en un modelo con la memoria consultada en capas altas, la
recurrencia ya movió información hacia adelante y el alcance efectivo crece con la profundidad. El
resultado se afirma para **consultas a memoria desde capas tempranas**, que es donde vive.
