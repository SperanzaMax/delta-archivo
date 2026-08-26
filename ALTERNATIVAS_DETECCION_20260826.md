# ALTERNATIVAS PARA DETECTAR LA ALUCINACIÓN — inventario con costo y retorno

**2026-08-26.** Pedido de Maxi: buscar una mejor manera de detectar cuándo el modelo alucina. Esto
ordena las vías que quedan, dice cuál cuesta cuánto, y separa las que se pueden correr hoy en CPU de
las que necesitan pool.

**Antes que nada, la distinción que reordena todo el problema** y que sale del censo del 26-ago.

| fallo | qué pasó | dónde está hoy |
|---|---|---|
| **`invento`** | la respuesta no estaba y el modelo contestó un valor | **es el que queda**, 0,19-0,51 según unidad |
| **`err_identidad`** | la respuesta estaba y trajo la de otra entidad | **`lat2` lo llevó a 0,0000** |

Esto **cambia el blanco**. Durante semanas «alucinación» significó mala atribución, y `lat2` la
resolvió. Lo que queda por detectar es contestar cuando no hay nada, que es otro problema.

---

## Las seis vías, ordenadas por costo

### A1 · Dos detectores especializados en vez de uno ★ CORRIENDO

`PREREG_DOS_DETECTORES.md` (SHA `91494aa0`). El detector único falla porque resuelve dos problemas
con un número solo: la confianza da AUC **0,8631** sobre un fallo y **0,7397** al mezclarlos. Es el
mismo argumento que ganó el trípode.

**Costo:** cero, CPU, checkpoints en disco. **Retorno:** alto si cumple, porque la solución es
barata (dos cabezas en vez de una).

### A2 · Pooling sobre las posiciones, en vez de leer sólo `pos_q` ★ SALE DE D-2

Verificado en el código el 26-ago: `entrenar.py:113` lee el logit de la cabeza **sólo en `pos_q`**, y
ahí la entropía de lectura es 1,71-1,77 contra un techo de 1,79 (casi uniforme), mientras el foco
real vive en posiciones intermedias. Medido hoy, **el foco coincide con `pos_q` en el 0,0-0,2 % de
los casos**: son lugares distintos casi siempre.

La bandera de Maxi transporta la señal de un punto al otro. **Hay una versión más simple que ni
siquiera necesita transportarla:** leer la cabeza en *todas* las posiciones y agregarlas con un
pooling aprendido. Evita el problema de que la posición de foco es un blanco móvil durante el
entrenamiento —depende de los pesos, que cambian— y es diferenciable de punta a punta.

**Costo:** 3 unidades de pool. **Retorno:** alto si D-2 cumple; **no se toca si D-2 falla.**

### A3 · Calibrar, que es donde está el techo real ★ GRATIS, Y ES LO QUE FALTA PARA QUE SIRVA

El techo medido **no es de capacidad**: el AUC del logit de la cabeza va de **0,777 a 0,998**. La
información para decidir está; lo que está mal puesto es el corte, y `sonda_umbral.py` ya mostró
—post-hoc— que la única unidad que falla la compuerta la pasa con otro umbral.

Nadie corrió una calibración en serio: Platt o isotónica ajustada en una mitad y juzgada en la otra,
con el umbral elegido **con margen** y no en el borde, que es la lección del 19-ago (*el óptimo
pegado al borde del criterio no generaliza*).

**Costo:** cero, post-hoc sobre checkpoints. **Retorno:** no descubre nada nuevo, pero es lo que
convierte un AUC alto en un sistema que se puede usar. Es la vía con mejor relación esfuerzo /
utilidad de toda la lista.

### A4 · Ensamble de semillas ★ GRATIS, Y ES EL BASELINE QUE FALTA

Tres modelos con semillas distintas contestan lo mismo; su desacuerdo detecta el error. Es la línea
clásica (*deep ensembles*), es el baseline fuerte contra el que cualquier detector propio tiene que
compararse, y **el proyecto nunca lo midió**. Los tres checkpoints ya están en disco.

**Cuidado declarado:** en este banco la variación entre semillas es **bimodal** y medida desde
E-I3c, o sea que las semillas difieren en *capacidad* y no sólo en ruido. Un ensamble puede estar
midiendo «esta semilla es la mala» en vez de «esta pregunta es difícil». Hay que reportarlo así.

**Costo:** cero. **Retorno:** medio como método (triplica el costo de inferencia), alto como
**contraste honesto**: si el ensamble gana a todo lo propio, eso hay que decirlo.

### A5 · Entrenar la cabeza contra el ERROR, no contra la ausencia ★ LA PREGUNTA CORRECTA

Hoy la cabeza se entrena con `es_nose`, o sea **¿hay respuesta?**. Pero lo que se quiere detectar es
**¿me voy a equivocar?**, que incluye los dos fallos. Es un cambio de una línea en
`perdida_cabeza`: cambiar el blanco de la BCE.

**Riesgo, declarado por adelantado:** el blanco depende del propio modelo y se mueve mientras
entrena, así que puede ser inestable o colapsar al prior, exactamente como le pasó al slot. Eso hay
que preverlo, no descubrirlo.

**Costo:** 3 unidades. **Retorno:** alto — es la única vía de la lista que apunta directo al
objetivo en vez de a un proxy.

### A6 · Perturbación DIRIGIDA en vez de aleatoria ⚠ BLOQUEADA POR EL CIERRE

El monitor de desacuerdo tapaba el 25 % de las entradas **al azar** y falló 0/8, porque eso mide
«¿la respuesta viene del archivo?» y la respuesta es que sí siempre.

**Tapar específicamente la entrada que GANÓ la lectura** mide otra cosa: si el modelo tenía una
alternativa igual de buena esperando. Esa es la firma exacta de flotar entre entradas reales, que es
la mala atribución. No es la misma perturbación ni mide lo mismo.

**⚠ Pero es sin etiquetas, y el `PLAN_FOCO` §8 cerró esa línea por seis meses, y el prereg del
monitor comprometió explícitamente no probar una tercera perturbación.** Va acá anotada, **no se
corre**, y la decisión de reabrirla es de Maxi y de nadie más. Si se reabre, tiene que ser con el
motivo escrito por adelantado, como se hizo con el test de k.

---

## Qué se corre hoy y qué no

| vía | costo | estado |
|---|---|---|
| A1 dos detectores | CPU | **corriendo** |
| A3 calibración | CPU | **se corre a continuación** |
| A4 ensamble | CPU | **se corre a continuación** |
| A2 pooling | 3 unidades | espera el veredicto de D-2 |
| A5 blanco = error | 3 unidades | espera decisión de Maxi |
| A6 perturbación dirigida | CPU | **bloqueada por el cierre del §8** |

**Lo que ninguna de estas contesta, y hay que seguir diciéndolo:** todas menos A6 son supervisadas.
Ninguna habilita la frase «el modelo sabe cuándo no sabe».
