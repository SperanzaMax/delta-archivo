# El tercer sustrato: un índice co-entrenado recuerda como un RAG y calla como un modelo paramétrico

**Borrador — 2026-08-16.** Maximiliano Speranza, investigador independiente.
Repositorio: https://github.com/SperanzaMax/delta-archivo
Material: `INFORME_SER_20260815.md`, `INFORME_MITIGACION_20260815.md`,
`INFORME_SCORE_ARCHIVO_20260816.md`, `INFORME_SONDA_VECINO_20260816.md`,
`INFORME_CORRECCION_PERDIDA_20260816.md`. Pre-registros con hash: `PREREG_SCORE_ARCHIVO.md`
(`fea5e061…`), `PREREG_SONDA_VECINO.md` (`faebb671…`), `PREREG_CORRECCION_PERDIDA.md` (`c51b36b4…`).

---

## 1. El problema

La memoria de un asistente conversacional se implementa hoy sobre dos sustratos, y hay evidencia
reciente de que se comportan de forma **asimétrica**: *Substrate Asymmetry in User-Side Memory*
(arXiv 2606.11712) compara memoria paramétrica (un adaptador LoRA por usuario) contra recuperación
densa top-K, y trata la **ausencia factual —abstenerse cuando el hecho no está— como un eje
ortogonal**, en el que **la recuperación gana decisivamente**.

La razón es estructural y vale enunciarla: **un recuperador externo puede devolver el conjunto
vacío.** Cuando no hay nada que traer, la abstención sale gratis. Un modelo que guarda el hecho en
sus pesos no tiene ese estado: siempre produce la continuación más probable.

Existe un **tercer sustrato** que ninguno de los dos describe y que este trabajo mide: un **índice
persistente co-entrenado dentro de la red**, ni externo ni paramétrico. Es la dirección natural
—elimina el retriever congelado y su *stale index*— y la pregunta que nadie contestó es de qué lado
de la asimetría cae.

**Resultado principal: cae del lado equivocado.** Recupera como un índice y, para la ausencia, se
comporta como la memoria paramétrica.

## 2. El modelo

Modelo de lenguaje **entrenado desde cero**, sin ningún componente preentrenado: 863 730 parámetros,
3,5 MB, idioma cerrado de 242 tokens legible por humanos. Regla delta + **archivo persistente
co-entrenado**, con sello de orden aprendido, leído por softmax e inyectado en el bloque 0.

La tarea es multi-sesión con **el estado recurrente reseteado entre sesiones**: los hechos se dicen
en una sesión, se corrigen en otra —con correcciones **elípticas**, del tipo «no, es beto», que no
nombran la entidad— y se preguntan en una tercera. **El único puente posible entre sesiones es el
archivo.** La respuesta es un token único, lo que hace la métrica exacta y determinista, sin juez LLM
ni parser interpretativo.

## 3. Qué funciona

| nivel | acierto | SER | err_versión | err_identidad | err_fuera |
|---|---:|---:|---:|---:|---:|
| N1 plantilla fija | 0,9980 | 0,0020 | 0,0020 | 0,0000 | 0,0000 |
| N2 paráfrasis | 1,0000 | 0,0000 | 0,0000 | 0,0000 | 0,0000 |
| N3 corrección elíptica | 0,7754 | 0,2246 | 0,0020 | **0,2227** | 0,0000 |
| N4 multi-sesión | 0,7598 | 0,2402 | 0,0078 | **0,2324** | 0,0000 |

**(a) El versionado está resuelto.** `err_versión ≤ 0,0078`. El control descarta que sea artefacto del
conteo: por azar el reparto sería versión 0,0741 / identidad 0,9259, y el observado es 0,0279 /
0,9721 — los errores de versión son **2,7× menos frecuentes que por azar**. Una segunda vía
independiente lo confirma: preguntando por la versión *anterior*, el modelo **nunca** devuelve la
vigente (**0,0000 exacto** en los dos checkpoints), lo que además descarta que el sello de orden se
esté usando como simple preferencia por lo último.

**(b) El archivo es lo que sostiene la respuesta.** Ablacionarlo cambia la predicción en el **100 %**
de las muestras.

**(c) El modelo nunca fabrica.** `err_fuera = 0,0000` en los cuatro niveles: toda respuesta errada es
un valor **real del archivo** puesto en la entidad equivocada. La alucinación, en este régimen, no es
inventar un dato sino **atribuir mal uno verdadero** — una forma más difícil de detectar, porque
cualquier verificación de tipo «¿este dato existe?» la da por buena. *(Alcance: con vocabulario
cerrado la fabricación libre está excluida por construcción; lo medible es que la salida queda
confinada al contenido archivado.)*

## 4. Qué falla, y dónde exactamente

Tres experimentos pre-registrados, todos sobre checkpoints ya entrenados y sin GPU.

**4.1 · El score del archivo no sabe si el hecho está.** Separando preguntas **con** respuesta de
preguntas **sin** respuesta, el máximo del score de matcheo contra las claves archivadas da
**AUC 0,4984** y **0,5022** — el azar exacto; el margen top-2 y el logsumexp, lo mismo. Las señales
de **salida** separan algo más (0,613 / 0,631), o sea **la poca discriminación que existe se arma
aguas abajo, no en la interfaz de memoria**.

Tres controles, todos capaces de fallar: los logits reconstruidos con el score extraído coinciden
**bit a bit** con los del modelo (0,000e+00), el score varía, y el archivo se usa. El azar es real.

**Mecanismo candidato**, declarado como no probado: la lectura es un **softmax sobre las entradas del
archivo y suma 1 siempre**. El modelo está obligado a leer algo aunque nada matchee, y como el
softmax es invariante a un desplazamiento constante, **nada en el entrenamiento presionó jamás a que
la magnitud del matcheo signifique «está»**. Es exactamente la propiedad que el recuperador externo
tiene gratis y que meter el índice adentro de la red destruye.

**4.2 · El error no es de escritura.** Sonda del vecino (n=4000): en los casos de `err_identidad` el
vecino —el hecho dueño del valor contestado— está **intacto en 0,8301**; el vecino corrupto explica
sólo 0,0482 y 0,1186. Contra su tasa de fondo entre los aciertos (0,07-0,09), un checkpoint da 1,68×
y **el otro va al revés**: no hay evidencia de que la corrupción de escritura tenga relación con el
error.

**4.3 · Tampoco se pierde la corrección: se pierde el hecho entero.** Si la revisión no se hubiera
ligado pero el hecho estuviera archivado, la versión vieja debería recuperarse bien. No ocurre: en los
casos fallidos la v1 está **también** degradada (0,5304 y 0,3193) contra **0,9498 y 0,9119** en los
casos donde el modelo acierta.

**Conclusión convergente:** el vecino se recupera, el hecho propio no se recupera **en ninguna de sus
versiones**, y el modelo contesta el valor del vecino. Es **competencia de claves —
direccionamiento**, no escritura.

**4.5 · El modelo enfoca, y el foco no sabe si el hecho está.** Recorriendo **todas** las posiciones
de la consulta, la lectura concentra hasta el **65 %** de la masa en una sola entrada del archivo
(entropía mínima 1,02-1,05 contra un techo de ln(6) ≈ 1,79), en posiciones **intermedias**. En la
posición de respuesta la distribución ya está difusa porque el estado recurrente integró lo leído.

Como el foco vive en otra posición que la usada en §4.1, la señal de ausencia podría vivir ahí. Se
re-midió (n = 4000, dos checkpoints):

| dónde se toma el score | `n4_s0` | `n3_s2` |
|---|---:|---:|
| `pos_q` *(réplica de §4.1)* | 0,4984 | 0,5022 |
| **posición de máximo foco** | **0,5007** | **0,5077** |
| máximo sobre todas las posiciones | 0,5293 | 0,5429 |

**Donde el modelo más concentra, el score sigue en el azar.** El resultado de §4.1 no es un artefacto
de la sonda, y queda enunciado con precisión:

> **El modelo selecciona una entrada con fuerza, y la fuerza de la selección no codifica si lo que
> buscaba está. Siempre encuentra algo, y encuentra con la misma convicción cuando no hay nada que
> encontrar.**

Es la consecuencia exacta del mecanismo candidato: con un softmax que suma 1, la mejor entrada gana
con masa comparable **haya o no haya un buen candidato**. Un recuperador externo, en cambio, tiene el
conjunto vacío como estado posible — y de ahí sale la asimetría que reporta arXiv 2606.11712.

**4.4 · El error es determinista.** Reformular la consulta acierta 0,958-0,978 donde el modelo ya
acertaba y sólo 0,052-0,105 donde fallaba: **cuando se equivoca en un episodio, se equivoca siempre**.
No duda entre dos candidatos — está comprometido con una asociación. Eso predice que el error llegue
con **alta confianza**, y explica el techo de la mitigación.

## 5. Mitigar sin reentrenar, y dónde se rompe

La confianza de salida separa aciertos de errores con **AUC 0,8631**. Con umbral calibrado en una
mitad y medido en la otra, contra el piso de abstenerse al azar: a cobertura **0,78**, el SER cae
0,2287 → **0,1073** (53,6 % de los errores silenciosos eliminados, 1,68× el azar).

**Pero se quiebra en el caso central.** Evaluado con preguntas sin respuesta, el AUC cae a **0,7397**,
el invento se apaga sólo **28,8 %** y la ventaja sobre el azar baja a 1,16×. Coherente con §4.1: si la
ausencia no tiene representación en la interfaz de memoria, el umbral no puede leerla.

## 6. Lo que se sigue

1. **Un slot nulo aprendido** —clave `k∅` con valor `NOSE` compitiendo en la recuperación— no es un
   detector sino **el marco de referencia que le da sentido a la magnitud del score**: reintroduce el
   «no devolvió nada» que el softmax eliminó. §4.5 muestra que la competencia es real —la masa se
   concentra hasta 0,65 en una entrada— así que un slot nulo puede efectivamente ganarla cuando nada
   matchea. Precedentes a citar, no a redescubrir: *pointer sentinel* (Merity et al. 2016) y el score
   de no-respuesta de SQuAD 2.0.
2. **Y no alcanza con leer: hay que crear la señal.** §4.1 refuta la hipótesis optimista de que la
   separación ya existiera y sólo faltara darle salida.
3. **El direccionamiento es un blanco distinto del `NOSE`** y §4.2-4.4 lo aíslan: por qué la clave de
   un hecho pierde sistemáticamente contra la de su vecino en ciertos episodios.

## 7. Límites

- Idioma cerrado de 242 tokens y respuesta de un token. Es **la interfaz de la prueba, no el objeto de
  estudio**: es lo que permite un SER exacto sin juez LLM. No se afirma transferencia a lenguaje
  natural abierto.
- Un checkpoint por nivel en N3 y N4; las tres semillas están en curso. Con bimodalidad ya medida en
  N2 (0,8028 contra 1,0000), un valor solo no distingue dificultad de no-convergencia. **Los números
  de N3/N4 son provisorios.**
- Todos los checkpoints se entrenaron con `p_nose = 0`: **nunca vieron una pregunta sin respuesta**.
  Cualquier señal de ausencia medida acá es **incidental, no supervisada** — y por eso el negativo de
  §4.1 acota la hipótesis del mecanismo, **no** la del entrenamiento.
- Los dos checkpoints difirieron en magnitud en las dos sondas. Se reporta por checkpoint y **no se
  promedia**.
