# Dossier de literatura II — las dos piezas que el dossier del 8-ago no cubrió

Fecha: 2026-08-23. Continúa `DOSSIER_LITERATURA_20260808.md`, que revisó **delta + archivo
recuperable** y concluyó que esa parte «ya está ocupada». Aquel dossier es anterior a las dos
piezas más propias del micro-LM, y un grep confirma que no las menciona ni una vez (cero apariciones
de abstención, alucinación, predicción selectiva o versionado):

1. **el sello de orden co-entrenado** en la clave del archivo (E-I3, 13-ago);
2. **la cabeza de abstención separada** (`PREREG_CABEZA_ABSTENCION.md`, 18-ago).

**Alcance declarado.** Esto es una búsqueda dirigida, no una revisión sistemática. Se leyeron los
abstracts o el cuerpo de los trabajos citados; uno (Wallat et al., ACL 2026) no se pudo descargar y
se resume por su ficha. Un trabajo que use otro vocabulario para lo mismo puede habérsenos pasado.

---

## PIEZA 1 · La cabeza de abstención separada

### 1.1 Lo que ya existe, y es mucho

**El antecedente estructural es viejo y directo: SelectiveNet** (Geifman & El-Yaniv, ICML 2019,
arXiv 1901.09192) ya tiene **tres cabezas: predicción, selección y auxiliar**, entrenadas
end-to-end, donde la cabeza de selección `g(x)` decide si el modelo se abstiene. Separar la
decisión «¿contesto?» de la decisión «¿qué contesto?» **no es nuevo** en clasificación selectiva, y
hay que decirlo así en cualquier paper. Lo que SelectiveNet no es: un modelo de lenguaje, ni tiene
softmax de vocabulario, ni memoria.

**El competidor directo en LM es el token [IDK]** — *"I Don't Know: Explicit Modeling of Uncertainty
with an [IDK] Token"* (arXiv 2412.06676). Y es importante porque **es exactamente la condición
`token` del micro-LM**: agrega [IDK] como una entrada más del vocabulario y desplaza masa de
probabilidad hacia él dentro de la misma cross-entropy.

Tres cosas de ese paper, verificadas leyéndolo:

- **No compara contra una cabeza binaria separada.** Trata la abstención como una decisión de
  vocabulario y ahí se queda.
- **No discute la competencia por la masa de probabilidad.** La mitiga de refilón con un
  hiperparámetro `Π` que capa en 0,5 la masa desplazable a [IDK], para que el token de oro «siga
  siendo competitivo» — o sea que la competencia está, pero se parchea sin nombrarla.
- **Reporta que el método FALLA en modelos muy chicos** (pythia-70m, pythia-160m), y lo atribuye a
  «problemas de precisión numérica con la inicialización de [IDK]».

Ese último punto es el que más nos toca: el micro-LM tiene 863.859 parámetros, o sea que vive en el
régimen donde el método publicado no funciona.

**El resto del campo de abstención en LLM es post-hoc o de prompting**, no arquitectónico: umbral
sobre la confianza, calibración conformal, o pedirle al modelo que diga que no sabe
(Uncertainty-Aware Abstention 2607.04430; I-CALM 2604.03904; Contrastive Decoding with Abstention
2412.12527; y la encuesta de reject option en ACM CSUR 3727633).

**Y hay un resultado del área de RAG que nos concierne directo:** *Sufficient Context*
(arXiv 2411.06037) encuentra que **RAG empeora la abstención** — agregar contexto sube la confianza
del modelo y produce más alucinación en vez de más «no sé». El micro-LM es exactamente un modelo con
archivo + abstención, así que ese es el efecto contra el que está trabajando, y conviene citarlo
como motivación en vez de descubrirlo de nuevo. Ver también RefusalBench (2510.10390).

### 1.2 El hueco que queda, y es defendible

> **Nadie comparó, dentro del mismo modelo con memoria, a igualdad de parámetros y con semillas, la
> abstención como entrada del vocabulario contra una cabeza binaria separada.**

SelectiveNet tiene la cabeza pero no el vocabulario; [IDK] tiene el vocabulario pero no la cabeza, y
nunca las contrasta. El micro-LM corrió las dos pareadas: `cabeza` pasa la compuerta en 4 de 5
unidades donde `token` y `escala` fallan 5 de 5, con **129 parámetros sobre 863.730 (0,015 %)**.

**Y hay algo mejor que el hueco: hay una discrepancia contrastable.** El paper [IDK] explica su
fallo en modelos chicos por la *inicialización / precisión numérica* del embedding de [IDK]. Esa
hipótesis, en nuestro banco, **es la condición `escala`** —renormalizar el vector de NOSE a la norma
media de los tokens de valor, que es la versión más generosa de «el problema es la escala del
vector»— y **falla igual que `token`**. Nuestra explicación es otra y está medida: no es la norma,
es que **dos decisiones comparten un mismo softmax**.

Eso es un resultado que le habla directamente a un paper publicado, con su misma pregunta y otra
respuesta. Es lo más publicable que tiene esta pieza.

**Advertencia honesta:** el techo que encontramos es de **calibración**, no de capacidad (AUC
0,77-0,99), y las cuatro vías de corte sin etiquetas están cerradas. O sea que el claim que se
sostiene hoy es *«separar la cabeza hace aprendible la abstención en este régimen»*, no *«el modelo
sabe cuándo no sabe»*. La segunda frase todavía no se puede escribir.

---

## PIEZA 2 · El sello de orden co-entrenado

### 2.1 El competidor más cercano, y lo que deja afuera

**"Unable to Forget: Proactive Interference Reveals Working Memory Limits in LLMs Beyond Context
Length"** (arXiv 2506.08184) es la coincidencia más cercana **en la tarea**: claves actualizadas
varias veces dentro del contexto, y el modelo tiene que devolver el valor **vigente**, con las
versiones viejas presentes como interferencia. Documenta el mismo atajo de recencia que E-I3d
encontró.

Lo que **no** hace, verificado leyéndolo:

- **No propone ningún arreglo arquitectónico.** Es benchmark y diagnóstico; los propios autores
  declaran que no ofrecen mitigación ni explicación mecánica.
- **Sólo evalúa LLM grandes preentrenados** (GPT-4-1, Claude). Ningún modelo chico entrenado desde
  cero, así que no puede separar «esto lo aprende el modelo» de «esto ya venía en los pesos».
- **Nunca pide el valor ANTERIOR.** Sólo el vigente.

Ese tercer punto es el hueco más limpio que tenemos. El micro-LM pregunta por la versión vieja en el
35 % de las preguntas, y de ahí sale **E-I3b: preferir-lo-último y usar-el-orden son DOS capacidades
distintas, que se aprenden en momentos distintos**. Un banco que sólo pide el valor vigente **no
puede** ver esa separación, porque el atajo de recencia le da la respuesta correcta. Y esa
separación es, además, la que hace posible el experimento escalonado de hoy.

### 2.2 El resto del terreno

- **Conflicto temporal de conocimiento** (When Facts Change, Findings ACL 2026; DYNAMICQA/MULAN
  2603.15892; Knowledge Conflicts survey EMNLP 2024; Right Knowledge Wrong Answer 2606.20959):
  todo esto es conflicto entre **memoria paramétrica y contexto**, resuelto por prompting o medido
  por benchmark. No es dos versiones del mismo hecho dentro de la memoria, y no toca la
  arquitectura. When Facts Change reporta algo útil igual: pedirle al modelo que considere la
  mutabilidad **aumenta la mención del cambio temporal pero no mejora la exactitud** — un desacople
  entre lo que verbaliza y lo que predice.
- **Time-Stamped Language Model** (2104.07635) sí mete marcas temporales en la entrada, pero para
  flujo de eventos (pasado / actual / futuro), no para versionar hechos en una memoria.
- **La familia delta** (DeltaNet; Gated DeltaNet 2412.06464; Gated DeltaNet-2 / EDA 2605.22791, que
  desacopla la dirección de borrado de la de escritura): el delta rule **sobrescribe**, y lo viejo
  **se pierde**. Por construcción no puede contestar «qué decía antes».
- **Fast-weight Product Key Memory** (2601.00671) y el trabajo de memorias híbridas: memoria
  episódica de fast weights, sin sello de orden en la clave ni pregunta por lo superado.

### 2.3 El hueco

> **Nadie pone un sello de orden co-entrenado en la clave de un archivo persistente, y nadie
> pregunta nunca por el valor superado.**

Es un hueco doble —mecanismo y medición— y las dos mitades se sostienen solas. El número que lo
respalda ya está: el sello levanta el conflicto de versiones de **0,4570 a 0,9956** (E-I3), y sin él
el modelo se queda en el azar.

---

## 3. Qué hacer con esto

**Lo que NO se puede decir:** «una arquitectura nueva». La regla delta es de Schlag/Schmidhuber, el
híbrido conv+linear-attention es Mamba/RWKV/GLA, y delta+archivo co-entrenado ya lo hicieron HOLA,
HAM, Tensor Cache y Google entre marzo y julio de 2026.

**Lo que SÍ se puede defender, en orden de filo:**

1. **La cabeza contra el token, pareada y con semillas** — con la discrepancia explícita contra el
   paper [IDK], que atribuye a la inicialización un fallo que nosotros medimos que no es de
   inicialización. Es el más publicable.
2. **Preguntar por lo superado** — E-I3b separa dos capacidades que ningún banco actual puede
   distinguir, porque todos preguntan sólo por el valor vigente. Y el sello de orden es el mecanismo
   que las une.
3. **La metodología** — sigue siendo la ventaja del 8-ago y ahora es más fuerte: el subcampo corre
   single-seed, y acá hay prereg, semillas, controles pareados y `stoppower` publicado. *Unable to
   Forget* declara que no explica el mecanismo; nosotros tenemos la colisión de clave medida y su
   causa (la query es función pura del token).

**Lo que hay que verificar antes de escribir cualquiera de las tres:** que no exista un trabajo con
otro vocabulario para lo mismo —«rejection head» en LM, «versioned memory», «fact supersession»— y
que HOLA no haya agregado abstención en alguna versión posterior a la de julio.

---

## 4. Fuentes

- SelectiveNet: https://arxiv.org/pdf/1901.09192
- [IDK] token: https://arxiv.org/html/2412.06676
- Unable to Forget (proactive interference): https://arxiv.org/pdf/2506.08184
- Revisiting associative recall in modern recurrent models: https://arxiv.org/abs/2508.19029
- Sufficient Context (RAG empeora la abstención): https://arxiv.org/pdf/2411.06037
- RefusalBench: https://arxiv.org/html/2510.10390
- Uncertainty-Aware Abstention: https://arxiv.org/pdf/2607.04430
- I-CALM: https://arxiv.org/html/2604.03904v1
- Contrastive Decoding with Abstention: https://arxiv.org/html/2412.12527
- Reject option survey (ACM CSUR): https://dl.acm.org/doi/10.1145/3727633
- When Facts Change (Findings ACL 2026): https://aclanthology.org/2026.findings-acl.103/
- Temporal Conflicts (DYNAMICQA + MULAN): https://arxiv.org/html/2603.15892
- Knowledge Conflicts survey: https://aclanthology.org/2024.emnlp-main.486.pdf
- Right Knowledge, Wrong Answer: https://arxiv.org/html/2606.20959
- Time-Stamped Language Model: https://arxiv.org/pdf/2104.07635
- Gated DeltaNet: https://arxiv.org/pdf/2412.06464
- Gated DeltaNet-2 (erase/write desacoplados): https://arxiv.org/pdf/2605.22791
- Fast-weight Product Key Memory: https://arxiv.org/html/2601.00671
