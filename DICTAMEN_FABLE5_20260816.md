# Dictamen de Fable 5 sobre el `NOSE` — qué se acepta, qué se corrige, qué se hace

**2026-08-16** · respuesta a `CONSULTA_FABLE5_NOSE.md`, verificada contra el código antes de asentar.

Regla de la casa: un dictamen externo no entra al programa sin pasar por su control. Dos de sus
afirmaciones se verificaron contra `mitigar.py`, `idioma.py` y `modelo.py`. **Una es correcta y nos
obliga a corregir un informe publicado internamente; la otra es falsa para nuestra arquitectura y
nos ahorra rediseñar la compuerta.**

---

## 1. Lo que hay que aceptar sin descuento

**El objetivo estaba mal enunciado, y el documento lo sabía mejor que el mensaje.** «Eliminar las
alucinaciones» no es lo que el instrumento permite afirmar. La abstención no elimina el error: lo
convierte de silencioso en avisado y compra una posición en la curva riesgo-cobertura. Nuestra propia
tabla lo dice — SER 0,0080 recién a cobertura 0,51, o sea callándose la mitad de las preguntas. **El
producto honesto es «SER X a cobertura Y, con la abstención calibrada», no «no alucina».**

**Y no se puede afirmar «eliminado», sólo acotarlo.** Con n preguntas sin respuesta y cero inventos
observados, la cota superior al 95 % es ≈ 3/n (regla de tres): para sostener «invento < 0,1 %» hacen
falta ~3000 preguntas y cero fallos. «Eliminado» siempre significa **«bajo el umbral de detección de
mi muestra»**, y dicho así nadie lo puede tocar.

**Las tres clases de alucinación, que son tres fenómenos con tres destinos distintos** — esta es la
mejor pieza conceptual del dictamen y reordena todo el plan:

| clase | qué es | destino |
|---|---|---|
| **1 · invento** | pregunta sin respuesta, contesta igual | **eliminable** hasta el límite de detección: la ausencia tiene firma (ninguna clave matchea) |
| **2 · confusión al leer** | dos claves compiten, margen bajo | **convertible** en abstención; el techo lo pone el AUC de la señal, no la voluntad |
| **3 · memoria falsa por escritura corrupta** | la corrección elíptica se ligó al dueño equivocado **al escribir** | **indetectable por abstención, por diseño**: al leer, el hecho falso tiene matcheo alto, margen alto y entropía baja — es mecánicamente idéntico a uno verdadero |

La clase 3 explica un número que teníamos sin explicar: el umbral apagó el **48 %** de los errores de
identidad. La mitad que carga señal es clase 2; **la otra mitad es probablemente clase 3, y ninguna
cantidad de abstención la va a tocar.** Eso asciende a P5 (escribir vs. leer) de «experimento
siguiente» a «el que decide qué fracción del 22 % es siquiera convertible».

**El resto que se acepta, en corto:**
- **P1** — (c) y (d) eran la misma respuesta: la pérdida auxiliar «¿está esta clave?» sobre el score
  de recuperación **es** la cabeza de abstención cuando se la usa en inferencia. El diagnóstico
  estructural es correcto y mejor que el nuestro: `NOSE` como token **entrelaza dos decisiones en un
  solo softmax** —binaria y ~balanceada («¿está?») contra 1-entre-100 («¿qué valor?»)— y esa
  asimetría de masa **es** lo que genera las dos cuencas. El currículum ataca el *timing*; esto ataca
  la *estructura*. Supervisión gratis (el generador sabe si hay respuesta) y pérdida de valor
  enmascarada en las preguntas sin respuesta.
- **P4** — entrenar a `p_nose = 0,4` **le hambrea datos a la cabeza de valores**: el 40 % de los pasos
  no le enseña nada al mecanismo que más cuesta entrenar. Con cabeza separada se repondera la pérdida
  binaria y se mantiene `p_nose` realista (0,1–0,2). No se nos había ocurrido y es correcto.
- **El costo asimétrico va en el umbral, no en la pérdida** — con score calibrado son la misma
  palanca (regla de Bayes), y el umbral es gratis y revisable después de entrenar.
- **P10** — AURC sobre la curva riesgo-cobertura como escalar titular; el modelo todo-`NOSE` colapsa a
  cobertura 0 y no puede gamearla. Y para los costos: **no elegirlos, barrerlos** — si la curva domina
  en todo ratio, no hay nada que justificar.
- **P2** — el control letal es el **gemelo de permutación de etiquetas**: entrenar a `p_nose = 0,4`
  asignando `NOSE` al azar al 40 % de las preguntas, tenga o no respuesta. Hereda el prior sin la
  señal. Si no le ganamos claramente en `nose_rel`, aprendimos frecuencia. 45 minutos.
- **P11** — falta el control de **archivo ablacionado** (ceroado o aleatorizado en evaluación): la
  señal de ausencia tiene que **morir**. Si sobrevive, está leyendo otra cosa.
- **P7/P12** — `err_fuera = 0,0000` es mitad real (un contestador al azar daría > 0: la mayoría de los
  valores no están archivados en un episodio dado → la salida está confinada al contenido archivado)
  y mitad construcción (vocabulario cerrado, respuesta de un token). La frase defendible: **«en este
  régimen, la alucinación se reduce a mala atribución de contenido real»** — una lente, no un logro.
  Y en escala: defender el idioma cerrado como aislamiento de variables, y comprar barato la versión
  **held-out** (entrenar con ~40 de los 60 nombres, evaluar ligadura y abstención sobre los 20 nunca
  vistos) → convierte «memorizó pares nombre-slot» en **sistematicidad**, con cero trabajo de
  tokenizador.
- **Literatura que hay que citar y no redescubrir con GPU:** el slot nulo es el **pointer sentinel**
  (Merity et al. 2016) y el **no-answer score de SQuAD 2.0**; el umbral sobre el score es **OOD por
  energía** (Liu et al. 2020) en la interfaz de memoria. Más predicción selectiva (Geifman &
  El-Yaniv 2017; SelectiveNet; Deep Gamblers 2019), auto-conocimiento (Kadavath 2022; Azaria &
  Mitchell 2023; entropía semántica), y entrenamiento de rechazo (R-Tuning).

---

## 2. CORRECCIÓN 1 — su crítica al informe es sana, pero el número que cita no la prueba

**Lo que dice:** que `INFORME_MITIGACION_20260815.md` se refuta a sí mismo — afirmamos «la ausencia
no tiene representación propia» y a la vez reportamos AUC 0,7397 sobre preguntas sin respuesta, que
está lejos de 0,5. Conclusión suya: la representación existe, es parcial, y sólo le falta lectura
dedicada.

**Verificado en `mitigar.py:42`:** `auc(v[ok], v[err])` — el AUC separa **aciertos de errores**, no
«preguntas con respuesta» de «preguntas sin respuesta». Y en ese régimen los checkpoints están
entrenados con `p_nose = 0`, así que **en las preguntas sin respuesta nunca aciertan**: todas son
`invento` y entran al AUC **sólo como negativos**.

→ El 0,7397 puede estar sostenido **enteramente por los errores de identidad**, que sí sabemos que
detecta (35,9 % apagado) mientras el invento se apaga apenas 28,8 %. De hecho **la caída 0,8631 →
0,7397 al incorporar los inventos es evidencia de que los inventos son MENOS separables**, no de que
la ausencia tenga señal propia. Ese número no puede decidir la cuestión en ninguna de las dos
direcciones.

**Es el mismo error de método que el propio `mitigar.py:21` tiene escrito como advertencia** (el
12-ago un AUC de 0,97 convivía con un top-1 de 0,13): *se mide lo que decide*.

**Qué queda en pie, que es lo importante:** nuestra frase **estaba sobre-afirmada igual**. No la
medimos — la dedujimos del mecanismo. La redacción correcta es que es una **hipótesis sin probar**, y
hay una medición barata que la decide, que es exactamente la que él propone en P3/P11: **AUC
separando preguntas con respuesta vs. sin respuesta, independiente de si acertó.** Su instinto era
correcto; su evidencia, no. Se corrige el informe en los dos sentidos.

---

## 3. CORRECCIÓN 2 — la compuerta NO hay que rediseñarla: su premisa es falsa acá

**Lo que dice (P8):** que en cualquier nivel de sesión única «todo está en la ventana de entrada, así
que hasta `nose_rel` es resoluble escaneando la entrada sin tocar el archivo», y que por eso la
compuerta necesita plantillas N1 en formato de dos sesiones.

**Verificado en dos lugares:**
- `idioma.py:169` — es cierta la premisa fáctica menor: `s = 0 if nivel < 4 else …`, o sea en N1/N2/N3
  **todos los enunciados van a la sesión 0**.
- `modelo.py:responder()` — **pero la consulta se procesa en un forward separado, con estado limpio, y
  el único acceso al contenido de las sesiones es el `archivo`.** `tronco(params, consulta, lectura,
  bloque)` recibe **la consulta sola**. Las sesiones se consumen en `escribir()`, que devuelve
  vectores; el estado recurrente no sobrevive.

→ **Escanear la entrada no es una opción disponible: el modelo no tiene la entrada.** Fable 5 razonó
sobre un LM estándar con el contexto concatenado al prompt. Acá la separación escritura/lectura es
arquitectónica, y por eso `nose_ent` en N1 **ya es** una operación sobre el archivo. No hay que
construir plantillas de dos sesiones — se ahorra el rediseño.

**Lo que sí se toma de P8, porque mejora el criterio:** exigir el umbral sobre **`nose_rel`** (el caso
que obliga a encontrar la entidad y verificar que la relación no está) en vez de sobre `nose`
agregado, y agregar un **piso de acierto** en las preguntas con respuesta. Nota: el criterio original
(`nose ≥ 0,50` **y** `falsa_abst ≤ 0,10`) **ya bloqueaba** al modelo todo-`NOSE`, que da
`falsa_abst = 1`; el agregado real del piso de acierto es contra un modelo que no abstiene de más
pero contesta cualquier cosa.

---

## 4. TENSIÓN INTERNA en el dictamen, que la Fase 0 puede resolver

Fable 5 promete que la clase 1 (`invento`) es eliminable **porque «la ausencia tiene firma mecánica:
ninguna clave del archivo matchea»**, y a la vez exige que la compuerta se mida sobre **`nose_rel`**.
Las dos cosas no conviven del todo:

- `nose_ent` — la entidad **nunca se nombró** → ninguna entrada del archivo la contiene → sí, la firma
  es «ninguna clave matchea». Limpio.
- `nose_rel` — **la entidad SÍ está archivada**, con otra relación → hay entradas que matchean
  parcialmente y **el score máximo va a ser alto**. La firma de ausencia no es la magnitud del
  matcheo, sino algo más fino (matchea la entidad pero no la relación).

Y hay un detalle de nuestra arquitectura que lo agrava y que **es el argumento mecánico más fuerte a
favor del slot nulo**: en `modelo.py:responder()` la lectura es un **softmax sobre las entradas del
archivo**, o sea **suma 1 siempre**. Hoy no existe la posibilidad de que la lectura devuelva «nada»:
el modelo está obligado a leer algo aunque nada matchee. El `sim` crudo (pre-softmax) sí conserva la
magnitud, y por eso es medible; pero la salida no la ve. El `penal` de `mask_arch` ya implementa el
mecanismo de una columna que no compite — **un slot nulo es agregar una columna aprendida a `ak`/`av`,
del orden de cinco líneas.**

**Predicción propia, para preregistrar:** el AUC del score de archivo va a separar bien `nose_ent` y
mucho peor `nose_rel`. Si se cumple, «eliminar la clase 1» vale para la mitad fácil del problema y el
caso difícil necesita el mecanismo entrenado, no el umbral. **La medición hay que desagregarla por
tipo o va a promediar dos fenómenos distintos** — que es, otra vez, la forma del error que este
programa cometió siete veces.

---

## 5. El plan, con lo corregido incorporado

**Fase 0 · medir antes de construir — hoy, CPU, cero GPU**
1. Borrar los 4 claims huérfanos (`x1_s0`, `x4_s0`, `x4_s1`, `x4_s2`).
2. **AUC del score de archivo** (máximo, margen top-2, logsumexp) sobre los checkpoints existentes,
   separando **con respuesta vs. sin respuesta** y **desagregado por `nose_ent` / `nose_rel`**.
   Predicción preregistrada: score-AUC > 0,7397 agregado, y `nose_ent` > `nose_rel`.
   *Bifurcación:* si da ≥ 0,85, el trabajo es **darle lectura a una señal que ya existe**; si queda en
   ~0,74, el slot nulo tiene que **crear** la separación entrenando.
3. **Sonda del vecino + atribución del delta** sobre errores de N3: después de la corrección elíptica,
   preguntar con consultas no ambiguas por la entidad correcta y por la que confunde. Vecino corrupto
   → falla al **escribir** (clase 3); vecino intacto → falla al **leer** (clase 2). Más: registrar qué
   entrada del archivo recibió la actualización en el paso de la corrección.
   *Bifurcación:* si la corrupción es al escribir, se abre la línea de ligadura en paralelo, porque la
   abstención no va a cubrir ese residuo nunca.

**Fase 1 · el mecanismo** — slot nulo (`k∅` aprendida, `v∅ = NOSE`) compitiendo en la recuperación +
cabeza binaria «¿está la clave?» sobre el score, con pérdida de valor enmascarada en las preguntas sin
respuesta. Entrenar a `p_nose` 0,1–0,2 con la binaria reponderada, **no** con mezcla inflada a 0,4.

**Fase 2 · la compuerta** — `x1` con el criterio corregido: **`nose_rel` ≥ 0,50 ∧ `falsa_abst` ≤ 0,10
∧ piso de acierto** en preguntas con respuesta. Sigue en N1 y en una sola sesión (§3). 45 min de GPU.
Si no pasa, se vuelve a Fase 1; no se sigue.

**Fase 3 · controles antes de creer nada** — gemelo de permutación de etiquetas · evaluación con
archivo ablacionado · barrido de régimen (entrenado a 0,2, evaluado en {0,1 · 0,2 · 0,4}: el AUC debe
quedar estable, la tasa de abstención puede moverse).

**Fase 4 · campaña y cifra** — las 3 semillas base faltantes (`n3_s0`, `n3_s1`, `n4_s2`), después N4 ×
3 semillas con el mecanismo. AURC contra las tres referencias (abstención al azar, baseline de
calibración, oráculo).

**Fase 5 · el residuo** — lo que quede de `err_identidad` con confianza alta es clase 3: se ataca por
el lado de la **escritura**, no con más abstención. Y el mapa 2-D **score × margen** da la abstención
por versión (P9) gratis: score máximo responde «¿está?», margen entre las dos versiones mejor
rankeadas del hecho matcheado responde «¿cuál rige?». **Un mecanismo, dos abstenciones.**

**Sobre P9 (abstención por versión), su recomendación y la aceptamos:** partir el **análisis** ahora,
partir el **token** sólo con datos nuevos. Entrenar un token de abstención sobre ambigüedad de versión
cuando el sello la hace resoluble sería enseñarle a abstenerse en preguntas que nuestra propia métrica
cuenta como `falsa_abst` — no se puede supervisar sin mentirle al instrumento. Lo honesto es un tipo
de pregunta nuevo (`p_ambiguo`: sello ausente o corrupto, dos correcciones en el mismo tick) como
experimento siguiente.

**Criterio de éxito a preregistrar** (estructura suya, números a ajustar): `invento` ≤ 3/n a cobertura
≥ 0,85 con n ≥ 3000 · supresión de invento ≥ 3× el baseline gratis de 28,8 % a cobertura igualada ·
SER total ≤ 0,05 en N4 a cobertura ≥ 0,80 · los controles de Fase 3 fallando como deben.

**Micro-ítem barato que vale la pena:** la semilla trabada de N2 (0,8028) y la ventana de lr filosa
(1e-3 bien, 3e-3 colapso) pueden ser el mismo fenómeno. Un micro-barrido de lr sobre esa semilla dice
si la «bimodalidad» es suerte de optimización — y eso cambia cómo interpretamos nuestra propia regla
de las tres semillas.

---

## 6. La frase que reemplaza al titular

> No es «un modelo sin alucinaciones». Es **un modelo cuyo error o viene avisado, o está acotado y
> localizado en la escritura.**

Esa sí se sostiene con datos.
