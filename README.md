# delta-archivo — memoria persistente sobre la regla delta

Trabajo experimental sobre una pregunta: **¿se puede acoplar un archivo externo a la regla delta para
que lo viejo importante no se pierda cuando el estado recurrente se llena?**

El punto de partida está medido en [telar-ligamento](https://github.com/SperanzaMax/telar-ligamento):
el estado recurrente de tamaño fijo tiene techo de capacidad y la regla delta **sobrescribe** — al
llenarse, lo viejo se pisa. La hibridación con softmax restituye el techo pero paga cuadrático. La vía
que se explora acá es la alternativa: **archivar afuera lo que va a ser desalojado** y recuperarlo por
similitud, con el índice **dentro** de la arquitectura y no como un pipeline RAG pegado al lado.

Autor: Maximiliano Speranza — investigador independiente (ORCID
[0009-0005-0413-8554](https://orcid.org/0009-0005-0413-8554)).

## La idea propia: «gemación»

Al revisar un recuerdo **no se sobrescribe**: se deposita una versión nueva en un lugar **cercano**,
porque la cercanía codifica la correlación. Linaje honorable, no scoop: la memoria distribuida disperso
de Kanerva escribe en direcciones vecinas y lee el vecindario, y el DNC resuelve el enlace temporal con
una matriz N×N que por eso no escaló. **El aporte propio es que la geometría reemplaza la matriz de
enlaces: O(1) por escritura.**

## Qué hay medido (R1–R13)

Detalle completo en `RESULTADOS_GEOMETRIA_20260808.md` e `INFORME_20260808.md`.

- **La geometría agrupa perfecto pero no ordena.** Se recupera la versión más vieja; hace falta un
  metadato para la recencia. Dictamen de diseño: **geometría para agrupar, metadato para ordenar.**
- **El eje por recuerdo es superior al eje global**, y el eje como campo determinista de la posición
  pierde — converge con el resultado independiente de que cualquier sistema de coordenadas aprendido
  junto al modelo es inestable.
- **La deriva del encoder es el obstáculo real.** Entrenando desde cero, el coseno contra el paso 0 cae
  a 0,727 en 25 pasos; el presupuesto se consume en decenas de pasos y el entrenamiento dura miles. Y
  **se acelera cuando el modelo aprende**: aprender *es* mover las coordenadas. Eso explica por qué el
  campo se quedó en caches intra-secuencia.
- **Pero sobre un modelo preentrenado que se afina, la memoria persistente es viable**: coseno 0,882 a
  400 pasos contra 0,207 desde cero, cómodamente sobre el umbral de tolerancia (~0,7).
- **La hipótesis GPS en forma fuerte está refutada** (la deriva no es rígida: tiene tantos grados de
  libertad como ítems), pero **en forma débil funciona**: la métrica correcta es el **rango**, no el
  coseno. Sin corregir, el ítem queda en la posición 12 de 1664; la corrección afín lo lleva a 7.
- **La fusión de cabezas es lo que hace escalar el mecanismo**: cruzar 4 mediciones parcialmente
  independientes lleva recall@1 de 0,209 a 0,502 y el rango mediano a 0.
- **Con un LLM real el espacio es extremadamente anisotrópico** (|cos| medio 0,547; dimensión efectiva
  ~17 de 2048 nominales) **y aun así escala perfecto con una sola medición** — la capacidad la da la
  dimensión *ambiente*, no la de la señal.

**Estado honesto:** lo sólido es **reducir 1664 candidatos a 25 conservando el 95 % de cobertura**, no
memoria exacta de un tiro. Ahí el modelo puede atender a los 25 y resolver por contenido.

## Estado actual (2026-08-10)

La tarea de **hechos versionados** (estilo VersionRAG) está pre-registrada con hash
(`PREREG_HECHOS.md`, `PREREG_HASH.txt`), con desviación y enmienda registradas, y **se detuvo dos veces
por compuerta**, sin analizar ninguna de las cuatro condiciones (`INFORME_FINAL_20260809.md`).

**El bloqueo quedó resuelto** (`HALLAZGO_TOKENIZADOR_20260810.md`): la causa no era el truncamiento que
se había diagnosticado, sino que **`nomic-embed-text` en Ollama colapsa a un único vector todo token que
empiece con mayúscula**. El arreglo es pasar el texto a minúscula, y con eso la compuerta abre con
top-1 0,975 y rango mediano 0.

**Lo que falta:** fijar por enmienda cómo se indexa cada condición —en particular si la entrada nueva
de `gemacion` lleva el contenido de v2 anclado a la posición de v1— congelarla, regenerar los
embeddings en minúscula y recién entonces correr P1–P4.

## Compuerta de admisión de encoders

`compuerta_encoder.py` es la lección de las dos detenciones convertida en código. Cuatro chequeos, y
cualquiera que falle aborta **antes** de generar un solo dato:

| | qué exige | por qué |
|---|---|---|
| **C1** | N textos distintos → ≈N vectores distintos | Una línea de código; detecta de un golpe toda la familia de fallos donde el encoder colapsa tokens. Es el chequeo que hubiera atajado las dos noches perdidas. |
| **C2** | v1 y v2 deben diferir | Sin discriminación de valores la tarea no tiene señal, por buena que sea la identificación. |
| **C3** | dos entidades distintas deben diferir | — |
| **C4** | **top-1 y rango mediano**, nunca AUC | Con AUC 0,97 convivía un top-1 de 0,13. |

## Literatura

`DOSSIER_LITERATURA_20260808.md` documenta el **scoop parcial**: la versión intra-secuencia de la idea
está ocupada por varios trabajos de 2026 (HOLA, HAM, Tensor Cache, Memory Caching). El hueco que queda
es **capacidad medida** (MQAR con N controlado, semillas y potencia) y el cruce que nadie hizo:
**índice persistente entre secuencias + regla delta, co-entrenados**. Los dos obstáculos reales, y la
razón probable de que nadie lo haya hecho, son que el gradiente no fluye por la selección top-k y el
*stale index*.

## Datos

Los arrays binarios están fuera del control de versiones (ver `.gitignore`, que explica cada uno). Son
regenerables desde los scripts, y **dos de ellos son inválidos**: los de `gemma:2b` (compuerta no
superada) y los de nomic sin minúscula (afectados por el colapso de tokens).

## Método

Régimen de pre-registro: predicción congelada con hash **antes** del dato, desviaciones registradas
cuando aparecen, y detención por compuerta cuando el instrumento no sirve — dos veces en este proyecto,
las dos a tiempo, con el costo en horas de cómputo en lugar de un resultado inventado.
