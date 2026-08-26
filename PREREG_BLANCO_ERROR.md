# PRE-REGISTRO · LA CABEZA APRENDE A PREDECIR SU PROPIO ERROR (A5)

**2026-08-26.** Se congela antes de lanzar y antes de mirar un solo número de las unidades nuevas.
Sale de `ALTERNATIVAS_DETECCION_20260826.md` §A5, escrito a las 09:32 —**11 minutos antes** de que
existiera el primer resultado de `PREREG_DOS_DETECTORES.md`, verificable por mtime— y de lo que esa
corrida midió después.

---

## 1. La pregunta

Hoy la cabeza de abstención aprende **«¿hay respuesta en el archivo?»** (`entrenar.py`, la BCE contra
`es_nose`). Pero lo que se le pide en producción es **«¿me voy a equivocar?»**, y no son lo mismo: el
modelo puede tener la respuesta archivada y traer la de otra entidad igual.

> **¿Entrenar la cabeza contra el error, en vez de contra la ausencia, produce un mejor detector de
> alucinación al mismo costo de parámetros?**

## 2. Lo que ya está medido y hace que valga la pena preguntarlo

De `dos_detectores/d1_p3_s*.json`, con el blanco sin contaminar y el nulo en 0,50:

| detector, evaluado sobre «¿me voy a equivocar si contesto?» | `p3_s1` | `p3_s2` |
|---|---:|---:|
| la cabeza **que existe hoy** | 0,7068 | 0,8105 |
| una sonda lineal sobre **el mismo estado que la cabeza lee** | 0,7986 | 0,8590 |
| cabeza + confianza de salida (4 escalares) | 0,8155 | 0,8722 |

**La información está en el estado que la cabeza ya lee, y la cabeza no la usa.** No es un techo de
capacidad: una sonda lineal sobre el mismo vector la recupera. Es el blanco.

Y el chequeo de instrumento del flag (`chequeo_blanco.py`, corrido antes de escribir esto) muestra
**cuánto deja afuera el blanco viejo**: la tasa del blanco `error` es 0,6406 contra 0,5000 del blanco
`ausencia` en el mismo lote. Ese 0,14 son casos donde el modelo se equivoca y su blanco actual
**no los contiene**.

## 3. Diseño

**Condición nueva:** `--abst cabeza --donde pre --blanco error`, nivel 3, `p_nose` 0,4, 26000 pasos,
horizonte 26000, semillas 0/1/2.

**Control:** `p3_s0/s1/s2`, **ya corridos**, idénticos en todo salvo `--blanco ausencia`. Mismo
generador, mismas semillas, mismo presupuesto. No hay que re-correr nada del control.

**Costo:** 3 unidades de pool. **Avisos: 3 rotadores, no 8** — arranque y cierre de tramo, nada más
(`INCIDENTE_AVISOS_20260824.md`).

**Instrumento ya validado** (`chequeo_blanco.py`, 4 de 4):
B-1 con el default la pérdida es **idéntica** a la fórmula anterior (maxabs 1,79e-07, ruido de fp32),
así que los controles ya corridos quedan protegidos; B-2 con `error` es distinta; B-3 el blanco vale 1
en todas las preguntas sin respuesta, por definición; B-4 el gradiente llega a la cabeza en las dos.
Y la guarda de reanudación está puesta, por la misma razón que las de `donde` y `mezcla`.

## 4. LA MÉTRICA, y por qué la compuerta de siempre NO sirve acá

Esto es lo más importante del pre-registro y se fija **antes** de ver nada.

La compuerta histórica (`nose` ≥ 0,50 y `falsa_abst` ≤ 0,10) **es injusta para esta condición, por
construcción**. Con blanco `error` la cabeza se activa también en preguntas que **sí** tienen
respuesta pero donde el modelo la erraría. `falsa_abst` cuenta eso como falsa abstención, cuando es
justamente la abstención **correcta** que la condición existe para producir.

> **Usar la compuerta vieja garantizaría que esta condición «falle», y el fallo no diría nada.**

**La métrica principal es el SER a COBERTURA IGUALADA.** Para cada unidad se barre el umbral de la
cabeza hasta alcanzar una cobertura fijada —fracción de preguntas contestadas— y se compara el SER
(error contestado con seguridad) de las dos condiciones **en el mismo punto de cobertura**. Las
coberturas se fijan acá: **0,60 · 0,70 · 0,80**.

Es la comparación que un usuario del sistema haría: *a igual cantidad de respuestas dadas, ¿cuántas
están mal?*

## 5. Predicciones

**E-0 · BLOQUEANTE.** `vigente` ≥ 0,70 en ≥ 2/3. Si el modelo no aprende la tarea, no se lee nada
más — es el error de `post`, que quedó no evaluable por no haber puesto esta guarda.

**E-1 · PRINCIPAL.** SER a cobertura 0,70 **más bajo** que su gemela `ausencia`, por ≥ 0,02
absoluto, en ≥ 2 de 3 semillas.

**E-2 · MECANICISTA.** El AUC del logit de la cabeza sobre el blanco «¿me voy a equivocar?» sube
≥ 0,05 contra la gemela, en ≥ 2/3. Referencias del control, ya medidas: 0,7068 y 0,8105.

**E-3 · NO-DAÑO.** `vigente` no cae más de 0,05 contra la gemela. La cabeza no debe comprarse la
detección rompiendo la recuperación.

**E-4 · RIESGO DECLARADO — el colapso al prior.** El blanco sale del argmax del propio modelo y **se
mueve mientras entrena**. El slot nulo colapsó al prior el 25-ago y eso fue la explicación entera de
su fallo. Se reporta, sin criterio de éxito: media y desvío del logit de la cabeza, y la fracción de
muestras que cruzan el umbral. **Si el desvío del logit es < 0,1 y la media está pegada a
`logit(tasa base)`, el resultado es "colapsó al prior" y no "el blanco no sirve".** Son cosas
distintas y hay que poder separarlas.

## 6. Cómo se lee cada desenlace, escrito ANTES

| celda | lectura |
|---|---|
| E-1 y E-2 cumplen | el blanco era el problema. Va al paper como el resultado que faltaba |
| E-1 no, E-2 sí | la información mejoró pero no se traduce en decisión. Es **calibración**, el mismo techo de siempre |
| E-2 no, con E-4 mostrando colapso | el blanco móvil es inentrenable así. Habría que congelarlo (etiquetar con un checkpoint fijo), y **eso sería otro experimento, no un rescate de este** |
| ninguna, sin colapso | el blanco no era el problema y la vía se cierra |

## 7. Criterio de abandono

> **Si E-1 y E-2 fallan sin colapso al prior (E-4), la vía del blanco se cierra y no se prueba una
> tercera definición de blanco.**

## 8. Lo que no contesta

Sigue siendo **supervisado**. No habilita «el modelo sabe cuándo no sabe». Y no dice nada sobre
escala: 863.730 parámetros, idioma sintético de 242 tokens.
