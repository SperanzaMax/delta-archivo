# PREREG · ¿filtrar antes de buscar corre el techo de capacidad? · 2026-09-04

**Congelado antes de implementar y antes de correr.** SHA en `SHA_FILTRADO_PREVIO.txt`.

## 1. De dónde sale

De una pregunta de Maxi el 4-sep, sobre cómo hacer que el archivo se pueda buscar rápido cuando tenga
muchísima información, tomando como inspiración cómo el cuerpo encuentra información en el ADN.

La revisión de esa idea dejó tres cosas, y las tres importan para entender qué se pregunta acá.

**Primero, la velocidad no es el cuello.** Leer el archivo cuesta `N × D` multiplicaciones por token.
Con `D = 128` y un millón de entradas son 128 millones de operaciones, o sea microsegundos en GPU.
Y si algún día lo fuera, el problema ya está resuelto por fuera con búsqueda de vecinos aproximados
(HNSW, IVF, product quantization), y no tiene sentido competir contra ese campo.

**Segundo, lo que sí se rompe es la precisión.** Con `N` grande el softmax reparte su masa entre más
candidatos y la recuperación cae.

**Tercero, y es lo que da origen a esta campaña, la lección del ADN no es sobre velocidad.** Un factor
de transcripción tarda minutos en encontrar su sitio. Lo que hace el cuerpo no es buscar rápido sino
**no tener que buscar**, porque la mayor parte del genoma está físicamente cerrada y la accesibilidad
está precompilada por contexto. Eso es **filtrar antes de buscar**, y es lo único de la analogía que
este banco no está aplicando ya. La otra lección del ADN, la coincidencia combinatoria en vez de una
clave sola, **ya está aplicada**, y es el sello de orden.

## 2. Hipótesis

**H.** Si antes del softmax la lectura se restringe a un subconjunto de entradas seleccionado por
contexto, el techo de recuperación se corre hacia arriba, porque la masa de atención deja de
repartirse entre candidatos que no podían ser la respuesta.

**H₀.** El techo no se mueve. La dilución no era el mecanismo, o el filtro pierde por falsos negativos
lo mismo que gana por concentración.

## 3. Diseño

Barrido de `N` (entradas del archivo) con dos condiciones a **igual presupuesto de parámetros**.

| condición | lectura |
|---|---|
| **completa** (control) | softmax sobre las `N` entradas, que es lo que hace hoy |
| **filtrada** | softmax sobre las `k` entradas mejor rankeadas por un puntaje barato, con el resto enmascarado |

El filtro tiene que ser **más barato que la lectura misma**, si no no es un filtro. Se fija de
antemano `k` como fracción de `N`, y el puntaje es el producto interno contra la clave sin la
proyección aprendida, o sea el candidato más simple posible. Elegir un filtro complicado antes de
saber si el mecanismo existe sería confundir dos preguntas.

## 4. Criterios, escritos antes del dato

| | criterio | qué decide |
|---|---|---|
| **F-1** principal | la recuperación de la condición filtrada supera a la completa en al menos **0,05** en el `N` más grande del barrido | el filtrado corre el techo |
| **F-2** | la ventaja **crece** con `N` en al menos 3 de 4 tramos | es dilución y no un efecto de tamaño fijo |
| **F-3 riesgo** | los falsos negativos del filtro, o sea la fracción de veces que la entrada correcta queda fuera de las `k`, se mide y se informa **siempre**, cumpla o no F-1 | sin esto una ventaja puede ser un artefacto del muestreo |
| **F-4 costo** | el filtro cuesta menos que la lectura que evita | si no, no es un filtro, es otra lectura |

**Regla de lectura.** F-1 y F-2 son la hipótesis. **F-3 se informa siempre**, y si los falsos
negativos superan el 5 % la ventaja no se atribuye al mecanismo sino al filtro habiendo tenido suerte
con el banco.

## 5. Abandono

Si F-1 falla y F-3 muestra falsos negativos bajos, entonces **la dilución no era el mecanismo del
techo**, y eso es un resultado en sí mismo que hay que informar como negativo. No se prueba un
segundo filtro más elaborado para rescatar la hipótesis.

## 6. Relación con TELAR-03, declarada y NO asumida

TELAR-03 tiene un plateau de ~67 % ya establecido como **techo de capacidad**, porque el oráculo en
forma cerrada lo reproduce sin gradiente, y su Fase 2 es un barrido `d × n` ya planificado.

⚠️ **Son dos bancos distintos y no se asume que sea el mismo fenómeno.** TELAR-03 mide capacidad en
atención lineal; acá se mide recuperación desde un archivo persistente con clave aprendida. Que los
dos tengan un techo no prueba que sea el mismo techo. **Si esta campaña sale, la comparación entre los
dos bancos es un experimento aparte y hay que diseñarlo como tal**, no leerlo de estos resultados.

## 7. Lo que esta campaña NO puede decidir

- **No dice nada sobre velocidad**, que es lo que motivó la pregunta original y que ya quedó
  establecido que no es el cuello.
- **No valida la analogía del ADN.** Prueba un mecanismo que la analogía sugiere. Sigue el precedente
  del paralelismo del GPS, donde la forma fuerte quedó refutada y la débil funcionó, así que **acá se
  prueba directamente la forma débil y falsable**.
- **No es una arquitectura de recuperación.** Si el mecanismo existe, recién ahí tiene sentido
  preguntarse cómo se elige el subconjunto, que es donde la eviction sorpresa-gated de VIGÍA-03 y la
  literatura de vecinos aproximados tienen algo que decir.
