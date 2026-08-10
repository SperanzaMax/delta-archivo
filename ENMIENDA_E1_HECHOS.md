# Enmienda E1 al pre-registro — cambio de sustrato de embeddings

**Fecha:** 2026-08-09. **Estado del experimento original:** DETENIDO por la compuerta (§7).

## Qué pasó

Con `gemma:2b`, la compuerta de identificabilidad falló: **AUC = 0.8937** (el prereg exigía
> 0.95). cos(consulta, su hecho) = 0.8351 contra cos(consulta, otro hecho) = 0.7779 — una
separación de apenas 0.057.

El experimento preregistrado se detuvo ahí y no se analizó ninguna de las cuatro condiciones.

## Exploratorio realizado (declarado como tal, no usado para sostener ninguna predicción)

| variante | AUC | top-1 |
|---|---|---|
| crudo | 0.8937 | 0.0900 |
| centrado | 0.9670 | 0.0940 |
| all-but-the-top (k=1) | 0.9708 | 0.1303 |

Corregir la anisotropía **rescata el AUC pero no el top-1**: 9-13 % de acierto sobre 3.000
entidades. Un archivo montado sobre eso recuperaría el ítem equivocado ~9 de cada 10 veces.

## Diagnóstico

`gemma:2b` es un modelo **generativo**; sus embeddings nunca fueron optimizados para que textos
semánticamente relacionados queden cerca. Con hechos que comparten plantilla, la señal que
distingue una entidad de otra queda por debajo de lo que los textos tienen en común.

Esto además acota el alcance de R12: aquellos 1.000 perfectos usaban wikitext, textos
**naturalmente diversos**. R12 midió un régimen fácil, no la capacidad del espacio en general.

## Cambio

**Sustrato:** `gemma:2b` → **`nomic-embed-text`** (modelo dedicado a recuperación, dim 768).
Se usan los prefijos que el modelo espera: `search_document:` para los hechos y `search_query:`
para las consultas.

**Lo que NO cambia:** la tarea, las cuatro condiciones, las predicciones P1–P4, las métricas,
el margen de 0.02, ε = 0.30, k = 5, las 10 semillas y la compuerta con umbral 0.95.

**Criterio de detención, reafirmado:** si la compuerta vuelve a fallar, el experimento se detiene
de nuevo. Un segundo fallo sería un resultado en sí mismo — que la identificación de entidades
por similitud no es viable en esta tarea con ningún sustrato accesible— y se reportaría como tal
en lugar de seguir cambiando de modelo hasta que dé.

**Registro:** ninguna condición del experimento fue analizada antes de esta enmienda. Lo único
observado son las métricas de la compuerta y el exploratorio de anisotropía de arriba.
