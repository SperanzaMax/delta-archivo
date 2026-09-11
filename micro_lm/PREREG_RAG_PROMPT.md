# Pre-registro (BORRADOR, congelar antes de correr) · RAG EN EL PROMPT contra ARCHIVO · 2026-09-11

Sale de la pregunta 3 de Maxi (`PLAN_PRUEBAS_20260911_TARDE.md` P5): RAG contra archivo, en
tokens, velocidad y exactitud, **en igualdad** de modelo, datos y presupuesto. Hasta hoy la
comparación es aritmética (2×10¹⁴ contra 3×10¹⁰ operaciones para 80B) más lo que E-I1 midió en
agosto: la receta del RAG estándar dentro del modelo —inyección tardía y top-k duro— da 0,0167
contra 1,0000. Falta el brazo que un RAG de verdad usa: **los hechos como texto en el prompt**.

## Los brazos

- **`archivo`**: el modelo de siempre (archivo co-entrenado, lectura en el bloque 0, top-2).
- **`prompt`**: el MISMO tronco sin archivo. Los enunciados del episodio se pegan como texto antes de
  la pregunta (T pasa de 48 a ~370 con 40 enunciados). Es lo que hace un RAG: lo recuperado entra
  como tokens y el modelo lo relee entero.
- **`prompt-topk`**: como `prompt`, pero sólo entran los dos enunciados más parecidos a la pregunta
  según un vector externo (`nomic-embed-text`, como en la transferencia del 10-sep). Es el RAG con
  recuperación, y es la comparación más justa con el top-2.

Mismos datos, mismas semillas, mismo presupuesto de pasos. Se mide exactitud (vigente, anterior,
NOSE), **tokens por consulta**, **tiempo por consulta** en la T4, y memoria.

## Las hipótesis

- **H1 · tokens y tiempo.** `archivo` procesa 48 tokens por consulta; `prompt` ~370; `prompt-topk`
  ~64 más el costo externo de vectorizar y buscar. Predicción: `archivo` es ≥ 5× más rápido por
  consulta que `prompt` en la T4 con archivo de 40, y la brecha crece con el archivo (P3 mide la
  curva contra N).
- **H2 · exactitud con versiones.** Con hechos corregidos («no, es …»), `prompt` y `prompt-topk`
  quedan por debajo de `archivo` en `anterior` y en `NOSE`, porque el texto pegado no lleva sello de
  orden y el modelo tiene que inferir la versión leyendo. Predicción: `archivo` ≥ `prompt` + 0,10
  en `anterior`.
- **H3 · la colisión.** Con 161 entradas viejas de otras conversaciones, `prompt` no cabe (T ≈
  1.400) o cae; `prompt-topk` trae las dos más parecidas, que con colisión son las equivocadas
  (RECUP 0,0117 del 5-sep): predicción ≤ 0,2 contra ≥ 0,9 de `archivo` (el resultado del 11-sep).

## Diseño (a congelar)

`datos.lote` con `modo_prompt` (concatenar enunciados a la consulta; T_Q nuevo), `entrenar.py
--sin-archivo`, brazo `prompt-topk` con `cache_emb_nl.npz`. Desde cero, 3 semillas, 26.000 pasos
para `archivo` y `prompt` (T=370 cuesta ~0,4 s/paso según el piloto del corpus, entra).
