# El bloqueo del 9-ago era un bug del tokenizador, no truncamiento

**2026-08-10.** Queda identificada la causa de la detención documentada en `INFORME_FINAL_20260809.md`,
y **el diagnóstico que había anotado ahí era equivocado**.

## Lo que decía el informe anterior

> «No es una propiedad del modelo: es un defecto de procesamiento, casi con seguridad **truncamiento
> del texto** (la entidad va al principio y sobrevive; el valor va al final y se pierde).»

Eso no se sostiene: los textos tienen 9 palabras. Nada se truncaba.

## La causa real

**`nomic-embed-text` en Ollama colapsa a un ÚNICO vector todo token que empiece con mayúscula.**

Medido localmente con el modelo bajado (`ollama pull nomic-embed-text`), cosenos entre vectores
normalizados:

| par | mayúscula | minúscula |
|---|---|---|
| Helios / Vantor | **1,000000** (bit a bit idénticos) | 0,407 |
| Laboratory / Foundry | **1,000000** | 0,480 |
| Ana / Beto | **1,000000** | 0,349 |
| Rotterdam / Sapporo | **1,000000** | 0,370 |
| Norvik Steel / Ardent Optics | **1,000000** | — |
| cat / dog (control, ya en minúscula) | 0,603 | 0,603 |

Y la misma palabra en sus dos formas **no** coincide: `Laboratory` vs `laboratory` da 0,445. O sea, la
forma capitalizada no es «otra versión» del token: es un vector degenerado compartido por todas.

**No es del modelo original ni del endpoint.** `/api/embeddings` (deprecado) y `/api/embed` (nuevo),
en llamada suelta y en lote, dan lo mismo; y `albert:v4.0` sobre los mismos textos da 0,605, así que
la máquina y el servidor están sanos. Es específico de ese modelo tal como lo sirve Ollama.

## Por qué encaja con todo lo observado

- **75 vectores únicos entre 3000.** Lo único que variaba sin mayúsculas eran las 5 plantillas de
  atributo y los 15 sufijos numéricos: **5 × 15 = 75**, exacto. Prefijos, tipos y valores —todos
  capitalizados— colapsaban.
- **`E1 == E2` bit a bit en 3000/3000:** v1 y v2 sólo difieren en el valor, siempre capitalizado.
- **`Helios Laboratory` vs `Helios Laboratory 2` sí difería** (0,817): el `2` no es una mayúscula.
- **La discrepancia de top-1 (0,0200 en la VM vs 0,708 local) queda explicada**: con sólo 75 vectores
  distintos hay empates masivos, y el ranking depende por completo de cómo cada implementación los
  rompe. Deja de ser un misterio y no hace falta perseguirla más.

## El arreglo

**Pasar los textos en minúscula.** Con la frase completa en minúscula:

- distinto valor (misma entidad): cos 0,835 — **discrimina**
- distinta entidad (mismo valor): cos 0,677 — **discrimina**

Bonus: el modelo ya está local, así que los 9000 embeddings se generan en la PC (~27 ms cada uno,
unos 4 minutos) y la tarea deja de depender de Colab.

## La lección, que es la que vale más

El informe del 9-ago cerró diciendo que las compuertas estaban **incompletas**: miraban identificación
pero no discriminación de valores, y usaban AUC donde hacía falta rango. Esto lo confirma y agrega
algo peor: **yo escribí un diagnóstico plausible («truncamiento») y lo dejé asentado sin verificarlo.**
Costó una noche de trabajo apuntando al lugar equivocado, y el chequeo que lo resolvía era de dos
minutos: embeber dos textos que difieren en una palabra y mirar si el vector cambia.

De ahí sale la compuerta de tres partes que ahora va antes de generar cualquier dato:

1. **Discriminación de valores** — v1 vs v2 deben diferir (cos < 0,99).
2. **Discriminación de entidades** — dos entidades distintas deben diferir (cos < 0,99).
3. **Identificación por top-1 y rango**, no por AUC (lección de R7: AUC 0,97 convivía con top-1 0,13).

Y una cuarta, específica de este hallazgo, porque es barata y ya nos costó una noche:
**censo de vectores únicos** — si N textos distintos producen muchos menos de N vectores distintos,
abortar. Es una línea de código y detecta toda esta familia de fallos de un golpe.
