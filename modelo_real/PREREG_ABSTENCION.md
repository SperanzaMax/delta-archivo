# Pre-registro · LA ABSTENCIÓN, con el presupuesto ya calibrado

Congelado el 2026-09-09, antes de correr. Es lo último que le falta a la vara del proyecto.

## Por qué recién ahora

La abstención se midió una sola vez hoy, con **2.000 pasos**, y el modelo se fue al **atractor mudo**:
`nose` 1,0000 y `acierto` 0,0000 desde el paso 100. Sobre eso se construyeron dos explicaciones que
después se cayeron, porque el problema era el presupuesto. Tres tareas distintas que a 2.000 pasos
estaban en el azar llegan al techo con 16.000 a 20.000.

Así que la pregunta sigue abierta y ahora se puede hacer bien.

## La vara, que es de Maxi y no se cambia

> «no queremos un modelo callado, queremos uno que responda bien o diga no sé»

    GLOBAL = (acierto + acierto_nose) / n

**El piso a batir es el del MUDO**, que es la fracción de casos sin respuesta. Abstenerse siempre da
exactamente eso. Un modelo por debajo del piso del mudo es peor que el silencio.

## Diseño

15 entidades · 8 valores · **2 relaciones** · batch 4 · archivo 8 · **16.000 pasos** · TinyLlama
congelado, escritura capa 11, lectura capa 2.

    p_dos 0,25 · p_una 0,35 · p_nose_rel 0,20 · p_nose_aus 0,20    ->  piso del mudo 0,4000

Dos brazos que difieren en **una** cosa.

| brazo | calentamiento |
|---|---|
| **`abst`** | ninguno, la abstención desde el paso 1 |
| **`abstcal`** | 3.000 pasos sin casos de abstención antes de habilitarla |

## Hipótesis

- **H1 · no se va al mudo.** `GLOBAL` supera el piso del mudo 0,4000 **y** `acierto` ≥ 0,80. Las dos
  condiciones juntas. Superar el piso callándose más no cuenta.
- **H2 · `nose_rel` es más difícil que `nose_aus`.** En `nose_rel` la entidad **sí** está en el
  archivo, con la otra relación, así que hay un valor tentador a mano y el parecido superficial
  apunta al lado equivocado. Es la alucinación en miniatura. Se predice `nose_rel < nose_aus`.
- **H3 · el calentamiento NO cambia el resultado.** Fue mi remedio de la mañana para un problema que
  era de presupuesto. Si los dos brazos llegan al mismo lugar, el remedio nunca hizo falta y queda
  descartado. **Refutación**: si `abstcal` llega y `abst` se va al mudo, el calentamiento era real.

## Riesgos declarados

1. Una semilla por brazo. Si la diferencia sale chica, hace falta variabilidad antes de concluir.
2. El piso del mudo, 0,4000, está **cerca** del acierto que un modelo a medio aprender puede sacar
   por otras vías. Por eso H1 exige las dos condiciones y no sólo `GLOBAL`.
3. `invento` se reporta desglosado en `invento_rel` e `invento_aus`, porque agregarlos esconde
   justamente el sabor que importa.
