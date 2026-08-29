# PRE-REGISTRO · ¿CON QUÉ FRECUENCIA UNA UNIDAD LOGRA CALLARSE BIEN?

**2026-08-28.** Se congela antes de lanzar y antes de mirar un solo número de las unidades nuevas.

Sale de `HALLAZGO_PUNTO_PROPIO_20260828.md`, que midió sobre archivos que ya estaban en disco lo que
las tres unidades de A5 hacen **en su propio punto de operación**, y encontró algo que la métrica de
cobertura igualada no podía mostrar porque no es su trabajo mostrarlo.

---

## 1. La pregunta

En su punto de operación propio, dos de las tres unidades con blanco `error` se callan casi perfecto:

| unidad | `vigente` | cobertura | SER | `nose` | `falsa_abst` |
|---|---:|---:|---:|---:|---:|
| **b3_s0** | 0,9996 | 0,5948 | 0,00050 | **0,9994** | **0,0000** |
| **b3_s1** | 0,9979 | 0,5940 | 0,00075 | **1,0000** | 0,0008 |
| b3_s2 | **0,6762** | 0,6058 | 0,20375 | 0,6097 | 0,2473 |

Con n=3 no se puede decir si eso es **el modo típico con una falla ocasional** o **una casualidad de
dos semillas**. Es exactamente la pregunta que la varianza medida de este banco impide contestar
promediando, y la §3.1 del informe de A5 ya mostró con Chebyshev que a esta escala la media no
autoriza nada.

> **¿Qué fracción de las unidades entrenadas con blanco `error` alcanza el régimen de abstención
> casi perfecta, y las que no lo alcanzan fallan por la abstención o por la recuperación?**

## 2. Diseño

**Unidades nuevas:** `b3_s3` … `b3_s8`, **seis**, con **exactamente** los flags de la campaña A5 —
`--abst cabeza --donde pre --blanco error`, nivel 3, `p_nose` 0,4, **26000 pasos, horizonte 26000**.
Lo único que cambia es la semilla. Cualquier otra diferencia invalida la comparación con `s0/s1/s2`.

**Control:** las tres `p3_s0/s1/s2` **ya corridas**, sin re-correr. Su punto propio ya está medido y
**ninguna de las tres alcanza el régimen** (`nose` 0,9069 · 0,5382 · 0,7195). Se declara acá, antes
de correr, que el control no se amplía en esta campaña: si el resultado pide más control, eso es otro
pre-registro y no un rescate de éste.

**Cómputo:** dos rotadores, uno con `ACEL=tpu` y otro con `ACEL=t4`, tres unidades cada uno. Los dos
aceleradores se racionan por separado, así que dos rotadores pidiendo cosas distintas se estorban
poco. **Dos rotadores y no más**, por `INCIDENTE_AVISOS_20260824.md`.

## 3. La definición del régimen, fijada ANTES

Un unidad **alcanza el régimen** cuando, en su punto de operación propio (`a > 0`, el umbral con el
que se entrenó, sin calibrar nada):

> **`nose` ≥ 0,99  Y  `falsa_abst` ≤ 0,01**

Los dos números salen de lo observado en `s0` y `s1` (0,9994/0,0000 y 1,0000/0,0008) y se fijan por
debajo de ellos, no encima, para que la definición no sea un molde de las dos unidades que la
inspiraron. `s2` queda fuera por lejos (0,6097 / 0,2473), así que el criterio no es ambiguo en
ninguna de las tres.

**Se mide con `ser_cobertura.py`, campo `propio`**, n=4000 y semilla de datos 54321, idénticos a los
del 27, para que las nueve unidades sean comparables entre sí.

## 4. Predicciones

**T-0 · BLOQUEANTE.** Al menos **4 de 6** unidades nuevas llegan a `vigente` ≥ 0,70. Si menos de
cuatro aprenden la tarea, algo cambió en el régimen de entrenamiento respecto del 26-27 y **no se lee
nada más**: primero habría que explicar por qué.

**T-1 · PRINCIPAL.** **≥ 3 de 6** unidades nuevas alcanzan el régimen del §3.

Fundamento del número, escrito antes: si la tasa real fuera la observada (2 de 3), el valor esperado
en seis es 4. Pedir 3 deja un margen de una unidad para no comprar la hipótesis con el ruido de un
n chico, y sigue siendo incompatible con la lectura pesimista (tasa ≤ 1/6 daría 1 esperado).

**T-2 · CONTRASTE, con el control que ya existe.** Ninguna de las tres `p3` alcanza el régimen —está
medido, 0 de 3. Si T-1 cumple, **la diferencia entre familias es el resultado**: el blanco `error` no
mejora el promedio, **habilita un régimen que el control no alcanza nunca**. Es la lectura correcta de
lo que A5 midió como «más inestable».

**T-3 · MECANICISTA, y es la hipótesis del hallazgo del 28.** En las unidades nuevas que **no**
alcanzan el régimen, se predice que el fallo es de **recuperación y no de abstención**: `vigente` <
0,70 **o** `err_identidad` > 0,02 en el punto propio, en ≥ 2 de cada 3 unidades que fallen.

> Si una unidad falla el régimen con `vigente` ≥ 0,70 **y** `err_identidad` ≤ 0,02, eso es una falla
> de abstención pura y **contradice** la lectura del 28. Se reporta destacado.

**T-4 · RIESGO DECLARADO.** Si T-1 falla con T-0 cumplido, el régimen bueno de `s0` y `s1` era
casualidad de dos semillas, **y el §2 del hallazgo del 28 se retira**. Ese es el desenlace que esta
campaña existe para poder producir.

**Sobre la varianza.** No se promedia. El resultado es un **conteo de unidades**, no una media de
métricas, precisamente porque con esta varianza la media no soporta afirmaciones. Las nueve unidades
se reportan una por una.

## 5. Cómo se lee cada desenlace, escrito ANTES

| celda | lectura |
|---|---|
| **T-1 y T-3 cumplen** | el modelo alcanza el régimen la mayoría de las veces, y cuando no lo alcanza es porque no recuperó. **La abstención no es el cuello de botella: la indexación sí.** Es el resultado, y es material de paper |
| **T-1 sí, T-3 no** | el régimen se alcanza seguido, pero las fallas son de abstención pura. La lectura del 28 se corrige y el problema vuelve a ser el detector |
| **T-1 no, T-0 sí** | `s0` y `s1` eran suerte. **Se retira el §2 del hallazgo del 28**, y la frase del cierre del 27 queda como estaba |
| **T-0 falla** | el régimen de entrenamiento no es reproducible. No se lee nada y se investiga eso primero |

## 6. Criterio de abandono

> **Si T-1 falla, no se prueban más semillas ni otro presupuesto para buscar el régimen.** La
> hipótesis de que el régimen bueno es típico queda refutada con nueve unidades, y el hallazgo del 28
> se reduce a «dos unidades lo lograron», que no sostiene ninguna dirección de trabajo.

## 7. Lo que NO contesta

**Sigue siendo supervisado.** La cabeza se entrena con etiquetas y nada de esto habilita la frase
«el modelo sabe cuándo no sabe». El §8 del `PLAN_FOCO_20260824.md` y su cierre de seis meses siguen
vigentes y esta campaña no los toca.

**No revive A5.** E-1 dio 1/3 y su criterio de abandono se aplicó. Esta campaña no mide SER a
cobertura igualada ni vuelve sobre esa comparación: mide **otra cosa**, que es con qué frecuencia
aparece el régimen.

**Y no dice nada sobre escala.** 863.730 parámetros, idioma sintético de 242 tokens, `p_nose` 0,4
fijo, un solo nivel.
