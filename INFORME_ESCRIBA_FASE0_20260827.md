# INFORME · Fase 0 de `escriba` — no hay señal de recuperabilidad en la escritura

Evalúa el §3 de `PREREG_ESCRIBA.md` (SHA `958ad236…`), congelado el 27-ago **antes** de escribir una
línea de `escriba_fase0.py`. CPU, checkpoints ya entrenados, **cero GPU**: A5 siguió corriendo sin
perder una sola cuenta de Colab mientras esto se medía.

Instrumento: `escriba_fase0.py`, con `sonda()` y `auc()` importadas de `sonda_dos_detectores.py` en
vez de reimplementadas. Su chequeo (`chequeo_sonda_lineal.py`) se corrió **antes** de leer nada y
pasa los cuatro casos (señal fuerte 0,9984 · sin señal 0,4840 · señal débil con clase rara 0,9927 ·
invariancia al orden con empates).

## 1. Resultado

E-1 pedía **AUC ≥ 0,65** en al menos 2 de 3 semillas. n = 3000 muestras por unidad, 1803 entradas
sondeadas, split mitad y mitad, semilla de datos 54321.

| unidad | tasa base | sonda REAL | permutada (E-0) | ciega (§7) | E-1 |
|---|---:|---:|---:|---:|---|
| `p3_s0` | 0,9712 | 0,6392 | 0,5105 | 0,5939 | **NO CUMPLE** |
| `p3_s1` | 0,7926 | 0,5560 | 0,4593 | 0,5093 | **NO CUMPLE** |
| `p3_s2` | 0,8375 | **0,5275** | 0,5437 | 0,5844 | **NO CUMPLE** |

**0/3.** Y en `s2` la sonda real queda **por debajo** de la permutada (0,5275 contra 0,5437), que no
es señal débil sino ruido.

**Los dos controles del pre-registro se comportan**, y es lo que hace que el negativo signifique algo
en vez de ser un instrumento roto:

- **E-0 cumple en las tres.** Con etiquetas permutadas la sonda queda en 0,5105 / 0,4593 / 0,5437,
  todas por debajo de 0,55. Lo medido no es la capacidad de la sonda.
- **Sin fuga de etiqueta.** La sonda ciega —sólo posición de la entrada y tamaño del episodio, ni una
  activación— queda bajo el umbral en las tres. E-1 es interpretable.

## 2. La explicación alternativa, buscada antes del veredicto y descartada

El script sondea `ks[-1]`, la última escritura del hecho consultado. Para una consulta por la versión
**vigente** esa es inequívocamente la entrada a recuperar; para una por la **anterior**, no. Mezclar
las dos podía diluir una señal real y producir este negativo **por culpa del instrumento**.

Se corrió el control restringido a `tipo == 0`, declarado como exploratorio y posterior al
pre-registro:

| unidad | REAL mezclado | REAL sólo vigente | permutada | ciega |
|---|---:|---:|---:|---:|
| `p3_s1` | 0,5560 | **0,5537** | 0,5192 | 0,5445 |
| `p3_s2` | 0,5275 | **0,5215** | 0,5307 | 0,6021 |

**No mueve nada.** La duda queda cerrada y no sostiene el resultado.

Y un detalle que apunta en la misma dirección: en `p3_s2` sólo-vigente la sonda **ciega** (0,6021)
predice **mejor** que las activaciones (0,5215). La posición de una entrada en el archivo dice más
sobre si se va a recuperar que el vector que se escribió en ella.

## 3. La unidad que NO se pudo medir, y por qué eso también es un dato

La Fase 0 se intentó primero sobre `lat2` (`v3_s0`), que era la condición adoptada. **La guarda de
reparto de clases abortó sola**: `lat2` acierta 0,9978 y deja **4 errores en 1803 casos**. Sin clase
negativa no hay nada que separar, y una AUC ahí habría sido un número presentable y sin contenido.

El pre-registro nombraba `p3_*` **y** `v3_*` como fuentes, así que mover el pie a `p3_*` no es una
desviación. Vale anotar el motivo de fondo, que es una buena noticia disfrazada de obstáculo: **`lat2`
resolvió tan bien el direccionamiento que dejó de haber fallas que predecir.**

## 4. Lo que el §6 manda, y se cumple

> **E-1 falla** → la línea se cierra acá y no se entrena nada. No hay señal de recuperabilidad en la
> escritura para que una cabeza lea, y construirla es otro proyecto. Se reporta como la octava vía y
> se suma a las siete, que es un resultado y no una decepción.

**La línea se cierra.** No se corre la condición `escriba`, no se escribe el pre-registro de la
Variante B, y no se le saca una sola cuenta de Colab a A5.

**La octava vía cae donde cayeron las siete.** Las siete tentativas sin etiquetas del
`PLAN_FOCO_20260824.md` aterrizaron entre 0,50 y 0,67 de AUC. Ésta da 0,6392 / 0,5560 / 0,5275, dentro
de la misma banda — y con la diferencia de que **ésta sí tenía etiqueta supervisada**, que era toda su
apuesta. Que ni con etiqueta la señal aparezca es más informativo que las siete anteriores juntas.

## 5. Lo que este informe NO dice

- **No dice que la idea de una cabeza en la escritura sea mala.** Dice que en **este** banco, en el
  vector que `modelo.escribir` archiva, no hay señal **linealmente decodificable** de recuperabilidad.
  Una sonda no lineal podría encontrar algo; el pre-registro pidió lineal y eso es lo que se midió.
- **No dice nada sobre `nose_rel`**, que es donde el §2.3 del pre-registro ubicó el margen. Esta Fase 0
  midió recuperabilidad, no detección de ausencia de la relación. Ese agujero sigue abierto.
- **No cierra la pregunta de Maxi, la cierra para esta vía.** El movimiento que propuso —separar la
  verificación en su propia cabeza— es el mismo que ganó en el trípode. Lo que este informe establece
  es que **del lado de la escritura no hay de dónde agarrarse**, y que si esa cabeza va a existir
  tendría que crear la representación en vez de leerla.
