# La ventana decide qué se puede APRENDER, y lo aprendido se usa donde la ventana no llega

**2026-09-02, cierre del día.** Tres campañas, y el arco entero cambió de forma dos veces.

## 1. El resultado, en una línea

**No hace falta que la parte discriminante de la pregunta entre en la ventana en todas las consultas.
Alcanza con que entre en algunas.** Después el modelo aplica lo aprendido incluso en las preguntas
donde su búsqueda es literalmente ciega.

## 2. Las tres condiciones, todas con kernel 3 y evaluadas en la forma `directa`

En la forma `directa` la relación queda a distancia 3 y el alcance es 2, así que **la búsqueda no la
ve nunca**, con sensibilidad `0,000000` exacto.

| condición | qué ve la query | `nose_rel` en `directa` |
|---|---|---|
| `v3` · una sola forma | la relación **nunca** | 0,6090 · 0,5850 · 0,7349 |
| `cf3` · `directa` + `invertida` | la relación **a veces**, en la invertida | **1,0000 · 0,9625 · 1,0000** |
| `cl3` · `directa` + `lejana` | la relación **nunca**, en ninguna de las dos | **0,5370** |

`cl3` es el control que adjudica y **se queda pegado al piso**: 0,5370 en `directa` y 0,5630 en
`lejana`, con `vigente` 1,0000 en las dos y `nose_ent` 0,9907 y 0,9610. O sea aprendió la tarea y
resuelve la ausencia fácil; lo único que no aprende es el caso que necesita la relación.

**La diversidad por sí sola no compra nada.** Lo que compra es que la relación entre en la ventana en
alguna de las formas.

**Legibilidad, declarada:** `cl3` va **1 de 3** semillas y el prereg pide 2, así que formalmente es
**NO EVALUABLE**. La segunda, a 5000 pasos, da 0,5912 y 0,6542, consistente con la primera. `cf3`
está cerrado 3 de 3.

## 3. El cruce falló, y falló de la forma más informativa posible

`PREREG_CRUCE_FORMAS.md` predecía que al dar vuelta la pregunta se daría vuelta cuál componente falla.
**X-1 y X-2 dan 0 de 3.** X-3 cumple 3 de 3, así que el montaje es legible y lo que falló fue el
pronóstico.

Y sin embargo la sensibilidad de la búsqueda **sí se da vuelta, exacto**:

| forma | sensibilidad a la entidad | sensibilidad a la relación |
|---|---:|---:|
| `directa` | 0,742 | **0,000000** |
| `invertida` | **0,000000** | 0,267 |

Las dos puntas medidas dan la disociación completa:

- **la ventana manda en la búsqueda**, y se invierte cuando se reordena la pregunta;
- **la ventana no manda en la abstención**, que no se invierte.

El error fue escribir la predicción sobre la métrica equivocada. La abstención no se computa en la
búsqueda, se computa aguas abajo, y por eso la geometría de la consulta no la determina. **Es el
tercer criterio del día con esa misma forma**, y el patrón ya no admite discusión: hay que preguntarse
*si la intervención funcionara perfecto, ¿esta métrica se mueve?* antes de congelar nada.

## 4. Un bug encontrado en el propio juez, y qué se salvó

Las métricas de la forma `lejana` salían **`nan` en silencio**. `datos.lote` guarda el índice de la
forma respecto de la lista **global**, y `evaluar` lo leía respecto de la lista **local** de cada
campaña. Con `directa` e `invertida` los dos órdenes coinciden por casualidad y no se notaba; con
`directa` y `lejana` no coinciden y esa columna entera se perdía.

El número principal **no está afectado**, porque `directa` es índice 0 en las dos listas. Se corrigió
traduciendo por nombre en vez de por posición.

## 5. Kernel 7, dos de tres

| unidad | `vigente` | `nose` | falsa |
|---|---:|---:|---:|
| `k73_s0` | 1,0000 | 1,0000 | 0,0000 |
| `k73_s1` | 0,9964 | 0,9667 | 0,0033 |
| control kernel 5 | 0,9964-1,0000 | 0,9697-0,9861 | 0,0000-0,0030 |

Por ahora **no hay diferencia legible** entre 7 y 5. La hipótesis de que más ventana ensucia no tiene
apoyo, y la de que mejora tampoco. Falta la tercera semilla.

## 6. Cómo queda el paper

El encuadre correcto ya no es «ensanchá la ventana». Es más preciso y más útil:

> **La ventana determina qué relaciones puede aprender el modelo a detectar. Si una parte de la
> consulta nunca entra en la ventana en ninguna de las formas en que se pregunta, esa parte no puede
> gobernar la abstención, y el modelo responde con confianza. Basta con exponerla adentro en una
> fracción de las consultas, sin tocar un solo parámetro de la arquitectura.**

Eso da dos remedios con mecanismo medido, uno de arquitectura y uno de datos, y el segundo es el que
alguien puede aplicar mañana sobre un modelo que ya tiene entrenado.

## 7. Lo que sigue

- Cerrar `cl3` con las tres semillas, que es lo único que falta para que §2 sea evaluable.
- La tercera semilla de kernel 7.
- El experimento en `mamba-130m`, corriendo, con las condiciones `una` y `dos`. El baseline ya está
  medido y el modelo preentrenado **nunca** dice `unknown`, así que todo lo que aparezca es aprendido.
