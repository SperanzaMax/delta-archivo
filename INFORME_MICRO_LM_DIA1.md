# MICRO-LM, día 1: el modelo existe y contesta — y tres de los cuatro niveles midieron mi padding

**2026-08-13/14.** Primera jornada del modelo entrenado desde cero con el archivo en la arquitectura,
que es la vara que puso Maxi: *«un modelo de cero con esto incorporado en su ADN, no algo adosado al
LLM, que no olvide lo que lee ni lo que hablamos»*.

Diseño en `DISENO_MICRO_LM.md`. Código en `micro_lm/`. Corridas en `micro_lm/corridas_20260813/`.

---

## 1. Lo que se construyó

| pieza | qué hace |
|---|---|
| `idioma.py` | idioma cerrado de **242 tokens**: 100 números, 58 nombres, 29 entidades, 6 relaciones, funcionales. Legible por un humano. |
| `datos.py` | episodios multi-sesión → tensores. Respuesta = **un solo token**. |
| `modelo.py` | delta rule + **archivo co-entrenado** + sello de orden, inyección temprana. Autocontenido. |
| `entrenar.py` | entrena igual en CPU y GPU. |

**863.730 parámetros · 3,5 MB.** Entra sobrado en Colab: T4 a 0,22 s/paso, 20000 pasos en ~73 min.

## 2. Resultado, y por qué sólo una fila es legible

| nivel | vigente | anterior | **enunciados truncados** |
|---|---|---|---|
| 1 plantilla fija | 0,6707 | 0,6914 | **33,9 %** |
| 2 paráfrasis | 0,5920 | 0,6341 | **33,6 %** |
| 3 elipsis | 0,6913 | 0,7218 | **33,6 %** |
| **4 multi-sesión** | **0,9881** | **0,9922** | **1,5 %** |

**El techo de los niveles 1-3 no era aprendizaje: era padding.** En esos niveles todos los hechos caen
en la misma sesión, y el armador de lotes cortaba a `E_MAX = 4` enunciados y `T_SES = 40` tokens. Un
tercio de lo dicho **nunca llegaba al archivo**, así que la tarea era literalmente irresoluble en esa
proporción.

El número cierra solo: **1 − 0,339 = 0,661** contra la accuracy medida de **0,6707**. No es
aproximado, es la misma cantidad. El nivel 4 reparte los hechos entre sesiones, truncaba el 1,5 %, y
por eso quedó limpio.

> **Séptima vez en el programa que un número limpio esconde un artefacto**, y la segunda en que el
> artefacto es del instrumento propio. Lo que lo delató fue que el nivel *más difícil* diera 0,988 y
> el *piso* 0,67: cuando el orden de dificultad sale al revés, el problema es del instrumento.

Corregido a `T_SES = 96`, `E_MAX = 10` → truncamiento **0,0 %** en los cuatro niveles. Los niveles 1-3
hay que **re-correrlos**; sus números actuales no significan nada sobre el modelo.

## 3. Lo que sí quedó medido, y no es poco

El nivel 4 es el más exigente del diseño: **paráfrasis + corrección elíptica + sesiones separadas**,
con el estado reseteado entre sesiones. Lo único que sobrevive de una sesión a la siguiente es el
archivo.

**0,9881 en la versión vigente · 0,9922 en la anterior**, sobre un vocabulario de 242 tokens (azar
0,004).

Dicho en la frase de Maxi: **el modelo contesta correctamente sobre algo que se le dijo en una sesión
anterior, cuyo estado ya no existe, incluso cuando eso fue corregido después.** Con un modelo de
3,5 MB entrenado desde cero.

**Precaución antes de festejar:** el nivel 4 fue el único que corrió sin el artefacto, así que su
0,988 todavía no tiene con qué compararse. Cuando 1-3 se re-corran limpios, si dan *menos* que el 4
habrá que explicar por qué el más difícil es el más fácil — y la explicación probable es que repartir
los hechos entre sesiones **baja la densidad por sesión**, o sea que el nivel 4 es más fácil en un eje
mientras es más difícil en otro. Eso hay que medirlo, no suponerlo.

## 4. Un patrón que aparece en los cuatro niveles

En **todos**, la pregunta por la versión **anterior** sale igual o mejor que por la vigente:

| nivel | vigente | anterior | diferencia |
|---|---|---|---|
| 1 | 0,6707 | 0,6914 | +0,0207 |
| 2 | 0,5920 | 0,6341 | +0,0421 |
| 3 | 0,6913 | 0,7218 | +0,0305 |
| 4 | 0,9881 | 0,9922 | +0,0041 |

Es el modo de falla de **R1** —«la geometría recupera la más vieja»— y el **atajo de la recencia
invertido** de E-I3d. El modelo tiende a preferir la primera entrada archivada en vez de comparar
turnos. En el nivel 4 la diferencia casi desaparece (+0,004), lo que sugiere que **con datos completos
el sesgo se diluye** — otra razón para re-correr 1-3 antes de sacar conclusiones.

## 5. Abstención: 0,0000 en los cuatro

El token `NOSE` existe y **nunca se usa**. El modelo siempre contesta algo. Todos sus errores son
silenciosos, que es exactamente el modo de falla que el preprint recién publicado
(DOI 10.21203/rs.3.rs-10669947/v1) mide en los sistemas desplegados. Para que aprenda a abstenerse
haría falta que el entrenamiento incluya preguntas **sin respuesta en el archivo**, que hoy no existen.

## 6. Para mañana

1. **Re-correr los niveles 1-3** con `T_SES=96` / `E_MAX=10` (el bug ya está corregido en `datos.py`).
2. Comparar con el nivel 4 y explicar el orden de dificultad que resulte.
3. **Semillas**: hay 8 cuentas de Colab vivas; una sola semilla no distingue un resultado de una
   casualidad.
4. Agregar **preguntas sin respuesta** para que `NOSE` tenga sentido y el error deje de ser silencioso.
5. Guardar los pesos, para poder mostrar diálogos concretos: pregunta, respuesta del modelo, acierto.
