# PREREG · ¿el score de matcheo del archivo separa presencia de ausencia?

**2026-08-16** · Fase 0 del plan de `DICTAMEN_FABLE5_20260816.md`. CPU, checkpoints existentes, cero
GPU. **Este archivo se hashea y se ancla ANTES de correr el script.**

## §1 · La pregunta

La tesis del programa dice que un modelo no distingue **recuperar** de **inventar** porque no hay
diferencia mecánica entre los dos casos. Un archivo co-entrenado tiene un lugar donde esa diferencia
podría vivir: **el score de la consulta contra las claves archivadas**, antes del softmax de lectura.

Nunca lo medimos. Medimos probabilidad, margen y entropía **de la salida**, que es el final de la
cadena.

## §2 · Por qué el 0,7397 del 15-ago no contesta esto

`mitigar.py:42` computa `auc(v[ok], v[err])`: separa **aciertos de errores**. Con checkpoints
entrenados a `p_nose = 0`, en las preguntas sin respuesta el modelo nunca acierta, así que **todas
entran sólo como negativos** y el AUC mezcla dos ejes. Acá el eje es otro y explícito:

> **con respuesta en el archivo (positivos) vs. sin respuesta (negativos)**, independiente de si
> acertó.

Para que la comparación sea legítima, **la confianza de salida se recomputa sobre ESTE mismo eje**.
Comparar el score nuevo contra el 0,7397 viejo sería comparar contra un número de otro eje — el error
que este mismo programa cometió el 12-ago (AUC 0,97 conviviendo con top-1 0,13).

## §3 · Qué se mide

Sobre `ckpts/n4_s0.pkl` y `ckpts/n3_s2.pkl` (los dos que ya se usaron en `mitigar.py`), con
`p_nose = 0.4`, semilla fija, n ≥ 4000 muestras:

En la posición `pos_q` de la consulta, con `sim` tomado **antes del softmax** y con las entradas
vacías ya penalizadas:

- `s_max` — máximo del score sobre las entradas del archivo
- `s_margen` — diferencia entre el primero y el segundo
- `s_lse` — logsumexp sobre las entradas (masa total de matcheo)
- `c_prob`, `c_margen`, `c_entropia` — las tres señales **de salida**, sobre el mismo eje

## §4 · Predicciones (comprometidas antes de mirar)

- **P-1 (principal).** El score del archivo separa presencia de ausencia por encima del azar:
  **AUC(`s_max`) ≥ 0,60**. Es la existencia de la señal mecánica que la tesis postula.
- **P-2 (la que decide el diseño).** El score del archivo separa **mejor que la salida**:
  **AUC(`s_max`) > AUC(`c_prob`) + 0,03**. Si se cumple, la señal está en la interfaz del archivo y
  el trabajo es **darle lectura**; si no, la señal no está más cerca del archivo que del final de la
  cadena y el slot nulo tiene que **crearla** entrenando.
- **P-3 (la tensión interna del dictamen).** La firma «ninguna clave matchea» vale para la entidad
  ausente y no para la relación ausente: **AUC(`s_max`, `nose_ent`) > AUC(`s_max`, `nose_rel`) +
  0,10**. Fable 5 promete eliminar la clase 1 apoyándose en esa firma **y** pide que la compuerta se
  mida sobre `nose_rel`, que es justo donde la firma no aplica: la entidad SÍ está archivada con otra
  relación, así que el score máximo debería seguir alto.

## §5 · Qué hace cada desenlace

| resultado | consecuencia |
|---|---|
| P-1 y P-2 cumplen | la señal existe en el archivo → Fase 1 es **darle salida** (slot nulo + cabeza binaria), y el paper tiene su evidencia mecánica **antes** de entrenar un solo token de `NOSE` |
| P-1 cumple, P-2 no | hay señal pero no privilegiada en el archivo → el slot nulo sigue valiendo, pero el argumento «la diferencia vive en la interfaz de memoria» **no se puede sostener** |
| P-1 no cumple | el archivo co-entrenado tampoco separa presencia de ausencia → hay que **crear** la separación entrenando, y nos ahorramos apuntar la campaña al mecanismo equivocado |
| P-3 cumple | «eliminar la clase 1» vale para la mitad fácil; `nose_rel` necesita mecanismo entrenado, no umbral. La compuerta sobre `nose_rel` queda **bien elegida y exigente** |
| P-3 no cumple | la firma de ausencia es más general de lo previsto, y el slot nulo puede cubrir los dos tipos |

## §6 · Controles

- **Direccionalidad declarada:** se espera que las preguntas CON respuesta tengan score **más alto**.
  Un AUC < 0,50 es señal invertida y se reporta como tal, no se da vuelta el signo.
- **Piso de sanidad:** el reparto `nose_ent` / `nose_rel` debe estar cerca de 50/50 por construcción
  del generador (`idioma.py:199`, `rng.random() < 0.5`). Si no lo está, el desglose de P-3 no es
  interpretable y se aborta.
- **Este análisis NO entrena nada** y no toca los checkpoints. Es de sólo lectura.

## §7 · Límite declarado por adelantado

Los checkpoints se entrenaron con `p_nose = 0`: **nunca vieron una pregunta sin respuesta**. Cualquier
señal que aparezca es **incidental**, no supervisada — y esa es exactamente la razón por la que la
medición es interesante: si la separación existe sin que nadie la haya pedido, es una propiedad del
mecanismo y no del entrenamiento. Un AUC bajo acá **no prueba** que un modelo entrenado con
`p_nose > 0` no pueda desarrollarla.
