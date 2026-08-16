# Fase 0 · el score del archivo NO sabe si el hecho está

**2026-08-16** · `score_archivo.py` + `control_score.py` · CPU, checkpoints ya entrenados, **cero GPU**
Pre-registro `PREREG_SCORE_ARCHIVO.md`, SHA `fea5e061…`, congelado 13:20 UTC **antes** de correr nada.

## La pregunta

La tesis del programa dice que un modelo no distingue **recuperar** de **inventar** porque no hay
diferencia mecánica entre los dos casos. El archivo co-entrenado tiene un lugar donde esa diferencia
debería vivir: **el score de la consulta contra las claves archivadas**, antes del softmax de lectura.
Nunca lo habíamos medido — medimos probabilidad, margen y entropía **de la salida**, que es el final
de la cadena.

Y era la Fase 0 recomendada por la revisión externa: **medir antes de construir**, porque el
resultado bifurca el diseño de toda la campaña.

## Resultado

Eje explícito: **con respuesta en el archivo (positivos) vs. sin respuesta (negativos)**,
independiente de si acertó. n = 4000 por checkpoint, `p_nose = 0,4`, semilla fija.

| señal | `n4_s0` | `n3_s2` | |
|---|---:|---:|---|
| `s_max` (archivo) | **0,4984** | **0,5022** | ← el máximo del matcheo |
| `s_margen` (archivo) | 0,5044 | — | |
| `s_lse` (archivo) | 0,4982 | — | |
| `c_prob` (salida) | 0,5960 | 0,6154 | |
| `c_margen` (salida) | 0,5845 | — | |
| `c_entropia` (salida) | 0,6132 | 0,6307 | |

**Las tres predicciones NO CUMPLEN, en los dos checkpoints:**

| | criterio | `n4_s0` | `n3_s2` |
|---|---|---|---|
| P-1 | AUC(`s_max`) ≥ 0,60 | 0,4984 ✗ | 0,5022 ✗ |
| P-2 | AUC(`s_max`) > AUC(`c_prob`) + 0,03 | 0,4984 vs 0,5960 ✗ | 0,5022 vs 0,6154 ✗ |
| P-3 | `nose_ent` > `nose_rel` + 0,10 | 0,5164 vs 0,4806 ✗ | 0,5102 vs 0,4944 ✗ |

Control de reparto `nose_ent`/`nose_rel`: 0,4981 y 0,4930 → dentro de rango, P-3 es interpretable.

## El control, porque era un cero demasiado limpio

`s_max` = 0,4984 es **el azar exacto**, y en este programa un número limpio escondió un artefacto
siete veces. Una extracción rota y una señal ausente producen exactamente el mismo AUC. Antes de leer
esto como resultado se corrió `control_score.py`, con tres comprobaciones **que podían fallar**:

| control | qué descarta | `n4_s0` | `n3_s2` |
|---|---|---|---|
| **C-1** · logits reconstruidos con *mi* `sim` vs. los del modelo | que mi copia de la lectura no sea la del modelo | **0,000e+00** | **0,000e+00** |
| **C-2** · variación del score | que sea un tensor constante (AUC 0,5 por falta de señal en la medición) | sd 0,4133 | sd 0,1325 |
| **C-3** · archivo ablacionado | que el archivo sea un canal muerto | cambia la predicción en **1,0000** | **1,0000** |

C-1 es el decisivo y da **cero absoluto**: los logits reconstruidos con el `sim` que extraigo son
idénticos bit a bit a los del modelo, así que estoy midiendo exactamente la lectura que el modelo
usa. **El 0,4984 es una propiedad del modelo, no de mi código.**

## Qué significa

**La diferencia entre recuperar e inventar no está en la interfaz de memoria.** El score de matcheo
—el lugar donde la tesis del programa la ubicaba— no distingue en absoluto si el hecho preguntado
está archivado. La poca señal que existe (AUC ≈ 0,61-0,63, en la entropía de salida) **se arma aguas
abajo**, en el cómputo posterior a la lectura.

Y hay un mecanismo candidato, visible en `modelo.py:responder()`: **la lectura es un softmax sobre las
entradas del archivo, o sea suma 1 siempre.** El modelo está obligado a leer algo aunque nada
matchee, y como el softmax es invariante a un desplazamiento constante de los scores, **nada en el
entrenamiento presionó jamás a que la magnitud del matcheo signifique «está»**. Sólo importan las
diferencias relativas entre entradas. Es coherente con que `s_margen` tampoco separe: no hay una
geometría de «nada matchea» porque nunca hizo falta que la hubiera.

**Advertencia sobre este párrafo:** es una explicación *candidata*, consistente con los datos pero no
probada. Lo medido es que las tres señales del archivo están en el azar; el *por qué* todavía no
tiene su control.

## Consecuencias, según el §5 del pre-registro

Comprometido por adelantado, se cumple la celda «P-1 no cumple»:

1. **El slot nulo tiene que CREAR la separación entrenando, no darle lectura a una que ya existe.**
   La revisión externa apostaba a lo contrario («si el AUC del score supera 0,7397, el trabajo es
   darle salida a una señal que ya está»). No está.
2. **El argumento «la diferencia vive en la interfaz de memoria» no se puede sostener** con estos
   datos, y no debe entrar en ningún paper como si estuviera medido.
3. **Se ahorraron ~13 h de GPU** apuntadas al mecanismo equivocado — que era exactamente para lo que
   la Fase 0 existía.
4. **Refuerza el slot nulo por otra vía**: si el problema es que el softmax obliga a leer algo y la
   magnitud nunca significó nada, entonces un slot nulo no es un detector — es **el marco de
   referencia que le da sentido a la magnitud**, convirtiéndola en una comparación contra una
   constante aprendida. Es la misma conclusión de la revisión externa, por el camino contrario.

## Lo que este informe NO dice

- **No dice que un modelo entrenado con `p_nose > 0` no pueda desarrollar la señal.** Los dos
  checkpoints se entrenaron con `p_nose = 0` y **nunca vieron una pregunta sin respuesta**: cualquier
  separación habría sido incidental, no supervisada. El resultado acota la hipótesis del mecanismo,
  no la del entrenamiento.
- **No mide `nose_rel` contra `nose_ent` de forma concluyente.** P-3 falla en magnitud (0,036 y 0,016
  contra el 0,10 exigido) aunque el signo va en la dirección predicha en los dos checkpoints. La
  tensión que se le marcó al dictamen —que la firma «ninguna clave matchea» sólo aplicaría a
  `nose_ent`— **queda sin resolver: no hay firma para ninguno de los dos.**
- **Dos checkpoints, un ckpt por nivel.** N3 y N4 todavía no tienen sus tres semillas.

---

## ⚠ CONTROL AGREGADO EL MISMO DÍA · el resultado NO depende de la posición de la sonda

Se detectó después que **el modelo enfoca la lectura en posiciones intermedias de la consulta**
(masa top-1 hasta 0,6492) y que en `pos_q` —donde se tomó este AUC— la distribución ya está difusa
(0,27). Si la señal de ausencia viviera donde vive el foco, este resultado sería un artefacto de dónde
se puso la sonda. Se re-midió en cuatro lugares (`score_pos_foco.py`, n = 4000):

| dónde se toma el score | `n4_s0` | `n3_s2` |
|---|---:|---:|
| `pos_q` *(réplica de este informe)* | **0,4984** | **0,5022** |
| **posición de máximo foco** | **0,5007** | **0,5077** |
| máximo sobre todas las posiciones | 0,5293 | 0,5429 |
| margen en la posición de foco | 0,5081 | 0,5377 |

La réplica de `pos_q` reproduce los valores originales **exactamente**. Y **donde el modelo más
concentra, el score sigue en el azar**. El resultado sobrevive y se enuncia mejor: no falta selección
—la hay, y fuerte— sino que **la selección no tiene un estado de vacío**. Ver
`INFORME_FOCO_LECTURA_20260816.md`.
