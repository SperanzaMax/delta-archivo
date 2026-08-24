# NOTA · cómo leer las curvas, y dos errores que ya cometimos

2026-08-24. Sale de una pregunta de Maxi —*«¿26000 pasos no es demasiado? ¿se degradan los pesos con
el sobreentrenamiento?»*— y de que la primera respuesta que le di **estaba mal**. Se escribe para que
las próximas corridas no repitan ninguno de los dos errores.

---

## 1. El máximo puntual NO es evidencia de degradación

**El error, tal cual se cometió.** Se comparó, para cada unidad, el **máximo** de una métrica a lo
largo del entrenamiento contra su **valor final**, y se reportó la diferencia como pérdida:

> «`p3_s1` alcanzó su mejor `nose` en el paso 3750 y terminó en 0,4878: perdió 0,1298 entrenando
> 22250 pasos más.»

**Por qué está mal.** Cada evaluación usa 512-2048 muestras, así que tiene ruido de ±0,02. El máximo
de **104 evaluaciones** no es el mejor estado del modelo: es **el pico más afortunado del ruido**. Y
el máximo de una serie ruidosa queda por encima del valor final **por construcción**, aunque la serie
sea perfectamente plana. Es sesgo de selección, no una medición.

**Cómo se lee bien.** Promediando tramos, que mata el ruido:

| | 18-22k → 22-26k, 6 unidades × 3 métricas |
|---|---|
| suben | **9** |
| planas | **9** |
| **bajan** | **0** |

Los dos casos que se habían reportado como degradación, releídos:

- `p3_s1` `nose`: tendencia **plana** (−0,0029). El 0,6176 del paso 3750 era un pico de ruido; la
  unidad nunca superó ~0,50 de forma sostenida.
- `w3_s2` `anterior`: **sube** +0,1001 (0,1023 → 0,2025). Aprende tarde y mal, pero aprende. Su
  0,4955 del paso 1500 también era ruido.

> **REGLA: no se reporta degradación desde un máximo puntual. Se compara el promedio de un tramo
> contra el promedio de otro, o no se afirma nada.**

## 2. La deriva de los pesos NO es degradación

La intuición que hay que desactivar es la del fine-tuning de un modelo denso, donde mover los pesos
pisa lo aprendido. **Acá se midió y pasa lo contrario.**

Entre los pasos 14000 y 20000, en las 8 unidades con checkpoint intermedio en disco:

| unidad | `\|Δw\|/\|w\|` | coseno | `vigente` | `anterior` | `nose` |
|---|---:|---:|---|---|---|
| `s4_s0` | **0,4034** | 0,9287 | +0,321 | +0,112 | +0,154 |
| `s4_s1` | 0,3959 | 0,9316 | +0,172 | +0,014 | +0,008 |
| `t4_s0` | 0,1218 | 0,9926 | +0,062 | +0,032 | +0,044 |
| `t4_s1` | 0,1313 | 0,9914 | +0,104 | +0,058 | +0,008 |
| `t4_s2` | **0,4072** | 0,9272 | +0,198 | +0,070 | +0,123 |
| `c4_s0` | 0,3813 | 0,9372 | +0,142 | +0,098 | +0,098 |
| `c4_s1` | 0,3864 | 0,9336 | +0,140 | +0,099 | +0,092 |
| `c4_s2` | 0,3844 | 0,9355 | +0,188 | +0,117 | +0,042 |

**Los pesos cambian hasta el 41 % de su norma y las tres capacidades mejoran en las ocho unidades,
sin una sola excepción.** Y la relación es al revés de lo temido: **las unidades que más derivaron
son las que más mejoraron.** La deriva **es** el aprendizaje.

Encaja con E-I4c (21-ago), que lo había medido por otra vía: coseno **0,8531** —pesos muy movidos—
con el conocimiento pasando de 0,9970 a **0,9922**. Y con su hallazgo más fino, que ahora tiene una
segunda confirmación: **a mismo coseno el daño es distinto según cómo se produjo la deriva**, o sea
**la magnitud del cambio de pesos no resume el daño** y no sirve para predecirlo.

**La razón mecánica, y es propia de esta arquitectura:** el conocimiento vive en el **archivo**, no
en los pesos. Por eso sobrevive a que los pesos se muevan, y por eso la analogía con el olvido
catastrófico del fine-tuning **no aplica acá**.

> **REGLA: `|Δw|/|w|` y el coseno miden CUÁNTO se movió el modelo, no CUÁNTO daño se hizo. Para
> daño hay que medir daño.**

## 3. Consecuencias para las próximas corridas

1. **26000 pasos no es demasiado, y el número no es arbitrario.** `vigente` y `anterior` tocan techo
   entre 11000 y 16250, pero **`nose` sigue subiendo hasta el final** (`p3_s0` +0,0938 y `p3_s2`
   +0,0432 en el último tramo). Cortar antes mataría justamente la métrica que las campañas de
   abstención existen para medir.

2. **No encender `--parar-si-estanca`.** La simulación del 23-ago ya lo decía; esto lo confirma por
   otra vía y con una razón más simple: no hay nada de qué proteger.

3. **No guardar «el mejor checkpoint».** Era la solución a un problema que no existe, y habría
   metido selección sobre el conjunto de prueba. Si alguna vez hiciera falta, va con el split que ya
   usa el proyecto (elegir con `90000+s`, juzgar con `77000+s`).

4. **Lo que SÍ conviene vigilar** es lo que ninguna de las dos cosas anteriores cubre: que una
   capacidad **caiga mientras otra sube**. El acierto global ponderado puede quedar plano tapando
   eso. Hoy no lo mira nadie, y es barato: comparar tendencias por tipo entre tramos.

## 4. La lección de proceso

Las dos veces el error fue el mismo: **tomar un estadístico y leerlo como si midiera lo que su nombre
sugiere.** «Máximo» suena a mejor estado y era el pico del ruido; «deriva de pesos» suena a daño y
era aprendizaje.

Es la misma familia que la advertencia ya escrita en `regla-verificar-antes-de-veredicto`: antes de
pasar una conclusión, correr su control y buscar la explicación alternativa. Acá el control era
trivial —promediar en vez de tomar el máximo, cruzar deriva con rendimiento— y no se corrió hasta que
Maxi preguntó.
