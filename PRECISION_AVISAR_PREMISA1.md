# Precisión a la premisa 1 del `PREREG_AVISAR_A_PRESUPUESTO.md` · antes de que llegue el dato

**2026-09-01, 08:45.** El prereg (SHA `cd7c78e9`) está congelado y **no se toca**. Esto se escribe
aparte, con el experimento ya corriendo y **antes** de que ninguna unidad llegue a 26000.

## Lo que decía y por qué está sobreestimado

La premisa 1 afirma que **`blanco=error` es lo que cierra el aviso**, con `nose` 1,0000 contra
**0,78-0,83** de `blanco=ausencia`, citando `v3_s0/s1` y `w3_s0/s1`.

**Ese contraste tiene un confound que no vi: `v3` y `w3` tienen `donde` = `lat2` y `lat`, mientras
`b3` tiene `donde=pre`.** Comparé blancos distintos con posiciones de lectura distintas.

## El pareo correcto, con `donde=pre` en las cuatro

| unidad | blanco | `donde` | vigente | `nose` | falsa_abst | exactitud |
|---|---|---|---:|---:|---:|---:|
| `b3_s0` | **error** | pre | 1,0000 | **1,0000** | 0,0000 | **1,0000** |
| `b3_s1` | **error** | pre | 1,0000 | **1,0000** | 0,0000 | **1,0000** |
| `p3_s0` | ausencia | pre | 0,9844 | **0,9674** | 0,0069 | 0,9781 |
| `p3_s2` | ausencia | pre | 0,8196 | 0,6902 | 0,0276 | 0,7944 |

**La brecha real es +0,0326 de `nose`, no +0,20.** Y **la dispersión entre semillas dentro de
`ausencia` (0,6902 a 0,9674) es diez veces la brecha entre blancos**, así que con dos unidades por
condición **no hay potencia para atribuirle nada al blanco**.

> **La ventaja de `blanco=error` sobre `ausencia` NO está establecida.** Lo que sí está medido es que
> las dos únicas unidades del banco con exactitud **1,0000** tienen `blanco=error`, y que la mejor con
> `ausencia` llega a 0,9781, muy cerca.

## Qué le hace esto al experimento que está corriendo

**No lo invalida, y conviene decir exactamente por qué.** Lo que se está midiendo es si **las
unidades que quedaron MUDAS se recuperan con presupuesto** — R-0, R-1, R-2 y R-3 comparan el
tratamiento contra su propio control `b3_s3…s8`, todas con `blanco=error`. Esa pregunta **no depende**
de si `error` le gana a `ausencia`.

Lo que sí cambia es **la conclusión que se podrá sacar si sale bien**: no será «hay que usar
`blanco=error`», sino «**el atractor mudo de `blanco=error` es reparable**». Para preferir un blanco
sobre el otro hace falta una corrida pareada por `donde` y con semillas suficientes, que no existe.

## Y una lectura mejor, que el pareo deja ver

Las dos rutas a la abstención buena llegan **por caminos opuestos**, y el predictor del paso 2500 lo
muestra:

- **`p3_s0` arrancó LOCUAZ** (468 respuestas de 512) y llegó a 0,9781 **sin riesgo de mudez**.
- **`b3_s0` arrancó casi MUDO** (4 de 512) y llegó a 1,0000, pero de sus ocho hermanas **4 murieron**.

Entonces la pregunta práctica es si el riesgo se paga: hoy cuesta 4 de 8 unidades para ganar ~0,02 de
exactitud. **Sólo si el experimento en curso muestra que el riesgo se elimina, `error` domina.** Es
justo lo que R-1 pregunta.
