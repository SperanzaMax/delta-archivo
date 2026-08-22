# NOTA · el exploratorio de abstencion queda NO EVALUABLE (2026-08-22)

`PREREG_ABSTENCION_QC.md` (SHA `bcc626ec`) + `ENMIENDA_E1_ABSTENCION_QC.md` (el veredicto se lee
contra el nulo, no contra 0,50). Sonda `sonda_abstencion_qc.py`, 2048 muestras por unidad.

## Los numeros

| unidad | | `s1` | nulo | **margen** | acierto de la unidad |
|---|---|---:|---:|---:|---:|
| `p3_s0` | pre | 0,5340 | 0,5854 | −0,0514 | 0,9705 |
| `p3_s1` | pre | 0,6537 | 0,6169 | +0,0368 | 0,7769 |
| `p3_s2` | pre | 0,5912 | 0,5992 | −0,0079 | 0,8351 |
| `q3_s0` | post | 0,4867 | 0,4835 | +0,0033 | 0,3880 |
| `q3_s1` | post | 0,4747 | 0,4733 | +0,0014 | 0,3979 |
| `q3_s2` | post | 0,4288 | 0,4489 | −0,0201 | 0,3675 |

A-1 no cumple (1/3), A-2 no cumple (0/3), **A-3 no pasa**: el margen en `post` es +0,003, +0,001 y
−0,020, o sea **cero**. No hay señal en la entrada mas alla de la escala de los scores.

## Por que esto NO se reporta como negativo

El brazo `post` **nunca aprendio la tarea**: acierto 0,37-0,40 contra 0,78-0,97 de `pre`, y plano
desde el paso 4000 (ver `INFORME_QUERY_CONJUNTA_20260822.md`). Preguntarle a un modelo que no
resuelve la tarea si **sabe cuando no sabe** no tiene contenido: la pregunta presupone competencia,
igual que RT-5 del round-trip presuponia que el modelo condicionaba en la entidad y por eso quedo no
evaluable.

**A-1..A-4 quedan NO EVALUABLES.** Es la tercera vez en el programa que una prediccion no se puede
evaluar porque su regimen no ocurrio (las otras: P-2 de E-I4/E-I4b/E-I4c, y RT-5).

La hipotesis —que una query conjunta haria visible en la ENTRADA que la respuesta no esta— sigue
**sin probar**, por la misma razon que la hipotesis principal: la condicion que iba a probarla
rompio el modelo por otro motivo.

## Lo unico que este analisis si aporta

En `pre` **maduro y competente** (`p3_s0`, acierto 0,9705) el margen es **−0,0514**: `s1` queda por
debajo de su propio nulo. Sumado a la linea de base sobre `c3_s*` del mediodia (margen −0,05 a +0,05
sobre tres unidades), **el cierre del 21-ago se confirma con un instrumento nuevo y sobre modelos
mejores que los de entonces**: en la arquitectura de query pura no hay señal sin etiquetas en la
entrada, y no es por falta de competencia del modelo.

## Para retomar

El experimento de la query conjunta por camino lateral (§ final del informe principal) reabre esta
pregunta gratis: si esa version aprende la tarea **y** forma una query conjunta, A-1..A-4 se pueden
correr tal cual estan escritas, sobre un modelo competente, que es lo que hoy falto.
