# Estado al cierre del 23-ago

## 1. Lo que se construyó hoy

**El entrenamiento escalonado, que era la idea de Maxi de anoche.** `--mezcla dinamica` en
`entrenar.py`: cada tipo de pregunta se muestrea proporcional a su error EMA, con piso y alpha
lentos. Verificado que `--mezcla fija` reproduce **bit a bit** la corrida anterior, así que no toca
nada de lo que ya estaba. Guardas de identidad para `mezcla` y `mezcla_piso` en el checkpoint, las
dos probadas contra un ckpt real.

**La celda `e0`** (`--mezcla-piso 0.0`), que contesta la pregunta de Maxi de si un tipo resuelto debe
bajar al piso o detenerse. Con piso 0 la parada es **suave** y **se auto-repara**, porque la
evaluación es de mezcla fija y al tipo se lo sigue midiendo aunque casi no se lo entrene.

**`--parar-si-estanca`, implementado y APAGADO a propósito.** La simulación sobre las curvas reales
dice que no se encienda: con N=10 habría cortado `w3_s1` en el paso 8750 con 0,7128 cuando la
corrida real llegó a 0,9330, y `w3_s2` perdiendo 0,2383. Con N=20 casi ninguna corta, con N=40
ninguna. **No hay ventana útil**, y la causa está en E-I3b: una meseta larga acá no es techo, es que
la capacidad que falta todavía no arrancó.

**`--idioma 3`**, 24 relaciones en vez de 6. Las relaciones se sortean con reemplazo, así que con 6
el **72,1 %** de los episodios tenía colisión de clave, y ahí es donde el modelo falla. La cuenta
cierra hacia atrás: 0,724 × 0,46 × 0,58 = **0,193**, que es el 0,19-0,21 global que se venía
midiendo. Con 24 la colisión baja al 23,1 %. Compuerta de padding: ABRE. **Implementado y verificado,
sin lanzar por falta de GPU.**

**Respaldo en TPU en `rotar_abst2.sh`.** Si T4 da 503, pide `--tpu v5e1` en la misma cuenta antes de
pasar a la siguiente. Y la precisión quedó fijada en `highest` para que la aritmética no dependa de
qué acelerador tocó — en TPU los matmul de float32 pasan por bf16 y eso habría contaminado la
comparación pareada. Verificado bit a bit que no mueve nada en GPU.

## 2. Las corridas

| unidad | pasos | qué es |
|---|---|---|
| `w3_s0/s1/s2` | **26000 · LISTAS** | camino lateral, quedaron de ayer |
| `ef3_s0` | **20000 · LISTA** | mezcla fija (control) |
| `ef3_s2` | 15750 | |
| `ed3_s0` | 15500 | mezcla dinámica |
| `ed3_s1` | 14750 | |
| `ed3_s2` · `ef3_s1` | 7750 | |
| `e03_s0/s1/s2` | **0** | piso 0, nunca consiguió GPU |

Faltan **98500 pasos**. Falta además el control `fijo_promedio` (`ep3_*`), que **no se puede lanzar
hasta que la dinámica termine** porque su mezcla es un resultado de ella.

**Lectura parcial, que NO es un resultado todavía.** Al último paso común, la dinámica gana en 2 de 3
semillas (+0,0140 · +0,2894 · −0,0136) y alcanza el nivel de la fija **27 %, 48 % y 29 % antes** con
criterio robusto. El +0,2894 de `s1` es un salto sospechoso y este proyecto tiene bimodalidad
conocida en el despegue, así que hasta 20000 y con el control corrido, esto no se cuenta.

## 3. El pool, que es el cuello de botella

Tercer día con la misma forma: **de mañana hay, de tarde se seca**. Hoy a las 9:00 salieron 9 T4 de 9
pedidos; a las 15:00, una sola; desde las 17:30, ninguna. **2121 intentos fallidos** entre las 16:08
y las 20:24, y **cero TPU** en todos ellos.

Medido cuenta por cuenta: las 13 tienen el mismo derecho y ninguna está mal configurada. T4 y TPU
v5e1 dan 503 por disponibilidad; L4, G4, H100, A100 y v6e1 contestan «you may not have quota or
entitlement», que es suscripción y no disponibilidad.

Los 8 rotadores quedaron relanzados con **200 vueltas y descanso de 7 minutos**, para llegar
despiertos a la ventana de la mañana. Si la PC se apaga, **mueren con ella** y hay que relanzarlos a
mano.

## 4. Publicación

**Mail enviado a los autores de [IDK]** (Cohen, Dobler, Biran, de Melo — NeurIPS 2024). La
discrepancia: ellos atribuyen el fallo en modelos chicos a la inicialización del embedding, y acá
está medido que la inicialización explica de dónde sale la asimetría pero **no** el fallo, porque
corregirla explícitamente no cambia nada. Queda pendiente el mail 2 (endorsement de arXiv a de Melo),
que va sólo si contestan.

**Dossier de literatura II** (`DOSSIER_LITERATURA_20260823.md`) sobre las dos piezas que el dossier
del 8-ago no cubría. Resumen: la cabeza separada tiene antecedente estructural viejo (SelectiveNet,
2019) pero nadie la contrastó contra el token de vocabulario en un LM con memoria; y del sello de
orden, el trabajo más cercano (*Unable to Forget*) **nunca pide el valor superado**, que es el hueco
limpio.

**UFLO: el resumen YA se había enviado el 1-ago** — lo decía mal la memoria y se verificó contra
Gmail. Queda la aceptación del 1-OCT y la ponencia completa hasta el 20-OCT, ya lista y en rango.

## 5. Para mañana, en orden

1. **Ver si el pool volvió.** Si sí, las 9 unidades siguen solas; si los rotadores murieron con la
   PC, relanzarlos.
2. **Terminar `ed3` y `ef3`**, y recién ahí leer S-1, S-2 y S-4.
3. **Lanzar `ep3_*`** (`fijo_promedio`) con la mezcla que imprima la dinámica al cerrar. Sin ese
   control, un positivo no dice si importó el escalonamiento o sólo la proporción.
4. **`e03_*`** sigue sin arrancar. Es la pregunta de Maxi sobre parar del todo.
5. Pendiente sin GPU asignada: la campaña de `--idioma 3`.
