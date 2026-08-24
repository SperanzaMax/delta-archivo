# INFORME · LA QUERY CONJUNTA POR CAMINO LATERAL

Campaña `lat` (`w3_s0/s1/s2`), 26000 pasos, contra el control `pre` (`p3_s0/s1/s2`) del mismo día,
mismo presupuesto y mismo generador. Pre-registro: `PREREG_CAMINO_LATERAL.md` (SHA `c440ec93`),
congelado antes de lanzar. Analizador: `analizar_camino_lateral.py`. Datos:
`camino_lateral_20260824.json` y `qc_26000/`.

Las tres unidades quedaron completas el 23-ago a la mañana y el análisis no se había corrido, así
que esto lee una campaña que estaba terminada en disco desde hace un día.

## 1. El resultado en una línea

**`err_identidad` se va a 0,0000 en las tres semillas, y la colisión de clave se disuelve entera.**
Es la primera intervención del proyecto que hace eso. Cuatro de las cinco predicciones cumplen, y la
que falla —W-4— falla por un canal que el análisis fino identifica y que no es el que W-1 mide.

| | W-0 bloqueante | W-1 principal | W-2 mecanicista | W-3 especificidad | W-4 no-intercambio |
|---|---|---|---|---|---|
| | **CUMPLE** 3/3 | **CUMPLE** 3/3 | **CUMPLE** 3/3 | **CUMPLE** 3/3 | **NO CUMPLE** |

## 2. W-0 · la compuerta que `post` no pasó

`lat` aprende la tarea, y no apenas: acierto **0,9975 · 1,0000 · 0,8843**, contra el 0,37-0,40 con
que `post` se invalidó a sí mismo el 22 a la mañana. Las tres semillas pasan la compuerta de 0,70.

Y sobre el mismo eje, contra su propio control pareado: `pre` da **0,9705 · 0,7769 · 0,8351**. `lat`
gana en las tres.

## 3. W-1 y W-2 · la colisión de clave, que era la causa

El 20-ago quedó medido que `err_identidad` no era marginalización sino **colisión de clave**: con
relación repetida el error saltaba a 0,38-0,54 (≈ azar entre las dos entradas que empatan) y con
relación única se quedaba en 0,005-0,014. El 23-ago se agregó que la colisión es una propiedad del
**vocabulario** (72,1 % de los episodios la tienen con 6 relaciones).

Pareado por semilla, como el §7 del prereg obliga (nunca por media, por la bimodalidad declarada):

| | `ident_rep` pre | `ident_rep` lat | |
|---|---:|---:|---|
| s0 | 0,0564 | **0,0000** | −0,0564 |
| s1 | 0,4683 | **0,0000** | −0,4683 |
| s2 | 0,2529 | **0,0069** | −0,2460 |

Y la brecha `acierto(única) − acierto(repetida)`, que es lo que separa «bajó el error» de «bajó por
haber disuelto la colisión»:

| | brecha pre | brecha lat | hacía falta |
|---|---:|---:|---:|
| s0 | 0,0558 | **0,0003** | ≤ 0,0279 |
| s1 | 0,4683 | **0,0000** | ≤ 0,2342 |
| s2 | 0,2503 | **0,0202** | ≤ 0,1252 |

**La bimodalidad del control desaparece.** `pre` iba 0,0564 / 0,4683 / 0,2529 —tres regímenes
distintos con la misma configuración—, y `lat` deja las tres en el mismo lugar. Eso es más fuerte
que el efecto medio: lo que `lat` elimina es justamente la varianza entre semillas que hacía que
este proyecto no pudiera afirmar nada con tres semillas.

W-3 cierra la lectura por el lado de la especificidad: con relación **única** `lat` no empeora
(`ident_unica` 0,0000 · 0,0000 · 0,0043, compuerta 0,03). No hubo daño general, hubo disolución de
la colisión.

## 4. W-4 falla, y el desglose dice exactamente por dónde

`falsa_abst` pasa holgado en las tres (0,0016 · 0,0000 · 0,0016 contra una compuerta de 0,10). Lo
que rompe W-4 es `nose`, y **sólo en s0**: 0,9119 → 0,7889, una caída de 0,1230 contra una
tolerancia de 0,05. En s1 sube +0,3209 y en s2 sube +0,0398.

El desglose de `nose` en sus dos mitades muestra que no es ruido sino **dos efectos opuestos**:

| | `nose_ent` pre → lat | `nose_rel` pre → lat |
|---|---|---|
| s0 | 0,9016 → **0,9725** | 0,9235 → **0,5842** |
| s1 | 0,4989 → **0,9908** | 0,5893 → **0,7194** |
| s2 | 0,6888 → **0,9382** | 0,7755 → **0,5816** |

**`nose_ent` mejora en las tres y se va casi a 1. `nose_rel` empeora en dos de tres.** Y esto es
mecánicamente lo que la query conjunta debería producir: al formar la query sobre entidad×relación,
el modelo distingue entidades como nunca —y por eso deja de confundir dos hechos que comparten
relación, que es el hallazgo de arriba— pero cuando la que falta es la **relación**, la mitad de su
query sigue coincidiendo con una entrada real del archivo y se ancla ahí.

Es la misma forma del hallazgo del 20-ago sobre el monitor de desacuerdo: **cuando no hay respuesta,
el modelo no inventa contenido, se ancla en otra entrada real.** `lat` no crea ese comportamiento;
lo concentra en el eje donde su query tiene coincidencia parcial.

## 5. La observación que W-4 debió vigilar y no vigilaba

Hueco propio, ya declarado el 22-ago: W-4 mira `falsa_abst` y `nose`, y **no** mira `anterior`. No
se cambia el criterio ahora —sería mover el arco— pero se reporta:

| | `anterior` pre → lat |
|---|---|
| s0 | 0,9471 → **1,0000** |
| s1 | 0,8317 → **1,0000** |
| s2 | 0,8125 → **0,3798** |

Dos semillas van al techo y una se desploma. Es el pago en `anterior` que Maxi destapó el 22 con la
pregunta de por qué no parar a los 4000, y la causa ya está diagnosticada en
`DIAGNOSTICO_CONV_COMPARTIDA_20260822.md`: **la query de `lat` y el mixer comparten la misma
`blk["conv"]`**, y esas dos cosas tienen balances opuestos. El mixer quiere el mix que le sirve a la
regla delta; la query quiere contexto para entidad×relación pero **poco** contexto para no diluir el
marcador temporal (`antes`, que además queda fuera de la ventana de la conv).

La corrección ya está comprometida y sigue sin correr: **`convq` propia, 3×D = 384 params (0,044 %),
inicializada en `[1,0,0]`**, con lo cual `lat2` contiene a `pre` como caso particular y no puede ser
estructuralmente peor.

## 6. Qué queda dicho y qué no

**Queda dicho.** La forma de la query **sí es** la causa de la colisión de clave. El §6 del prereg
había comprometido la lectura contraria por adelantado —si W-0 pasaba y W-1 fallaba, el mecanismo
del 21-ago quedaba como correlación y la línea se cerraba— y no es lo que pasó: W-1 cumple 3/3
pareado. Esto **acota hacia atrás** todas las lecturas de `err_identidad` del proyecto, que venían
leyéndose como alucinación o marginalización.

Y acota también el cierre del 21-ago. Las cuatro vías de aquel día fallaban todas en el mismo punto
—separaban estados del modelo, no aciertos de errores—. `lat` no es una quinta vía de detección: es
la intervención que **elimina la condición** que las cuatro intentaban detectar.

**No queda dicho.** Que `lat` sea mejor en conjunto. Paga en `nose_rel` (dos de tres semillas) y
paga en `anterior` (una de tres, hasta 0,3798). W-4 falla y se reporta como falla, no como matiz.
Y las tres unidades corren con `idioma 2`, o sea el vocabulario chico cuya colisión del 72,1 % es
justamente lo que se está midiendo: con `--idioma 3` la colisión baja al 23,1 % por construcción y
habría que ver cuánto del efecto queda.

## 7. Lo que sigue, en orden

1. **`lat2` con `convq` propia**, contra el mismo control `p3_s*`. Es la corrección comprometida y
   ahora tiene además una segunda pregunta que responder: si al desacoplar la conv el modelo puede
   elegir cuánto contexto usar, ¿recupera `nose_rel` y `anterior` sin perder el 0,0000 de
   `err_identidad`?
2. **Vigilar `anterior` y `nose_rel` explícitamente** en el prereg de `lat2`. El hueco de W-4 no se
   repite dos veces.
3. `lat` sobre `--idioma 3`, para separar «disolvió la colisión» de «la colisión era del
   vocabulario».
