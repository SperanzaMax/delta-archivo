# Escalón 1 del Micro LM de Frontera · el acceso global aprendido, medido

2026-09-08. Evalúa `PREREG_FRONTERA_E1.md` (SHA `050341b4`) con sus dos enmiendas, `61e88855`
(selectividad) y `077cf358` (el N de la sonda). Las tres congeladas antes de tener el dato que
juzgan.

Tres unidades `at3_s0/s1/s2`, `--donde attn`, 26.000 pasos, desde cero, con los mismos
hiperparámetros que los dos brazos ya corridos.

## 1. H1 · CONFIRMADA, y el número final dice más que el veredicto

`nose_rel`, media de tres semillas.

| paso | kernel 3 | kernel 5 | atención completa | attn − k3 | attn − k5 |
|---|---|---|---|---|---|
| 4.000 | 0,5737 | 0,6411 | 0,5331 | −0,0407 | −0,1080 |
| 8.000 | 0,6360 | 0,7858 | 0,8833 | +0,2474 | +0,0975 |
| 12.000 | 0,6544 | 0,9432 | 0,9841 | +0,3297 | +0,0409 |
| 16.000 | 0,6506 | 0,9613 | 0,9910 | +0,3404 | +0,0297 |
| 20.000 | 0,6471 | 0,9907 | 0,9930 | +0,3459 | +0,0023 |
| **26.000** | **0,6430** | **0,9977** | **0,9977** | **+0,3547** | **+0,0000** |

Semillas de `at3` en el paso final, **1,0000 · 0,9931 · 1,0000**. El criterio de refutación pedía
menos de 0,80 en dos de las tres. Ninguna baja de 0,9931.

**Lo que decide no es que gane sino que EMPATE EXACTO.** La atención completa y el kernel 5 terminan
en el mismo valor, `0,9977` contra `0,9977`, mientras el kernel 3 se queda 0,3547 abajo. Eso es
exactamente lo que predice la ley enunciada como cobertura. El kernel 5 tiene alcance 4 y **ya cubre**
la relación, que cae a distancia 3, así que llega al techo. La atención cubre las 23 posiciones, o sea
cubre de más, **y cubrir de más no compra nada, cero exacto**. El kernel 3 tiene alcance 2, deja la
relación afuera, y 26.000 pasos no lo arreglan.

**El techo depende de SI la posición relevante está cubierta, no de CUÁNTO más se cubra.** La ley es
una condición binaria y no una escala, y este empate en cuatro decimales es la evidencia más limpia
que dio la campaña.

## 2. El transitorio, que no estaba pre-registrado

`attn` **arranca más lento**. En el paso 4.000 va 0,0407 por debajo del kernel 3 y 0,1080 por debajo
del kernel 5, y recién los cruza entre el 4.000 y el 6.000. Después sube más rápido que los dos y
llega al techo antes, con la diferencia contra el kernel 5 cerrándose de +0,0975 en el 8.000 a
+0,0000 en el final.

Lectura, y va como post-hoc. Es consistente con lo medido en `attn_concentracion.py`, porque sin
`wq`/`wk` propias el modelo no puede aprender a qué atender y sólo puede mover la geometría de las
embeddings, que es un camino más largo. **Es un argumento a favor del escalón 2 que no estaba en el
pre-registro**, y hay que tratarlo como hipótesis y no como resultado.

## 3. H2' · REFUTADA, y era la rama más informativa de las dos

La razón de selectividad `TV(d=3)/TV(d=5)` sobre los pesos de `at3`, con N = 600 como fija la
enmienda 2.

| | paso 11.500 | **paso 26.000** |
|---|---|---|
| `at3_s0` | 1,00 | **0,98** |
| `at3_s1` | 1,01 | **0,99** |
| `at3_s2` | 0,94 | **0,97** |
| cuatro controles sin acceso global | | 0,94 a 1,05 |

**Son el mismo número, y se mantiene de punta a punta del entrenamiento.** Ni una sola de las siete
unidades medidas pasa 1,05, y el umbral pre-registrado era 1,5. El modelo entrenó 26.000 pasos con
acceso global y no lo usó para pesar más la relación que el ruido.

La enmienda 1 pre-registró esta rama textualmente como «el resultado más informativo de los tres, y
el que obliga a buscar la causa alternativa antes de escribir nada».

## 4. La causa alternativa, buscada como el pre-registro manda

`masa_sensibilidad.py`, N = 300, distancias 1 a 20. **Post-hoc declarado.**

| | pico | masa acumulada | distancias con señal |
|---|---|---|---|
| kernel 5 | **0,0447** | 0,1432 | 4 de 20 |
| atención completa | 0,0110 | **0,2021** | **20 de 20** |

**Pico cuatro veces menor, masa 1,41 veces mayor.** La atención gana por área y pierde por altura. La
razón de selectividad comparaba dos distancias y por eso no podía ver lo que había cambiado, porque
el kernel concentra mucho en cuatro posiciones y da cero exacto en el resto mientras la atención
reparte poco en todas.

Y encaja con el §1. Si lo que decide fuera el pico, el kernel 5 tendría que ganar. Empatan, porque lo
que decide es **que no haya ceros**.

## 5. Un control que salió gratis y no estaba planeado

`donde=lat2` sobre los pesos de `at3` da **cero exacto en las seis distancias, incluso adentro de la
ventana**, con N = 600. La causa es que `convq` nunca entró en la pérdida y quedó en su
inicialización, `[0,9827, 0, 0, 0, 0]`, o sea la identidad menos el weight decay.

Eso prueba por el lado contrario la lección del 4-sep. **La ventana existe sólo si el entrenamiento
abre los taps.** En un modelo que nunca usó `convq` no hay ninguna ventana, hay una identidad. Y de
paso, `at3` viene con su propio brazo `pre` adentro sin que nadie lo haya construido.

## 6. Lo que este informe NO puede concluir, y estaba escrito antes

`attn` **no contiene a `pre` como caso particular**, a diferencia de `lat2` con `convq` en
`[1,0,...,0]`. Acá no hizo falta invocarlo porque el resultado fue positivo, pero la limitación queda
en pie para el escalón 2, y por eso ahí las proyecciones arrancan **en la identidad**.

Y esto **no es el Micro LM de Frontera**. Es cerrar la ley de la ventana por el otro lado. El tronco
sigue siendo recurrente y el vocabulario sigue cerrado, que son las dos diferencias que deciden.

## 7. Archivos

`corridas_20260908/at3_s*.json`, `curvas_frontera.py`, `control_attn_at3_mid.json`,
`control_attn_at3_final.json`, `control_attn_ruido_n600.json`, `masa_sensibilidad_20260908.log`,
`selectividad_attn.py`.
