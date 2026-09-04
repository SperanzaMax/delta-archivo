# Auditoría completa del paper de la ventana · 4-sep-2026, DESPUÉS de enviarlo

Se auditaron **las siete tablas** contra los datos crudos, no contra los informes propios. Motivo: el
paper ya había salido a Research Square (`rs-10929866`) y a TMLR (paper `11988`) esa misma mañana, y
Maxi pidió controlar el Resultado 7 porque había salido de dos fotos suyas de una clase de
probabilidad.

## Lo que dio bien

| tabla | qué se verificó | resultado |
|---|---|---|
| 1 · ley de la ventana | `micro_lm/salidas/ley_ventana_0902.log` | **60/60 celdas.** `v3` d=1 `0.706`–`0.744` (pub. 0.71–0.74), d=2 `0.154`–`0.254` (pub. 0.15–0.25), d≥3 cero exacto; `kq3` d=1 `0.359`–`0.459`, d=2 `0.0997`–`0.1476`, d=3 `0.055`–`0.348`, d=4 `0.067`–`0.327`, d≥5 cero exacto |
| 2 · abstención por kernel | 6 JSON de entrenamiento | **24/24 exactos** |
| 4 · atenuación | `escalera_v2_130m/370m.json` | **12/12 exactos.** Pendientes recalculadas: `1.0773` y `1.0283` (pub. 1.077 y 1.028), r `0.9784` y `0.9777`, correlación entre curvas `0.9549`, control del relleno `1.9819` |
| 5 · near/far | 6 JSON de `modelo_real` | **25/25 exactos**, incluidas las 15 diferencias por semilla |
| 6 · permanente/transitorio | `cl3_*`, `cf3_*` | **6/6 exactos** |
| 7 · corpus reales | re-corrido de punta a punta | **12/12 exactos**, más tres controles nuevos, abajo |

## Los tres defectos, y lo que se hizo con cada uno

### 1 · «two blocks deep» era falso: el micro-LM tiene CUATRO bloques

Estaba **en el abstract** y en la Tabla 6, o sea que salió en los dos envíos. Lo contradicen tres
fuentes independientes: el `config` de las seis corridas (`capas: 4`), el código (`for i in range(NB)`
y el comentario «los CUATRO bloques» en `modelo.py`) y **el Setup del propio paper**, que dice «four
blocks». Corregido en los tres manuscritos.

No cambia ningún resultado —el contraste es «pocas capas contra 24»— pero era una afirmación falsa
sobre la arquitectura propia, en el abstract, y verificable abriendo el repo que el paper enlaza.

### 2 · la fila «layer output» de la Tabla 3 no salía de ningún estadístico

Publicaba `9e-2 · 2e-2 · 2e-2 · 1e-2 · 9e-2 · 1e-2` para d = 1, 2, 3, 5, 7, 8. **No es el mínimo, ni
el máximo, ni la mediana, ni la media**: para d=7 publicaba `9e-2` cuando el mínimo es `1.2e-2`, la
mediana `4.0e-2` y el máximo `1.0e-1`. Parecen valores sueltos tomados de filas distintas.

Lo que la tabla **afirma** sigue en pie y es lo que importa: la salida de la capa nunca es cero,
**80 celdas de 80**, con un mínimo global de `6.3e-3`. Reemplazada por mediana + rango, ambos
declarados, y el mínimo global agregado al texto.

### 3 · el Resultado 7: el `n` de X3 y la falta de una cota inferior

- **El `n` estaba mal atribuido.** El paper daba `10 570 · 3 610 · 12 000 · 7 405` junto a una tabla
  con columna `P(X3 > 2)`, pero X3 se computó sobre `6 610 · — · 8 326 · 5 269`, o sea el **62 %, 69 %
  y 71 %** de cada corpus, porque sólo queda definida cuando hay interrogativo y después una mayúscula.
  Corregido, con la razón explicada.
- **Las tres definiciones eran todas del lado alto.** El texto prometía «believe only what survives all
  three» pero ninguna acotaba por abajo. Se agregó la cota inferior —las dos **últimas** palabras de
  contenido— que da `0.186 · 0.211 · 0.295 · 0.264`, y se dice explícitamente que el número real
  depende de qué dos partes discriminan, cosa que los corpus no anotan.

## Los controles nuevos del Resultado 7, que el resultado PASÓ

1. **Tokenizador.** No agrega BOS ni EOS y el round-trip es idéntico, así que `X2` no está corrido.
2. **Truncado.** TriviaQA completo (17 944 preguntas, no 12 000) da `0.9941` contra `0.9943`.
3. **★ Sub-tokens contra palabras.** Es el control que decidía. Medido por **palabras** en vez de
   piezas BPE, `W1` da `0.9637 · 0.9970 · 0.9923 · 1.0000` contra `X1` `0.9625 · 0.9978 · 0.9943 ·
   1.0000`. El resultado principal **no depende de la unidad**.

Y de paso descartó mi propia primera objeción: sobre sub-tokens la cota inferior daba `0.10`–`0.17`,
pero **en tres de los cuatro corpus los dos últimos tokens de contenido son la misma palabra el 90 %
al 94 % de las veces**, así que eso medía el ancho de una palabra y no acotaba nada. Corregida a
palabras, la cota sube a `0.19`–`0.30`. Por eso la que se publica es la de palabras.

## Lo que queda anotado y NO se cambió

- El control del relleno se reporta con la dispersión de `130m` (`1.98`). En `370m` es **`2.99`**. La
  conclusión aguanta igual —`2.99` sigue siendo mucho menor que el rango de `1.04` a `6.47` entre
  distancias— pero el paper no aclara de qué modelo es el `1.98`.
- Para el mecanismo del micro-LM, donde la consulta se forma en el último token, la cantidad correcta
  no es el span entre partes sino la distancia de la parte más lejana **al final de la pregunta**, que
  es `X1 + X2` y es todavía mayor. Ahí el paper **subestima** su propio efecto.

## La regla que deja

> **Una tabla publicada tiene que decir QUÉ estadístico es cada fila.** Las seis tablas que lo decían
> reprodujeron exacto; la única que no lo decía fue la única que no reprodujo. No hizo falta buscar el
> error: se delató sola al no poder nombrarlo.

Ver `preprint/ventana/METADATOS_RESEARCH_SQUARE.md` y `preprint/ventana/ENVIO_TMLR_PASO_A_PASO.md`.
