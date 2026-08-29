# PRE-REGISTRO · CALIBRAR EL CORTE, Y CUÁNTO SOBREVIVE SIN LAS ETIQUETAS DE LA UNIDAD

**2026-08-28.** Se congela antes de correr y antes de mirar un solo número de las muestras nuevas.

Sale de tres campañas ya cerradas y de un hueco que las tres dejaron declarado:

- `INFORME_UMBRAL_PROSPECTIVO_20260819.md` — el corte calibrado es prospectivo, el nulo permutado
  está limpio (0 de 20), y `a*` transfiere entre unidades del mismo nivel. Dejó escrito que `a*` se
  elige **con etiquetas**, y que de dónde sale el corte en uso real «es una pregunta abierta que este
  informe no toca».
- `INFORME_SIN_ETIQUETAS_20260820.md` — el corte estimado de la **forma de la densidad** es negativo
  en 5 de 5 criterios, y el post-hoc dice por qué con precisión: las dos poblaciones están a 1,15-1,27
  σ y una mezcla necesita ~2 σ para ser bimodal, así que **no hay valle que encontrar**. Lo único que
  sobrevivió fue U-2, la **transferencia** de un corrimiento constante (7/8).
- `INFORME_CALIBRA_ENSAMBLE_20260826.md` §A3 — calibrar cierra ~75 % de la brecha al oráculo y sube
  la detección entre +0,035 y +0,056 sin costo. **Es exploratorio y sin pre-registro, así que hoy no
  confirma nada.** Es el único positivo de calibración sobre las unidades actuales.

---

## 1. La pregunta

> **¿La ganancia de calibrar el corte se sostiene prospectivamente sobre las unidades a 26000, y
> cuánto de ella sobrevive cuando la unidad NO aporta sus propias etiquetas?**

Son dos preguntas y se contestan con la misma corrida. La primera convierte A3 en confirmatorio. La
segunda es la que decide si esto sirve para algo fuera del banco, porque en uso real no se sabe qué
preguntas tenían respuesta.

## 2. Un defecto del instrumento de A3, y la corrección

`sonda_calibra_ensamble.py` fija `SEM_A, SEM_P = 55000, 66000` **sin sumar la semilla de la unidad**,
y lo hace por una razón correcta: A4 mide acuerdo entre semillas y necesita que las tres unidades vean
**el mismo lote**.

Pero para A3 eso significa que las seis unidades comparten muestra. Para la réplica no es grave
—ajuste y prueba siguen siendo independientes entre sí— pero **para la transferencia sí lo es**: un
corte calibrado en `s0` y aplicado a `s1` se estaría juzgando sobre **las mismas preguntas** con las
que se eligió, y la transferencia quedaría inflada por construcción.

**Corrección, fijada acá:** las muestras se generan **por unidad**, sumando la semilla.

| muestra | generador | uso |
|---|---|---|
| ajuste | `31000 + s` | elegir el corte |
| prueba | `42000 + s` | juzgarlo |
| estabilidad | `63000 + s` | post-hoc de K-4, si hace falta |

Verificado antes de escribir esto: 31000, 42000 y 63000 **no aparecen en el repo**. Las usadas hasta
hoy son 54321, 55000, 66000, 77000 y 90000. Ninguna de las muestras de esta campaña fue mirada nunca.

## 3. Diseño

**Unidades:** los 6 checkpoints ya entrenados a 26000 pasos — `p3_s0/s1/s2` (blanco `ausencia`) y
`b3_s0/s1/s2` (blanco `error`). Las seis del nivel 3, `p_nose` 0,4.

**Costo: cero GPU y cero pool.** Todo es post-hoc sobre checkpoints que ya existen. CPU.

**n:** 6000 de ajuste + 6000 de prueba por unidad, igual que A3.

**Procedimiento de calibración**, idéntico a A3, que es la lección del 19-ago aplicada: el corte se
elige en la muestra de **ajuste** pidiendo **margen** (`falsa_abst` ≤ 0,07) y se juzga en la de
**prueba** contra el criterio real (`falsa_abst` ≤ 0,10). El óptimo pegado al borde no generaliza.

**Se incluyen las `b3_*` a propósito.** Su blanco es distinto y su varianza entre semillas es la que
cerró A5. Si la calibración sólo funciona en las `p3_*`, es un dato y no un accidente.

## 4. Predicciones

**K-0 · BLOQUEANTE, el nulo.** Con el logit permutado contra sus etiquetas, el mismo buscador de 400
cortes no encuentra corte válido en más de **1 de 20** repeticiones por unidad. Es el control C-C del
19-ago, que ahí dio 0 de 20 en las cinco. **Si el procedimiento se pasa a sí mismo, no se lee nada
más.**

**K-1 · PRINCIPAL, la réplica de A3.** El corte calibrado sube `nose` en la muestra de **prueba**
contra σ>0,5, por **≥ 0,03 absoluto**, en **≥ 5 de 6** unidades, con `falsa_abst` ≤ 0,10.

Referencias ya publicadas de A3 sobre `p3_*`, con muestra compartida: +0,0353 · +0,0562 · +0,0391.
Las `b3_*` no tienen referencia y por eso el criterio pide 5 de 6 y no 6 de 6.

**K-2 · TRANSFERENCIA, y es la que decide si esto sirve.** Corte **leave-one-out**: para cada unidad
`j`, el corrimiento se calibra con las etiquetas de **las otras cinco** —mediana de sus `z*` en
unidades de desvío, que es la forma que sobrevivió el 20-ago— y se aplica a `j` **sin mirar una sola
etiqueta suya**. Retiene **≥ 60 %** de la ganancia medida en K-1, en **≥ 4 de 6**, con `falsa_abst`
≤ 0,10.

**K-3 · NO-DAÑO.** `falsa_abst` en prueba ≤ 0,10 en las 6, con los dos cortes. Un corte que detecta
más rompiendo el criterio no cuenta como ganancia.

**K-4 · RIESGO DECLARADO, la dificultad.** El 19-ago midió que `z*` es **negativo en las tareas
fáciles y positivo en las difíciles**, ≈0,3 σ en los dos grupos. Las 6 unidades de acá son todas del
nivel 3, o sea del mismo grupo. Se reporta media y desvío de `z*`, **sin criterio de éxito**.

> **Si los seis `z*` no caen dentro de una banda de ±0,15 σ, K-2 no se lee como transferencia.** Con
> los cortes dispersos, que la mediana ajena funcione sería suerte de muestreo y no un sesgo
> compartido por el nivel. Son cosas distintas y hay que poder separarlas.

**Sobre la varianza, que en este banco no es un detalle.** No se promedian las semillas y no se
reporta una media sola. La §3.1 del informe de A5 lo dejó como consecuencia aritmética y no como
convención: con la varianza medida de este banco y n=3, la cota de Chebyshev a la escala del efecto
buscado supera 1 y no autoriza ninguna afirmación. Los seis valores se reportan uno por uno.

## 5. Cómo se lee cada desenlace, escrito ANTES

| celda | lectura |
|---|---|
| **K-1 y K-2 cumplen** | calibrar es real y **no necesita las etiquetas de la unidad**. Es el resultado usable, y va al paper junto con el 19 y el 20-ago como la pieza que faltaba |
| **K-1 sí, K-2 no** | calibrar funciona pero cada unidad necesita sus propias etiquetas. Es un resultado de ingeniería, **no** responde «el modelo sabe cuándo no sabe», y hay que decirlo así |
| **K-1 no** | A3 era un artefacto de la muestra compartida. **El positivo del 26-ago se retira**, y eso es lo que la campaña vino a poder hacer |
| **K-0 falla** | el buscador se pasa a sí mismo. No se lee nada, se arregla el instrumento y se vuelve a congelar |

## 6. Criterio de abandono

> **Si K-1 falla, no se prueba una segunda regla de elección del corte sobre estas unidades.** La
> calibración post-hoc queda cerrada como vía, y lo que sigue es el experimento que el 19-ago dejó
> explícitamente afuera — entrenar con el corte corregido — que es otro pre-registro y cuesta campaña.

## 7. Lo que NO contesta

**Sigue usando etiquetas.** K-2 las toma de otras unidades, no de la propia, que es exactamente la
pregunta más débil que el 20-ago identificó en U-2: no es «el modelo sabe cuándo no sabe», es «un
corte calibrado en otros modelos de la misma familia transfiere». No hay que venderlo como otra cosa.

**No toca la estimación de dificultad sin etiquetas**, que es donde quedó la pared del 27. Si K-2
cumple, el hueco que queda es cómo elegir el signo del corrimiento cuando no hay una familia de
unidades hermanas ya etiquetadas.

**Y no dice nada sobre escala.** 863.730 parámetros, idioma sintético de 242 tokens, `p_nose` 0,4
fijo, un solo nivel de dificultad.
