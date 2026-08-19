# El techo de `c3_s0` es de calibración — ahora prospectivo, y el corte se puede fijar de una vez

**Prereg:** `PREREG_UMBRAL_PROSPECTIVO.md`, congelado (SHA `b0ef03b9…`) antes de escribir una línea
del script y antes de mirar un número que no estuviera ya publicado.
**Unidades:** 5 checkpoints ya entrenados. **Muestras:** 2048 por unidad y por muestra, ocho veces las
de la sonda del 18-ago. **Sin GPU.**

---

## §1 · Qué se estaba cerrando

`INFORME_CABEZA_20260819.md` §4 bis dijo que el techo de `c3_s0` —la única unidad que no pasa con
`cabeza` en toda la serie— parecía de **calibración** y no de capacidad, y dijo también que no valía
como confirmación: era post-hoc, y elegía y juzgaba el umbral sobre **dos mitades del mismo muestreo**.

Acá las dos muestras son independientes en los datos: el umbral se elige con el generador `90000+s` y
se juzga con el `77000+s`, fijado en el prereg antes de correr. Son sesiones y hechos distintos, no
dos vistas de los mismos.

## §2 · Las cuatro predicciones cumplen

| unidad | AUC (prueba) | con σ>0,5 · `f_abst` / `nose` | `a*` | en PRUEBA con `a*` | |
|---|---:|---|---:|---|---|
| c1_s0 | 0,983 | 0,0113 / 0,9261 · pasa | −2,234 | 0,0647 / 0,9643 | pasa |
| c2_s0 | 0,980 | 0,0048 / 0,8460 · pasa | −1,955 | 0,0724 / 0,9329 | pasa |
| **c3_s0** | **0,825** | **0,1177 / 0,6115 · falla** | **+0,419** | **0,0536 / 0,5502** | **pasa** |
| c3_s1 | 0,821 | 0,0900 / 0,5866 · pasa | +0,145 | 0,0727 / 0,5747 | pasa |
| c3_s2 | 0,809 | 0,0937 / 0,5533 · pasa | +0,171 | 0,0627 / 0,5343 | pasa |

- **U-1 cumple.** AUC de `c3_s0` 0,825, sobre el 0,75 pedido.
- **U-2 cumple, y es la principal.** `a*` = +0,419 se eligió en ajuste y se imprimió antes de tocar la
  muestra de prueba. Ahí `c3_s0` cumple la compuerta con holgura: `falsa_abst` **0,0536** contra el
  límite de 0,10.
- **U-3 cumple.** `a*` > 0, o sea el corte correcto pide **más** evidencia para callarse. Era la
  dirección predicha: la unidad fallaba por abstenerse de más.
- **U-4 cumple.** Las dos unidades fáciles no se rompen.

## §3 · Los controles, que es donde el resultado se gana el peso

**C-C · el nulo está limpio.** Con el logit permutado contra sus etiquetas, el mismo buscador de 400
cortes **no encuentra ni un solo `a*` válido**: 0 de 20 repeticiones en las cinco unidades. El
procedimiento no se pasa a sí mismo. Sin esto, U-2 no diría nada.

**C-B · transfiere, y esto es lo operativamente útil.** El `a*` de `c3_s0` aplicado **tal cual** a las
otras dos unidades del nivel 3 las hace pasar a las dos (0,0446/0,5544 y 0,0468/0,5228). El sesgo
pertenece al **nivel**, no a la unidad: se fija una vez y sirve para las tres.

**C-A · no dio lo que el prereg planteaba, y dio algo mejor.** El prereg preguntaba si el `z*` caía
cerca de 0 en todas —lo que habría significado un bias global. No cae en 0: cae **separado por
dificultad**.

| | c1_s0 | c2_s0 | c3_s0 | c3_s1 | c3_s2 |
|---|---:|---:|---:|---:|---:|
| `z*` | −0,368 | −0,255 | **+0,379** | **+0,280** | **+0,342** |

**Negativo en las dos fáciles, positivo en las tres difíciles.** σ>0,5 no está mal puesto en general:
está mal puesto **en direcciones opuestas según la dificultad de la tarea**. En las fáciles habría que
abstenerse algo más de lo que dicta el corte; en las difíciles, bastante menos. Y en unidades de
desvío el corrimiento es del mismo tamaño en los dos grupos, ≈0,3 σ.

Es el mismo patrón que `sonda_normas.py` encontró el 18-ago por otra vía —una intervención que hace
cosas opuestas según la unidad— y refuerza que `NOSE` arrastra su inicialización de la campaña base.

## §4 · La alternativa que había que descartar, y cómo se descartó

`c3_s0` fallaba el criterio **por 0,0177**. Está pegada al borde, y la lección que `sonda_umbral.py`
dejó el 18-ago —y que la celda cruzada de esta mañana repitió del otro lado— es que **lo pegado al
borde no es estable**. La alternativa obvia era: no se movió nada, la unidad oscila alrededor del
límite de muestra en muestra, y el «pasa» con `a*` es esa oscilación.

`verificar_umbral_estable.py` (post-hoc, declarado como tal) la mide con tres muestras independientes
—se agrega el generador `55000+s`— y la descarta **por los dos lados**:

**1 · `a*` es estable.** En `c3_s0` los tres valores son **0,419 · 0,301 · 0,420**, desvío 0,056. Hay
un umbral que fijar, no un número que salta.

**2 · La falla con σ>0,5 tampoco es ruido.** `falsa_abst` de `c3_s0` en las tres muestras:

| | m1 | m2 | m3 | media | desvío |
|---|---:|---:|---:|---:|---:|
| c3_s0 | 0,1240 | 0,1177 | 0,1356 | **0,1258** | **0,0074** |

**Falla 3 de 3**, y el desvío entre muestras (0,0074) es menos de un tercio de lo que le falta para el
criterio (0,0258 desde la media). No es que a veces pase: **falla siempre, y por un margen que el
ruido de muestreo no explica**.

**3 · Nueve de nueve.** `a*` elegido en cada muestra y medido en las tres, incluidas las dos que no
vio: `c3_s0` **pasa en las 9 combinaciones**, y lo mismo las otras cuatro unidades. 45 de 45 cruces.

→ **Las dos cosas son estables**: la falla con el corte del prereg y el pase con el corte calibrado.
Ahí sí se puede escribir la conclusión.

## §5 · Qué queda dicho

**La información para decidir cuándo callarse está en el logit de `c3_s0`. σ>0,5 no es donde había que
leerla, y el corte correcto se puede fijar una sola vez para todo el nivel 3.** La campaña de la
cabeza pasa de «4 de 5» a **«5 de 5 con el corte calibrado»**, con el asterisco de que la
calibración es un paso posterior al entrenamiento y no estaba en el prereg de la campaña.

**Y lo que no queda dicho**, porque el prereg lo acotó antes:

- Esto mide **checkpoints ya entrenados**. **No** demuestra que entrenar con el umbral corregido dé un
  modelo mejor. Ese es otro experimento y otro prereg.
- El `a*` se elige **con etiquetas**, o sea sabiendo qué preguntas tenían respuesta. En uso real esa
  información no está, y de dónde sale el corte —una porción de validación, una regla sobre el
  cuantil— es una pregunta abierta que este informe no toca.
- Son **5 unidades**, todas del mismo micro-LM y del mismo archivo.
