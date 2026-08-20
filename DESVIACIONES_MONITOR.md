# Desviaciones · `PREREG_MONITOR_DESACUERDO.md` (SHA `b259fd0d…`)

---

## D-1 · El prereg v1 queda ANULADO: la perturbación era una no-operación

**Qué pasó.** El smoke de una unidad (`c3_s0`, 64 muestras) devolvió **cero en todo**: AUC 0,000,
`falsa_abst` 0,0000, `nose` 0,0000 y la diferencia de M-3 en +0,0000. O sea `consistencia = 1` en
todas las muestras, incluso al tapar una entrada del archivo.

**Diagnóstico, hecho antes de tocar nada** (porque en este proyecto un cero limpio ya fue siete veces
un artefacto). Había dos explicaciones posibles y se separaron con una medición:

- que la permutación **no se estuviera aplicando** (bug de indexado), o
- que se aplicara y **no cambiara nada** (equivarianza).

Resultado: el tensor del archivo **sí cambia** al permutar (`allclose` = falso) y aun así
`max|logit_original − logit_permutado| = 5,7e-06`. **Equivarianza exacta.**

**Por qué era inevitable, y esto es lo importante:** la lectura del archivo es una atención softmax,
`Σ_n softmax(sim)_n · v_n`, que es **invariante a permutar `n` por construcción algebraica**. El §2
del prereg usó exactamente esa propiedad para argumentar que la respuesta *correcta* no cambia al
permutar — **y esa misma propiedad hace que NINGUNA respuesta cambie, esté anclada en evidencia o
sea pura adivinanza.** El argumento que justificaba la perturbación es el que la vacía.

**Por qué esto no es «M-1 falla».** El §5 dice que si M-1 falla no se prueba una segunda
perturbación. Esa regla existe para impedir que, ante un resultado adverso, uno siga probando
variantes hasta que alguna dé. **Acá no hay resultado adverso: no hay resultado.** El estadístico
vale 1 idénticamente por una identidad algebraica, así que el experimento no podía producir ni un
positivo ni un negativo — es un instrumento vacío, no una medición. Se anula el v1 entero y se
escribe `PREREG_MONITOR_DESACUERDO_V2.md` con una perturbación que **rompe** la equivarianza.

**Lo que se salva del v1, y no es poco:** la motivación del §1 (el desacuerdo tiene un cero
interpretable y el logit no tiene escala absoluta) sigue en pie sin cambios, y **M-4 hizo su trabajo**:
el nulo con permutación identidad dio 1,000 exacto, que es lo que descartó el ruido numérico y dejó
la equivarianza como única explicación.

**Costo real de la equivocación: un smoke de 64 muestras.** Es la razón por la que el smoke corre
sobre una unidad antes que sobre las ocho.
