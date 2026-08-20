# El corte sin etiquetas — NEGATIVO, y el nulo explica por qué

`PREREG_CORTE_SIN_ETIQUETAS.md` (SHA `17e0a35e…`, congelado antes de mirar un solo logit) ·
desviaciones en `DESVIACIONES_SIN_ETIQUETAS.md` · datos en `corte_sin_etiquetas_20260820.json`.

8 unidades de la familia `c` (condición `cabeza`), **todas a 14000 pasos**, 2048 muestras de ajuste y
2048 de prueba por unidad, con generadores independientes (rng 90000+s y 77000+s).

---

## 1 · El resultado

| | resultado | criterio | |
|---|---|---|---|
| **S-1** principal · U-1 pasa | **2 / 8** | ≥ 6/8 | **NO CUMPLE** |
| **S-2** costo contra el oráculo | caída media −0,053, pero `falsa_abst` hasta **0,3093** | ≤ 0,10 en las dos | **NO CUMPLE** |
| **S-3** el nulo | **2 / 8** unidades con tasa > 0,5 | ≤ 1/8 | **NO CUMPLE** |
| **S-4** necesidad · σ>0,5 falla | 2 / 8 | ≥ 6/8 | **NO CUMPLE** (criterio mío mal puesto, ver D-2) |
| **S-5** el signo | **5 / 8** | ≥ 7/8 | **NO CUMPLE** |

**Según el §5, comprometido por adelantado: S-1 y S-5 fallan ⇒ la idea no cierra.** Se reporta el
negativo, **no se prueba un cuarto estimador**, y el corte sin etiquetas queda como problema abierto.

## 2 · Lo que hace que el negativo sea informativo y no sólo un «no dio»

**U-1 pierde contra su propio nulo.**

| unidad | U-1 pasa | el nulo pasa |
|---|---|---:|
| `c1_s0` | **sí** | **99 / 100** |
| `c2_s0` | **sí** | **100 / 100** |
| `c3_s0` · `c3_s1` · `c3_s2` | no | 6 · 12 · 2 / 100 |
| `c4_s0` · `c4_s1` · `c4_s2` | no | 7 · 2 · 0 / 100 |

**Las dos únicas unidades donde U-1 pasa son exactamente las dos donde pasa cualquier corte.** Con el
logit reemplazado por una gaussiana de igual media y desvío —o sea sin ninguna estructura que
estimar— la compuerta se pasa igual de bien. En las seis unidades donde el nulo sí discrimina, U-1 no
pasa en ninguna. **No hay una sola unidad en la que la mezcla aporte información.**

Esto es exactamente lo que el nulo estaba puesto para poder mostrar, y por eso importa haberlo
elegido bien: **permutar etiquetas no habría servido de nulo**, porque U-1 no las mira y el corte no
se habría movido. Habría dado «limpio» sin significar nada.

**Y el piso está más alto que el estimador.** σ>0,5 —el criterio sin calibrar, que tampoco usa
etiquetas— pasa en **6 de 8**. La mezcla de dos gaussianas es **peor que no hacer nada**.

## 3 · El signo, que el prereg declaró como el test real

El §2 lo dejó escrito antes de correr: «el signo depende de la dificultad, y la dificultad tampoco se
conoce en producción; si el signo no se puede inferir de la propia distribución, la idea no cierra.
**Ése es el test real, no la magnitud.**»

La asimetría da **positiva en siete de las ocho unidades**, así que la regla predice el mismo signo
para todas y se equivoca justo en las dos fáciles, que son las de `z*` negativo.

**Y hay algo peor que descubrió la desviación D-1.** La asimetría de `c4_s2` vale **−0,120** con el
checkpoint del paso 14000 y **+0,075** con el del 15000 que se había colado en la primera corrida:
**mil pasos de entrenamiento le dan vuelta el signo al estadístico del que dependía toda la regla.**
No es sólo que no infiera la dificultad — no es estable ni dentro de la misma unidad. Ese dato salió
de un error de procedimiento, pero una vez medido vale, y vale en contra de U-2.

## 4 · Lo que sí sobrevive

**U-2 pasa en 7 de 8**, mejor que σ>0,5 (6/8) y que U-1 (2/8). La constante `z̄ ≈ 0,35 σ` transferida
entre unidades funciona. **Pero necesita etiquetas de otras unidades para estimar `z̄` y el signo
puesto a mano**, así que responde una pregunta más débil que la del §1 del plan: no es «el modelo sabe
cuándo no sabe», es «un corte calibrado en otros modelos de la misma familia transfiere». Es un
resultado útil de ingeniería y no hay que venderlo como otra cosa.

**U-3 (cuantil de la tasa base) pasa 2/8**, las mismas dos fáciles. Como el prereg obligaba a decirlo:
U-3 no es mejor que U-1, pero tampoco hacía falta — las dos son irrelevantes frente a σ>0,5.

## 5 · Alcance, sin estirar

Ocho unidades del mismo modelo (863.730 parámetros) sobre un idioma cerrado de 242 tokens, con
`p_nose = 0,4` fijo. Es un negativo de **mecanismo**: dice que **este** estimador no encuentra el
corte, no que ningún estimador sin etiquetas pueda hacerlo. La sensibilidad a la tasa base no se
midió y sigue siendo la continuación natural.

## 6 · Post-hoc declarado (§7) · por qué falla, y es más preciso de lo esperado

`posthoc_mezcla.py`, sobre la misma muestra de ajuste. **No decide ningún veredicto**: los criterios
ya se juzgaron arriba.

| unidad | separación de la mezcla | ¿bimodal? | valle `z` | separación REAL de las dos poblaciones |
|---|---:|---|---:|---:|
| `c1_s0` | **5,59 σ** | sí | −0,059 | +1,889 |
| `c2_s0` | **4,06 σ** | sí | +0,043 | +1,758 |
| `c3_s0` | 0,55 σ | no | +0,081 | +1,207 |
| `c3_s1` | 0,66 σ | no | +0,107 | +1,161 |
| `c3_s2` | 0,56 σ | no | +0,154 | +1,213 |
| `c4_s0` | 0,63 σ | no | +0,108 | +1,188 |
| `c4_s1` | 0,61 σ | no | +0,058 | +1,271 |
| `c4_s2` | 0,21 σ | no | +0,051 | +1,151 |

**Seis de ocho mezclas no son bimodales**, y las dos que sí lo son (5,59 y 4,06 desvíos de
separación) son exactamente las dos unidades fáciles — o sea aquellas donde el nulo ya pasaba
99-100/100 y no hacía falta estimar nada. **U-1 acierta sólo donde el problema no existe.**

**Lo que lo explica del todo está en la última columna.** Las dos poblaciones reales —preguntas con
respuesta y sin respuesta— están separadas **1,15 a 1,27 σ** en las seis unidades difíciles: la
estructura *existe*, y es la misma que el 19-ago se midió como AUC 0,825. Pero una mezcla de dos
gaussianas necesita ~2 σ de separación para que la densidad tenga dos modas; con 1,2 σ **la suma es
unimodal**. No hay ningún valle que encontrar, y el EM termina partiendo en dos una masa de una sola
cima: ajusta componentes con 0,6 σ de separación cuando las clases están a 1,2 σ, o sea **ni siquiera
recupera las poblaciones que existen**. Por eso el corte cae en `z` entre +0,05 y +0,15 —casi la
media, donde está el grueso de la masa— en vez del +0,35 del oráculo, y por eso abstiene de más.

**La moraleja, que es lo que hay que llevarse para el próximo intento: la información está, pero no
en forma de valle.** Cualquier estimador que la busque en la *forma de la densidad* va a fracasar por
la misma razón; el que la busque tiene que apoyarse en otra cosa.
