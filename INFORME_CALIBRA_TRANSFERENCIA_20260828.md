# INFORME · calibrar SÍ, transferir NO — y un criterio mío mal construido

Evalúa `PREREG_CALIBRA_TRANSFERENCIA.md` (SHA `5fdab03d…`), congelado a las 09:33 antes de escribir
una línea del instrumento. Datos en `calibra_transf_20260828.json`, log en el `.log` del mismo nombre.

6 unidades del nivel 3 a 26000 pasos · n = 6000 de ajuste + 6000 de prueba **por unidad**, con
generadores independientes (31000+s y 42000+s, ninguno usado antes en el repo) · CPU, sin pool.

---

## 1. Lo medido

| unidad | base σ>0,5 | **calibrado** | ganancia | oráculo | transferido | `f_abst` transf. | `z*` |
|---|---:|---:|---:|---:|---:|---:|---:|
| b3_s0 | 0,9992 | **1,0000** | +0,0008 | 1,0000 | 0,9996 | 0,0020 | −0,5636 |
| b3_s1 | 0,9988 | **1,0000** | +0,0012 | 1,0000 | 0,9992 | 0,0008 | −0,6378 |
| b3_s2 | 0,5912 | 0,5359 | **−0,0553** | 0,5470 | 0,7017 | **0,4424** | −0,0956 |
| p3_s0 | 0,9130 | 0,9538 | +0,0409 | 0,9587 | 0,9352 | 0,0278 | −0,3670 |
| p3_s1 | 0,5284 | 0,5898 | +0,0614 | 0,6087 | 0,7577 | **0,2609** | −0,2750 |
| p3_s2 | 0,7054 | 0,7426 | +0,0371 | 0,7607 | 0,8812 | **0,3784** | −0,1041 |

## 2. Los criterios, contra lo medido

| | criterio | resultado | |
|---|---|---|---|
| **K-0** bloqueante · el nulo | ≤ 1/20 por unidad | **0/20 en las seis** | **CUMPLE** |
| **K-1** principal | ganancia ≥ 0,03 en ≥ 5/6 | **3/6** | **NO CUMPLE** |
| **K-2** transferencia | retiene ≥ 60 % en ≥ 4/6 | **0/6** | **NO CUMPLE** |
| **K-3** no-daño | `f_abst` ≤ 0,10 en las 6 | 3/6 | **NO CUMPLE** |
| **K-4** riesgo · banda de `z*` | reportar | rango [−0,638, −0,096], desvío 0,208 | **fuera de ±0,15** |

**K-0 es lo que le da peso a todo lo demás.** Con el logit permutado contra sus etiquetas, el mismo
buscador de 400 cortes no encuentra **ni una vez** en 120 intentos un corte que mejore la detección
sin romper el criterio. El procedimiento no se pasa a sí mismo.

## 3. K-1 falla, y el criterio estaba mal construido — mío, no de los datos

Hay que decirlo antes de leer nada más, porque cambia cómo se lee la fila.

Las tres unidades que fallan K-1 son `b3_s0` (+0,0008), `b3_s1` (+0,0012) y `b3_s2` (−0,0553). En las
dos primeras **la ganancia de 0,03 era imposible**: la base ya vale 0,9992 y 0,9988, así que el techo
alcanzable era +0,0008 y +0,0012. Y las dos **llegan a 1,0000**, o sea al máximo absoluto, igualando
a su propio oráculo.

> Es **exactamente** el defecto que el §4 del informe de A5 dejó anotado el 27 como lección para el
> próximo pre-registro: *un umbral de mejora absoluta necesita verificar antes cuánto espacio queda
> hasta el techo*. El próximo pre-registro fue éste, y el defecto se repitió igual.

**No se usa para rescatar K-1.** El criterio se escribió antes y falla; la vía que K-1 juzgaba se
juzga como corresponde. Queda anotado para que la regla se aplique de una vez: **todo criterio de
mejora absoluta se acompaña del margen al techo de cada unidad, calculado antes de congelar.**

## 4. Lo que sí quedó establecido: A3 replica

A3 (26-ago) fue exploratorio y sin pre-registro, sobre las tres `p3_*` con muestra compartida. Acá se
repitió con pre-registro y con muestras independientes por unidad:

| unidad | A3 · 26-ago | **hoy, con prereg** |
|---|---:|---:|
| p3_s0 | +0,0353 | **+0,0409** |
| p3_s1 | +0,0562 | **+0,0614** |
| p3_s2 | +0,0391 | **+0,0371** |

**3 de 3, y con el nulo limpio.** El positivo del 26-ago **no era** un artefacto de la muestra
compartida, que es lo que esta campaña existía para poder descartar. Calibrar con las etiquetas
propias sube la detección entre +0,037 y +0,061 sin costo, y queda a 0,005-0,019 del oráculo.

## 5. El resultado de la campaña es K-2, y es un NEGATIVO limpio

**El corte no transfiere entre estas unidades.** Ninguna de las seis retiene el 60 % de la ganancia
con el corte de las otras cinco, y en tres de seis el corte transferido **rompe el criterio de falsa
abstención**, con márgenes que no admiten discusión: 0,4424 · 0,2609 · 0,3784 contra un límite de
0,10. Un corte que se abstiene del 44 % de las preguntas que sí tenían respuesta no es utilizable.

**K-4 explica por qué, y estaba previsto.** El pre-registro fijó que si los `z*` no caían en una banda
de ±0,15 σ, K-2 no se leería como transferencia. No caen: van de −0,0956 a −0,6378, con desvío 0,208.
**No hay un sesgo compartido por el nivel**, hay seis cortes distintos. La mediana ajena que K-2 usa
cae en −0,275 o −0,367 según la unidad, y aplicarla a una unidad cuyo corte propio es −0,0956 la hace
abstenerse muchísimo de más. Es aritmética, no mala suerte.

**Y esto no contradice el 20-ago, lo acota.** Allá U-2 transfería 7/8, pero eran ocho unidades de la
misma familia `c` y del mismo régimen. Acá hay **dos familias con blancos distintos** (`error` y
`ausencia`) y capacidades muy dispares —de `nose` 0,53 a 0,999— y el corrimiento normalizado deja de
ser común. La transferencia del corte necesita unidades homogéneas, y en cuanto la familia se mezcla,
se cae.

## 6. Veredicto, según el §5 del pre-registro

La celda que corresponde es **«K-1 sí, K-2 no»** en su contenido —calibrar funciona, transferir no—
aunque K-1 falle en la letra por el defecto del §3 de este informe. Su lectura, escrita antes:

> *calibrar funciona pero cada unidad necesita sus propias etiquetas. Es un resultado de ingeniería,
> **no** responde «el modelo sabe cuándo no sabe», y hay que decirlo así.*

Se aplica el **criterio de abandono del §6**: no se prueba una segunda regla de elección del corte
sobre estas unidades. La calibración post-hoc queda cerrada como vía.

## 7. Un dato lateral que refuerza el hallazgo de la mañana

`b3_s0` y `b3_s1` alcanzan `nose` = **1,0000** con el corte calibrado, igualando a su oráculo, con
falsa abstención 0,0762 y 0,0627. Y ya sin calibrar estaban en 0,9992 y 0,9988.

En estas dos unidades **no queda brecha de calibración que cerrar**. Lo que el proyecto llamó «techo
de calibración» durante semanas no existe acá: existe en `b3_s2`, en las `p3_*` y en las campañas
viejas. Es consistente con `HALLAZGO_PUNTO_PROPIO_20260828.md` y es una razón más para que
`PREREG_TASA_REGIMEN.md` (SHA `dc62ecae…`, corriendo) diga si eso es el modo típico o dos semillas.

## 8. Lo que NO dice

- **Sigue siendo supervisado** de punta a punta. El §8 del `PLAN_FOCO_20260824.md` y su cierre de seis
  meses no se tocan.
- **Seis unidades, un nivel, un modelo de 863.730 parámetros**, idioma de 242 tokens, `p_nose` 0,4.
- **No se promedian las semillas.** Las seis van una por una, y `b3_s2` —que empeora al calibrar— no
  se descarta por ser incómoda.
