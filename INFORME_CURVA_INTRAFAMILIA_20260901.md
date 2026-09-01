# La pendiente floja era el artefacto · recuperar mejor SÍ compra detección, y no alcanza

**2026-09-01.** §4 del `PLAN_20260901.md`, hecho bien. Lectura congelada en
`NOTA_CURVA_INTRAFAMILIA.md` (SHA **ee781db0**) antes del dato. 17 unidades en Colab
(`medir_en_colab.sh`, cuenta C, T4, ~8 min contra ~85 min de CPU en la PC).

---

## 1. El punto de partida, y por qué no se podía leer

La curva de 10 unidades dio **r = +0,8643 · pendiente +0,1308** y sugería *debilitar* la frase «toda
mejora de la abstención pasa por recuperar mejor». **Esa lectura era inválida por dos razones**, las
dos escritas antes de rehacer la medición: el confound de T-4 no se había resuelto sino **replicado**
(RECUP bajo = 3000 pasos con recompensa, RECUP alto = 12000 sin ella), y la `r` estaba **inflada por
tener dos nubes separadas en el eje x**, con dispersión intra-nube (0,063) del mismo tamaño que la
distancia entre nubes (0,060).

## 2. La medición limpia

Las 17 unidades comparten **presupuesto (26000 pasos), pérdida, arquitectura, `lr` y `p_nose`=0,4**.
Lo único que varía es `donde` (la posición de la lectura) y la **semilla**. El confound de T-4 no
existe en este conjunto.

| conjunto | n | r | Spearman | pendiente |
|---|---:|---:|---:|---:|
| p3+q3+v3+w3 | 12 | +0,9758 | +0,8601 | **+0,4403** |
| las 17 (con `b3`) | 17 | +0,9721 | +0,8676 | +0,5046 |
| **WITHIN** (residuos centrados por familia) | 12 | +0,8031 | — | **+0,5350** |

**La pendiente WITHIN es la que decide**: al centrar por familia elimina el efecto de `donde`, así
que mide sólo lo que la semilla mueve dentro de condiciones idénticas. Da **+0,5350**, cuatro veces
la pendiente global confundida.

> **La pendiente floja era el artefacto, no el fenómeno.** La frase «toda mejora de la abstención
> pasa por recuperar mejor» **queda en pie**, y ahora sin el confound que invalidaba T-4.

Por familia: `p3` (pre) +0,7477 con **r = +0,9993** sobre tres puntos · `w3` (lat) +0,1632 pero **no
monótona** (RECUP 0,9979 tiene MENOS techo que 0,8812) · `q3` y `v3` **no informan** por rango de
RECUP < 0,10, tal como el riesgo declarado en la nota anticipaba. `b3`, analizada aparte por entrenar
con `blanco=error`, es la de rango más grande (0,3666→0,9996) y da **+0,5838 con r = +0,9831**.

**★ Y haberla dejado aparte resultó más importante de lo que decía la razón que se dio.** `b3` **no es
una familia homogénea**: `ENMIENDA_PERDIDA_CABEZA.md` (29-ago) ya había establecido que **`b3_s0/s1/s2`
arrancaron desde un modelo con RECUP ≈ 0,78 y `b3_s3`…`b3_s8` arrancaron de cero** — dos poblaciones,
no nueve semillas. Su pendiente de +0,5838 mezcla «la semilla» con «el punto de partida», así que **no
habría podido usarse** aunque no tuviera el `blanco=error`. La homogeneidad de `p3`, en cambio, está
documentada desde el `INFORME_BIMODALIDAD_20260822.md`: son «las tres unidades `pre` de la campaña de
la mañana, a igual presupuesto y con todo lo demás igual», y su dispersión entre semillas tiene causa
identificada (el **atajo de la relación**, `ac_repetida` 0,9425 · 0,5317 · 0,7471).

**Pero ese contraste de poblaciones aporta algo que la curva sola no puede dar, y va en la misma
dirección: es una INTERVENCIÓN, no una correlación.** Lo que se manipuló fue el punto de partida, y
el desenlace se alineó casi perfecto — **arrancar con RECUP ≈ 0,78 dio 3 de 3 unidades útiles; arrancar
de cero dio 2 útiles y 4 mudas**. Es evidencia causal, y del único tipo que hay en todo esto, a favor
de que **la recuperación previa decide si la abstención llega a existir**.

## 3. ⚠ El veredicto automático, otra vez, no se puede leer como salió

El juez imprimió **«m = +0,4555 ≥ 0,40 → SE REFUERZA»**, y el criterio pre-registrado se cumple
formalmente. **Pero ese número es la mediana de dos valores que caen en lados opuestos de los
umbrales** (`p3` +0,7477 → refuerza; `w3` +0,1632 → casi debilita), y con dos familias la mediana es
el promedio: un número que no representa a ninguna de las dos.

**No se cuenta como confirmación del criterio.** Lo que sostiene la conclusión es la estimación
WITHIN (+0,5350) sobre las 12 unidades a la vez, que apunta al mismo lado con mucha mejor evidencia.
La conclusión no cambia; **la razón por la que se la cree, sí**.

Es la cuarta vez que un juez automático de este proyecto imprime algo que sus propios números no
sostienen. **Se agregó una guarda de discordancia dentro del juez** —si con menos de tres familias
caen en lados opuestos, devuelve NO LEÍBLE en vez de una mediana—, y se declara que **la guarda es
post-hoc**: se escribió al ver el defecto, no antes.

## 4. ★ El hallazgo que acota la tesis, y no depende de ninguna pendiente

**`v3` (lat2) tiene RECUP = 1,0000 EXACTO en sus tres semillas, y su techo es 0,9258 · 0,9274 ·
0,9638.**

> **Con recuperación perfecta la ausencia todavía no es perfectamente legible.** Queda entre 4 y 7
> puntos que la recuperación no explica.

Así que la frase correcta ya no es «toda mejora pasa por recuperar mejor» sino la más precisa:
**recuperar mejor compra detección a razón de ~0,53 de AUC por punto de RECUP, y aun así no la
agota.** Es una tesis más fuerte que la anterior porque dice cuánto, y más honesta porque dice dónde
se corta.

El complemento está en `b3`: con **`blanco=error` entrenado** llega a techo **1,0000 exacto** (RECUP
0,9996). Coherente con el precedente del proyecto —el blanco `error` da 0,65 post-hoc y 1,0000
entrenado— y sugiere que ese residuo se cierra **entrenando la detección**, no mejorando la búsqueda.

## 5. Lo que esto NO dice

- **No reinterpreta el techo de 0,7003 del 31-ago.** Es tentador leer «`n3_s0` da 0,70 porque recupera
  0,785, y con RECUP 1,0 daría 0,93», pero **`n3_s0` tiene `p_nose`=0,0** (entrenado sin una sola
  pregunta sin respuesta) contra 0,4 de todas las de hoy, así que la comparación confunde RECUP con
  `p_nose` —y además con el presupuesto (12000 contra 26000) y con la interfaz (`token` contra
  `cabeza`)—.

  **★ Corrección a este mismo informe, sobre su primera versión:** decía que la comparación «queda
  como hipótesis barata de medir, bastan unidades de `p_nose`=0,4 con RECUP ≈ 0,78». **Ese dato ya
  está en la tabla de arriba y no lo vi: `p3_s1` tiene RECUP 0,7945**, a nueve milésimas del 0,7850
  de `n3_s0`, **y su techo es 0,8441 contra 0,7003**. O sea, **a recuperación prácticamente idéntica
  el techo difiere en +0,1438**. Eso NO se puede atribuir a `p_nose` porque los tres confounds viajan
  juntos, pero sí deja algo que no depende de cuál sea la causa: **RECUP no determina el techo por sí
  sola**, y el mismo punto del eje x admite techos muy distintos. Es evidencia independiente del
  hallazgo de `v3` en el §4, y por el otro lado: allá la recuperación estaba saturada y acá está en
  el medio. **Para separar las tres causas hace falta una corrida nueva variando una sola.**
- **La variación de RECUP viene de la semilla**, o sea de la biestabilidad ya documentada, no de una
  intervención. Es correlacional: nadie movió RECUP a propósito.
- `q3` y `v3` no informaron la pendiente, y `w3` es no monótona. La evidencia intra-familia limpia
  descansa sobre todo en `p3`.
- Un solo nivel (3), una sola arquitectura, un solo `p_nose`.

## 6. Infraestructura: el objetivo del §3 del plan quedó cumplido

`medir_en_colab.sh` **probado y en uso**: la cuenta A dio 503, la C consiguió T4, y las dos corridas
(10 y 17 unidades) subieron, corrieron y bajaron el JSON. La curva de 17 tardó **~8 minutos** contra
los ~85 que habría costado en la PC. Se le agregó pasar los checkpoints **como argumentos** al script
de medición, para no depender de listas hardcodeadas adentro de cada script.
