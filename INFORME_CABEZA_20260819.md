# La cabeza de abstención — informe final, con las 21 unidades cerradas

**Prereg:** `PREREG_CABEZA_ABSTENCION.md`, congelado 2026-08-18 antes de implementar.
**Desviaciones:** `DESVIACIONES_CABEZA.md`.
**Campaña:** 7 unidades × 3 condiciones = **21**, corridas el 18-ago sobre 13 cuentas de Colab;
la última (`c4_s1`) se verificó el **19-ago**.

---

## §1 · La pregunta

Del 17-ago quedó un negativo incómodo: de nueve modelos entrenados con `NOSE` como una entrada más
del vocabulario, **cinco no aprendían a callarse**, y lo que los separaba era el «margen sobre el
atajo» — cuánto le sacaba el modelo a la política de *no abstenerse nunca*. `nose` nunca era el
problema (0,57-0,98 en los nueve: **todos detectan la ausencia**); lo que fallaba siempre era
`falsa_abst`, o sea **se callaban de más**.

Y había una pista leída de los pesos: el vector de `NOSE` medía **0,367** de norma contra 1,011 de
«ana» y 1,028 de «beto». Competía en el mismo softmax con un vector 3× más corto.

De ahí las tres condiciones, todas partiendo **del mismo checkpoint base, con Adam reinicializado en
las tres** (por eso la campaña `token` del 17-ago no se reusa como línea de base):

- **`token`** — control pareado: `NOSE` es una entrada más del softmax de valores.
- **`escala`** — la explicación barata: se renormaliza el vector de `NOSE` a la norma media de los
  valores, en entrada y en salida. **Es el control que podía ahorrar la arquitectura entera.**
- **`cabeza`** — salida binaria separada, **+129 params sobre 863.730 = 0,015 %**, con `NOSE`
  excluido del softmax de valores.

Compuerta, fijada el 15-ago y sin tocar desde entonces: **`nose` ≥ 0,50 y `falsa_abst` ≤ 0,10**. Las
dos mitades juntas, porque un modelo que contesta `NOSE` a todo saca `nose` = 1,000 y hay que poder
verlo.

---

## §2 · Las 21 unidades

Todas a 14000 pasos, `p_nose` 0,4, truncamiento 0,000.

| unidad | condición | `vigente` | `nose` | `falsa_abst` | compuerta |
|---|---|---:|---:|---:|:--|
| 1_s0 | token | 0,9157 | 0,9046 | 0,0667 | pasa |
| 1_s0 | escala | 0,9286 | 0,9258 | 0,0606 | pasa |
| 1_s0 | **cabeza** | 0,9789 | 0,9064 | 0,0118 | pasa |
| 2_s0 | token | 0,9762 | 0,8919 | 0,0205 | pasa |
| 2_s0 | escala | 0,9682 | 0,8498 | 0,0231 | pasa |
| 2_s0 | **cabeza** | 0,9686 | 0,8449 | 0,0029 | pasa |
| 3_s0 | token | 0,5828 | 0,6378 | 0,2246 | falla |
| 3_s0 | escala | 0,5501 | 0,6538 | 0,2805 | falla |
| 3_s0 | **cabeza** | 0,6496 | 0,5826 | 0,1189 | **falla** |
| 3_s1 | token | 0,6388 | 0,5765 | 0,1757 | falla |
| 3_s1 | escala | 0,5976 | 0,5927 | 0,2086 | falla |
| 3_s1 | **cabeza** | 0,6384 | 0,5406 | 0,0970 | **pasa** |
| 3_s2 | token | 0,5656 | 0,6379 | 0,2237 | falla |
| 3_s2 | escala | 0,5064 | 0,6664 | 0,2951 | falla |
| 3_s2 | **cabeza** | 0,6584 | 0,5962 | 0,0841 | **pasa** |
| 4_s0 | token | 0,7047 | 0,7106 | 0,1342 | falla |
| 4_s0 | escala | 0,6025 | 0,7128 | 0,2436 | falla |
| 4_s0 | **cabeza** | 0,7028 | 0,6150 | 0,0898 | **pasa** |
| 4_s1 | token | 0,6909 | 0,7000 | 0,1713 | falla |
| 4_s1 | escala | 0,6583 | 0,6996 | 0,1691 | falla |
| 4_s1 | **cabeza** | 0,7081 | 0,6189 | 0,0746 | **pasa** |

Se reporta **por unidad y nunca sólo la media**: la bimodalidad entre semillas está medida desde
E-I3c y una media taparía justamente lo que interesa.

---

## §3 · Los cuatro veredictos

**P-1 (principal) CUMPLE.** Sobre las 5 unidades difíciles:

| condición | pasa la compuerta |
|---|---|
| `token` | **0 de 5** |
| `escala` | **0 de 5** |
| `cabeza` | **4 de 5** (3_s1 · 3_s2 · 4_s0 · 4_s1) |

Mismo checkpoint base, mismo presupuesto de 2000 pasos, Adam reinicializado en las tres.
**Lo único que cambia es dónde se toma la decisión de abstenerse.** La única que no pasa, 3_s0, se
queda en `falsa_abst` 0,1189 contra el 0,10 exigido.

**P-2 (el control que podía ahorrar la arquitectura) CUMPLE con holgura.** `escala` promedia
**0,2394** de `falsa_abst` en las cinco difíciles contra **0,0929** de `cabeza`: una brecha de
**+0,1465**, casi 3× el 0,05 que pedía el prereg. **No era la norma del vector: era que dos
decisiones de naturaleza distinta competían por la misma masa de probabilidad.**

**P-3 (no-daño) cumple, raspando.** En las dos unidades fáciles la cabeza no puede costar más de
0,05 de `nose`: en 1_s0 gana (+0,0018) y en 2_s0 pierde **0,0470** — dentro del criterio, pero por
poco.

**P-4 (sanidad) FALLA en su forma literal, y falla hacia arriba.** Pide `vigente` dentro de ±0,05 de
`token` y se pasa en 3 de 7 unidades (+0,0633 · +0,0668 · +0,0928), **siempre por exceso**: la cabeza
recupera mejor. Se registra como está en `DESVIACIONES_CABEZA.md`; no se toca el umbral después de
ver el número.

---

## §4 · Lo que el resultado explica hacia atrás

**`sonda_normas.py` — la premisa de `escala` mezclaba dos matrices.** El 0,367 que motivó la
condición es de la matriz de **entrada** (`emb`). Pero el softmax que decide la respuesta usa la de
**salida** (`head`), y ahí `NOSE` ya era **1,77-2,05× más largo** que un valor promedio en las cinco
unidades que fallan (y 0,825 en las fáciles). → **`escala` subía la norma en las unidades fáciles y
la bajaba en las difíciles: la misma intervención hace cosas opuestas según la unidad.** La causa es
que `NOSE` nunca se entrenó durante la campaña base y quedó en su inicialización, mientras los
vectores de valor se encogieron más en los niveles difíciles.

Y el negativo de P-2 es **limpio, no un fallo de implementación**: la renormalización sobrevive al
entrenamiento (entra en 1,000 por construcción y cierra en 0,899 de entrada / 1,131 de salida). La
norma se quedó donde se la puso y aun así no alcanzó.

**El control pareado era imprescindible, y quedó demostrado.** `t3_s0` replicó el fallo del 17-ago
(0,2246 hoy contra 0,2109 ayer): reiniciar Adam **no** arregla la unidad difícil. Pero en las fáciles
reiniciar Adam **cuesta** (t1_s0 0,0667 contra 0,0095 de x1_s0, con 1000 pasos menos). **Si se
hubiera reusado la campaña `x` del 17-ago como línea de base, `cabeza` habría parecido no aportar
nada.** La decisión del prereg de re-correr `token` es lo único que deja ver el efecto.

---

## §4 bis · El techo de 3_s0 es de CALIBRACIÓN, no de capacidad (exploratorio)

`sonda_umbral.py`, re-corrida hoy sobre los checkpoints ya entrenados (CPU, sin GPU). Mide el AUC del
logit de la cabeza y después elige el umbral en una mitad de los datos y lo juzga en la otra:

| unidad | AUC | con σ>0,5 (prereg) | con umbral elegido en A, medido en B |
|---|---:|---|---|
| c1_s0 | 0,998 | pasa | pasa (`f_abst` 0,0625 · `nose` 1,0000) |
| c2_s0 | 0,973 | pasa | pasa (0,0132 · 0,9038) |
| **c3_s0** | 0,807 | **falla** | **pasa** (0,0714 · 0,6379) |
| c3_s1 | 0,777 | pasa | no |
| c3_s2 | 0,841 | pasa | no (0,1882 · 0,5349) |
| c4_s0 | 0,854 | pasa | pasa (0,0429 · 0,6034) |
| c4_s1 | 0,850 | pasa | pasa (0,0568 · 0,6000) |

**Lo que importa: c3_s0 —la única que falla bajo el prereg— pasa con otro corte.** Con AUC 0,807 la
información para decidir está en el logit; lo que está mal puesto es el punto de corte. σ>0,5 no es
el óptimo.

**Dos salvedades que no se pueden saltear.** (1) Es **post-hoc y exploratorio**: no está en el
prereg y no confirma nada (D-C4). (2) La sonda usa 8 lotes de 32 = 256 muestras por unidad, mucho
menos que la evaluación oficial, y se nota: le da a c3_s1 un `nose` de 0,4757 donde la evaluación
completa da 0,5406. **Los números de esta tabla no son comparables con los de §2**, y las dos
unidades que acá «no pasan» con umbral propio sí pasan el prereg. Sirve para la pregunta
capacidad-vs-calibración, no para reordenar el ranking.

**Y una lección de método que vale sola: el óptimo pegado al borde del criterio no generaliza.**
Eligiendo el umbral con el criterio real (`f_abst ≤ 0,10`) pasaban 2 de 6; pidiendo **margen** al
elegirlo (0,07) y juzgando con 0,10, pasan 5. El borde es el punto más frágil de la curva.

---

## §5 · Qué queda dicho, y qué no

**Dicho:** con 129 parámetros —el 0,015 % del modelo— la abstención pasa de imposible a posible en
4 de 5 modelos donde antes no lo era, sin costar precisión. El margen sobre el atajo, que el 17-ago
parecía una barrera del aprendizaje, **es una barrera de la interfaz**: con `token` predice perfecto
(0 de 5), y esos mismos cinco modelos con `cabeza` pasan cuatro.

**No dicho, y es lo que sigue:**
- **La frontera no está muestreada.** Entre +0,2358 y +0,4071 de margen no hay ni un punto medido, y
  es justo donde estaría el corte. Es lo que corre ahora bajo `PREREG_FRONTERA.md`.
- **El presupuesto de fase lo elegí yo** (D-C2): no se puede distinguir «`cabeza` llega y `token`
  no» de «`token` llega más tarde».
- **3_s0 sigue afuera**, y la sonda de umbral sugiere —post-hoc, exploratorio— que su techo es de
  calibración y no de capacidad.
