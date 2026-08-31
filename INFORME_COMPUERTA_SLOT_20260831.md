# La compuerta W-0 ABRE, y encuentra dos cosas que el pre-registro no había previsto

**2026-08-31, noche.** `compuerta_slot_orden.py` sobre `b3_s3` y `b3_s6`, CPU, antes de congelar
`PREREG_SLOT_ORDEN.md` y antes de gastar una unidad de GPU. Es la compuerta que decide si el diseño
hace lo que dice.

---

## 1. Lo que sale bien, y es lo que el diseño necesitaba

**W-0(c) CUMPLE limpio, y era la que decidía.** El gradiente del término de orden, |g| medio por
elemento:

| | `b3_s3` | `b3_s6` |
|---|---:|---:|
| `kw` (keys del archivo) | 1,7369e−02 | 2,4762e−02 |
| `qr` (query de búsqueda) | 6,5555e−03 | 3,8194e−03 |
| `k_nulo` (el slot) | 6,4961e−03 | 3,0066e−04 |
| **`head.w` (la salida)** | **0,0000e+00** | **0,0000e+00** |

**Todo el gradiente entra en la búsqueda y nada en la salida**, exactamente al revés de lo que pasaba
hoy con `token`. La idea de «enseñarle a buscar diferente» está mecánicamente implementada: el
término reconfigura las keys y las queries, no sólo el slot.

**W-0(a) CUMPLE, pero el criterio estaba mal escrito y hubo que corregirlo antes de aplicarlo.** La
primera versión pedía distancia **0,0 exacta** al valor inicial y **fallaba** en las dos semillas por
1,4e−02. La explicación alternativa resultó ser la correcta: **weight decay**. 26000 pasos encogen
`k_nulo` aunque no reciba un solo gradiente, y sobre `v_nulo`, que vale cero, no dejan marca.

> Lo que separa las dos lecturas es la **dirección**: coseno con `init_params` de su propia semilla
> **+1,000000 exacto** en las dos, razón de normas **0,866972 idéntica en las dos**. Un gradiente lo
> habría girado; el decay sólo lo encoge. **El slot nunca recibió gradiente**, que es lo que había
> que verificar.

## 2. ★ Hallazgo 1 · el logit de abstención arranca siendo una variable de DOS VALORES

**W-0(b) cumple su criterio literal y el criterio no medía lo que su nombre dice.** Pedía «desvío
> 0», y el desvío da 0,364 y 0,462. Pero:

| | `b3_s3` | `b3_s6` |
|---|---:|---:|
| masa del slot, media | 0,1627 | 0,3147 |
| masa mínima · máxima | 0,00000 · 1,00000 | 0,00000 · 1,00000 |
| **muestras pegadas al clip** (\|logit\| = 13,8155) | **0,8438** | **0,6250** |
| **valores distintos del logit** | **10 de 64** | **8 de 64** |
| AUC del logit vs la ausencia | 0,4956 | 0,5451 |

**El desvío es alto por saturación, no por graduación.** La masa del slot vale 0 o 1 y casi nada en
el medio, así que el logit toma ocho o diez valores distintos sobre 64 muestras y su AUC es azar.
Medido sin el clip, por `sim_nulo − logsumexp(sim_resto)`, el log-odds real va de **−834 a +395**:
el slot compite contra un `logsumexp` cuyo rango es de **cientos de nats**.

**Y el 25-ago NO arrancaba así.** Aquella campaña entrenó con slot desde el principio y su masa se
quedó en 0,4074 / 0,4046 / 0,4020 con **máximo 0,4679**, suave y clavada en el prior. Acá el slot
entra a un softmax **ya entrenado sin él** y se satura.

> **Consecuencia para el pre-registro, y hay que decirla antes: W-3 ya no replica el 25-ago.**
> W-3 pedía que la masa «deje de estar clavada en el prior», con desvío > 0,10. **Ese criterio ya se
> cumple en la siembra, antes de entrenar nada**, y por saturación. Tal como está escrito no puede
> cerrar ni reabrir la vía del 25-ago: mide otra cosa.

**Lo que sí redefine, y no invalida el experimento.** Que el slot arranque saturado no lo rompe: el
término de orden le pone gradiente a `kw` y a `qr`, así que **puede comprimir la escala de los scores
del archivo** para que el slot compita de forma graduada. Eso es más duro que lo del 25-ago y es la
tarea real. Pero el punto de partida es peor de lo que el pre-registro suponía, y se ve en el propio
término: vale **3,4056** y **5,6468** contra `log 2 = 0,6931`, o sea **5 a 8 veces peor que cualquier
constante**. Con `token` arrancaba en 0,83 y 1,03.

## 3. ★ Hallazgo 2 · el peso NO queda determinado por el criterio, y es consecuencia del anterior

El criterio quedó escrito antes de correr nada: `rec_rank* = |g|` base en `kw` sobre `|g|` del orden
en `k_nulo`, medido en el checkpoint de siembra. El resultado:

| | `b3_s3` | `b3_s6` | dispersión |
|---|---:|---:|---:|
| **`rec_rank*`, criterio principal** | **1,5617** | **5,4459** | **3,49×** |
| referencia sobre `qr` | 0,7316 | 2,4113 | 3,30× |
| referencia sobre la búsqueda entera | 0,6227 | 0,0827 | 7,53× |
| *el mismo criterio con `token`, hoy* | *0,00805* | *0,00720* | ***1,12×*** |

**Ninguna variante del criterio da un peso reproducible**, y las tres discrepan entre sí. La causa es
el hallazgo 1: el gradiente que el término pone en `k_nulo` depende de cuántas muestras zafan de la
saturación, que es 16 % en una semilla y 37 % en la otra. Con `token` el mismo criterio daba 1,12×.

**Promediar da 3,5, y sería inventar un número que no describe a ninguna de las dos unidades.**
(Y la coincidencia con el 3,5 del ratio de gradientes del 30-ago es casualidad aritmética, no una
relación: son cocientes de cosas distintas. Se aclara antes de que alguien las junte.)

## 4. Veredicto

| | veredicto | |
|---|---|---|
| **W-0(a)** el slot está sin usar | **CUMPLE** | coseno 1,0 exacto; lo que se movió fue weight decay |
| **W-0(b)** la masa no es constante | **CUMPLE**, y el criterio era débil | saturada: 84 % y 63 % en el clip |
| **W-0(c)** el gradiente va a la búsqueda | **CUMPLE** | 0,0 exacto en `head`, todo en `kw`/`qr`/`k_nulo` |
| **W-0(d)** el peso derivado | **NO QUEDA DETERMINADO** | 1,56 contra 5,45, dispersión 3,49× |

**COMPUERTA ABIERTA en lo que decidía —el diseño hace lo que dice— y BLOQUEADA en el peso.**

## 5. Lo que hay que decidir antes de lanzar

1. **El peso.** O una unidad con su propio `rec_rank*` (1,56 en s3 y 5,45 en s6), que es lo más fiel
   al criterio ya declarado; o un único número, que no está sostenido por la medición.
2. **W-3 se reescribe o se retira.** Como está, se cumple en la siembra y no mide lo que dice.
3. **La saturación se declara en el pre-registro**, porque cambia lo que el experimento puede medir:
   no es «¿el orden crea la señal de ausencia?» sino «¿el orden puede primero desaturar el slot y
   después crear la señal?».

## 6. Lo que NO dice

- **No dice que el experimento vaya a fallar.** Dice que arranca de un punto peor que el supuesto, y
  que el peso no sale de donde se pensaba sacarlo.
- **No toca el resultado de hoy.** El informe de `token` queda como está.
- **No mide nada entrenado.** Todo es sobre los checkpoints de siembra, un lote de 64 muestras
  (256 en los controles de saturación), nivel 3, `p_nose` 0,4.
