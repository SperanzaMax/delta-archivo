# PRE-REGISTRO · quitarle a la cabeza la posibilidad de cobrar el prior

**2026-08-29.** Se congela **antes de escribir una línea del instrumento** y antes de correr nada.

Sale de una observación de Maxi sobre el hallazgo del día, textual:

> «estudiar tiene que tener una retribución mucho mayor que no estudiar, y si el modelo ya sabe de
> antemano que en el total de las respuestas hay un porcentaje verdadero de aciertos diciendo "no
> estudié", pero nosotros buscamos que él se esfuerce en mejorar ese valor que para él es piso. [...]
> la forma de entrenamiento con valores de errores que él tiene que descubrir pero para descubrir
> tiene que saber y eso lo lleva a saber»

---

## 1. El diagnóstico que esto ataca

`INFORME_ATRACTOR_MUDO_FASE1_20260829.md` estableció que con blanco `error` la cabeza de 4 de 9
unidades **colapsa al prior** y que ese estado es absorbente. El punto que hace falta subrayar para
entender el diseño de abajo es **por qué** es estable:

> Con la pérdida actual, $\mathrm{BCE}(a, b)$, una cabeza que ignora la entrada tiene un óptimo, y
> ese óptimo es la constante $\mathrm{logit}\,\mathbb{E}[b]$. **El colapso no es una meseta ni un
> fracaso de optimización: es un mínimo real de la función que se está optimizando.**

Y tiene un agravante medido: cuanto **peor** recupera el modelo, más **fácil** le queda el problema
de la cabeza, porque el blanco se vuelve casi constante. Mejorar la recuperación le complica la tarea
a la cabeza. Los dos términos están en tensión y el estado degenerado es el cómodo para ambos.

**La idea que se prueba:** si la cabeza no puede acertar con una constante, tiene que mirar la
entrada; para que mirar la entrada sirva, la representación compartida tiene que llevar señal; y esa
señal sólo existe si la recuperación funciona.

## 2. Las dos condiciones, y por qué son dos y no una

Se comparan **dos formas distintas de eliminar el pago por la constante**, porque no son equivalentes
en fuerza y conviene saber si alcanza con la débil.

**C1 · `balance` — BCE balanceada por clase.** Cada muestra se pesa por el inverso de la frecuencia
de su clase en el lote, con los pesos renormalizados a media 1 para no cambiar la escala de la
pérdida. Efecto: la mejor constante pasa a valer $\log 2$ **cualquiera sea el prior**. Cobrar el
prior deja de pagar. Es la intervención mínima.

**C2 · `ranking` — sustituto del AUC por pares.** Sobre los pares $(i, j)$ del lote con $b_i = 1$ y
$b_j = 0$:
\[
\mathcal{L}_{\text{cabeza}} = \frac{1}{|P|}\sum_{(i,j) \in P} \mathrm{softplus}\big(-(a_i - a_j)\big).
\]
Efecto más fuerte: **toda constante da exactamente el mismo valor, y es el peor alcanzable**. No hay
prior que cobrar en absoluto, y el único modo de bajar la pérdida es ordenar. Es la traducción
literal de «para descubrir tiene que saber».

**Control:** las **nueve unidades `b3_s0`…`b3_s8` ya corridas**, que son exactamente esta receta con
$\mathrm{BCE}$ sin balancear. **No se re-corren.** Su comportamiento en el paso 2500 está medido y
publicado en el informe del 29.

## 3. Diseño

**Semillas: las mismas nueve, 0 a 8.** El diseño es **pareado**: para cada semilla se sabe qué hace
el control, y la pregunta es qué hace la misma semilla con la pérdida cambiada. Con semillas nuevas
la comparación perdería la mitad de su fuerza, porque la tasa base de degeneración se estimaría en
lugar de conocerse.

**Flags: idénticos a la campaña de control** —`--abst cabeza --donde pre --blanco error`, nivel 3,
`p_nose` 0,4, **`--horizonte 26000`**— cambiando **sólo** `--perdida-cabeza`. El horizonte se
mantiene en 26000 aunque se corran 3000 pasos, para que la curva de lr en el tramo medido sea la
**misma** que la del control. Poner horizonte 3000 haría decaer la lr dentro del tramo y la
comparación mediría eso.

**Presupuesto: 3000 pasos por unidad.** $9 \times 2 \times 3000 = 54000$ pasos.

> **Por qué 3000 y no 26000.** El predictor del paso 2500 separa 40 de 40 corridas del banco, así que
> el desenlace se lee ahí. **Pero el predictor se validó sobre corridas con $\mathrm{BCE}$ sin
> balancear, y estas condiciones cambian la dinámica que él describe.** Por eso se corre hasta 3000 y
> no hasta 2500 exactos, y por eso el criterio principal de abajo **no** se apoya sólo en el hito de
> 2500 sino en la trayectoria completa del tramo. Si una condición sale del silencio **después** de
> 3000, esta campaña no lo verá y se declara como su límite.

## 4. Predicciones, fijadas ANTES

**P-0 · BLOQUEANTE, no-daño.** En las **cinco** semillas que el control lleva a un estado útil
(0, 1, 2, 4, 5), ninguna de las dos condiciones puede empeorarlas a abstención total. Se permite
como mucho **una** de cinco por condición. Si una condición rompe dos o más unidades que funcionaban,
**no se lee su resultado principal**: cambiar una pérdida para arreglar 4 y romper 2 no es un arreglo.

**P-1 · PRINCIPAL.** De las **cuatro** semillas que el control deja mudas (3, 6, 7, 8), al menos
**3 de 4** emiten respuestas dentro de los 3000 pasos, es decir `abstencion` < 1,0000 en algún hito.

Fundamento del número, escrito antes: el control da **0 de 4**. Pedir 3 de 4 es pedir un efecto
grande, y es deliberado. Un efecto chico en este banco no se puede distinguir del ruido entre
semillas, cosa que la §3.1 del informe de A5 ya acotó con Chebyshev.

**P-2 · CONTRASTE entre las dos condiciones.** Se predice `ranking` $\geq$ `balance` en el conteo de
P-1, porque elimina el pago por la constante de forma total y no parcial. **Si `balance` alcanza,
gana `balance`** por ser la intervención mínima, y así se declara antes de ver los datos.

**P-3 · MECANICISTA.** En las unidades que salgan del silencio, se predice que el AUC de la cabeza
sobre su propio blanco supera **0,60** al paso 3000, contra los 0,525–0,571 del control degenerado.
Salir del silencio sin que la cabeza discrimine sería otra cosa y hay que poder distinguirlo.

**P-4 · RIESGO DECLARADO, y es el que más me preocupa.** Quitar el incentivo a callarse puede empujar
al otro extremo. Se reporta en las mismas unidades **`invento` y la exactitud global contra su piso
de 0,4065**, y se declara ahora que **una unidad que sale del silencio inventando no cuenta como
éxito de P-1**. La lección del 28 se aplica a esta campaña desde el diseño y no después.

**P-5 · RIESGO, el nulo del propio método.** Eliminar la solución constante **no crea el camino a la
buena**. Es posible que la cabeza pase de una constante con AUC 0,5 a ruido con AUC 0,5. P-3 existe
para detectar exactamente eso.

## 5. Cómo se lee cada desenlace, escrito ANTES

| celda | lectura | qué se hace |
|---|---|---|
| **P-0 falla** en una condición | esa condición rompe lo que andaba | se descarta esa condición, sin leer su P-1 |
| **P-1 cumple en `balance`** | alcanza la intervención mínima | **se adopta `balance`** y se corre a 26000 con pre-registro nuevo |
| **P-1 cumple sólo en `ranking`** | hace falta eliminar el prior del todo | se adopta `ranking`, misma continuación |
| **P-1 cumple en las dos** | el mecanismo es el pago por la constante, y está confirmado por dos vías | se adopta `balance` por mínima (P-2) |
| **P-1 falla en las dos, P-3 bajo** | el prior no era el mecanismo; el cuello es la recuperación y no la cabeza | **la vía se cierra** y el foco vuelve a la indexación |
| **P-1 cumple pero P-4 se dispara** | se cambió mudez por invención | no se adopta; el resultado es sobre el trade-off y se informa así |

## 6. Criterio de abandono

> **Si P-1 falla en las dos condiciones, no se prueba una tercera forma de descontar el prior.** La
> hipótesis de que el pago por la constante es lo que sostiene el atractor queda refutada con dos
> implementaciones independientes, una mínima y una total, sobre las mismas nueve semillas.

## 7. Lo que NO contesta

- **No revive nada.** `PREREG_TASA_REGIMEN` está juzgado (T-0 fallido) y `PREREG_ATRACTOR_MUDO` cerró
  con la Fase 2 cancelada. Esto mide **otra cosa**, que es si el atractor se puede eliminar por diseño
  de la pérdida en vez de por presupuesto.
- **No dice que el modelo sepa cuándo no sabe.** Sigue siendo supervisado, y el §8 del
  `PLAN_FOCO_20260824.md` con su cierre de seis meses no se toca.
- **No mide calidad final.** 3000 pasos no alcanzan para `vigente` ni para SER. Lo único que se juzga
  acá es **si el atractor mudo desaparece**, y todo lo demás necesita una campaña a 26000.
- **Y no dice nada sobre escala.** 863.730 parámetros, idioma de 242 tokens, `p_nose` 0,4, un nivel.
