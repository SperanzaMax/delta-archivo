# PRE-REGISTRO · pagarle al modelo por acertar y cobrarle por mentir, sin vigilante

**2026-08-29.** Se congela **antes de escribir una línea del instrumento** y antes de correr nada.

Sale de una idea de Maxi, textual:

> «el tema del premio para que los aciertos que el modelo hace en decir nose cuando realmente no sabe
> y también cuando dice la respuesta correcta lo incentiven a querer siempre dar su mejor esfuerzo en
> saber, y resta cuando miente o da una respuesta incorrecta»

y de su criterio sobre qué vale la pena construir:

> «siempre las mejores cosas son las simples y las que se puede escalar [...] que no sólo un micro LM
> lo pueda hacer sino que lo podamos escalar en un LLM de frontera»

---

## 1. Qué se aprendió hoy, y que este diseño tiene que respetar

**`PREREG_PERDIDA_CABEZA` cumplió P-1 y disparó P-4.** Las dos pérdidas nuevas sacaron a las cuatro
unidades mudas del silencio, 4 de 4 en las dos condiciones y desde el paso 250. **Y las seis unidades
de `balance` terminaron por DEBAJO del piso trivial** (exactitud global 0,2361–0,3536 contra 0,4065)
con invención de 0,0617 a 0,1966, donde el control tenía 0,0000.

> **Eliminar el atractor mudo no alcanza. Se cambió mudez por invención.** Es exactamente el riesgo
> que P-4 declaró antes de correr, y se cumplió.

La razón es que aquellas dos pérdidas tocan **sólo al vigilante**. La pérdida del valor y la de la
cabeza son dos términos que no se hablan, así que nada le dice al modelo que **equivocarse sea peor
que callarse**. Se le sacó el pago por callarse y se fue al otro extremo, que es lo único que quedaba.

**Este pre-registro ataca eso: una recompensa sobre el resultado FINAL, que acopla las dos
decisiones.**

## 2. La recompensa, y la condición derivada ANTES de elegir los pesos

Cuatro casos, normalizados con el acierto en $+1$:

| caso | valor |
|---|---|
| acertó la respuesta | $+1$ |
| dijo NOSE y de verdad no estaba | $+L$ |
| contestó y erró, inventar incluido | $-M$ |
| dijo NOSE pero la respuesta **sí** estaba | $-F$ |

Con $q$ la probabilidad que el modelo le da a abstenerse y $c$ su probabilidad de acertar si contesta,
la recompensa esperada por muestra es

$$\mathbb{E}[R] \;=\; (1-\pi)\Big[q(-F) + (1-q)\big(c - (1-c)M\big)\Big] \;+\; \pi\Big[qL + (1-q)(-M)\Big],$$

con $\pi = 0{,}4065$ medido en este banco. La pérdida es $-\mathbb{E}[R]$. Derivando en $q$:

$$\frac{\partial \mathbb{E}[R]}{\partial q} = (1-\pi)\big[-F - c + (1-c)M\big] + \pi\big[L+M\big].$$

El modelo se calla de más cuando esa derivada es positiva, así que la condición para que **no** le
convenga es

$$\boxed{\;F \;>\; \frac{\pi(L+M) + (1-\pi)\big[(1-c)M - c\big]}{1-\pi}\;}$$

**Evaluada en este banco**, el $F$ mínimo según cuánto sabe el modelo:

| $L$ | $M$ | $c=0$ | $c=0{,}1$ | $c=0{,}35$ |
|---:|---:|---:|---:|---:|
| 1 | 1 | 2,370 | 2,170 | 1,670 |
| 0,5 | 0,5 | **1,185** | 1,035 | 0,660 |
| 0 | 0,5 | 0,842 | 0,692 | 0,317 |

> **Lo que dice, y es contraintuitivo:** para que un modelo que todavía no sabe nada ($c=0$) no se
> calle de más, **callarse teniendo la respuesta tiene que costar más que equivocarse**. Con premios
> simétricos hace falta $F = 2{,}37$. Ése es el precio de eliminar el atajo, y sale de la aritmética,
> no de una corazonada.

**Pesos elegidos, y se fijan acá antes de correr:** $L = 0{,}5$, $M = 0{,}5$, $F = 1{,}5$.
El mínimo en $c=0$ es $1{,}185$, así que $1{,}5$ deja **27 % de margen** sin irse a los extremos. Se
elige el juego simétrico y suave porque $L=M=1$ obligaría a $F>2{,}37$, y un castigo tan grande a la
abstención es precisamente lo que empuja a inventar, que es el fracaso de hoy.

## 3. La decisión de diseño principal, y por qué

**La condición principal NO usa vigilante.** La probabilidad de abstenerse es la masa que el softmax
ya le da al token `NOSE`. No hay cabeza, no hay parámetro nuevo, no hay umbral que calibrar.

**El motivo es escalar, y es de Maxi.** Un LLM de frontera ya tiene vocabulario y ya se entrena con
recompensas sobre su salida; esta pérdida entra sin tocar la arquitectura. Una cabeza binaria no.

Y hay un motivo interno además. `speranza2026abstention` midió que el token **pierde** contra la
cabeza, y la explicación que encontró es que las dos decisiones **competían por la misma masa de
probabilidad** sin que nada dijera cómo repartirla. **Una recompensa esperada es exactamente lo que
dice cómo repartirla.** Si el token funciona acá, esta pérdida rescata la vía barata que aquel trabajo
había descartado, y lo hace por el mecanismo que aquel trabajo identificó.

## 4. Diseño

**Condiciones.** `--perdida-cabeza recompensa`, con dos interfaces:

| | interfaz | prefijo | rol |
|---|---|---|---|
| **T** | `--abst token` · sin cabeza | `tk3_sX` | **PRINCIPAL** |
| **H** | `--abst cabeza` | `hd3_sX` | contraste |

**Semillas: 3 a 8**, las seis sin base, por `ENMIENDA_PERDIDA_CABEZA`. **`SEMBRAR=0`**, declarado.

**Control:** las que ya existen y no se re-corren. `b3_s3`…`b3_s8` (BCE, cabeza) y las `bl3`/`rk3` de
hoy. El control de la exactitud a 3000 pasos es **exacto y no hace falta medirlo**: una unidad muda da
$0{,}4065$ por definición.

**Presupuesto en DOS ETAPAS, y ésta es la corrección de método más importante del día.**

> El pre-registro de hoy puso P-4 —invención y exactitud— en una campaña de **3000 pasos**, y a esa
> altura **P-4 no es decidible**: un modelo que va bien todavía no supera el piso. El criterio estaba
> mal presupuestado y se leyó igual. **Es la cuarta vez que un umbral se escribe sin verificar si es
> alcanzable** (E-2 de A5, K-1 de CALIBRA, F-1 de ATRACTOR, y ésta).

- **Etapa 1, 3000 pasos, las 12 unidades.** Sólo decide **W-1**, que es la pregunta barata.
- **Etapa 2, hasta 12000 pasos, sólo las que pasen W-1.** Decide **W-2**, que es la pregunta real y
  necesita presupuesto. Se lanza sólo si W-1 cumple.

## 5. Predicciones, fijadas ANTES

**W-0 · BLOQUEANTE, la compuerta del instrumento.** Antes de gastar un paso de GPU, un chequeo
verifica sobre vectores sintéticos que (i) la recompensa esperada es diferenciable y su gradiente en
$q$ tiene el signo que predice el §2, (ii) con $F=1{,}5$, $L=M=0{,}5$ la política «abstenerse de todo»
**no** es el óptimo, y (iii) el flag llega hasta `entrenar.py`. Si algo falla, no se lanza.

**W-1 · ETAPA 1.** Al menos **4 de 6** unidades de la condición **T** salen del silencio a 3000 pasos
(`abstencion` < 1,0000 en algún hito). El control da 0 de 4 sobre las mudas.

**W-2 · PRINCIPAL, ETAPA 2.** A 12000 pasos, al menos **4 de 6** unidades de **T** tienen
**exactitud global > 0,4065**, el piso trivial.

> **Éste es el criterio que importa y es el que ninguna campaña anterior puso como principal.** El 28
> se descubrió que `nose` premia la degeneración; hoy se descubrió que sacarle el pago a la abstención
> lleva a inventar. **Las dos fallas son invisibles para todo criterio que no mire la exactitud global
> contra su piso.** Acá va de principal y no de riesgo.

**W-3 · CONTRASTE T contra H.** Se predice que **T ≥ H − 1 unidad** en W-2. **Si empatan, gana T**,
por simple y escalable, y se declara acá antes de ver nada.

**W-4 · MECANICISTA.** En las unidades de T que cumplan W-2, se predice `invento` < 0,05, contra
0,0617–0,1966 de `balance` hoy. Salir del silencio sin inventar es la definición operativa del
objetivo del proyecto.

**W-5 · RIESGO DECLARADO.** La recompensa acopla las dos decisiones, así que puede degradar la
recuperación para mejorar la mezcla. Se reporta **RECUP** en las mismas unidades, y **una unidad que
cumple W-2 con RECUP por debajo del control se informa destacada**, porque significaría que la mejora
vino de repartir mejor y no de saber más.

**W-6 · RIESGO, el nulo.** Es posible que la recompensa produzca un modelo que contesta lo justo y
sigue sin saber, con exactitud apenas sobre el piso. Por eso W-2 pide superar el piso y W-4 pide que
además no invente; cumplir uno solo no es cumplir.

## 6. Cómo se lee cada desenlace, escrito ANTES

| celda | lectura | qué se hace |
|---|---|---|
| **W-0 falla** | el instrumento no hace lo que dice | no se lanza nada |
| **W-1 falla** | la recompensa no saca del silencio | la vía se cierra; `balance` y `ranking` sí lo hacían y son más simples |
| **W-1 sí, W-2 no** | saca del silencio pero no compra exactitud | **mismo fracaso que hoy con otra fórmula**, y el problema no es el incentivo sino la recuperación |
| **W-2 cumple en T** | **el resultado**, y en la vía escalable | se escribe, y se diseña la réplica fuera de este banco |
| **W-2 sólo en H** | hace falta la cabeza | se adopta H y se anota que la vía barata no alcanzó |
| **W-2 cumple, W-5 se dispara** | mejoró repartiendo, no sabiendo | se informa así y no se vende como aprendizaje |

## 7. Criterio de abandono

> **Si W-2 falla en las dos interfaces, se cierra la línea de «arreglarlo desde la función de
> pérdida».** Tres formas independientes —balancear, ordenar y premiar— habrían movido la decisión sin
> mover la exactitud, y eso ubicaría el cuello de botella en la **recuperación**, que es la otra mitad
> del proyecto y tiene sus propias campañas.

## 8. Lo que NO contesta

- **No dice que el modelo sepa cuándo no sabe.** Sigue siendo supervisado. El §8 del
  `PLAN_FOCO_20260824.md` y su cierre de seis meses no se tocan.
- **No prueba que escale.** Que la interfaz sea escalable es un argumento de diseño, no un resultado.
  Un positivo acá justifica **probarlo** en un modelo mayor; no lo demuestra.
- **No mide calidad final.** 12000 pasos no son los 26000 de las campañas de referencia, así que
  ningún número de acá es comparable con el SER de A5.
- **Y no dice nada sobre escala.** 863.730 parámetros, idioma de 242 tokens, $p_{\mathrm{nose}}$ 0,4,
  un nivel.
