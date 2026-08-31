# El slot con orden COLAPSA el logit a una constante · y el control es BIESTABLE

**2026-08-31, cierre de la noche.** `PREREG_SLOT_ORDEN.md` (SHA `b7471e02`) + compuerta W-0. Cuatro
unidades, 3000 pasos, `n=3072` held-out pareado.

---

## 1. Los números

| | `k03_s3` ctrl | `k03_s6` ctrl | `w03_s3` trat (1,56) | `w03_s6` trat (5,45) |
|---|---:|---:|---:|---:|
| **AUC del logit vs la ausencia** | 0,4815 | 0,5133 | **0,4865** | **0,5175** |
| abstención | **0,9932** | **0,0075** | 0,0000 | 0,0003 |
| pegadas al clip (siembra 0,8438 · 0,6250) | 0,0055 | 0,9912 | **1,0000** | **0,9997** |
| **valores distintos del logit, sobre 3072** | 12 | 8 | **1** | **2** |
| invento | 0,0026 | 0,3968 | 0,3997 | 0,3994 |
| exactitud (piso 0,4065) | 0,3984 | 0,2048 | 0,2021 | 0,2171 |
| exactitud, mejor umbral | 0,3997 | 0,3997 | 0,3997 | 0,3997 |
| RECUP (origen 0,3654 · 0,3835) | 0,3379 | 0,3373 | 0,3368 | 0,3612 |
| término de orden (constante = 0,6931) | 0,8804 | 0,9010 | **0,6931** | **0,6926** |

## 2. ⚠ La trampa que el juez estuvo a punto de imprimir, y es la CUARTA del proyecto

W-7 decía: *«si W-1 falla pero el término de orden bajó de `log 2`, es presupuesto y no un
negativo»*. Los tratamientos dan **0,6931** y **0,6926**, y `log 2 = 0,693147`. Un `< log 2` ingenuo
lo lee como **«ordenó»**, por una diferencia en el quinto decimal.

**Es al revés.** El propio diseño del término dice que *«toda constante da el mismo valor y es el
PEOR alcanzable»*. El logit toma **1 y 2 valores distintos sobre 3072**: es una constante. **El
término vale `log 2` porque colapsó, no porque haya ordenado** — y de hecho se quedó en el peor
valor posible.

> **Cuarto veredicto automático engañoso del proyecto, y el segundo del mismo día.** La guarda que se
> agregó al juez no mira el número sino la degeneración: si el logit toma menos de 10 valores
> distintos o la saturación pasa de 0,95, **W-7 es NO EVALUABLE**.

## 3. Veredicto

| | | |
|---|---|---|
| **W-1** principal | **NO EVALUABLE** | precondición W-6 rota |
| **W-2** mecanicista, la que decidía | **NO EVALUABLE** | precondición W-6 rota |
| **W-3** desaturar | **NO CUMPLE, y al revés** | 0,8438 → **1,0000** y 0,6250 → **0,9997** |
| **W-4** control | **NO EVALUABLE** | precondición W-6 rota |
| **W-5** nulo | **CUMPLE** | RECUP intacta: 0,3368 y 0,3612 |
| **W-6** precondición | **NO CUMPLE** | abstención 0,0000 y 0,0003, extremo locuaz |
| **W-7** presupuesto | **NO EVALUABLE** | el término vale `log 2` por constante (§2) |

**La cláusula de cierre del §5 exige que W-3 falle *y* W-1 también. W-1 no falla: es NO EVALUABLE.
Así que el cierre de la vía NO se adjudica**, exactamente por la misma razón por la que esta tarde no
se aplicó el criterio de abandono. Lo que sí queda medido es más específico y más útil.

## 4. Lo que este negativo SÍ deja establecido

**1 · El argumento mecánico del pre-registro era cierto en su premisa y falso en su conclusión.** La
compuerta verificó que el gradiente entra entero en la búsqueda (`head` recibe **0,0 exacto**). Y
aun así el resultado es azar. **Que el gradiente llegue al mecanismo no alcanza:** el camino de menor
resistencia resultó ser **apagar el slot**, no aprender a usarlo.

**2 · La interfaz `slot` es PEOR que `token` para esto, y era la predicción al revés.** Con `token`
el mismo término dio AUC **0,6620 / 0,6681**; con `slot` da **0,4865 / 0,5175**, que es azar. El
pre-registro esperaba lo contrario por el argumento de que el gradiente entra «más cerca».

**3 · ★ El slot nulo sembrado en un modelo entrenado sin él es BIESTABLE.** Las dos unidades del
CONTROL B —misma condición, mismo peso 0,0, sólo cambia la semilla— se fueron a extremos **opuestos**:
`k03_s3` al **mudo** (abstención 0,9932) y `k03_s6` al **locuaz** (0,0075). Es la firma del atractor
absorbente del 29-ago, ahora con los dos polos visibles en la misma condición. **Un mecanismo cuya
salida depende de la semilla y no de la pregunta no es un detector.**

**4 · La saturación diagnosticada por la compuerta era el problema, y el término la EMPEORÓ.** No
desaturó: llevó 0,84 y 0,63 a **1,00**. La compuerta lo había declarado como el riesgo del diseño
antes de lanzar.

## 5. ⚠ Lo que este experimento NO puede decidir, y por qué se corre un brazo más

**El peso puede ser el culpable, y no la interfaz.** Los tratamientos corrieron con **1,56 y 5,45**,
derivados del criterio declarado; en `token` el mismo criterio daba **0,008**, o sea entre **200× y
680× menos**. Un gradiente tan grande sobre `k_nulo` es una explicación completamente suficiente del
colapso, y no está descartada por nada de lo medido.

**El CONTROL B lo hace verosímil en vez de especulativo:** con `--rec-rank 0` el logit NO colapsó
(12 y 8 valores distintos, y una unidad ni siquiera está en el clip). **El colapso a constante lo
produjo el término de orden con estos pesos**, no el slot por sí solo.

> **Por eso se corre la ENMIENDA W-8 antes de escribir cualquier cierre: dos unidades más, `slot`
> con `--rec-rank 0,008` —el peso de `token`, no derivado de nuevo—. Si con ese peso el logit no
> colapsa, lo que falló fue la DERIVACIÓN DEL PESO; si colapsa igual, falló la interfaz.**
> Sin ese brazo, cerrar la vía sería atribuirle a la idea de Maxi un fracaso que puede ser mío.

## 6. Lo que NO dice

- **No cierra el slot nulo** ni la vía de la búsqueda: la cláusula de cierre exige W-1, que no es
  evaluable.
- **No dice nada sobre `--abst slot` entrenado desde cero**, que es lo que se hizo el 25-ago y llegó
  al prior en vez de a un extremo.
- **3000 pasos**, dos semillas, comparables sólo entre sí.
