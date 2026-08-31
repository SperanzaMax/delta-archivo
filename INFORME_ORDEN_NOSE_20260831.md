# El término de orden ORDENA, y ordenar no alcanza · el cuello pasa a la EVIDENCIA

**2026-08-31, cierre.** `PREREG_ORDEN_NOSE.md` (SHA `9e5659e5`) + `PRECISION_ORDEN_NOSE_O3.md`.
Dos unidades, `r03_s3` y `r03_s6`, 3000 pasos, contra el control ya en disco (`t03_s3`, `t03_s6`).

---

## 1. El desenlace, y estaba escrito por adelantado

**Es la celda O-7 del §5 del pre-registro**, textual: *«O-1 falla PERO el término de orden bajó de
log 2: el cuello no es la decisión sino la evidencia. Pasa a la recuperación.»*

| | control | `r03_s3` | `r03_s6` |
|---|---:|---:|---:|
| término de orden (constante = 0,6931 · oráculo = 0) | **11,2053** | **0,6295** | **0,6341** |
| **AUC del logit de `NOSE` vs la ausencia** | **0,5103** | **0,6620** | **0,6681** |
| abstención | 0,4945 | **0,0000** | **0,0000** |
| RECUP | 0,3579 | 0,3590 | 0,3688 |
| exactitud, umbral actual | 0,2965 | 0,2151 | 0,2202 |
| **exactitud, MEJOR umbral posible** | 0,4099 | **0,4119** | **0,4102** |

**El mecanismo funciona:** el orden pasa de 11,2 —quince veces PEOR que cualquier constante— a 0,63,
por debajo de `log 2`, y el AUC de **0,5103 (azar) a 0,6620 y 0,6681**. Replicado en dos semillas.

**Y no alcanza:** con el mejor umbral posible la exactitud llega a 0,4119 contra un piso trivial de
**0,4065**. Son **+0,0054**, abstiniéndose del 87 % de las preguntas. En la práctica, el modelo mudo.

## 2. Por qué no alcanza, y el número que lo explica

Medido el mismo día con una sonda lineal held-out sobre el estado interno (ridge, solución cerrada,
con techo y nulo):

| | techo «¿el argmax es un nombre?» | **señal «¿NO hay respuesta?»** | nulo |
|---|---:|---:|---:|
| `n3_s0` (base sana, RECUP 0,7885) | 0,9983 | **0,7003** | 0,4994 |

**La ausencia es decodificable del estado con AUC ≈ 0,70. El término de orden llegó a 0,66.**

> **Extrajo casi toda la señal que había.** El cuello de botella no es la función de pérdida ni la
> interfaz: es que **la información de ausencia en el estado sólo da 0,70**, y con 0,70 no se
> construye una abstención útil.

Y la confianza de salida no la tiene en absoluto: AUC **0,4599 / 0,4459 / 0,4486** en `t03_s3` y
**0,6147 / 0,5982 / 0,5876** en `n3_s0`, o sea entre el azar y apenas por encima, y en dos de tres
casos **por debajo de 0,50** (más confianza cuando la respuesta NO está).

## 3. ⚠ El veredicto automático del juez es INCORRECTO, por tercera vez en el proyecto

`juzgar_orden.py` imprimió **«O-3 SE DISPARA → SE CIERRA la línea»**. Sus propios números lo
desmienten y no se adjudica.

**Con `abstencion = 0,0000`, tres criterios dejan de medir y pasan a ser aritmética:**

| criterio | valor | azar / nulo | por qué no se lee |
|---|---:|---:|---|
| O-1 acuerdo | 0,6003 | **0,6003** | con `calla` constante, acuerdo = P(hay). **Idénticos** |
| O-2 pureza | 1,0000 | **1,0000** | todo grupo es unánime si nadie se calla. **Idénticos** |
| O-3′ invento | 0,3997 | — | contestar todo cuando el 40 % no tiene respuesta **da 0,40 por definición** |

**O-3′ se disparó por aritmética, no por el fenómeno que quería capturar.** Lo que O-3 vigilaba era
que el término desacoplara la decisión del valor, como pasó con `balance` y `ranking` el 29-ago. **Eso
NO pasó: O-4 CUMPLE** (RECUP 0,3590 y 0,3688 contra orígenes de 0,3654 y 0,3835). La recuperación
quedó intacta.

**Lo que falló es O-6:** la abstención se fue al extremo locuaz. El pre-registro ya decía que en ese
caso **O-1 no se adjudica aunque cumpla**; lo que no dejó escrito es que **O-2 y O-3 quedan igual de
ciegos**.

> **Octavo defecto de pre-registro del mes, y de clase nueva: tres criterios compartían un supuesto
> implícito —que la abstención no esté en un extremo— y sólo uno de los tres lo declaraba.
> O-6 no era un riesgo más de la lista: era la PRECONDICIÓN de los otros tres.**
>
> **Regla que deja: cuando un criterio de riesgo protege la legibilidad de otros, hay que decir
> CUÁLES, y el juez tiene que devolver NO EVALUABLE en vez de un número.**

## 4. Veredicto, criterio por criterio

| | veredicto | |
|---|---|---|
| **O-1** principal | **NO EVALUABLE** | precondición O-6 rota |
| **O-2** mecanicista | **NO EVALUABLE** | precondición O-6 rota |
| **O-3′/O-3″** control | **NO EVALUABLE** | aritmética de abstención 0; lo que vigilaba lo cubre O-4 |
| **O-4** nulo | **CUMPLE** | RECUP intacta: el valor **no** se desacopló |
| **O-5** exactitud | **NO CUMPLE** | 0,2155 y 0,2214 contra el piso 0,4065 |
| **O-6** riesgo | **NO CUMPLE** | abstención 0,0000, extremo locuaz |
| **O-7** riesgo | **SE ACTIVA** | el término ordenó (0,63 < log 2) y no alcanzó |

**El criterio de abandono del §6 NO se aplica.** Exigía que O-3 se disparara, y O-3 no es evaluable.
Y lo que O-3 vigilaba —el desacople del valor— está descartado por O-4.

## 5. Lo que este negativo deja establecido, y es bastante

1. **La degeneración era real y el término la rompe.** Orden 11,2 → 0,63 y AUC 0,51 → 0,66. La
   pérdida plana, probada por la mañana, tenía la corrección que se derivó de ella.
2. **Ordenar y calibrar el umbral son DOS cosas.** El término de orden pide sólo la primera —«toda
   constante da el mismo valor»— y por eso el modelo mejoró el orden **bajando todo el nivel**. Hace
   falta un segundo término que fije dónde cortar.
3. **★ El techo está en la EVIDENCIA, no en la decisión.** Con la ausencia decodificable a 0,70 y el
   término llegando a 0,66, no queda margen del lado de la pérdida. **Toda mejora futura de la
   abstención pasa por que el modelo detecte mejor que algo falta, y eso depende de la recuperación.**
4. **Y la detección escala con la recuperación:** la señal es 0,70 en `n3_s0` (RECUP 0,7885) y la
   confianza está anti-correlacionada en las unidades degradadas (RECUP 0,36). **Para saber que algo
   no está, primero hay que buscarlo bien.**

## 6. Lo que NO dice

- **No cierra la línea de la función de pérdida.** El criterio de abandono exige lo que no se pudo
  medir. Lo que sí queda es que **el margen que quedaba ahí es chico**, acotado por el 0,70.
- **No prueba que 0,70 sea el techo real.** Es el techo de un lector **lineal** sobre el estado final
  de **este** modelo. Una sonda no lineal, o una capa distinta, podrían dar más.
- **No toca la fase H.** `cabeza` sigue NO EVALUABLE por presupuesto desde el 30.
- **3000 pasos**, semillas sin base, comparables sólo contra sí mismas y su par de `rec_rank`.
