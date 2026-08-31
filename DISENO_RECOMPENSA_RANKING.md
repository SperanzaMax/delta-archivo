# La síntesis que falta · `recompensa` acopla pero no discrimina, `ranking` discrimina pero no acopla

**2026-08-31.** Escrito **antes** de ver el resultado de `prueba_perdida_plana.py`, que está corriendo.
No es un pre-registro todavía: es el diseño que se deriva del diagnóstico, para que quede fechado
antes del dato que lo habilita o lo tumba.

---

## 1. Los dos negativos que ya están medidos, y son complementarios

**`balance` y `ranking` (29-ago, seis unidades).** Está escrito en `entrenar.py:113`:

> aquellas dos tocan **SOLO al vigilante**, así que la pérdida del valor y la de la cabeza nunca se
> hablan y **nada le dice al modelo que equivocarse sea PEOR que callarse**. Les sacamos el pago al
> silencio y las seis unidades se fueron a inventar (exactitud 0,2361-0,3536 contra un piso de
> 0,4065, invento hasta 0,1966).

`ranking` es, textual del código, «el sustituto del AUC por pares, donde **toda constante da el mismo
valor y es el PEOR alcanzable**, así que no hay prior que cobrar y el único modo de bajar la pérdida
es **ordenar**». Es decir: **`ranking` es exactamente la pérdida que premia discriminar.**

**`recompensa` (29 al 31-ago).** Arregla el acople —una sola pérdida que mira el resultado final, con
los cuatro casos y la CE del valor viva— y por eso las unidades dejaron de inventar. Pero es
**LINEAL en `q`**:

$$R = \text{hay}\big[q(-F) + (1-q)(c-(1-c)M)\big] + (1-\text{hay})\big[qL + (1-q)(-M)\big]$$

y al promediar, **si `q` es independiente de (hay, c), $E[R]$ depende de `q` sólo a través de su
media**. Dos modelos con la misma tasa de abstención y particiones distintas valen igual.

| familia | ¿acopla valor y silencio? | ¿premia discriminar? | qué pasó |
|---|---|---|---|
| `bce` / `balance` / `ranking` | **NO** | **SÍ** | se fueron a inventar (29-ago) |
| `recompensa` | **SÍ** | **NO** | partición arbitraria y saturada (31-ago) |
| **`recompensa` + ranking sobre `q`** | **SÍ** | **SÍ** | **nunca se probó** |

## 2. Por qué la síntesis no es «una quinta variante de lo mismo»

El §6 de `PREREG_RECOMPENSA_L` dejó escrito que si L-1 falla en las dos interfaces se cierra
«arreglarlo desde la función de pérdida», porque serían cuatro formas independientes de mover la
decisión sin mover la exactitud. **Esa cláusula sigue en pie y hay que respetarla.** Lo que la
distingue de este caso es el diagnóstico: las cuatro variantes anteriores movían **dónde está el
óptimo de la tasa**; ninguna tocaba **la degeneración**. Con la pérdida plana medida, agregar un
término de orden no es probar otro peso: es **quitarle la planitud al eje que hoy no tiene gradiente**.

Si la prueba de planitud **no** confirma, este documento se archiva sin correr nada.

## 3. Forma concreta

Al `-E[R] + \text{CE}$ de hoy se le suma un término de orden **sobre `q`**, no sobre la cabeza, para
que funcione con la interfaz `token` que es la principal:

```python
# i sin respuesta, j con respuesta: q_i deberia ser MAYOR que q_j
dif = s[:, None] - s[None, :]                    # s = logit de NOSE (no q, por condicionamiento)
par = es_nose[:, None] * hay[None, :]
rank = (softplus(-dif) * par).sum() / maximum(par.sum(), 1.0)
perdida = -rec.mean() + REC_CE * ce + REC_RANK * rank
```

**Se usa el logit y no `q`** porque `q` está saturado en 0 y 1 (medido hoy: 86-99 % de la masa en los
extremos) y una diferencia de probabilidades saturadas no tiene gradiente. El logit sí: hoy va de
−18,4 a +22,3.

**`--rec-rank` arranca en un valor derivado, no elegido:** el mismo criterio de
`PRECISION_RECOMPENSA_L_CE`, o sea igualando su gradiente en la columna de `NOSE` al de la recompensa,
medido con `medir_ratio_ce.py` **antes de entrenar**. Y esta vez el ratio se mide **en el checkpoint
de siembra**, no a mitad de corrida, que fue el error del 30.

## 4. Las predicciones que habría que fijar

**R-1 · PRINCIPAL.** El acuerdo pareado entre «se calla» y «no hay respuesta» supera **0,60** (hoy es
0,4985 contra un azar de 0,5004) en al menos 3 de 4 unidades.

**R-2 · MECANICISTA.** La pureza de la decisión agrupando por relación **cae por debajo de 0,70** (hoy
0,977-0,982 contra un nulo de 0,517). Es la medición de `sonda_volado.py` y es la que dice que se
rompió el atajo, no sólo que se movió la tasa.

**R-3 · EL CONTROL QUE PUEDE FALLAR, y es el negativo del 29.** `invento` **no supera 0,10**. Si el
término de orden vuelve a desacoplar la decisión del valor, las unidades se van a inventar como en
`balance` y `ranking`, y eso sería el mismo negativo por tercera vez.

**R-4 · NULO.** RECUP no cae más de 0,05 respecto del origen.

**R-5 · RIESGO.** El término de orden puede empujar a `q` a los extremos por otra vía (separar
maximizando el margen), dando abstención 0 o 1. Se exige abstención estrictamente entre 0,05 y 0,95,
igual que L-3.

## 5. Lo que este diseño NO arregla

- **No hace que el modelo sepa cuándo no sabe.** Sigue siendo supervisado, y el cierre de seis meses
  del `PLAN_FOCO_20260824.md` no se toca.
- **No dice nada de la interfaz `cabeza`**, cuya fase H sigue **NO EVALUABLE por presupuesto**.
- **Y arrastra el límite de siempre:** 3000 pasos, semillas sin base, comparables sólo contra sí mismas.
