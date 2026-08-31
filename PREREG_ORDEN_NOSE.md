# PRE-REGISTRO · romper la degeneración de la recompensa con un término de ORDEN

**2026-08-31.** Se congela **antes de lanzar** y **después** de las mediciones en CPU del §2, todas
sobre checkpoints ya en disco y sin gastar un paso de GPU.

Deriva de `DISENO_RECOMPENSA_RANKING.md` y de `INFORME_RATIO_CE_20260831.md`. No reemplaza a
`PREREG_RECOMPENSA_L` (`96e750b6`): ataca un defecto **distinto** del que aquél atacaba.

---

## 1. La pregunta

> La recompensa esperada es **lineal en `q`**, así que cuando la decisión de callarse resulta
> independiente de la evidencia, la pérdida depende **sólo de la tasa** de abstención y no de
> **cuáles** preguntas se callan. **¿El bloqueo de la abstención es esa degeneración?**

## 2. Lo medido hoy en CPU, antes de escribir esto

**(a) La pérdida es PLANA, y el control tenía potencia de sobra** (`prueba_perdida_plana.py`,
n=3072, 200 barajadas):

| | `t03_s3` | `t03_s6` |
|---|---:|---:|
| pérdida con el `q` real | +0,182406 | +0,191432 |
| pérdida con el `q` **barajado** | +0,186155 ± 0,004170 | +0,189159 ± 0,004349 |
| distancia | **0,90 σ** | **0,52 σ** |
| pérdida con el `q` **oráculo**, misma tasa | +0,024874 | +0,029466 |
| el oráculo baja la pérdida | **78,6 σ** | **73,7 σ** |
| margen aprovechado | **51,96 %** | **49,45 %** |

Barajar no cambia nada; el oráculo baja la pérdida 74-79 σ **con la misma tasa**. Los dos modelos
aprovechan **el 50 % del margen, que es lo que daría el azar**.

**(b) Y la degeneración se ve por dónde cayó el modelo.** Las unidades se callan según la **relación
preguntada** (pureza 0,977 y 0,982 contra un nulo de 0,517, `sonda_volado.py`), corte que **no
correlaciona con nada relevante**, verificado por cuatro vías que se escribieron para poder fallar y
fallaron todas:

| explicación candidata | medición | veredicto |
|---|---|---|
| sigue la ausencia | ganancia del atajo **+0,0000** exacto sobre el generador | **no** |
| sigue la dificultad | RECUP nombres 0,4272 contra números 0,4472 (**−0,0200**, al revés) | **no** |
| es aritmética del softmax | la brecha está en `l_NOSE` (38 nats), no en el `logsumexp` (1,4, y al revés); truncar a k=58 no mueve nada | **no** |
| sigue su propia confianza | `c` 0,2916 contra 0,2680 en s3 y **−0,0029** en s6, con **signos opuestos** | **no** |

**(c) El punto de partida ordena PEOR que una constante.** El término de orden vale **0,8288**
(`b3_s3`) y **1,0257** (`b3_s6`) contra `log 2 = 0,6931` de cualquier constante: las unidades de
siembra le dan **más** masa de `NOSE` a las preguntas que **sí** tienen respuesta.

**(d) El peso, derivado y no elegido.** Criterio del §4 de `PRECISION_RECOMPENSA_L_CE`: igualar el
gradiente en la columna de `NOSE` con el **gradiente medio del resto del vocabulario**, medido **en el
checkpoint de siembra** —la corrección al error del 30, donde el 3,5 se midió a mitad de corrida y
resultó no ser una constante—.

| | `|g|` resto (recompensa+CE) | `|g|` en NOSE (orden) | `rec_rank*` |
|---|---:|---:|---:|
| `b3_s3` | 6,774e−06 | 8,418e−04 | 0,00805 |
| `b3_s6` | 6,679e−06 | 9,275e−04 | 0,00720 |

**Se fija `--rec-rank 0,008`.** Deja el gradiente en `NOSE` en 6,734e−06 contra 6,774e−06 del resto:
**ratio 0,994**.

> **Y una advertencia sobre la regla del 30-ago, para no aplicarla mal:** con este peso el término
> aporta el **0,33 %** del *valor* de la pérdida. La regla decía que «un contraste sobre el 3 % de la
> pérdida no es un contraste», pero lo que aquella regla medía era el **gradiente**, no el valor, y
> acá el gradiente está igualado por construcción. **Se declara antes: si alguien lee el 0,33 % como
> violación de la regla, la regla está mal citada.**

## 3. Diseño, y es barato porque el control ya está corrido

| | `--rec-rank` | origen | prefijo |
|---|---:|---|---|
| **CONTROL**, ya en disco | **0,0** | `b3_s3` / `b3_s6` | `t03_s3` · `t03_s6` |
| **TRATAMIENTO** | **0,008** | `b3_s3` / `b3_s6` | `r03_s3` · `r03_s6` |

Todo lo demás **idéntico y heredado de la campaña L**: interfaz `token`, `L=0`, `M=0,5`, `F=0,2`,
`CE=1,0`, `p_nose=0,4`, nivel 3, lr 1e-3, horizonte 12000, **3000 pasos**. **Dos unidades de GPU.**

**Compuerta O-0, ya corrida y verificada:** con `--rec-rank 0` la pérdida es **bit a bit** la anterior
(11.2584819794 en las dos ramas); toda constante da exactamente `log 2` sea −5, 0 o +5; el oráculo da
**0,000000**; el gradiente es finito; y `rec_rank` entra en la guarda de identidad del checkpoint.

## 4. Predicciones, fijadas ANTES

**O-1 · PRINCIPAL.** En las **dos** unidades de tratamiento, el acuerdo pareado entre «se calla» y
«no hay respuesta» supera **0,60**. Hoy el control da 0,4985 y 0,4985 contra un azar de 0,5004.

**O-2 · MECANICISTA, y es la que hace válido el resultado.** La pureza de la decisión agrupando por
relación cae **por debajo de 0,70** (control: 0,977 y 0,982, nulo 0,517). Dice que se **rompió el
atajo**, no sólo que se movió la tasa. Sin O-2, un O-1 que cumpla no distingue entre las dos cosas.

**O-3 · EL CONTROL QUE PUEDE FALLAR, y es el negativo del 29-ago.** `invento` **no supera 0,10**. Si
el término de orden vuelve a desacoplar la decisión del valor, las unidades se van a inventar igual
que con `balance` y `ranking` (invento hasta 0,1966). **Sería el mismo negativo por tercera vez y hay
que poder verlo.**

**O-4 · NULO.** RECUP no cae más de 0,05 respecto del origen (0,3654 y 0,3835).

**O-5 · EXACTITUD.** La exactitud global supera el piso trivial **0,4065**. Alcanzable sin aprender
nada nuevo: con la RECUP que ya tienen, repartir bien da **0,6234**.

**O-6 · RIESGO DECLARADO, y va contra la hipótesis.** El término de orden separa maximizando margen,
así que puede empujar `q` a los extremos por otra vía. Se exige abstención **estrictamente entre 0,05
y 0,95**; si sale de ahí, O-1 no se adjudica aunque cumpla.

**O-7 · RIESGO.** 3000 pasos pueden no alcanzar. Si O-1 falla **pero** el término de orden bajó de
`log 2` (o sea empezó a ordenar), es **falta de presupuesto y no un negativo** — la misma lectura que
la `NOTA_LECTURA_FASE_H` del 30, y se declara antes para no repetir el quinto negativo por impaciencia.

## 5. Cómo se lee cada desenlace, escrito ANTES

| desenlace | lectura | qué se hace |
|---|---|---|
| **O-1, O-2 y O-3 cumplen** | la degeneración era el bloqueo | es el resultado de la línea: la abstención se calibra rompiendo la planitud, no reponderando |
| **O-1 sí, O-2 no** | mejoró la tasa sin romper el atajo | no se vende como calibración; se informa la brecha |
| **O-3 se dispara** | el orden volvió a desacoplar del valor | negativo por tercera vez; **ahí sí se cierra la línea de la función de pérdida** |
| **O-1 no, y el término no bajó de log 2** | no hubo presupuesto | se extiende, no se adjudica |
| **O-1 no, y el término SÍ bajó de log 2** | ordenó y no alcanzó | el cuello no es la decisión sino la evidencia, y pasa a la recuperación |

## 6. Relación con el criterio de abandono del §6 de `PREREG_RECOMPENSA_L`

Aquél cierra «arreglarlo desde la función de pérdida» si L-1 falla en las dos interfaces. **Sigue en
pie y no se toca.** Lo que lo distingue de este caso está escrito antes de correr: las cuatro
variantes anteriores movían **dónde está el óptimo de la tasa**; ninguna tocaba la **degeneración**,
que recién se midió hoy. Esto no es un quinto peso: es quitarle la planitud a un eje que no tenía
gradiente. **Si O-3 se dispara, el criterio se aplica y la línea se cierra.**

## 7. Lo que NO contesta

- **No dice que el modelo sepa cuándo no sabe.** Sigue supervisado; el cierre de seis meses del
  `PLAN_FOCO_20260824.md` no se toca.
- **No dice nada de `cabeza`**, cuya fase H sigue NO EVALUABLE por presupuesto.
- **No prueba que escale.** El término de orden es **cuadrático en el lote** (pares), y eso hay que
  decirlo: con B=64 son 4096 pares, barato acá y no necesariamente barato a escala.
- **Y arrastra el confound del 28:** semillas sin base, comparables sólo contra sí mismas y contra su
  par de `rec_rank`.
