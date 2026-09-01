# El desacuerdo no GANA a la confianza, la COMPLEMENTA — y donde el modelo está confiado, marca ausencia

**2026-09-01, mañana.** Control declarado faltante en el §4 del `INFORME_DOS_VECES_20260831.md` y
primer punto del `PLAN_20260901.md`. Criterios congelados en `NOTA_PRECISION_DESACUERDO.md`
(SHA **9188e02d**) **antes** de correr. `precision_desacuerdo.py`, `n3_s0`, n=1536, σ=0,4, sin GPU.

---

## 1. El control principal se dispara: P-1 NO CUMPLE

| a igual cobertura **0,0781** (120 preguntas) | precisión |
|---|---:|
| desacuerdo | **0,8750** |
| confianza baja · pasada limpia | 0,8417 |
| confianza baja · pasada ruidosa `r1` | 0,8417 |
| marcar al azar (= tasa base) | 0,5286 |

**diferencia +0,0333, IC95 bootstrap pareado [−0,0418, +0,1102]** → cruza el cero.

> **El 0,8980 del 31-ago no se atribuye al desacuerdo como detector superior.** Marcar por confianza
> baja el mismo 7,8 % de las preguntas da lo mismo dentro del error. Es el **mismo dictamen que D-4**
> dio para el AUC (0,6054 contra 0,6054), ahora replicado en precisión, que era la métrica bajo la
> cual el detector se veía útil.

**P-2 CUMPLE** (0,8750 contra tasa base 0,5286, IC95 [0,8125, 0,9310] muy por encima): el detector es
real, no es ruido. **P-3 CUMPLE**: el 0,8980 de ayer cae dentro del IC95, así que **el número de
`n=512` replica** y no era un artefacto de muestra chica.

## 2. Lo que el mismo control destapa: no son la misma medida, son COMPLEMENTARIAS

**P-4 (descriptivo, anticipado por escrito antes del dato): Jaccard = 0,2308** — sólo 45 preguntas en
común de 195. Los dos detectores marcan grupos **mayormente distintos** con precisión casi igual.

Eso vuelve legítima la pregunta condicional, que es la correcta para «¿aporta señal independiente?»:

**Dentro de las 1416 preguntas que la confianza NO marca** (confianza alta, tasa de error propia
0,5021):

| | |
|---|---:|
| el desacuerdo marca | 75 |
| precisión ahí | **0,8533** |
| donde no marca | 0,4825 |
| **ganancia condicional** | **+0,3512** IC95 **[+0,2694, +0,4280]** |

> **El desacuerdo separa en el territorio donde la confianza ya no discrimina.** P-1 preguntaba si
> *gana*; la respuesta es no. Pero *no es redundante*: aporta donde la otra medida es ciega.

## 3. ★ El hallazgo, y toca el objetivo declarado de la línea

De esas **75 preguntas marcadas con el modelo CONFIADO**:

| | |
|---|---:|
| **sin respuesta en el archivo (ausencia)** | **0,7200** |
| con respuesta, valor errado | 0,1333 |
| aciertos (falso positivo del detector) | 0,1467 |

**Control de la explicación alternativa, corrido antes de llamarlo hallazgo** (la regla del proyecto):
¿el grupo de confianza alta ya venía enriquecido en ausencia? **No, al contrario** — tiene 0,3870,
algo **menos** que el 0,4043 global.

**Enriquecimiento real contra su propia base: 1,86× · ganancia +0,3330, IC95 [+0,2298, +0,4310].**

> **Cuando el modelo está seguro y las dos búsquedas no coinciden, casi tres de cada cuatro veces es
> porque el dato NO ESTÁ.** Ése es exactamente el régimen de la alucinación confiada, y es donde la
> confianza de salida no ve nada por construcción.

Esto **matiza el límite declarado ayer** («el detector no separa error de ausencia»): sin condicionar
no los separa, pero **condicionado a confianza alta el enriquecimiento en ausencia es de 1,86×**.

Y **no contradice el techo de 0,7003**: aquél acota a los lectores del **estado** —funciones de un
punto—, y esto mide la **estabilidad del mecanismo** alrededor del punto. Es la propiedad por la que
el plan de ayer lo eligió como la única candidata que podía pasar el techo sin romper nada.

## 4. Lo que NO dice

- **Un modelo, un σ, una corrida.** Falta el degradado, donde el 31-ago se midió que la inestabilidad
  es 50× menor y el detector se apagaría.
- **La sección 3 es post-hoc.** El subgrupo «confianza alta» no estaba en los criterios; lo habilitó
  P-4, que sí estaba escrito antes, pero eso no lo convierte en confirmatorio. **Genera hipótesis, no
  adjudica**, y necesita su propio pre-registro para valer como resultado.
- **La cobertura sigue siendo baja:** 75 de 1416 es el 5,3 % del grupo confiado.
- Sigue siendo la versión **post-hoc por ruido**, la más débil de las tres. Las dos fuertes —dos
  queries `qr1`/`qr2` aprendidas con el desacuerdo en la pérdida, y buscar por entidad contra buscar
  por relación— siguen sin probar, y ahora con una razón más para probarlas.

## 5. Nota de método: la guarda de NO EVALUABLE funcionó, y se puede comprobar

El smoke con n=128 marcó 17 preguntas y el juez devolvió **NO EVALUABLE** en vez de una precisión
sobre 17 casos. Es la primera aplicación de la regla que dejó el O-6 del 31-ago, y **la guarda podía
fallar** — que es justo lo que le faltaba al control `m=1` del 12-ago.

Arrays en `salidas/precision_desacuerdo_n3_s0.npz`: todo análisis posterior sale de ahí sin recomputar.
