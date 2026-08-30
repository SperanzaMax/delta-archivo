# PRE-REGISTRO · `L` es el subsidio al silencio, y las mudas no son ignorantes

**2026-08-30.** Se congela **antes de lanzar la campaña** y **después** de los tres chequeos en CPU que
se reportan en el §2, que están hechos sobre checkpoints ya en disco y no gastaron un paso de GPU.

Deriva de `PREREG_RECOMPENSA.md` (`f1f7bb66`) y su `ENMIENDA_RECOMPENSA_F.md`. No los reemplaza:
corrige un parámetro más y cambia el punto de partida.

---

## 1. La pregunta

> El 29 se midió que **mudez e invención son los dos topes de una misma perilla**, y ninguna de las
> tres pérdidas nuevas la dejó en el medio. **¿La mudez persiste porque callarse es la política óptima
> por muestra, o porque el silencio cobra un premio que nadie le sacó?**

## 2. Lo que se midió hoy en CPU, antes de escribir esto

**(a) `mapa_recompensa.py` — el umbral que gobierna no era el que se usó ayer.** La ventaja de la
política del oráculo sobre el silencio total es

$$R_{\text{oráculo}}(c) - R_{\text{mudo}} = (1-\pi)\big[(1+M)c - M + F\big],$$

que cruza cero **exactamente en $c^{*}=(M-F)/(1+M)$**, el umbral POR MUESTRA. El umbral **global**
(0,657 con los pesos de ayer) gobierna sólo a un modelo que **no distingue** ausencia de error, y este
proyecto midió que sí distingue (AUC 0,9998–1,0000, `INFORME_A5_BLANCO_ERROR_20260827.md`). **La
predicción del 29 a la noche leyó la mudez de `f23_s3` contra el umbral equivocado.** El desenlace
operativo que anotó era correcto; la razón, no.

**(b) `medir_confianza.py` — `c` real, y su distribución, no su media.** n=4000, semilla 54321 pareada:

| unidad | paso | RECUP | `c` mediana | **frac. `c` > 0,200** | q (sin − con) |
|---|---:|---:|---:|---:|---:|
| `f23_s3` recompensa F=0,2 | 3000 | 0,2300 | 0,0405 | **0,119** | +0,0002 |
| `b3_s3` "muda", absorbente | 26000 | 0,3654 | 0,2324 | **0,544** | −0,0393 |
| `b3_s6` "muda", absorbente | 26000 | 0,3835 | 0,2541 | **0,580** | −0,0422 |
| `b3_s0` sana | 26000 | 0,9996 | 1,0000 | 0,9996 | +0,9988 |
| `n3_s0` base | 12000 | 0,7851 | 0,9671 | 0,8919 | — |

**Dos lecturas, y las dos cambian el tablero:**

1. **`f23_s3` hizo lo correcto.** Con el 88 % de las preguntas por debajo de $c^{*}$, callarse **era**
   la política óptima. No fue un fracaso de la pérdida: fue la pérdida funcionando.
2. **Las mudas no son ignorantes.** `b3_s3` y `b3_s6` superan $c^{*}$ en el 54 % y el 58 % de las
   preguntas. **El atractor absorbente vive en la cabeza, no en el generador.** El veredicto del 29
   («RECUP no se mueve dentro de una unidad») queda intacto y **acotado**: es cierto de RECUP bajo BCE,
   y no implica que al modelo le falte con qué hablar bajo otra pérdida.

**(c) El defecto aritmético: $L$ le paga al silencio.** Con $L=0{,}5$, $M=F$ aparte, una unidad que se
calla siempre cobra

$$R_{\text{mudo}} = \pi L - (1-\pi)F = 0{,}4065\cdot 0{,}5 - 0{,}5935\cdot 0{,}2 = \mathbf{+0{,}0845}.$$

**Es el piso trivial 0,4065 metido adentro de la recompensa como premio.** Con $L=0$ pasa a −0,1187 y
**$c^{*}$ no se mueve un decimal**, porque $L$ no entra en $(M-F)/(1+M)$. La intervención cambia
**un** parámetro y deja la política óptima por muestra idéntica.

> **Cómo se eligieron los pesos, y por qué no es ajustar sobre la marcha.** $L$ sale de una
> **identidad aritmética** ($R_{\text{mudo}}\le 0$), no del desenlace de `f23_s3`. $M$ y $F$ **no se
> tocan**. Verificar que el umbral sea alcanzable **desde el punto de partida medido** es justamente
> la corrección a los cinco defectos de pre-registro del mes, que fueron umbrales fijados sin
> comprobar que se pudieran alcanzar.

## 3. Diseño

**Siembra.** Las ocho unidades parten de `b3_s3` y `b3_s6` —las dos unidades **declaradas atractor
absorbente el 29**— pasadas por `sembrar.py`, que conserva los pesos, reinicia Adam, pone el paso en 0,
declara la procedencia en el checkpoint y **borra las claves de la corrida vieja** para que la guarda
de identidad no lea dos regímenes como una curva.

**★ Y el smoke destapó una simetría que no estaba buscada y que hay que declarar antes de correr:**
sembradas con `--abst token` las unidades arrancan **locuaces** (`abstencion` 0,0000 medido), porque
bajo `cabeza` el token `NOSE` estaba fuera del softmax de valores y su logit nunca se entrenó; con
`--abst cabeza` arrancan **mudas** (q ≈ 0,77), porque heredan la cabeza colapsada al prior. **Las dos
interfaces atacan el mismo punto desde los dos extremos opuestos de la política.**

| | interfaz | $L$ | arranca en | prefijo |
|---|---|---:|---|---|
| **T0** | `token`, sin cabeza | **0,0** | locuaz | `t0` |
| **T5** | `token`, sin cabeza | 0,5 | locuaz | `t5` |
| **H0** | `cabeza` heredada | **0,0** | mudo | `h0` |
| **H5** | `cabeza` heredada | 0,5 | mudo | `h5` |

$M=0{,}5$, $F=0{,}2$, CE$=1{,}0$, `p_nose` 0,4, nivel 3, lr 1e-3, **horizonte 12000, 3000 pasos**,
semillas **3 y 6**. Ocho unidades. **T es la condición principal**, por escalable, igual que en
`PREREG_RECOMPENSA`.

**Prioridad si Colab escasea** (ayer dio 503 en las trece cuentas): primero los **cuatro pares T0/T5**
y **H0/H5 en s3**, que son los que deciden L-1 y L-2. `H` en s6 es lo primero que se suelta.

## 4. Predicciones, fijadas ANTES

**L-0 · COMPUERTA, ya corrida.** El chequeo aritmético de `entrenar.py` aborta si $F\ge M$ y anuncia
$c^{*}$ y $R_{\text{mudo}}$ en cada arranque. **Verificado: aborta con la configuración exacta que
ayer costó ocho unidades de GPU.**

**L-1 · PRINCIPAL.** Con $L=0$, al menos **3 de 4** unidades **T0 y H0** superan la **exactitud
global 0,4065** a 3000 pasos.

> **Alcanzable, y con el número por delante:** callándose en las sin-respuesta y contestando en las
> con-respuesta al RECUP que ya tiene, `b3_s3` daría $0{,}4065 + 0{,}5935\times 0{,}3654 =
> \mathbf{0{,}6234}$. El criterio no pide aprender nada nuevo: pide **repartir** lo que ya sabe.

**L-2 · CONTRASTE PAREADO, y es lo que hace válido el resultado.** En ≥ **3 de 4** pares con el mismo
origen y la misma semilla, la unidad con $L=0$ supera en exactitud global a su par con $L=0{,}5$.
**Si L-1 cumple pero L-2 no, la causa fue la siembra y no $L$**, y se informa así.

**L-3 · MECANICISTA.** En las unidades que cumplan L-1, `abstencion` cae **estrictamente entre 0,05 y
0,95**. Ni muda ni locuaz: es la primera vez que el proyecto pediría el intermedio como predicción.

**L-4 · CONVERGENCIA DESDE LOS DOS EXTREMOS.** Si T0 y H0 cumplen L-1, sus `abstencion` finales
distan **menos de 0,20**. Sería evidencia de que el intermedio es el óptimo de la pérdida y no un
resto del arranque — el argumento de robustez que ninguna campaña anterior pudo hacer.

**L-5 · RIESGO DECLARADO, y va en contra de la hipótesis principal.** $L=0$ **debilita** el gradiente
que empuja a callarse donde no hay respuesta: $\partial R/\partial q$ pasa de $L+M=1{,}0$ a $0{,}5$.
**Predicción incómoda: T5 debería alcanzar la abstención ANTES que T0.** Si las $L=0$ terminan
locuaces (`abstencion` < 0,05), ésa es la causa y no se disfraza de otra cosa.

**L-6 · RIESGO, el warmup.** La lr arranca en 0 y sube a 1e-3 sobre una base ya entrenada. Se reporta
RECUP: **si cae más de 0,05 por debajo del origen, la campaña se lee como dañada** y L-1 no se
adjudica.

**L-7 · RIESGO, el nulo.** Puede que superen el piso apenas y sin saber más. Por eso L-1 pide el piso
**y** L-3 pide el intermedio; cumplir uno solo no es cumplir.

## 5. Cómo se lee cada desenlace, escrito ANTES

| celda | lectura | qué se hace |
|---|---|---|
| **L-1 y L-2 cumplen** | **$L$ era el subsidio al silencio**, y quitarlo alcanza | se escribe; es un resultado sobre por qué el atractor era estable |
| **L-1 sí, L-2 no** | la siembra explica todo y $L$ no aporta | se informa así, y el hallazgo pasa a ser sobre el punto de partida |
| **L-1 no, en T y en H** | ni el subsidio ni la confianza alta bastan | **se cierra la línea de la función de pérdida** (§7 de `PREREG_RECOMPENSA`), y el cuello queda en la recuperación |
| **L-3 falla con L-1 cumplida** | superó el piso sin llegar al intermedio | no se vende como abstención calibrada |
| **L-6 se dispara** | el warmup rompió lo sembrado | se repite con lr menor, y nada de esta corrida se adjudica |

## 6. Criterio de abandono

> Si **L-1 falla en las dos interfaces**, se cierra «arreglarlo desde la función de pérdida». Serían
> **cuatro** formas independientes —balancear, ordenar, premiar y quitarle el subsidio al silencio—
> moviendo la decisión sin mover la exactitud, **partiendo de un modelo cuya confianza ya está por
> encima del umbral**. Eso ubicaría el cuello de botella en la recuperación, sin margen para una
> quinta variante de lo mismo.

## 7. Lo que NO contesta

- **No dice que el modelo sepa cuándo no sabe.** Sigue siendo supervisado; el cierre de seis meses del
  `PLAN_FOCO_20260824.md` no se toca.
- **No mide calidad final.** 3000 pasos no son los 26000 de las campañas de referencia.
- **No prueba que escale.** Que el token sea la interfaz escalable es un argumento de diseño.
- **Y arrastra el confound del 28:** estas unidades vienen de semillas **sin base**, así que no son
  comparables con `b3_s0/s1/s2`. Se comparan sólo contra sí mismas y contra su par de $L$.
