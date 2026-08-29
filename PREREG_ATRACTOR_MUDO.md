# PRE-REGISTRO · ¿el atractor mudo es ABSORBENTE o sólo LENTO?

**2026-08-29.** Se congela antes de lanzar y antes de mirar un solo número nuevo.

Sale de `INFORME_BIFURCACION_20260829.md`, que es post-hoc sobre archivos ya en disco y dejó **una**
pregunta abierta que no se puede contestar con lo que hay.

---

## 1. La pregunta, y por qué no se puede contestar con lo que hay

Las cuatro unidades mudas tienen una recuperación que **no es cero**, y que ordenada por paso queda:

| unidad | paso | RECUP |
|---|---:|---:|
| b3_s8 | 8000 | 0,3050 |
| b3_s7 | 13500 | 0,3218 |
| b3_s3 | 22000 | 0,3665 |
| b3_s6 | 25000 | 0,3960 |

Ajuste lineal r = 0,9836, pendiente **+0,0052 cada 1000 pasos**.

> **Pero son cuatro unidades DISTINTAS en cuatro pasos DISTINTOS. No es una trayectoria.** La
> correlación es igual de compatible con «una unidad muda mejora despacio» que con «las semillas que
> se rompen más tarde se rompen menos», y los checkpoints intermedios se pisan, así que **con lo que
> hay en disco no se desconfunde**.

De ahí las dos lecturas, que llevan a proyectos distintos:

- **ABSORBENTE** — el punto fijo degenerado es estable y no se sale con presupuesto. El blanco `error`
  queda descartado como mecanismo y lo que importa es evitar entrar, no salir.
- **LENTO** — es un cuello de botella y con horizonte suficiente la unidad sale. Entonces el blanco
  `error` no está roto, está **mal presupuestado**, y el 26000 de todas las campañas es el problema.

## 2. Por qué esto va en dos fases y no en una

**La continuación directa de `b3_s3` más allá de 26000 NO es posible sin cambiar de corrida**, y lo
impide el propio código: `entrenar.py:443` aborta si el horizonte pedido difiere del del checkpoint,
porque `HOR` es el `decay_steps` del `warmup_cosine_decay_schedule` (línea 351). Extender a 80000 con
un checkpoint de horizonte 26000 sería continuar con la lr ya en su piso, que es exactamente lo que la
guarda existe para impedir y lo que la D-1 del 22-ago costó.

Así que la Fase 2 tiene que **entrenar desde cero con el horizonte largo**, y eso cuesta. La Fase 1
existe para decidir si vale la pena pagarlo, y **de paso cierra dos cosas que ya estaban pendientes**.

---

## FASE 1 · barata, sin desviación, y cierra tres cosas a la vez

**Qué se corre:** `b3_s3` de 22000 a **26000** y `b3_s6` de 25000 a **26000**. Flags **idénticos** a
`PREREG_TASA_REGIMEN` —`--abst cabeza --donde pre --blanco error`, nivel 3, `p_nose` 0,4, horizonte
26000— o sea que **esto es terminar la campaña para esas dos unidades, no una condición nueva.**

**Qué se agrega, y es lo único nuevo:** un archivador que **guarda copia de cada checkpoint parcial**
en `ckpts_traza/b3_sX_<paso>.pkl` a medida que bajan, para poder medir **RECUP como trayectoria dentro
de una misma unidad**, que es lo que falta. No toca el entrenamiento ni el rotador.

**`s7` y `s8` no se corren.** T-0 y T-1 de `PREREG_TASA_REGIMEN` ya están fallados por aritmética
(§1 del informe del 29) y esas dos no pueden cambiarlos. Se declara como desviación en
`DESVIACIONES_TASA_REGIMEN.md`.

### Predicciones de la Fase 1, fijadas ANTES

**F-0 · EL TEST DURO DEL PREDICTOR, y es el que puede tumbarlo.** Se predice que `b3_s3` y `b3_s6`
llegan a 26000 **sin salir de la abstención total** (`abstencion` ≥ 0,999 en todos los hitos).

> Si alguna de las dos emite respuestas antes de 26000, **el predictor del paso 2500 pasa de 40/40 a
> 40/41 y deja de ser un predictor**. Se reporta destacado y el §2 del informe del 29 se corrige.

**F-1 · PRINCIPAL, la pendiente intra-unidad.** En `b3_s3`, RECUP medido en 22000 · 23000 · 24000 ·
25000 · 26000 **crece de forma monótona en al menos 3 de los 4 intervalos**, y el total sube
**≥ +0,010** entre 22000 y 26000.

Fundamento del número, escrito antes: la pendiente entre unidades predice +0,0052 × 4 = **+0,021** en
esos 4000 pasos. Se pide la mitad porque **los últimos 4000 pasos corren con la lr en el piso del
cosine** (`lr × 0,1`), así que lo que se mida acá es un **límite inferior** de la pendiente real y
pedir el valor completo sería pedir de más.

**F-2 · CONTRASTE.** Lo mismo en `b3_s6`, que sólo tiene 1000 pasos. Se reporta pero **no cuenta para
el veredicto**: un intervalo no hace una pendiente. Está para ver el signo.

**F-3 · RIESGO DECLARADO.** Si RECUP **baja** o queda plana (< +0,003 en 4000 pasos), la correlación
del §1 era **confound entre semillas** y la lectura «lento» se cae sin necesidad de la Fase 2.

### Cómo se lee cada desenlace, escrito ANTES

| celda | lectura | qué se hace |
|---|---|---|
| **F-0 falla** (alguna despierta) | el predictor no es un predictor | se corrige el informe del 29 y **la Fase 2 no se lanza**: la pregunta cambia |
| **F-1 cumple** | la pendiente es intra-unidad y sobrevive al control | **se lanza la Fase 2** |
| **F-1 falla, F-3** | era confound entre semillas | **atractor ABSORBENTE**, la Fase 2 no se lanza y el blanco `error` queda cerrado también por acá |
| **F-1 falla sin F-3** (entre +0,003 y +0,010) | ambiguo a esta escala | no se lanza la Fase 2 con este dato; hace falta un rango de pasos mayor, y eso es otro pre-registro |

---

## FASE 2 · condicional, se lanza SÓLO si F-1 cumple

**Diseño:** cuatro unidades nuevas `b3_s9` … `b3_s12`, flags idénticos salvo **`--horizonte 60000` y
`--pasos 60000`**. La semilla y el horizonte son lo único que cambia.

**El cribado, que es lo que la hace pagable, y de paso VALIDA el predictor.** Las cuatro se corren
**sólo hasta 2500 pasos**. Ahí se mira el hito del paso 2500:

- las que emitieron **≥ 1** respuesta de 512 se **matan** (se predice que van a vivir, no sirven acá);
- las que emitieron **0** siguen hasta 60000.

> Esto usa el predictor como **instrumento** y al mismo tiempo lo somete a su primera prueba
> **prospectiva**, que es exactamente la advertencia que el informe del 29 dejó escrita sobre sí mismo.
> **G-0:** de las que se dejan correr, se predice que **ninguna** sale de la abstención total antes de
> 26000. Cada una que salga es un fallo del predictor y se reporta.

**Costo:** 4 × 2500 = 10000 pasos de cribado, más 60000 por cada muda que siga. Con la tasa observada
(4 de 9) se esperan **1 o 2** mudas. Si el cribado deja **cero** mudas, se declara y no se re-tira:
buscar semillas hasta que salga una muda sería elegir la muestra después de verla.

**Por qué 60000 y no 80000.** No hace falta llegar a RECUP 0,70 para separar las dos lecturas, basta
con ver si la pendiente se sostiene. A 60000 la extrapolación da ≈ 0,57 contra ≈ 0,40 de hoy, que es
una diferencia que no se confunde con ruido. 20000 pasos más costarían un tercio más para afinar un
número que no cambia la decisión.

### Predicciones de la Fase 2, fijadas ANTES

**G-1 · PRINCIPAL.** RECUP de la unidad muda crece de forma sostenida entre 26000 y 60000, y llega a
**≥ 0,50** a los 60000.

**G-2 · CONTRASTE con el techo.** Si además llega a `vigente` ≥ 0,70 **y** sale de la abstención total,
el atractor no sólo es no-absorbente sino **transitorio**, y eso reabre el blanco `error` como vía con
presupuesto mayor. Es el desenlace más fuerte posible y el menos esperado.

**G-3 · RIESGO.** La curva de lr de horizonte 60000 **no es** la de 26000: es más plana durante más
tiempo. Una unidad que salga podría estar saliendo por la lr y no por el presupuesto, y eso **no se
puede separar en esta campaña**. Se declara ahora, antes de correr, y si G-1 cumple hay que decir en
el informe que la atribución queda abierta.

## 3. Criterio de abandono

> **Si F-1 falla, no se lanza la Fase 2 y no se prueban más semillas.** La lectura «lento» queda
> refutada con el único control que la puede refutar, y el blanco `error` queda cerrado también por
> esta vía, que se suma a la del 27.

> **Si G-1 falla, el atractor es ABSORBENTE y se termina.** No se prueba 100000, no se prueba otra lr.

## 4. Lo que NO contesta

- **No revive A5 ni `PREREG_TASA_REGIMEN`.** A5 cerró el 27 y la campaña de la tasa se juzga como
  T-0 fallido. Esto mide **otra cosa**.
- **Sigue siendo supervisado.** El §8 del `PLAN_FOCO_20260824.md` y su cierre de seis meses no se tocan.
- **No dice nada sobre escala.** 863.730 parámetros, idioma de 242 tokens, `p_nose` 0,4, un nivel.
- **Y no valida el predictor para otros bancos.** Ni siquiera para este con otro blanco: las 31
  corridas de blanco `ausencia` nunca entran en la fase muda, así que ahí el predictor no tiene nada
  que predecir.
