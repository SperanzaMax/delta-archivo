# INFORME · Fase 1 — el atractor mudo es ABSORBENTE, y la extrapolación de la mañana estaba mal

Evalúa `PREREG_ATRACTOR_MUDO.md` (SHA `2be4a610`, congelado 10:40:59) con
`PRECISION_ATRACTOR_MUDO_N.md` (SHA `5e413fef`, congelada 10:55:50 **antes de medir** un solo
checkpoint nuevo).

`b3_s3` de 22000 a **26000** y `b3_s6` de 25000 a **26000**, flags idénticos a `PREREG_TASA_REGIMEN`.
Instrumentos: `archivar_traza.sh` (nuevo) y `traza_recup.py` (nuevo), **n=8000, semilla de datos
54321, diseño pareado**.

---

## 1. Los criterios, contra lo medido

| | criterio | resultado | |
|---|---|---|---|
| **F-0** el test duro del predictor | `abstencion` ≥ 0,999 en todos los hitos hasta 26000 | **NINGÚN hito por debajo, en las dos** | **CUMPLE** |
| **F-1** PRINCIPAL | ΔRECUP ≥ +0,0100 (22000→26000) y monotonía ≥3/4 | **−0,0021** y monotonía **2/4** | **NO CUMPLE** |
| **F-2** contraste (no cuenta) | lo mismo en `b3_s6`, 1 intervalo | +0,0080 en 1000 pasos (1,6 σ) | se reporta |
| **F-3** riesgo · confound | se dispara si < +0,0030 | **−0,0021** | **SE DISPARA** |

**Celda del §«Cómo se lee cada desenlace», escrita antes de correr:** *F-1 falla con F-3 → la
correlación del §1 era confound → atractor **ABSORBENTE**, la Fase 2 no se lanza y el blanco `error`
queda cerrado también por esta vía.*

**Se aplica el desenlace. Pero la razón NO es la que la celda decía, y eso hay que corregirlo abajo.**

### La trayectoria de `b3_s3`, que es el dato que nunca se había podido medir

| paso | RECUP | Δ | p_flip | σ de la diferencia |
|---:|---:|---:|---:|---:|
| 22000 | 0,3841 | | | |
| 23000 | 0,3778 | −0,0063 | 0,1472 | 0,0056 |
| 24000 | 0,3844 | +0,0065 | 0,1347 | 0,0053 |
| 25000 | 0,3812 | −0,0032 | 0,1166 | 0,0050 |
| 26000 | 0,3820 | +0,0008 | 0,1181 | 0,0050 |
| **total** | | **−0,0021** | 0,1526 | 0,0057 → **0,4 σ** |

**Las cuatro diferencias son ruido** (0,1–1,2 σ) y el total está a 0,4 σ de cero. En 4000 pasos la
recuperación de una unidad muda **no se mueve**.

`p_flip` se reporta **medido**, como la precisión se comprometió: **0,11–0,15**, tres veces el 5 % que
se había supuesto. Aun así σ de la diferencia pareada queda en 0,0057, o sea que **F-1 era decidible
con N=8000** — con el N=2000 original no lo habría sido, y ése es el motivo de la precisión.

---

## 2. La corrección al informe de la mañana, y la hipótesis que se cayó por el camino

### Lo que sospeché primero, y es FALSO
Al ver que `b3_s3` a 22000 daba **0,3665** con n=2000 y **0,3841** con n=8000, sospeché que la
pendiente entre unidades del §5 del informe del 29 (r = 0,9836) fuera **ruido de medición**. Se
re-midieron las cuatro unidades con n=8000:

| unidad | paso | RECUP n=2000 | **RECUP n=8000** |
|---|---:|---:|---:|
| b3_s8 | 8000 | 0,3050 | **0,3040** |
| b3_s7 | 13500 | 0,3218 | **0,3432** |
| b3_s3 | 22000 | 0,3665 | **0,3841** |
| b3_s6 | 25000 | 0,3960 | **0,3884** |

**r = 0,9864 y el rango vale 12,5 σ. La correlación es real y mi sospecha era falsa.** Queda anotado
porque el control se corrió antes del veredicto y dio vuelta la lectura que ya tenía escrita, que es
la tercera vez este mes.

### Lo que sí está mal, y es otra cosa
**No es una recta.** Las pendientes por tramo decrecen de forma monótona:

| tramo | ΔRECUP por 1000 pasos |
|---|---:|
| s8 → s7 (8000→13500) | +0,0071 |
| s7 → s3 (13500→22000) | +0,0048 |
| s3 → s6 (22000→25000) | +0,0014 |
| **dentro de `b3_s3`** (22000→26000) | **−0,0005** |

Un ajuste `RECUP = A − B·e^(−paso/τ)` da **SSE 12,6 veces menor** que el lineal (τ ≈ 13200,
asíntota ≈ 0,42).

> **La extrapolación del §6 del informe del 29 —«~84000 pasos para llegar a RECUP 0,70»— era un
> artefacto de ajustar una recta a una curva cóncava.** Se retira.

**Con la salvedad honesta de que el ajuste de saturación NO es lo que sostiene la conclusión:** son 4
puntos para 3 parámetros, y su predicción para 26000 (0,4535) ni siquiera acierta el valor medido
(0,3820). Lo que sostiene la conclusión es el dato **directo y sin confound posible**: dentro de una
sola unidad, con las mismas preguntas y 4000 pasos de por medio, **RECUP no se mueve**.

### Y por eso la celda del pre-registro acierta el desenlace por la razón equivocada
F-3 estaba redactado como «la correlación era confound entre semillas». **No lo era.** La lectura
correcta es: la correlación entre unidades es real, pero **no es una trayectoria temporal
extrapolable** — es una curva que ya saturó, y el tramo 22000–26000 está en la meseta. El desenlace
operativo (no lanzar la Fase 2) es el mismo; la frase no.

---

## 3. Qué se cierra

**La Fase 2 NO se lanza,** por el criterio de abandono del §3 del pre-registro, que dice exactamente
esto y se aplica sin discusión. Son **70000 pasos** que no se corren.

Y ahora hay una **predicción cuantitativa** de lo que habría dado, que es mejor que una corazonada:
G-1 pedía RECUP ≥ 0,50 a los 60000 pasos, y todo lo medido dice que la unidad se queda alrededor de
**0,38–0,42**. La campaña habría gastado 70000 pasos para fallar su criterio principal.

> **El atractor mudo es ABSORBENTE.** No es un cuello de botella lento: es un punto fijo, y con este
> presupuesto y esta curva de lr no se sale de él.

**Y F-0 es el otro resultado del día.** El predictor del paso 2500 se formuló ayer post-hoc sobre 40
corridas y **hoy sobrevivió su primera prueba con unidades llevadas a horizonte completo después de
formularlo**: las dos mudas más viejas llegaron a 26000 sin un solo hito por debajo de 0,999. Sigue
siendo una regla post-hoc —la prueba verdaderamente prospectiva era el cribado de la Fase 2, que ya no
se corre— pero pasó el test más duro que había disponible.

## 4. Lo que NO autoriza

- **No dice que el blanco `error` no sirva nunca.** Dice que **una unidad que entró en el atractor no
  sale** en 22000–26000 pasos con esta curva de lr. Nada de esto habla de las 5 unidades que **no**
  entraron.
- **No prueba la saturación.** El ajuste exponencial es sugerente y está sobreparametrizado. Lo
  medido es que la pendiente intra-unidad es cero en ese tramo.
- **No separa presupuesto de curva de lr.** Los últimos 4000 pasos de `b3_s3` corren con la lr en el
  piso del cosine. Que RECUP no se mueva ahí es compatible con «el atractor es absorbente» **y** con
  «con lr en el piso nada se mueve». Era el riesgo G-3, declarado antes de correr, y **la Fase 2 era
  justamente lo que lo iba a separar**. Al no lanzarse, **queda abierto**, y por eso el §3 dice
  «con este presupuesto y esta curva de lr» y no «nunca».
- **`b3_s6` va en la dirección contraria** (+0,0080 en 1000 pasos, 1,6 σ). El pre-registro ya había
  dicho que un intervalo no cuenta, y no cuenta, pero se reporta porque es el único dato del informe
  que empuja para el otro lado.
- **Sigue siendo supervisado, y sigue sin decir nada sobre escala.**

## 5. Nota de infraestructura

**Cada tramo desperdicia VM.** `MIN = TRAMO/1000*10 + 20` fija 40 minutos de polling para
`TRAMO=2000`, y el polling **no corta cuando el entrenamiento termina**: `VIVO=` nunca pasó a False en
ninguno de los tres tramos de hoy. `b3_s6` necesitaba 1000 pasos (~12 min) y ocupó la VM 40. Son
~16 min tirados por tramo en el caso normal y ~28 en el de `s6`.

**El comando del §6 del `ESTADO_20260828_NOCHE` no lanza** (ver D-3 de
`DESVIACIONES_TASA_REGIMEN.md`): `env $COM ... LOG_ROTADOR=$PWD/...` sin comillas, con un espacio en la
ruta. Reemplazado por `lanzar_fase1_0829.sh`.
