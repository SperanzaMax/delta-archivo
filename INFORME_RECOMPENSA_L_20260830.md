# INFORME · `PREREG_RECOMPENSA_L.md` (SHA `96e750b6`) · las dos interfaces

**2026-08-30.** Ocho unidades sembradas desde `b3_s3` y `b3_s6` —las dos declaradas **atractor
absorbente** el 29— cruzando `L` ∈ {0 · 0,5} por interfaz ∈ {`token` · `cabeza`}. 3000 pasos,
horizonte 12000, `M`=0,5, `F`=0,2, CE=1,0.

---

## 1. Veredicto

| | criterio | `token` (T) | `cabeza` (H) |
|---|---|---|---|
| **L-1** principal | exactitud global > 0,4065 | **NO CUMPLE 0/4** | **NO EVALUABLE** (§3) |
| **L-3** | abstención en (0,05 · 0,95) | **CUMPLE 4/4** | **NO CUMPLE 0/4** |
| **L-2** pareado | L=0 supera a L=0,5 | 1 de 2 | no evaluable |
| **L-4** convergencia | \|abst(T0) − abst(H0)\| < 0,20 | **NO EVALUABLE** (§3) | |
| **L-6** riesgo | RECUP no cae > 0,05 | no se dispara | — |

## 2. `token` · llegó al intermedio y sigue sin discriminar

| unidad | exactitud | abstención | falsa_abst | invento | RECUP |
|---|---:|---:|---:|---:|---:|
| `t03_s3` (L=0) | 0,3020 | 0,4918 | ~0,48 | 0,2130 | 0,3675 |
| `t53_s3` (L=0,5) | 0,2938 | 0,4933 | ~0,48 | 0,2110 | 0,3364 |
| `t03_s6` (L=0) | 0,3005 | 0,4960 | ~0,48 | 0,2105 | 0,3650 |
| `t53_s6` (L=0,5) | 0,3030 | 0,4968 | ~0,48 | 0,2102 | 0,3621 |

**L-3 cumple 4/4 — primera vez en el proyecto que la abstención cae en un valor intermedio.**
Y **L-1 falla 0/4**: ninguna supera el piso trivial 0,4065.

> **★ EL HALLAZGO, y sale de mirar las cuatro juntas: `q` se clava en ~0,50 sin importar la semilla,
> el origen ni el valor de `L`** (0,4918 · 0,4933 · 0,4960 · 0,4968), **con `falsa_abst` ≈ 0,48 en
> todas.** O sea: se calla en el 48 % de las preguntas que **sí** tienen respuesta. La decisión de
> abstenerse es casi **independiente** de si hay respuesta.
>
> **Mudo, locuaz y medio son la MISMA patología con distinto valor: `q` es una CONSTANTE, no una
> función de la pregunta.** Cambiar la pérdida mueve el valor de la constante y nada más.

Ordenados por exactitud: **mudo 0,4065 > medio-sin-discriminar 0,3020 > locuaz 0,2181.**
El silencio total sigue ganando, y ahora se entiende por qué: con 40,65 % de preguntas sin respuesta,
callarse es un piso caro de superar.

**L-2 da 1 de 2 pares** (+0,0082 y −0,0025) = ruido. **Confirma la `PRECISION` (`4b61894e`), que lo
había declarado NO DECIDIBLE antes de mirarlo**: con CE=1,0 la recompensa es el **7,3 %** de la
pérdida y el logit de `NOSE` recibe **3,5× menos gradiente** que un token de valor cualquiera.

## 3. `cabeza` · las cuatro terminan MUDAS, y eso NO se lee como fracaso

| unidad | exactitud | abstención | acierto | invento | **RECUP** |
|---|---:|---:|---:|---:|---:|
| `h03_s3` (L=0) | 0,4055 | 1,0000 | 0,0000 | 0,0000 | **0,3541** |
| `h53_s3` (L=0,5) | 0,4055 | 1,0000 | 0,0000 | 0,0000 | **0,3583** |
| `h03_s6` (L=0) | 0,4055 | 1,0000 | 0,0000 | 0,0000 | **0,3549** |
| `h53_s6` (L=0,5) | 0,4055 | 1,0000 | 0,0000 | 0,0000 | **0,3553** |

`abstencion` = **1,0000 exacta** en las cuatro → exactitud clavada en **0,4055**, que es el piso
**muestral** de este lote (la fracción de preguntas sin respuesta en las 4000 muestras; el piso
poblacional declarado es 0,4065). El §4 del pre-registro ya lo anticipaba: *«el control de la
exactitud a 3000 pasos es exacto y no hace falta medirlo: una unidad muda da 0,4065 por definición»*.
Las cuatro dan **idéntico a cuatro decimales**, y `L` no las mueve ni una milésima — consistente con
que la salida no depende de la entrada.

> **★ Y el dato que importa: RECUP se mantiene en 0,354-0,358**, contra 0,3654 y 0,3835 de los
> orígenes `b3_s3`/`b3_s6`. **La recuperación NO se rompió: la mudez está en la cabeza, no en el
> generador.** Es la misma conclusión que la medición de `c` de la mañana, ahora por otra vía.

**La lectura estaba comprometida ANTES del dato** en `NOTA_LECTURA_FASE_H_20260830.md` (SHA
`4a0900bf`), congelada con la campaña ya corriendo:

> *«si `cabeza` termina muda, NO se lee como fracaso de la interfaz. Es indistinguible de "no le
> alcanzó el presupuesto para salir del atractor", y este proyecto ya tiene cuatro negativos que eran
> impaciencia.»*

**Las dos interfaces no parten del mismo lugar.** `token` arrancó **locuaz** (0,0000, el logit de
`NOSE` nunca se entrenó bajo `cabeza`) y `cabeza` arranca **muda** (hereda la cabeza colapsada al
prior). Con 3000 pasos para las dos, no recorren la misma distancia — y el aviso del 26-ago dice
textual que unidades así **se abstienen del 100 % durante ~3000 pasos y después aflojan solas**.

> **Conclusión: L-1 en `cabeza` queda NO EVALUABLE por presupuesto, y con ella L-4** (que da
> \|d\| = 0,5082 y 0,5040, pero mide la distancia entre un punto medido y uno no evaluable, así que
> no informa sobre convergencia). **El criterio de abandono del §7 NO se aplica**, porque exige las
> dos interfaces y una no fue medida en condiciones comparables.

**Y L-2 pasa a 1 de 4** al sumar las celdas de `cabeza`, pero **esas dos aportan 0,0000 exacto de
diferencia** —las dos mudas dan el mismo número— así que no son un contraste, son un empate
estructural. El pareado sigue decidido por las dos celdas de `token`, que ya estaban declaradas NO
DECIDIBLES por magnitud de gradiente.

## 4. Qué haría falta para cerrarlo

Extender las cuatro unidades de `cabeza`. **El horizonte ya está en 12000 en su config**, así que
extender **no toca la curva de lr** —la guarda de identidad lo permite y la lección D-1 del 22-ago
queda respetada—. Con eso la comparación sería a distancia recorrida y no a pasos iguales.

**Lo que NO corresponde hacer:** contar la mudez de `cabeza` como evidencia contra la interfaz. Sería
el quinto negativo por impaciencia del proyecto.

## 5. Lo que el conjunto sí deja establecido

1. **`q` es una constante en la interfaz `token`**, robusto a semilla, origen y `L`. Es el diagnóstico
   más preciso que tiene la línea de la abstención.
2. **`L` era efectivamente un subsidio al silencio** —con L=0,5 el modelo mudo cobra **+0,0845** de
   recompensa neta, o sea el piso trivial metido adentro de la pérdida como premio— pero quitarlo
   **no alcanza**: saca del extremo y deposita en otra constante.
3. **El bloqueo está medido y es de magnitud**, no de forma: cualquier intervención sobre `F`, `L` o
   un schedule opera dentro del 7,3 % de la pérdida que hoy ocupa la recompensa.

**Próximo paso derivado, no elegido mirando resultados:** bajar `--rec-ce` con el valor sacado del
**ratio de gradientes medido** (≈3,5), con pre-registro propio. Y después, la única candidata escrita
que puede romper la constante: un **castigo superlineal en la confianza** (que es lo rescatable de la
devolución de Gemini, ver `DICTAMEN_GEMINI_20260830.md`), porque hace que el costo dependa de la
pregunta concreta en vez del reloj.
