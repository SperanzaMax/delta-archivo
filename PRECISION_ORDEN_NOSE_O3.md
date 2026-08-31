# PRECISIÓN a `PREREG_ORDEN_NOSE.md` (SHA `9e5659e5`) · **O-3 está mal fijado y no discrimina**

**2026-08-31, tarde.** Se escribe con la campaña **en curso** (`r03_s3` corriendo en la cuenta A) y
**antes** de que exista un solo checkpoint de tratamiento en disco. Motivo: al validar el juez contra
el control aparece que **el criterio O-3, tal como lo escribí, lo falla también el control**, y eso
hay que decirlo antes y no después.

---

## 1. Qué disparó la corrección

`juzgar_orden.py` corrido sobre el control, que ya estaba en disco (n=1024, semilla pareada):

| | `t03_s3` | `t03_s6` |
|---|---:|---:|
| acuerdo con «no hay respuesta» | 0,5098 (azar 0,4997) | 0,5059 (azar 0,4992) |
| pureza por relación | 0,9785 (nulo 0,5304) | 0,9824 (nulo 0,5297) |
| **`invento`** | **0,2100** | **0,2100** |
| exactitud global | 0,3203 | 0,3281 |
| RECUP | 0,3711 | 0,3625 |
| abstención | 0,5020 | 0,5059 |

Todo replica lo medido el 30 y el 31 por otras vías. **El problema es `invento` = 0,2100.**

O-3 pedía `invento ≤ 0,10` y lo justificaba como «el control que puede fallar, y es el negativo del
29-ago», donde `balance` y `ranking` llegaron a 0,1966. **Pero el control de esta campaña ya está en
0,2100, o sea por encima del umbral y por encima del negativo que O-3 quería detectar.**

## 2. Por qué pasó, y no es un descuido de aritmética

O-3 se derivó del número del 29 **sin medir el mismo estadístico en el control de hoy**. Y tiene una
razón mecánica que ahora es obvia: un modelo que se calla en la mitad de las preguntas **sin mirar
cuáles** contesta la mitad de las que no tienen respuesta, y `p_nose = 0,4`, así que
`invento ≈ 0,5 × 0,4 = 0,20`. **El 0,21 del control no es una patología: es la aritmética de decidir
al azar**, y es exactamente el fenómeno que la campaña quiere arreglar.

Un umbral absoluto de 0,10 le pedía al tratamiento algo que **sólo se puede cumplir si O-1 ya cumplió**
(hay que discriminar para dejar de inventar). O-3 dejaba de ser un control independiente y pasaba a
ser una consecuencia de O-1.

## 3. Qué se corrige, y qué NO

**O-3 pasa a ser RELATIVO al control, con el mismo espíritu con el que se escribió** (detectar que el
término de orden desacopló la decisión del valor, como pasó el 29):

> **O-3′ · CONTROL.** `invento` del tratamiento **no supera al del control por más de 0,02**
> (0,2100 + 0,02 = **0,2300**). Si lo supera, el término de orden empeoró la alucinación y **se aplica
> el criterio de abandono del §6 de `PREREG_RECOMPENSA_L`**.

El umbral de 0,02 no se elige mirando nada: es el orden de la dispersión entre las dos semillas del
control (0,2100 y 0,2100, idénticas a cuatro decimales acá, y 0,2130 / 0,2105 en la medición del 30
con otro `n`), o sea **el ruido del estadístico**.

**Y se agrega el criterio que O-3 quería expresar y no expresaba:**

> **O-3″ · el `invento` DEBE BAJAR si O-1 cumple.** Si el acuerdo supera 0,60 pero `invento` se queda
> en ~0,21, las dos cosas son incompatibles y el resultado se declara **inconsistente**, no positivo.
> Con acuerdo 0,60 el `invento` esperado cae a **≈0,16** o menos.

**Lo que NO se toca, y se deja escrito:**

- **O-1, O-2, O-4, O-5, O-6 y O-7 quedan exactamente como estaban.** Ninguno de ellos lo falla el
  control: acuerdo 0,5098 contra el 0,60 pedido, pureza 0,978 contra el 0,70 pedido, exactitud 0,3203
  contra el piso 0,4065. **Los cinco pueden fallar y el control efectivamente los falla, que es lo que
  un criterio tiene que poder hacer.**
- **El umbral de O-1 no se afloja.** 0,60 sigue siendo 0,60.

## 4. Un dato nuevo del control, que no estaba medido y vale por sí solo

**El término de orden vale 10,3858 y 10,9448 en el control**, contra `log 2 = 0,6931` de cualquier
constante y 0 del oráculo. **El control está 15× PEOR que no decidir nada.**

No está sólo desordenado: está **anti-ordenado y saturado**, dándole masa de `NOSE` a las preguntas
que sí tienen respuesta de forma sistemática. Es coherente con los logits de ±20 medidos hoy: cuando
la partición arbitraria cae al revés, la diferencia es de −20 y el softplus la cobra entera.

**Y tiene una consecuencia práctica que mejora la campaña:** con `rec_rank = 0,008` el término aporta
`0,008 × 10,4 = 0,083` sobre una pérdida de ~2,0, o sea el **4 %** y no el 0,33 % que el §2(d) del
pre-registro calculó **usando el valor en el checkpoint de siembra (0,83) en vez del valor donde el
modelo se degrada**. El término tiene **más** autoridad justo cuando el problema aparece, que es lo
que se quería. **Se declara acá porque el número del prereg quedó desactualizado, no porque convenga.**

## 5. Lección

Van **siete** defectos de pre-registro este mes. Éste es de la misma clase que el sexto —el criterio
está bien derivado y el problema es su calibración— pero con un agravante propio: **el número contra
el que se calibró venía de OTRA campaña, con otra pérdida y otro punto de partida.**

> **Regla que deja: todo umbral absoluto se mide primero EN EL CONTROL de la propia campaña, antes de
> congelar el pre-registro. Si el control lo falla, el criterio no discrimina y hay que hacerlo
> relativo.** Sale gratis cuando el control ya está en disco, que es justamente el caso que esta
> campaña eligió por barato.
