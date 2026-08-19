# PREREG — La celda cruzada: ¿el margen, o la dificultad de la tarea?

**Estado:** CONGELADO 2026-08-19, **antes de correr un solo paso** y antes de mirar ningún número de
`f2_s1` con `p_nose` > 0.
**Depende de:** `PREREG_FRONTERA.md` (§4, el confound declarado) e `INFORME_FRONTERA_20260819.md`.

---

## §1 · El cabo suelto que dejó la campaña de la frontera

`PREREG_FRONTERA.md` declaró en su §4, **antes de correr**, que el eje A confunde **margen sobre el
atajo** con **grado de entrenamiento**, y fijó el criterio de decisión:

> «Si el margen predice igual en ambas [vías], es el margen. Si no, el margen era un proxy — y eso
> también es un resultado, probablemente mejor.»

**Lo medido el 19-ago:** el margen predice igual por las dos vías, **sin una sola inversión en 13
unidades**. Pero eso es evidencia a favor, **no una demostración**, porque las dos vías quedaron
confundidas en el rango muestreado:

| | margen bajo (≤ +0,2358) | margen alto (≥ +0,2826) |
|---|---|---|
| **tarea difícil** (nivel 3-4) | 5 unidades, `token` falla 5/5 | *vacío* |
| **tarea fácil** (nivel 2) | **la celda que falta** | 8 unidades, `token` pasa 8/8 |

La celda vacía de arriba a la derecha **es inalcanzable por construcción**: los niveles 3 y 4 se
estancan en `vigente` 0,74-0,83, así que nunca llegan a margen alto. La de abajo a la izquierda, en
cambio, **ya existe y no se usó**.

## §2 · `f2_s1` es esa celda, y no fue construida para esto

`f2_s1` se entrenó como base de la campaña de la frontera y **falló su propósito**: nunca cruzó 0,85,
ni siquiera extendida dos veces (6000 → 0,7777; 14000 → 0,8234; 20000 → meseta y baja). Cerró en
**`vigente` 0,8030, margen +0,2124**.

Eso la deja **dentro del rango bajo** (+0,1489 a +0,2358) pero llegando por la vía opuesta a todas
las demás: **nivel 2 y entrenamiento completo**, en vez de nivel 3-4 y entrenamiento completo.

**Es una celda encontrada, no fabricada.** Nadie la diseñó para ganar un contraste; salió del fracaso
de otra cosa. Eso la hace mejor evidencia, no peor.

## §3 · Diseño

Dos fases pareadas desde `ckpts/f2_s1.pkl` (paso 18000), **2000 pasos** cada una hasta el paso 20000,
`p_nose` 0,4, Adam reinicializado, idénticas en todo salvo la condición:

- **`mt2_s1`** — `token`
- **`mc2_s1`** — `cabeza`

El presupuesto de 2000 pasos y el horizonte de 20000 son los mismos de las 12 fases anteriores, y
acá además **coinciden exactamente**: 18000 + 2000 = 20000 = horizonte de la tasa de aprendizaje, así
que la curva no se altera.

## §4 · Predicciones, y qué mata qué

Compuerta de siempre, sin cambios: **`nose` ≥ 0,50 y `falsa_abst` ≤ 0,10**.
Corte medido de `token`: entre +0,2358 y +0,2826. Corte medido de `cabeza`: entre +0,1489 y +0,1672.
`f2_s1` entra con **+0,2124**, o sea **debajo del corte de `token` y encima del de `cabeza`**.

- **C-1 (si gobierna el MARGEN).** `mt2_s1` **falla** la compuerta y `mc2_s1` **la pasa**. Es lo que
  predice la frontera medida, aplicada a una unidad que llega al margen por una vía nunca probada.
- **C-2 (si gobierna la DIFICULTAD DE LA TAREA).** **Las dos pasan.** Todas las unidades de nivel 2
  medidas hasta hoy pasan la compuerta con las dos condiciones, sin excepción; si lo que manda es el
  nivel, `f2_s1` debería comportarse igual pese a su margen bajo.
- **C-3 (el resultado incómodo).** **Las dos fallan.** Ni el margen ni el nivel alcanzan, y lo que
  gobierna es algo específico de esta semilla, que además es la que no convergió. Sería el peor caso
  y obliga a mirar la semilla, no el eje.

**C-1 y C-2 son mutuamente excluyentes y las dos son informativas.** C-1 convierte la evidencia
correlacional del 19-ago en un contraste con la vía cruzada. C-2 tumba el margen como variable causal
y deja que lo que separa sea la dificultad de la tarea, que es **un resultado mejor y más simple**,
tal como el prereg anterior anticipó.

## §5 · Compromisos

Se reporta **por unidad**, con los tres números juntos (`vigente`, `nose`, `falsa_abst`), pasen o
fallen. **Son 2 unidades y una sola semilla**: no alcanza para una media ni para hablar de
convergencia, y así se va a decir. Este experimento **no puede confirmar** nada por sí solo; lo que
puede hacer es **falsar** la lectura del margen si sale C-2. No se toca la compuerta ni el
presupuesto después de ver los números.
