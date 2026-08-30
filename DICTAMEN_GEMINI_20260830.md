# Dictamen sobre la devolución de Gemini · 2026-08-30

Maxi le pasó la bitácora a Gemini. La devolución trae una **propuesta concreta** y se la evalúa con el
mismo criterio que a cualquier dictamen externo: verificar antes de aceptar, y buscar la explicación
alternativa. Ver [[regla-verificar-antes-de-veredicto]] y el precedente del
`DICTAMEN_FABLE5_20260816.md`, donde su crítica era sana y su evidencia no.

---

## 1. Lo que hay que conceder, y no es poco

**La idea del `F` con schedule es NUEVA, no la probamos, y su dirección es correcta.**

Probamos `F` **fijo** dos veces: 1,5 (mandó al extremo locuaz, por mi error de derivación) y 0,2
(quedó mudo a 3000 pasos). Un `F` que **crece** con el entrenamiento nunca se probó.

Y la aritmética la respalda. Con $c^{*} = (M-F)/(1+M)$:

| F | c\* (umbral para hablar) | qué favorece |
|---:|---:|---|
| 0,00 | 0,3333 | callarse |
| 0,10 | 0,2667 | medio |
| 0,35 | 0,1000 | medio |
| 0,49 | 0,0067 | hablar |
| ≥ 0,50 | ≤ 0 | contestar siempre (el error de ayer) |

> Arrancar con `F` bajo deja que el modelo se calle **mientras efectivamente no sabe nada, que es lo
> correcto**, y subirlo baja el umbral y lo empuja a hablar cuando ya debería saber.

**Y converge de forma independiente con el criterio operativo que derivamos el 15-ago por otra vía**
(«introducir `NOSE` sólo cuando `vigente` supere la tasa de preguntas sin respuesta»). Que dos
razonamientos distintos lleguen al mismo curriculum es a favor.

## 2. Su diagnóstico está desactualizado, y es la parte que invalida la justificación

Gemini escribe: *«la cabeza de abstención se trabó entre los extremos del atractor mudo y la
locuacidad»* y justifica el castigo severo al error porque *«el castigo inicial fuerte a los errores
elimina la locuacidad»*.

**La bitácora que leyó llega al 29-ago. El 30 se midió otra cosa.**

- La **locuacidad ya no existe**: fue mi error de derivación con $F > M$, corregido el 29 a la noche.
- Con $L=0$, $M=0,5$, $F=0,2$ las cuatro unidades **llegaron al intermedio**: `abstencion` 0,4918 ·
  0,4933 · 0,4960 · 0,4968.
- **Y ahí está el problema real, que Gemini no podía tener:** llegaron al medio **y siguen sin
  discriminar** (`falsa_abst` ≈ 0,48). Se callan en el 48 % de las preguntas que **sí** tienen
  respuesta.

> **El problema dejó de ser «está trabado en un extremo». Es que `q` es una CONSTANTE y no una
> función de la pregunta.** Mudo, locuaz y medio son la misma patología con distinto valor.

## 3. La objeción de fondo a su propuesta

**Un $F(t)$ mueve el umbral a lo largo del TIEMPO. En cada instante sigue siendo un umbral GLOBAL,
igual para todas las preguntas.**

Si `q` es constante por muestra —y eso es lo que muestran las cuatro unidades— entonces un schedule
produce `q(t)` constante en cada paso. **No produce `q(x)` función de la entrada.** Movería la
constante más despacio; no la convertiría en decisión.

Es la misma forma del cierre del 29: *el blanco `error` tiene dos puntos fijos y la semilla decide a
cuál cae*. Cambiar la trayectoria hacia el punto fijo no cambia que sea un punto fijo.

## 4. Dos errores factuales en el «control crítico»

**(a) El horizonte 60000 NO EXISTE.** Gemini dice *«mantén el learning rate y el horizonte (60000)
intactos frente al control»*. Ese horizonte era el diseño de la **Fase 2 del atractor mudo, que se
CANCELÓ** el 29 cuando F-1 no cumplió (−0,0021 en 4000 pasos, 70000 pasos ahorrados). **Verificado en
disco: 0 checkpoints a 60000.** Los que hay son 12000 (8), 20000 (67) y 26000 (44). No hay control
contra el cual mantenerlo intacto.

**(b) El paso 2500 está fuera de su régimen validado.** El predictor de la bifurcación es específico
del **blanco `error` con cabeza binaria** —el informe lo dice: la fase muda temprana es *exclusiva*
de ese blanco, y las 31 corridas con blanco `ausencia` están en 0,02-0,32 a los 2500—. La condición
principal de la recompensa es **`token`, sin cabeza**. Y el propio informe declara que el predictor
**se eligió DESPUÉS de ver los desenlaces** y que usarlo exige pre-registrarlo primero.

## 5. Lo que sí hay que rescatar, y no es el schedule

Gemini mezcla **dos** ideas en una, y la que vale es la que menos desarrolla:

> *«el castigo por un error con alta confianza sea severo»*

**Eso es por MUESTRA, no por tiempo, y es lo único de la propuesta que ataca el problema real.** Hoy
$M$ es constante y el término del error entra como $(1-c)M$: la confianza pesa **linealmente**. Un
castigo **superlineal en la confianza** —del tipo $M(c) = M \cdot c^{k}$ con $k>1$— hace que el costo
dependa de **la pregunta concreta**, no del reloj.

Esa sí es una intervención que puede romper la constante, porque le da al gradiente una razón para
distinguir una muestra de otra. **Es la primera candidata escrita que lo hace.**

## 6. El orden correcto, y por qué el schedule no va primero

Antes de cualquier variante sobre `F` o `M` hay un bloqueo medido hoy
(`PRECISION_RECOMPENSA_L_CE.md`, `4b61894e`): con `--rec-ce 1.0` **la recompensa entera es el 7,3 %
de la pérdida**, y el logit de `NOSE` recibe **3,5× menos gradiente** que un token de valor
cualquiera.

> **Cualquier schedule sobre `F` opera DENTRO de ese 7 %.** Y la regla que quedó escrita hoy dice que
> un contraste sobre el 3-7 % de la pérdida no es un contraste.

**Orden propuesto:** (1) bajar `--rec-ce` con el valor derivado del ratio de gradientes medido (≈3,5),
con pre-registro propio; (2) recién ahí probar el castigo superlineal en la confianza del §5; (3) el
schedule de `F` sólo si (2) rompe la constante, como refinamiento y no como mecanismo.

## 7. Resumen del dictamen

| lo que dice | veredicto |
|---|---|
| rigor metodológico, falsabilidad, auditoría | descriptivo y correcto, no aporta acción |
| **`F` con schedule creciente** | **idea nueva y coherente, se conserva** — pero no ataca la constante y va tercera |
| «eliminar la locuacidad» | **desactualizado**: ya no existe, era mi error de derivación |
| «se trabó entre los extremos» | **desactualizado**: llegó al medio y sigue sin discriminar |
| **castigo severo al error con alta confianza** | **★ lo más valioso, y lo que menos desarrolla** — es por muestra y puede romper la constante |
| control con horizonte 60000 | **imposible**: la Fase 2 se canceló, 0 checkpoints |
| evaluar en el paso 2500 | **fuera de régimen**: el predictor es del blanco `error` con cabeza, y exige pre-registro |

**Y una advertencia de método que vale para todo dictamen externo:** Gemini leyó una bitácora que
llega al 29 y razonó sobre el estado de ese día. No es un defecto suyo. **Es la razón por la que un
dictamen externo se verifica contra el disco antes de convertirlo en campaña**, que es exactamente lo
que pasó con Fable 5 el 16-ago.
