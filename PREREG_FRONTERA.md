# PREREG — ¿Qué decide que el micro-LM aprenda a callarse? La frontera del margen

**Estado:** CONGELADO 2026-08-18 antes de implementar la infra y antes de correr un solo paso.
Aprobado por Maxi tal como está (sin cambios al diseño ni a las predicciones).
**Fecha:** 2026-08-18.

## §1 · El objetivo, y por qué esta pregunta es el cuello

Lo que se busca del micro-LM son tres cosas que **son la misma medida vista de tres lados**:

| lo que se pide | cómo se mide hoy |
|---|---|
| que no olvide lo que se le dijo o leyó | `vigente` y `anterior` |
| que sepa decir «no sé» | `nose` |
| que no alucine | responder un valor cuando no hay → `1 − nose`; responder el valor equivocado → error de identidad |

Y ya está medido de dónde sale cada una:

- **No olvida por escritura.** `INFORME_RANK_HECHO_20260816`: la entrada del hecho preguntado está
  ausente del archivo en **0,0000** de 8000 muestras. El hecho SIEMPRE se escribe.
- **Alucina al leer.** Cuando se equivoca de identidad, el hecho correcto está en el archivo con rango
  mediano 2 y gana la selección sólo el 14-18 % de las veces. **El error es enteramente de lectura →
  convertible en abstención.**
- **Puede callarse, si se le da interfaz propia.** La campaña del 18-ago: `cabeza` pasa la compuerta
  en 4 de 5 unidades donde `token` y `escala` fallan en 5 de 5.

Queda un cabo suelto, y es el que bloquea todo lo demás: **el 17-ago la separación entre los modelos
que aprenden a abstenerse y los que no era perfecta, y la explicaba el «margen sobre el atajo»** —
`vigente` al cerrar la base, menos 0,5906, que es lo que vale no abstenerse nunca con `p_nose` 0,4.
Los de margen ≥ +0,4071 pasaban; los de margen ≤ +0,2358 fallaban. **Pero no hay ni un punto medido
entre +0,2358 y +0,4071**, así que no se sabe si el margen es un **umbral** o una **pendiente**, ni
dónde está el corte.

**Por qué importa para el objetivo y no es curiosidad:** si hay que entrenar el modelo hasta casi la
perfección ANTES de poder enseñarle a decir «no sé», el método no sirve para nada real — ningún
modelo útil llega a `vigente` 1,0 en su dominio. Si en cambio la cabeza corre la frontera hacia
abajo, la abstención se puede introducir temprano y **el modelo aprende a callarse mientras todavía
está aprendiendo a acordarse**, que es la única versión que escala.

## §2 · Lo que la campaña del 18-ago ya dice, y que reformula la pregunta

Con `token`, el margen predice **perfecto**: 0 de 5 con margen bajo. Con `cabeza`, **4 de 5 de esos
mismos modelos pasan**. → El margen no es una barrera del aprendizaje: **es una barrera de la
interfaz `token`**. La pregunta ya no es «¿dónde está la frontera?» sino **«¿la arquitectura la
mueve, y cuánto?»**.

## §3 · Diseño

**Eje A — margen al introducir `NOSE`.** Se muestrea el hueco guardando el checkpoint base cuando
`vigente` **cruza por primera vez** 0,85 · 0,90 · 0,95 (márgenes ≈ +0,26 · +0,31 · +0,36), más el
punto ya existente de 12000 pasos. El corte se hace **por valor de `vigente`, no por número de paso**:
es el margen lo que se quiere controlar, y fijar el paso lo dejaría al azar de la semilla.

**Eje B — condición:** `token` (control) y `cabeza`. `escala` no entra: falló 5 de 5 y P-2 la
descartó.

**Unidades:** nivel 2, semillas 0/1/2 (llega a saturar, así que su curva atraviesa todo el hueco).
**3 márgenes × 3 semillas × 2 condiciones = 18 fases** de 2000 pasos, más 3 corridas base que se
detienen al cruzar 0,95.

**Gratis y se agregan:** `n2_s1` (margen +0,2122) y `n4_s2` (+0,2163) están entrenados a 12000 y
nunca se usaron. Dan dos puntos más en el grupo bajo sin entrenar base.

## §4 · El confound, declarado antes de correr

Un modelo detenido en el paso 4000 con `vigente` 0,90 **no es equivalente** a uno que a 12000 pasos
llega a 0,90 por dificultad de la tarea: el primero está sub-entrenado en todo, no sólo en `vigente`.
**El eje A confunde margen con grado de entrenamiento, y no hay forma de separarlos dentro de este
diseño.** Lo que sí se puede es medir si las dos vías coinciden: los puntos del grupo bajo vienen de
tareas difíciles a entrenamiento completo, y los del hueco vendrán de tarea fácil a entrenamiento
incompleto. **Si el margen predice igual en ambas, es el margen. Si no, el margen era un proxy** —
y eso también es un resultado, probablemente mejor.

## §5 · Predicciones (a congelar)

- **F-1 (forma de la frontera, sobre `token`).** Con los 5 puntos bajos + 3 del hueco + los altos,
  `falsa_abst` contra margen es **monótona decreciente** (Spearman ρ ≤ −0,70). Si además hay un salto
  y no una pendiente, el corte cae **dentro** del hueco medido.
- **F-2 (la principal).** **`cabeza` corre la frontera hacia abajo:** en los tres márgenes del hueco
  pasa la compuerta en al menos 2 de 3 semillas, y en el margen más bajo (0,85) `cabeza` le gana a
  `token` en `falsa_abst` por **≥ 0,05**.
- **F-3 (control de sanidad, PUEDE fallar).** En el margen más alto las dos condiciones pasan. Si
  `token` falla también ahí, el punto de introducción no es el eje que gobierna y la campaña del
  17-ago necesita otra explicación.
- **F-4 (lo que haría útil el método).** Existe **algún** margen del hueco donde `cabeza` pasa y
  `token` falla en las 3 semillas. Ese margen es la recomendación operativa: *hasta acá hay que
  entrenar antes de enseñarle a callarse, si tiene cabeza propia*.

## §6 · Qué mata qué

- **F-2 cumple** → la abstención se puede introducir antes con arquitectura correcta. Es el resultado
  que el objetivo necesita.
- **F-2 falla con F-1 cumpliendo** → el margen es una barrera real del aprendizaje y no se mueve con
  la interfaz; entonces el camino no es arquitectónico sino de currículum.
- **F-1 falla (sin monotonía ni salto)** → el margen nunca fue la variable; la separación perfecta del
  17-ago era coincidencia con n=9, y hay que buscar qué la explica de verdad.

## §7 · Compromisos

Se reporta **por unidad, nunca sólo la media** — la bimodalidad entre semillas está medida desde
E-I3c. Los tres números (`vigente`, `nose`, `falsa_abst`) van juntos siempre. El presupuesto de la
fase se fija en **2000 pasos**, igual que la campaña del 18-ago, y se declara acá porque no lo fijaba
el prereg anterior.
