# PREREG — ¿La abstención necesita una cabeza propia, o le alcanza con un vector más largo?

**Congelado:** 2026-08-18, antes de implementar nada y antes de ver un solo número.
**Scripts:** `micro_lm/modelo.py` + `micro_lm/entrenar.py` con el eje `--abst` (hashes en
`PREREG_CABEZA_ABSTENCION_HASH.txt`, tomados antes de correr).

## §1 · De dónde sale

El 17-ago la campaña `x` dejó un patrón sin excepciones en 9 modelos: **`nose` nunca es el problema**
(0,57-0,98 en todos) y **lo que falla siempre es `falsa_abst`** (0,13-0,22 en los cinco que no pasan la
compuerta). El diagnóstico correcto no es «no aprende a abstenerse» sino **«abstenerse le sigue
pagando demasiado»**.

Y hay una pista estructural medida sobre los pesos: **el vector del token `NOSE` tiene norma 0,367,
contra 1,011 de «ana» y 1,028 de «beto»**. `NOSE` compite en el mismo softmax que los valores con un
vector tres veces más corto, y ese softmax mezcla dos decisiones de naturaleza distinta: una binaria y
balanceada («¿está?») con una de 1-entre-100 («¿qué valor?»).

El 18-ago quedó establecido, con el control que faltaba (`PREREG_INYECCION.md`), que **la abstención
consulta el archivo de verdad**: con el hecho disponible la tasa de `NOSE` se derrumba de ~0,90 a
0,006-0,022. O sea que la señal existe y el problema es de **interfaz de salida**, no de acceso a
memoria. Por eso este experimento se corre ahora y no antes.

## §2 · Las tres condiciones (pareadas, todas desde el mismo checkpoint base)

- **`token`** — lo de hoy: `NOSE` es una entrada más del softmax de vocabulario.
- **`escala`** — idéntico, pero al arrancar la fase de abstención la fila de `NOSE` en `head.w` se
  **renormaliza a la norma media de los tokens de valor**. Es la explicación barata de la pista
  estructural: si el problema *es* la norma, esto solo alcanza y no hace falta arquitectura nueva.
- **`cabeza`** — una **cabeza binaria separada**: un escalar por muestra, proyección propia desde el
  mismo estado final. Se responde `NOSE` si σ(a) > 0,5; si no, el argmax del softmax de valores **con
  `NOSE` excluido**. La pérdida pasa a ser `BCE(a, es_nose)` más la CE del valor **aplicada sólo
  cuando hay respuesta**. Las dos decisiones dejan de competir por la misma masa de probabilidad.

**Justicia del contraste, declarada por adelantado:** las tres se corren **desde cero desde el
checkpoint base**, con el estado de Adam reinicializado en las tres. La campaña `token` del 17-ago
**no** se reusa como línea de base, porque no reinicializó Adam y la comparación no sería pareada.
La cabeza agrega **129** parámetros sobre 863.730 (0,015 %): no es una condición con más capacidad,
y se declara para que no se lea así. *(Corregido el mismo día: al congelar el prereg escribí 193, calculado sobre D=192; el modelo real es D=128, o sea D+1 = 129. Verificado al construirlo: 863.859 parámetros.)*

## §3 · Unidades

**Predicción principal sobre las cinco que HOY fallan** la compuerta: `x3_s0`, `x3_s1`, `x3_s2`,
`x4_s0`, `x4_s1` (`falsa_abst` 0,13-0,22).
**Predicción de no-daño sobre dos que HOY pasan**: `x1_s0`, `x2_s0`.
Compuerta idéntica a la del 17-ago: **`nose` ≥ 0,50 Y `falsa_abst` ≤ 0,10**.

## §4 · Predicciones (congeladas)

- **P-1 (principal).** En las cinco que fallan, `cabeza` baja `falsa_abst` a **≤ 0,10** manteniendo
  `nose` **≥ 0,50** en **al menos 3 de las 5**. Es decir: la compuerta pasa a pasarse por
  arquitectura, sin tocar el margen ni el punto de introducción.
- **P-2 (el control que puede ahorrar la arquitectura).** `escala` **no** alcanza:
  su `falsa_abst` medio queda **> 0,10**, y **`cabeza` le gana por ≥ 0,05** en `falsa_abst` medio.
  Si `escala` iguala a `cabeza`, **el hallazgo es que era la norma y no la arquitectura**, y la cabeza
  separada se descarta por innecesaria.
- **P-3 (no-daño).** En `x1_s0` y `x2_s0`, `cabeza` no empeora `nose` ni `falsa_abst` en más de 0,05
  respecto de `token` corrido en las mismas condiciones.
- **P-4 (control de sanidad, PUEDE fallar).** `vigente` bajo `cabeza` se mantiene dentro de ±0,05 del
  de `token`. Sacar `NOSE` del softmax de valores no debe degradar la recuperación; si la degrada, la
  ganancia en `falsa_abst` está comprada con precisión y no se lee como mejora.

## §5 · Qué mata qué

- **P-1 y P-2 cumplen** → la abstención necesita una interfaz propia; el «se callan de más» era un
  artefacto de mezclar dos decisiones en un softmax. Es un resultado de arquitectura, del tipo que
  pide [[objetivo-memoria-persistente-llm]].
- **P-2 falla porque `escala` iguala** → era la norma. Resultado más barato y igual de publicable, y
  la cabeza no se construye.
- **P-1 falla con P-2 cumpliendo** → ni la norma ni la interfaz: el «se callan de más» viene del
  currículum o del margen, y la línea vuelve a la frontera del margen sin muestrear.
- **P-4 falla** → cualquier ganancia queda suspendida hasta explicar la pérdida de `vigente`.

## §6 · Compromiso

Se reporta **por unidad, nunca sólo la media** — la bimodalidad entre semillas está medida desde
E-I3c y una media taparía exactamente lo que interesa. Los tres números de cada corrida (`vigente`,
`nose`, `falsa_abst`) se publican juntos: `nose` solo no significa nada, que es la lección del 15-ago.
