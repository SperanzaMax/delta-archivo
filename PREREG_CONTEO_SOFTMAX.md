# PRE-REGISTRO · la abstención de `token` la decide el NÚMERO DE CANDIDATOS, no la evidencia

**2026-08-31.** Se congela **antes de correr la intervención** y **antes** de mirar el resultado de la
prueba algebraica de `prueba_conteo_softmax.py`, que está corriendo mientras se escribe esto.

---

## 1. De dónde sale

Medido hoy sobre `t03_s3` y `t03_s6` (`INFORME_RATIO_CE_20260831.md`), las dos a 3000 pasos:

| relación | valor | se calla `s3` | se calla `s6` | P(sin respuesta) | RECUP |
|---|---|---:|---:|---:|---:|
| altura | número | 0,9851 | 1,0000 | 0,4371 | 0,4722 |
| clave | número | 0,9883 | 1,0000 | 0,4506 | 0,4711 |
| precio | número | 0,9885 | 0,9951 | 0,4441 | 0,3984 |
| director | **nombre** | 0,0000 | 0,0000 | 0,4450 | 0,4164 |
| dueño | **nombre** | 0,0000 | 0,0000 | 0,4595 | 0,4654 |
| guardia | **nombre** | 0,0000 | 0,0000 | 0,4901 | 0,4000 |

El corte es **exactamente** `PERSONALES` (`idioma.py:53`), replica idéntico en dos semillas, y:

- **no sigue la ausencia:** P(sin respuesta) es 0,44-0,49 en las seis, y el valor del atajo medido
  sobre el generador es **+0,0000 exacto** (decidir sólo por la relación vale 0,5967, igual que no
  decidir nunca);
- **no sigue la dificultad:** RECUP nombres 0,4272 contra números 0,4472, o sea brecha **−0,0200** y
  **del lado contrario** — se calla donde acierta un poco MÁS;
- **no está en la tarea:** en `b3_s3` (el origen) y en `n3_s0` (base sana, RECUP 0,7885) la abstención
  es 0,0000 en las seis relaciones y la brecha nombre/número es −0,0047. Aparece **al entrenar la
  abstención**, no antes.

## 2. La hipótesis mecánica

Con `--abst token`, `q = softmax(lg)[NOSE]`, y por lo tanto, **exactamente**:

$$q > 0{,}5 \iff l_{\text{NOSE}} > \operatorname{logsumexp}_j\big(l_j\big)$$

**`NOSE` no compite contra el mejor candidato: compite contra la SUMA de todos.** Y el idioma tiene
**100 números contra 58 nombres** (`idioma.py:115` y `:28`), así que del lado numérico la suma tiene
**1,72×** más términos. Con logits parecidos, más términos ⇒ `logsumexp` más alto ⇒ `NOSE` gana.

> **Si esto es así, callarse no es una decisión del modelo: es una consecuencia de cuántas respuestas
> plausibles hay.** Y entonces el defecto no es de la función de pérdida ni del presupuesto —los dos
> lugares donde el proyecto viene buscando desde el 15-ago— sino **de la interfaz**.

## 3. Diseño de la intervención

**`I.fijar_pool_numeros(58)`** limita los valores numéricos que el generador sortea, **sin tocar el
vocabulario**: `V` sigue en 242, los 42 números restantes siguen existiendo como tokens y el modelo es
**idéntico** en arquitectura, así que la guarda de identidad no ve un idioma distinto y la comparación
es directa. Verificado antes de congelar: con `k=58` aparecen exactamente 58 valores numéricos
distintos (0-57).

| | pool numérico | pool de nombres | prefijo |
|---|---:|---:|---|
| **C100** control, lo de hoy | 100 | 58 | `c100` |
| **C58** tratamiento | **58** | 58 | `c58` |

Todo lo demás **igual y heredado de la campaña L**: interfaz `token`, `L=0`, `M=0,5`, `F=0,2`,
`CE=1,0`, `p_nose=0,4`, nivel 3, lr 1e-3, horizonte 12000, **3000 pasos**, sembradas desde `b3_s3` y
`b3_s6`. Cuatro unidades.

## 4. Predicciones, fijadas ANTES

**K-0 · COMPUERTA, ya corrida.** El pool truncado da 58 valores distintos y ni uno de los otros 42.
**Verificado.**

**K-1 · PRINCIPAL.** En **C58**, la brecha de abstención entre relaciones de nombre y de número
**cae por debajo de 0,20** (hoy es 0,9888 − 0,0000 = **0,9888**), en al menos **3 de 4** comparaciones
por semilla.

**K-2 · MECANICISTA, y es la que hace válido el resultado.** Si el mecanismo es el conteo, en C58 el
`logsumexp` de las dos clases se acerca: la diferencia `LSE(números) − LSE(nombres)` cae **al menos a
la mitad** de la de C100. Sin esto, un K-1 que cumpla podría ser cualquier otra cosa.

**K-3 · NULO, y puede fallar.** RECUP **no cambia** más de 0,05 entre C100 y C58. Truncar el pool
hace la tarea **más fácil** (58 opciones en vez de 100), así que si RECUP sube mucho, K-1 se explica
por la tarea más fácil y no por el conteo. **En ese caso K-1 no se adjudica.**

**K-4 · RIESGO DECLARADO.** Puede que la abstención se vaya al **extremo mudo o locuazo** en las dos
clases a la vez. Eso **cumpliría K-1 por la razón equivocada** (brecha chica porque no hay variación),
así que K-1 exige además que la abstención global quede **estrictamente entre 0,05 y 0,95**.

**K-5 · La predicción cuantitativa, que es lo más falsable que hay acá.** Con logits de valor
comparables, `logsumexp` sobre $n$ términos crece como $\log n$. Pasar de 100 a 58 debería bajar el
`logsumexp` numérico en $\log(100/58) = \mathbf{0{,}545}$ nats. **Se predice el número, no el signo.**

## 5. Cómo se lee cada desenlace, escrito ANTES

| desenlace | lectura | qué se hace |
|---|---|---|
| **K-1 y K-2 cumplen** | el conteo era la causa | **`token` queda descartada como interfaz de abstención por razón MECÁNICA**, no por presupuesto; la prioridad pasa a `cabeza` |
| **K-1 sí, K-2 no** | la brecha se movió por otra vía | no se adjudica el mecanismo; se informa la brecha y se busca la otra vía |
| **K-1 no** | el conteo NO era la causa | el corte nombre/número es aprendido y hay que preguntarse por qué; **se cae la hipótesis de la interfaz** |
| **K-3 se dispara** | la tarea se volvió más fácil | nada se adjudica y se repite con el pool de NOMBRES subido a 100 en vez del numérico bajado |

## 6. Lo que NO contesta

- **No dice cómo arreglarlo.** Igualar los pools es un instrumento de diagnóstico, no una solución:
  en un modelo real el número de continuaciones plausibles varía por pregunta y no se puede fijar.
- **No prueba que `cabeza` esté libre del defecto.** Eso es una medición aparte, sobre `h03_s3` y sus
  hermanas, y hay que hacerla sabiendo que esas unidades están 100 % mudas.
- **No toca el criterio de abandono del §7 de `PREREG_RECOMPENSA_L`**, que exige las dos interfaces.
- **3000 pasos** no son los 26000 de las campañas de referencia, y estas unidades vienen de semillas
  sin base, así que se comparan sólo contra sí mismas.
