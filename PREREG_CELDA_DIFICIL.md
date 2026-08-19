# PREREG — La celda difícil: mismo margen, otra dificultad de tarea

**Estado:** CONGELADO 2026-08-19 antes de correr un solo paso y antes de mirar ningún número de
`n4_s2` con `p_nose` > 0.
**Fecha:** 2026-08-19.

## §1 · De dónde sale, y por qué no es una repetición

`PREREG_FRONTERA.md` §3 declaró, antes de correr la campaña de la frontera:

> **Gratis y se agregan:** `n2_s1` (margen +0,2122) y `n4_s2` (+0,2163) están entrenados a 12000 y
> nunca se usaron. Dan dos puntos más en el grupo bajo sin entrenar base.

**`n4_s2` nunca se corrió.** La campaña del 18-ago cerró con `t4_s0`, `t4_s1`, `c4_s0`, `c4_s1` y el
nivel 4 quedó con dos semillas. Este prereg corre la unidad que faltaba, y la corre porque hoy sirve
para algo más que sumar un punto.

`n2_s1` **no** se agrega: su `vigente` a 12000 es 0,8028 y el de `f2_s1` a 18000 es 0,8030, o sea el
mismo margen y la misma dificultad. Es el punto que la celda cruzada de esta mañana ya midió, no una
celda nueva.

## §2 · La celda que llena

El informe de la celda cruzada de hoy descartó C-2 con `f2_s1`: **nivel 2 (tarea fácil), entrenada a
fondo, margen bajo**. Lo que quedó sin medir es el mismo margen **por el otro lado de la dificultad**.

| | tarea fácil (nivel 2) | tarea difícil (nivel 4) |
|---|---|---|
| **margen ≈ +0,21** | `mt2_s1` / `mc2_s1` ✅ medido hoy | **`t4_s2` / `c4_s2` ← este prereg** |
| **margen ≈ +0,17** | — | `t4_s0` / `t4_s1` ✅ medido el 18-ago |

`n4_s2` cierra la base en `vigente` 0,8069 → **margen +0,2163**. `f2_s1` cierra en 0,8030 → **+0,2124**.
**Difieren en 0,0039 de margen y en dos niveles de dificultad.** Es el par mismo-margen/distinta-tarea
que ningún punto de la serie tiene hoy.

## §3 · Procedimiento

Idéntico al de la campaña de la cabeza del 18-ago, sin una sola variación, porque el valor de esta
unidad es ser **comparable** con `t4_s0`/`c4_s0`/`t4_s1`/`c4_s1`:

- parte de `ckpts/n4_s2.pkl` (12000 pasos, campaña base, `p_nose` 0)
- `p_nose` 0,4 · `p_vieja` 0,35 · presupuesto **2000 pasos** (12000 → 14000) · `cada` 250 · horizonte 20000
- Adam reiniciado en la siembra, igual que las otras
- dos condiciones: `token` (`t4_s2`) y `cabeza` (`c4_s2`). `escala` no entra (P-2 la descartó)
- compuerta, como siempre: **`falsa_abst` ≤ 0,10 y `nose` ≥ 0,50**

## §4 · Predicciones

- **D-1 (el margen gobierna).** `t4_s2` cae cerca de `mt2_s1` (0,1885) y no de sus hermanas de nivel:
  `falsa_abst(t4_s2)` ∈ [0,1385 ; 0,2385]. Y falla la compuerta, como todo el grupo de margen
  ≤ +0,2358 bajo `token`.
- **D-2 (la dificultad gobierna).** `t4_s2` cae junto a `t4_s0` (0,1342) y `t4_s1` (0,1713), media
  0,1528, y **por debajo** de `mt2_s1` pese a tener el mismo margen que ella.
- **D-3 (la principal, y la que no depende del poder estadístico).** `cabeza` le gana a `token` en
  `falsa_abst` también en esta unidad. Es el patrón que se cumplió en 7 de 7 puntos de la celda
  cruzada y en 4 de 5 unidades de la campaña del 18-ago; si se rompe justo acá, el efecto de la
  arquitectura depende de la dificultad de la tarea y eso cambia la recomendación operativa.
- **D-4 (orden dentro de la misma dificultad).** Entre las tres semillas de nivel 4 —márgenes +0,1672
  (s0), +0,1787 (s1), +0,2163 (s2)— `falsa_abst` bajo `token` es monótona decreciente con el margen.

## §5 · Poder, dicho antes de mirar

**D-1 y D-2 predicen valores que distan 0,036 entre sí, y son 2 unidades de una sola semilla.** Con la
dispersión que muestran los datos existentes, este contraste **no puede resolverse**: se declara acá
para que el resultado no se lea después como si hubiera sido decisivo. Lo que este experimento sí
puede hacer es **falsar D-3 y D-4**, que son afirmaciones sobre orden y no sobre distancia.

**D-4 ya arranca en contra.** Entre `t4_s0` (margen +0,1672 → `falsa_abst` 0,1342) y `t4_s1` (+0,1787
→ 0,1713) el orden **ya está invertido**: más margen, más abstención falsa. Con dos puntos eso no
significa nada; con tres se puede decir algo. Si `t4_s2` no queda por debajo de las dos, **dentro de
una misma dificultad el margen no ordena**, y entonces lo que ordenaba en las 13 unidades del informe
de la frontera era la dificultad de la tarea, con el margen sólo acompañándola.

## §6 · Qué mata qué

- **D-3 falla** → el aporte de `cabeza` no es general; hay que acotarlo por dificultad antes de
  recomendarlo.
- **D-4 falla** → el margen deja de ser variable explicativa dentro de la dificultad, y la lectura del
  informe de la frontera se debilita, aunque la celda cruzada de hoy siga en pie.
- **D-3 y D-4 cumplen** → el margen ordena también dentro de la tarea difícil, y con eso el margen
  queda como la variable en las dos direcciones del cuadro del §2.

## §7 · Compromisos

Se reporta **por unidad y nunca sólo la media**, con los tres números (`vigente`, `nose`,
`falsa_abst`) juntos. Se reporta además la **trayectoria de los 8 puntos de evaluación**, no sólo el
último, porque la celda cruzada de esta mañana mostró que el veredicto binario de una unidad puede
depender de dónde cayó el corte. Si el resultado queda pegado al límite de la compuerta, se dice.
