# Gemación con desplazamiento acotado — resultados

Prereg `PREREG_GEMACION_ACOTADA.md` (SHA ab3115e2…), congelado antes del dato.
ε = 0.3 (sin re-ajustar) · γ = 0.5 · k = 5 · 10 semillas × 1000 · margen 0.02 · IC95 t de Student, 9 gl

## COBERTURA (ambas versiones en el top-k)

| K | `g_orbita` | `g_decay` | `g_fija` | `duplicados` |
|---|---|---|---|---|
| 1 | 0.9946 | 0.9946 | 0.9946 | 0.9991 |
| 2 | 0.9747 | 0.6805 | 0.0546 | 0.9914 |
| 4 | 0.8628 | 0.0282 | 0.0000 | 0.9482 |
| 8 | 0.1958 | 0.0000 | 0.0000 | 0.2822 |

## Coseno de la versión vigente contra la consulta (P-A4)

| K | `g_orbita` | `g_decay` | `g_fija` | `duplicados` |
|---|---|---|---|---|
| 1 | +0.8177 | +0.8177 | +0.8177 | +0.8540 |
| 2 | +0.8178 | +0.7722 | +0.7126 | +0.8537 |
| 4 | +0.8176 | +0.7265 | +0.3362 | +0.8536 |
| 8 | +0.8178 | +0.7103 | -0.1513 | +0.8545 |

## Veredictos pre-registrados

**P-A3 (control de mecanismo, BLOQUEANTE)** `g_fija` a K=4 = 0.0000 [0.0000, 0.0000] · exige < 0,10 → **reproduce el colapso, el harness es comparable**

**P-A1 (PRINCIPAL)** COBERTURA a K=8, `g_orbita` − `duplicados` = **-0.0864** IC95 [-0.1014, -0.0714] · exige ≥ 0.02 sin cruzar cero → **NO CONFIRMA**

**P-A2 (no-regresión)**
  - K=1: -0.0045 IC95 [-0.0059, -0.0031] · piso −0.02 → ok
  - K=2: -0.0167 IC95 [-0.0197, -0.0137] · piso −0.02 → ok

**P-A4 (mecanicista)** pendiente del coseno por revisión (exige ≥ −0,01 en `g_orbita`)
  - `g_orbita`: +0.0000 por revisión
  - `g_fija`: -0.1410 por revisión
  → P-A4 **CUMPLE**

**Secundaria** `g_decay` − `duplicados` a K=8 = -0.2822 IC95 [-0.2912, -0.2732] → no supera

## Falsación global (§5 del prereg)

`g_orbita` **no supera a `duplicados` en ningún K**. Según lo comprometido por adelantado, la gemación queda **descartada como mecanismo de indexación en este régimen**, y no se prueba una tercera geometría.

---

## Por qué este negativo es más fuerte que el anterior

**La reparación funcionó.** P-A4 lo confirma sin ambigüedad: el coseno de la versión vigente contra la
consulta en `g_orbita` es **plano** — 0,8177 · 0,8178 · 0,8176 · 0,8178 para K = 1, 2, 4, 8, pendiente
**+0,0000** por revisión. Contra **−0,1410** por revisión en `g_fija`. La deriva acumulativa que P4
había identificado como causa del colapso **está eliminada por completo**.

**Y el mecanismo pierde igual.** `g_orbita` queda por debajo de `duplicados` en los cuatro K, y a K = 8
la diferencia es **−0,0864 IC95 [−0,1014, −0,0714]**.

Eso descarta la hipótesis de rescate. Si el desplazamiento no acotado hubiera sido el problema,
acotarlo tenía que alcanzar. No alcanza, así que **el problema no era la deriva: es el desplazamiento
en sí**.

## El mecanismo del fracaso, medido

La tabla de cosenos lo dice de un vistazo:

| | coseno con la consulta (constante en K) |
|---|---|
| `duplicados` | **0,8540** |
| `g_orbita` | **0,8177** |

Las dos son planas. La diferencia es un **peaje constante de ~0,036** que paga la gemación por
desplazar la entrada ε del ancla. Y ese peaje alcanza para perder la competencia del top-k contra las
entradas de otras entidades.

**La razón de fondo:** `emb(v_r)` —el embedding del texto real de esa versión— ya está **óptimamente
colocado**, porque contiene la entidad que la consulta menciona. Cualquier desplazamiento artificial,
por acotado que sea, sólo puede alejarlo. La geometría no tenía nada que agregar: el encoder ya había
puesto cada versión donde correspondía.

## Cierre de la línea, según lo comprometido

§5 del pre-registro se comprometió por adelantado: si `g_orbita` no superaba a `duplicados` en ningún
K, la gemación quedaba descartada como mecanismo de indexación **sin probar una tercera geometría**.
Es lo que ocurre, y así se cierra.

**Lo que queda establecido, con dos experimentos pre-registrados y sus mecanismos identificados:**

1. **Con paso fijo** (P4): colapsa por deriva acumulativa del clúster.
2. **Con desplazamiento acotado** (P-A1): no colapsa, pero pierde por el peaje constante del
   desplazamiento.
3. **La conclusión operativa es la del prereg original §4**, ahora sostenida por evidencia y no por
   defecto: *«el mecanismo se reduce a "guardá las dos versiones"»* — con su metadato de versión, que
   es exactamente lo que hacen Zep/Graphiti y lo que la tesis determinista ya publicada recomienda.

**Lo que NO queda establecido, y hay que decirlo:** esto vale para un **índice no paramétrico sobre un
encoder congelado**. No dice nada sobre un índice **co-entrenado dentro de la red**, donde las
direcciones no serían embeddings de texto fijos sino representaciones aprendidas — y donde el
argumento de «el encoder ya lo puso donde corresponde» deja de aplicar por construcción. Ese sigue
siendo el hueco del camino B, con sus dos obstáculos intactos (el gradiente no fluye por la selección
top-k, y el *stale index*).
