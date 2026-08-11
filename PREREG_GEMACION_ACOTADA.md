# Pre-registro — Gemación con desplazamiento ACOTADO

**Congelado el 2026-08-10, antes de generar un solo dato de este experimento.** Los embeddings de las
9 versiones ya existen (`hechos_revisiones.npz`, generados para P4), pero **ninguna de las condiciones
de acá se corrió**: lo que se pre-registra es la geometría del índice, que todavía no se construyó.

Continúa `PREREG_HECHOS.md` (+ D1, D2, enmiendas E1 y E2). Reutiliza su corpus, su encoder, sus
métricas y sus umbrales.

---

## 1. El diagnóstico del que sale

P4 falsó la gemación **tal como estaba especificada**: con paso ε fijo por revisión, la caminata sobre
la esfera **se aleja acumulativamente del ancla**. Medido, coseno de la versión vigente contra la
consulta:

| r | gemación de paso fijo | duplicados |
|---|---|---|
| 0 | 0,811 | — |
| 2 | 0,679 | 0,768 |
| 4 | **0,324** | 0,806 |
| 6 | **−0,139** | 0,819 |
| 8 | **−0,139** | 0,802 |

Cobertura resultante: `gemacion` 0,0567 a K=2 y **0,0000** desde K=4, contra `duplicados` 0,9914 /
0,9482 / 0,2822.

**La causa no es la idea de depositar-al-lado: es que el desplazamiento no está acotado.** El arco
recorrido crece sin cota (~0,30 rad por revisión, 2,36 rad a r = 8) y la versión vigente termina más
lejos de la consulta que cualquier distractor.

Este pre-registro prueba la única reparación que no cambia la idea: **acotar el desplazamiento**.

## 2. Las variantes, con su orden fijado de antemano

Todas comparten el principio de la gemación —las versiones de un mismo recuerdo viven **cerca** unas de
otras, y la cercanía codifica que son el mismo recuerdo— y difieren sólo en cómo acotan el
desplazamiento.

| condición | dónde va la revisión r | cota del desplazamiento |
|---|---|---|
| **`g_orbita`** (**PRINCIPAL**) | `normalizar(E₀ + ε·t̂_r)`, con `t̂_r` una dirección tangente **nueva** por revisión | **ε siempre**, respecto del ancla original |
| `g_decay` (secundaria) | caminata desde la anterior con paso `ε·γ^(r−1)`, γ = 0,5 | arco total ≤ `2ε` (serie geométrica) |
| `g_fija` (control de mecanismo) | la caminata de paso ε **falsada por P4** | **no acotada** |
| `duplicados` (referencia a batir) | `emb(v_r)`, el embedding real del texto de esa versión | — (posición medida, no inducida) |
| `sobrescritura` | `emb(v_r)` reemplazando la entrada previa | — |

**`g_orbita` es la principal y así queda fijado.** Razón declarada antes del dato: es la única que
acota el desplazamiento **por construcción y no asintóticamente** — toda versión está exactamente a ε
del ancla, sin importar cuántas revisiones haya. `g_decay` acota la suma pero deja que el centro del
clúster se corra hasta 2ε, así que es una reparación más débil del defecto diagnosticado.

**Si las dos superan a `duplicados`, el resultado principal es el de `g_orbita`.** `g_decay` se reporta
igual, y si la única que gana es `g_decay`, se reporta como tal y se dice explícitamente que la
predicción principal cayó.

## 3. Parámetros: NADA se re-ajusta

- **ε = 0,30**, el mismo de `PREREG_HECHOS.md` §6, tomado de R2 y no re-ajustado. **No se prueban otros
  valores.** Es la tentación obvia después de un negativo y es exactamente lo que invalidaría el
  experimento.
- **γ = 0,5** para `g_decay`: elegido para que la cota sea `2ε` (un número redondo), no ajustado a
  ningún resultado.
- k = 5 · 10 semillas × 1000 entidades · margen absoluto **0,02** · IC por t de Student con 9 gl.
- Corpus, encoder (`nomic-embed-text` en minúscula) y embeddings: los de `hechos_revisiones.npz`.
- El **orden de versiones lo sigue dando el metadato entero**, no la geometría (dictamen R1+R4 y §2.4
  de la enmienda E2). La geometría sólo agrupa.

## 4. Predicciones

**P-A1 (principal, falsable).** En **COBERTURA a K = 8**, `g_orbita` **supera** a `duplicados` con
diferencia media **≥ 0,02** e IC95 apareado que no cruce cero.

*Por qué K = 8 y no otro:* es el único punto donde `duplicados` **tiene margen** (0,2822 medido). A
K ≤ 4 está en 0,95–0,99 y un empate ahí no distinguiría nada — es la lección de P1, donde las dos
condiciones empataron en el techo y el negativo resultó sin potencia. Fijar el punto de test **antes**
del dato evita elegir después el K que convenga.

**Falsa si** la diferencia no supera el margen, o si el IC cruza cero.

**P-A2 (no-regresión).** A **K = 1 y K = 2**, `g_orbita` **no queda por debajo** de `duplicados` en más
del margen (≥ −0,02). Acotar el desplazamiento no debe costar nada donde no hacía falta.

**P-A3 (control de mecanismo, bloqueante).** `g_fija` **reproduce el colapso** de P4: cobertura
**< 0,10 a K = 4**. Si no lo reproduce, algo cambió en el harness y **ninguna comparación de este
experimento es válida** — se detiene y se investiga, como en las dos detenciones por compuerta previas.

**P-A4 (mecanicista).** El coseno de la versión vigente contra la consulta **no decae con r** en
`g_orbita`: pendiente ≥ **−0,01** por revisión, contra −0,17 por revisión medido en `g_fija`. Es la
verificación directa de que la cota hace lo que se supone que hace.

## 5. Falsación global, comprometida por adelantado

**Si `g_orbita` no supera a `duplicados` en K = 8 ni en ningún otro K, la gemación queda descartada
como mecanismo de indexación en este régimen**, y se reporta así, sin buscar una tercera variante.

Sería el segundo intento fallido del mismo mecanismo, y la conclusión honesta pasaría a ser la que el
prereg original ya anticipaba en §4: *«el mecanismo se reduce a "guardá las dos versiones", que no
necesita nada de este trabajo»* — reforzada, además, por la tesis determinista ya publicada
(`DOSSIER_CAMINO_B_20260810.md`).

**No se probará una tercera geometría en esta línea.** Si `g_orbita` y `g_decay` caen, el resultado es
un negativo con mecanismo identificado, que es publicable y es lo que hay.

## 6. Lo que este experimento NO puede afirmar

- **No mide un sistema de memoria completo.** Mide la geometría del índice con todo lo demás fijo.
  LongMemEval mide lo otro, y no son comparables (ver `DOSSIER_CAMINO_B_20260810.md`).
- **No dice nada sobre co-entrenamiento.** El índice es no paramétrico y el encoder está congelado. Los
  dos obstáculos reales del camino B —que el gradiente no fluye por la selección top-k, y el *stale
  index*— quedan intactos y fuera de alcance.
- **`duplicados` no es un competidor débil**: es exactamente lo que hacen los sistemas de producción
  (guardar cada versión con su metadato). Ganarle es el mínimo para que la geometría valga la pena.

## 7. Registro

Congelado con hash y anclado por push antes de construir ningún índice, bajo el mismo régimen que
`PREREG_HECHOS.md`, la enmienda E2 y E-006 de telar-ligamento.
