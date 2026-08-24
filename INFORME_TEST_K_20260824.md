# TEST DE k — NEGATIVO. La hipótesis de la colisión es falsa y la línea se cierra

`PREREG_TEST_K.md` (SHA `d7ca4455`, congelado antes de correr una sola pasada) · datos en
`test_k_20260824.json`, `test_k_pre_20260824.json`, `test_k_lat_20260824.json`.

Seis unidades a 26000 pasos, `sonda_desacuerdo.py` con sus parámetros originales (K = 16, `f` = 0,25,
512 muestras por unidad). Brazos: `lat` (`w3_s*`) contra su control pareado `pre` (`p3_s*`).

---

## 1. El resultado

| | criterio | resultado | |
|---|---|---|---|
| **K-1** PRINCIPAL | `k(lat) ≤ 1,10` en ≥ 2/3 | **0 / 3** — 1,118 · 1,201 · 1,133 | **NO CUMPLE** |
| **K-2** pareada | `k(lat) < k(pre)` en ≥ 2/3 | **3 / 3** | **CUMPLE** |
| **K-3** cuantitativa | en `pre`, \|k − (1+P_rep)\| ≤ 0,15 | **1 / 3** | **NO CUMPLE** |
| **K-5** la que importa | `AUC(lat) ≥ 0,70` en ≥ 2/3 | **0 / 3** — 0,599 · 0,602 · 0,571 | **NO CUMPLE** |
| **K-4** el nulo | `f=0` da 1,000 exacto | **3 / 3** en los dos brazos | **OK** |

**Por el §5 del pre-registro y el §8 de `PLAN_FOCO_20260824.md`, comprometidos por adelantado: K-1
falla ⇒ la hipótesis de la colisión como causa del fracaso del monitor es FALSA y la línea se cierra
por seis meses.** No se prueba una sexta vía, no se ajusta `f`, no se cambia la sonda.

## 2. Los números

| sem | brazo | consistencia | k | P_rep | k estimado | AUC | `ident_rep` |
|---|---|---:|---:|---:|---:|---:|---:|
| s0 | pre | 0,6931 | 1,274 | 0,4243 | 1,424 | 0,609 | 0,0564 |
| s0 | **lat** | 0,7250 | **1,118** | 0,4243 | 1,424 | 0,599 | **0,0000** |
| s1 | pre | 0,6333 | 1,588 | 0,4087 | 1,409 | 0,648 | 0,4683 |
| s1 | **lat** | 0,7078 | **1,201** | 0,4087 | 1,409 | 0,602 | **0,0000** |
| s2 | pre | 0,6884 | 1,298 | 0,4268 | 1,427 | 0,586 | 0,2529 |
| s2 | **lat** | 0,7219 | **1,133** | 0,4268 | 1,427 | 0,571 | **0,0069** |

## 3. Lo que hace que el negativo sea informativo

### 3.1 El efecto existe y va en la dirección correcta, pero no alcanza

**K-2 cumple 3 de 3**, y no por poco: `k` baja **−0,156 · −0,386 · −0,166**. La colisión de clave
**sí contribuye** a que la respuesta dependa de más de una entrada, y la semilla donde más baja
(`s1`, −0,386) es exactamente la que tenía la colisión más grande (`ident_rep` 0,4683).

Pero eliminarla **entera** —`ident_rep` a 0,0000— deja `k` en **1,15 de promedio**, no en 1,0. Queda
un residuo que la colisión no explica. La hipótesis no era «la colisión aporta a k», que es cierta:
era «la colisión **es** el `k > 1` que mató al monitor», y eso es falso.

### 3.2 El dato que la mata, y es el que no se puede discutir

**El AUC BAJA al quitar la colisión, en las tres semillas:** 0,609 → 0,599 · 0,648 → 0,602 ·
0,586 → 0,571.

Si la colisión fuera lo que impedía al monitor distinguir aciertos de errores, quitarla tendría que
**subir** el AUC. Bajó en las tres. No es un fallo por margen ni por umbral mal puesto: **el efecto
va en la dirección contraria a la predicha.**

### 3.3 El instrumento andaba

**M-3 pasa 3/3 en los dos brazos** (`pre` 0,982-0,989 · `lat` 0,989-0,993): tapar la entrada que
originó el hecho cambia la respuesta en el 98-99 % de los casos. **M-4 da 1,000 exacto** con `f = 0`.
El modelo lee la evidencia, la perturbación funciona, y aun así el desacuerdo no distingue — igual
que el 20-ago, pero ahora también sin colisión.

### 3.4 Y la aritmética que motivó todo no predice el control

**K-3 falla 2 de 3.** En `pre`, `k` observado (1,274 · 1,588 · 1,298) contra lo que predice su propio
`P_rep` (1,424 · 1,409 · 1,427): las diferencias son 0,150 · 0,179 · 0,129.

Esto importa más que K-1, porque K-3 era la que separaba «la explicación es correcta» de «los números
coincidieron». **La coincidencia original del §2 del prereg —k ≈ 1,34 observado contra 1,42
predicho— era en buena parte casualidad**, y esas ocho centésimas se leyeron como confirmación
cuando el mismo cálculo, aplicado a seis unidades nuevas, falla en dos de tres.

Eso es un error mío de razonamiento, no del experimento: propuse la línea con confianza sobre una
coincidencia numérica de dos informes que no se hablaban, y el test la desarmó en horas. Que sea
barato y rápido descubrirlo es exactamente para lo que estaba el criterio de abandono.

## 4. Lo que queda dicho

- **La colisión de clave contribuye a `k` pero no es la causa del fracaso del monitor.** El
  desacuerdo interno no separa aciertos de errores ni siquiera en un régimen sin colisión.
- **El cierre del 21-ago se refuerza en vez de acotarse.** Aquella lectura decía que las cuatro vías
  separan estados del modelo y no aciertos de errores. Este test agrega que **eso se sostiene incluso
  cuando se elimina la fuente dominante de error**, que era la mejor objeción que le quedaba.
- **`lat` sigue siendo un buen resultado por sus propios méritos** (`err_identidad` 0,0000, la
  bimodalidad disuelta) — pero no compra detección.

## 5. Lo que NO se hace ahora

Por el compromiso del §5, y esto se escribe para que quede constancia: **no se propone acá la
siguiente vía.** El patrón que llevó a cinco fracasos fue encadenar una idea nueva a cada negativo
sin parar. El residuo de `k > 1` en `lat` admite una lectura tentadora —que sea el anclaje espurio de
`nose_rel`, o sea la tesis de recuperación-vs-generación del §5 del plan— y **esa lectura es
post-hoc y no puede usarse para salvar la hipótesis ni para lanzar el experimento siguiente**.

Queda anotada como observación, sin estatus, para cuando la decisión de reabrir sea de Maxi y con su
propio pre-registro.

## 6. Nota de instrumento

`sonda_desacuerdo.py` es del 20-ago, anterior a que existieran `lat` y `lat2`, y llamaba a
`responder_con_abst` **sin `donde`**, quedándose con el default `pre`. Aplicada tal cual a las
unidades `lat` habría medido la arquitectura equivocada — el mismo bug contra el que
`analizar_query_conjunta.py` puso una guarda explícita el 22-ago. Se corrigió antes de correr: ahora
lee `donde` **del checkpoint** y lo imprime en cada línea. Para las unidades `pre` no cambia nada, y
el control lo confirma: reproduce el patrón del 20-ago (AUC 0,586-0,648 contra 0,502-0,669 entonces,
con otras unidades y otro presupuesto).
