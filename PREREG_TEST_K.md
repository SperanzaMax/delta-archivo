# PRE-REGISTRO · TEST DE k — ¿el monitor de desacuerdo falló por la colisión de clave?

Escrito el 2026-08-24, **antes de correr la sonda sobre `lat`**. Se congela y se hashea. Sale del §4
y del §7 de `PLAN_FOCO_20260824.md`.

Es el primer experimento del foco nuevo, y el que decide si la línea sigue o se cierra por seis
meses.

## 1. La pregunta

El monitor de desacuerdo se cerró el 20-ago con **AUC 0,502-0,669 en 8 de 8 unidades** — azar. Su
propio informe explicó por qué no discriminaba:

> *«La premisa cuantitativa del v2 era que una respuesta anclada en UNA entrada sobrevive en
> `1 − f = 0,75` de las pasadas. La consistencia media observada es 0,65-0,71, por debajo de 0,75 en
> las ocho unidades. Si se la lee como `(1−f)^k`, da **k ≈ 1,3** entradas de las que depende la
> respuesta.»*

Ese informe es **anterior** a que se entendiera la colisión de clave (21-ago) y muy anterior a que se
midiera su magnitud (23-ago). La pregunta de acá es si esas dos cosas son la misma:

> **¿El `k ≈ 1,3` que mató al monitor era la colisión de clave?**

## 2. La aritmética que motiva el test

`diag_relacion` mide la fracción de preguntas con **relación repetida**: `P_rep` = 0,4243 · 0,4087 ·
0,4268, o sea **≈ 0,42**. Si en esas preguntas la respuesta depende de las **2** entradas que empatan
y en el resto de **1**:

```
k esperado = 0,42 × 2 + 0,58 × 1 = 1,42
k observado (monitor, 20-ago)     = 1,34
```

Ocho centésimas de diferencia sobre una predicción hecha con dos informes que no se hablaban. Puede
ser casualidad — por eso esto es un test y no una conclusión.

**Y `lat` permite intervenir en vez de correlacionar:** lleva `ident_rep` de 0,0564 / 0,4683 / 0,2529
a **0,0000 / 0,0000 / 0,0069**. Si la colisión es la causa de `k > 1`, en `lat` tiene que desaparecer.

## 3. Lo que se corre

`sonda_desacuerdo.py`, **la misma del 20-ago, sin tocar**, con sus mismos parámetros declarados:
K = 16 pasadas, `f = 0,25` de las entradas tapadas, 512 muestras por unidad.

| brazo | unidades | qué es |
|---|---|---|
| `lat` | `w3_s0` · `w3_s1` · `w3_s2` (26000) | tratamiento: sin colisión de clave |
| `pre` | `p3_s0` · `p3_s1` · `p3_s2` (26000) | control pareado, mismo día y presupuesto |

**El control `pre` se corre de nuevo y no se compara contra los números del 20-ago.** Aquellas ocho
unidades eran de la familia `c`, a 14000 pasos y con otra configuración; usarlas como control sería
comparar contra otro experimento. Los números del 20-ago entran sólo como **origen de la hipótesis**,
no como brazo.

## 4. Predicciones

- **K-1 · PRINCIPAL.** En `lat`, `k ≤ 1,10` en al menos 2 de 3 semillas. (`k` se despeja de la
  consistencia media como `k = ln(c) / ln(1−f)`.)

- **K-2 · PAREADA.** `k(lat) < k(pre)` en al menos 2 de 3 semillas, semilla contra semilla. Es la que
  descarta que el efecto sea del checkpoint y no de la condición.

- **K-3 · CUANTITATIVA, y es la que hace falsable la explicación.** En `pre`, `k` observado cae
  dentro de **±0,15** de lo que predice su propio `P_rep` medido por `diag_relacion`
  (`k̂ = 1 + P_rep`). No alcanza con que `lat` baje: la aritmética tiene que **acertar el número en el
  control**, o la coincidencia del §2 fue casualidad.

- **K-4 · EL NULO, que se reusa porque ya está validado.** Con `f = 0` la consistencia debe dar
  1,000 exacto en las seis unidades. El 20-ago dio 1,000 en las ocho (M-4).

- **K-5 · LA QUE IMPORTA DE VERDAD, y va declarada aunque sea secundaria.** El AUC del desacuerdo en
  `lat` supera **0,70** en al menos 2 de 3 semillas. Es el criterio M-1 original, que falló 0/8.

## 5. Regla de decisión, comprometida por adelantado

- **K-1 falla** → la hipótesis de la colisión es **falsa**. Por el §8 de `PLAN_FOCO_20260824.md`,
  **la línea se cierra por seis meses**. No se prueba una sexta vía, no se ajusta `f`, no se cambia
  la sonda. Se escribe el negativo y se archiva.

- **K-1 pasa y K-3 falla** → `k` baja pero la aritmética no explica el control. Entonces `lat` mejora
  la consistencia por algún otro motivo y **la explicación de la colisión no se puede afirmar**. Se
  reporta como efecto sin mecanismo, igual que se hizo con W-1/W-2 el 24-ago.

- **K-1 y K-3 pasan y K-5 falla** → la colisión **sí** era el `k > 1`, pero **quitarla no alcanza**
  para que el desacuerdo distinga aciertos de errores. Es un resultado real y acota el cierre del
  21-ago: separaría «el monitor medía la colisión» de «el monitor sirve». La línea sigue, pero hacia
  la señal de correspondencia entidad×relación del §5 del plan, **no** hacia una variante del
  monitor.

- **K-1, K-3 y K-5 pasan** → el monitor de desacuerdo vuelve a estar vivo, y por primera vez hay una
  señal interna que separa aciertos de errores. Es el mejor caso y el menos probable.

## 6. Riesgos declarados

- **Tres semillas y bimodalidad conocida.** Todo se reporta pareado por semilla, nunca por media.
  Es la regla de E-I3c y vale igual acá.
- **`w3_s2` es la semilla rara**: tiene `anterior` en 0,3798 y `ac_unica` 0,8944, o sea no aprendió
  la tarea tan bien como sus hermanas. Si K-1 pasa 2/3 y la que falla es `s2`, se reporta así y no se
  la descarta.
- **`k` es una lectura indirecta.** Sale de un modelo de la consistencia (`(1−f)^k`) que supone que
  las entradas se tapan de forma independiente y que la respuesta cambia si y sólo si se tapa una de
  las que la sostienen. Ese modelo es del prereg del monitor v2 y no se re-litiga acá, pero **es un
  supuesto y no una medición directa**, y por eso K-3 pide que acierte el número en el control.
- **Confusión posible con la calidad del modelo.** `lat` acierta más que `pre` (0,9975 / 1,0000 /
  0,8843 contra 0,9705 / 0,7769 / 0,8351). Un modelo mejor podría tener mayor consistencia por
  razones ajenas a la colisión. K-3 es justamente lo que separa las dos cosas.
