# El día en que las dos paredes cayeron y el efecto conductual no apareció

**2026-09-03.** El plan de anoche pedía destrabar el experimento en modelo real. Se destrabó, y lo
que había del otro lado no es lo que se esperaba. Éste es el cierre.

## 1. Las dos paredes, y ninguna cayó por donde estaba escrito

| pared | remedio previsto | lo que pasó |
|---|---|---|
| el paso cuesta 9,7 s en T4 | compilar `mamba-ssm`, 15-30 min y puede fallar | **`mambapy`**, pip puro, 40 kB, **8,7× en T4** y 9,6× en CPU |
| con 4 hechos la tarea satura | subir a 16 hechos, ~30 h de GPU | **no era techo: el montaje no tenía condición ciega** |

El backend estaba nombrado en el propio mensaje de error que se venía imprimiendo hace dos días
(*«as use_mambapy is set to False»*). Se verificó **equivalente antes de mirar la velocidad**: logits
3,3e−6 relativo en CPU y 3,5e−6 en T4, gradientes 9e−6. Es ruido de fp32.

Y la saturación tenía causa: el prereg contaba la distancia de cada pieza **al signo de pregunta**,
que es la correcta en el micro-LM porque ahí la consulta se forma en el último token y lee un archivo
**externo**. En Mamba la memoria es el estado, se actualiza en cada posición y cada token condiciona
la búsqueda en la suya, así que la distancia que decide es la que separa **la relación de la
entidad**. Verificado por intervención, no derivado: `conv1d` en la posición de la entidad, capa 0,
al cambiar la relación → `directa` 4,77e−01 · `lejana` 4,77e−01 · `d5` **0,0 exacto**.

**Las tres formas del montaje anterior eran no-ciegas (d=2) o ciegas por causalidad.** Subir a 16
hechos habría gastado ~30 h de GPU midiendo un contraste que no existía.

## 2. Lo mecanicista, que es lo que queda firme

`escalera_v2.py`: 15 formas, 6 distancias, 8 contextos, todas las capas, dos modelos.

| d(rel↔ent) | capa 0 | atenuación capa 1, `130m` | atenuación capa 1, `370m` |
|---:|---|---:|---:|
| 2 | se mueve | 1,10 | 1,04 |
| 3 | **0,0 exacto** | 2,31 | 3,21 |
| 4 | **0,0 exacto** | 3,06 | 3,17 |
| 5 | **0,0 exacto** | 4,64 | 4,47 |
| 6 | **0,0 exacto** | 4,65 | 5,72 |
| 7 | **0,0 exacto** | 6,92 | 6,47 |

Pendientes **1,077** y **1,028** por token, r = 0,978 en los dos, correlación entre curvas **0,9549**.
Tres veces los parámetros y el doble de capas **no cambian la tasa**.

> **En un modelo profundo la ventana no BLOQUEA: ATENÚA. La forma dura vale por capa.**

**El control que adjudica:** con d=5 y cuatro rellenos distintos la dispersión es ×1,98 contra un
rango de ×1,10 a ×6,92 entre distancias. Manda la distancia, no las palabras del medio.

## 3. Lo conductual, y es NEGATIVO

`PREREG_DISTANCIA_REAL.md` (SHA `19a9b8a1`) + `ENMIENDA_DISTANCIA_REAL.md` (SHA `ddb4f4b9`), los dos
congelados antes. Doce unidades, `mamba-130m`, 800 pasos, 512 ejemplos por evaluación.

| condición | `nose_rel` final (3 semillas) | AUC de la curva |
|---|---|---|
| `cerca` (d2) | 0,8571 · 1,0000 · 0,9882 | 0,9583 · 0,9947 · 0,9926 |
| `lejos` (d5) | 0,9881 · 1,0000 · 0,9882 | 0,9360 · 0,9548 · 0,9721 |
| `lejos_dos` (d5+d2) | 1,0000 · 1,0000 · 1,0000 | 0,9494 · 0,9641 · 0,9691 |

- **G-1 · NO EVALUABLE por techo.** Las tres semillas de `lejos` terminan en ≥0,95, así que no queda
  margen para que `cerca` sea mejor. La guarda G-L estaba declarada en el prereg y se disparó.
- **G-1v · NO CUMPLE, 0 de 3.** Pero **los tres signos son positivos**: +0,0223 · +0,0399 · +0,0206.
  El efecto de velocidad existe y es de 2-4 puntos, contra los 10 que pedía el criterio.
- **G-2 y G-2v · NO CUMPLE.** La diversidad de formas no compra nada acá (+0,013 · +0,009 · −0,003).
- **G-4 · CUMPLE.** Sin daño: `falsa_abst` 0,0060 · 0,0063 · 0,0031.

> **En un Mamba de 24 capas con presupuesto suficiente, la distancia relación-entidad NO produce
> falla conductual apreciable. La profundidad paga el impuesto, que es exactamente lo que la §2
> predecía.**

Es un negativo limpio y **acota la ley** en vez de tumbarla: vale para **acceso**, medido y exacto, y
no se traduce en comportamiento cuando hay capas de sobra con que pagar.

## 4. ⚠ QUINTO veredicto automático mal leído, y esta vez el defecto es de clase nueva

El juez imprimió **«G-1 NO CUMPLE 0/3»** sobre semillas que **su propia guarda G-L** acababa de
declarar no evaluables. La guarda estaba escrita, calculada e impresa — **y no conectada al
criterio**. Los cuatro casos anteriores fueron criterios escritos sobre la métrica equivocada; éste
es distinto: la guarda era correcta y sólo faltaba **aplicarla**.

**Regla que deja:** una guarda que se imprime pero no filtra es decorativa. Al escribir un juez, cada
guarda tiene que **modificar** el conjunto de datos sobre el que se calcula el criterio, no
acompañarlo con una advertencia.

## 5. El micro-LM cerró la tabla

`cl3_s2` llegó a 26000 pasos, así que el control ciego queda **3 de 3** y el resultado del 2-sep pasa
de NO EVALUABLE a evaluable.

| condición | la relación entra en la ventana | `nose_rel` |
|---|---|---|
| `cl3` · directa + lejana | **nunca** | 0,5440 · 0,6777 · **0,5064** |
| `cf3` · directa + invertida | a veces | 0,9934 · 0,9799 · 1,0000 |

**Sin un solo solape.** Y `k73` (kernel 7) también está 3 de 3 —1,0000 · 1,0000 · 0,9761—, solapado
con kernel 5: **7 no agrega nada sobre 5**.

## 6. Dos controles que se cayeron, y se informan

1. **El «control de largo» era causalidad disfrazada de control.** Alargar la pregunta *después* de
   la entidad dio idéntico a la base **hasta el último decimal en las 24 capas**, y tenía que darlo.
   La prueba que caza esta clase: *si la hipótesis fuera falsa, ¿este control podría dar distinto?*
2. **El titular «~4 capas por token» (r = 0,971) no aguanta el barrido de umbral**: de 1,2 a 2,5 la
   pendiente va de 3,97 a 1,54 y el orden se rompe. Descartado antes de publicarlo.

## 6b. Cierre de `lejos_relleno`, y G-3 tampoco adjudica

Las dos semillas cortadas se completaron. `lejos_relleno` da 0,9643 · 0,9894 · 1,0000, y **G-3 no
adjudica en ninguna de las tres**: la diferencia entre «se parece a `lejos_dos`» y «se parece a
`lejos`» es de 0,0000 a 0,0119, contra un error típico de 0,0216 por celda. Con las tres condiciones
saturadas, adjudicar ahí es aritmética sobre ruido.

**Y hubo un sexto defecto, en la guarda que escribí para cazar esto:** la primera versión calculaba
el error binomial con la proporción observada, que vale exactamente 1,0000 en varias celdas, así que
la varianza daba **cero** y la guarda no filtraba nunca — justo en el caso donde más falta hacía. Se
acota la proporción antes de calcular el error. **Regla: una guarda estadística hay que probarla en
el régimen degenerado que pretende cubrir, no sólo en el caso típico.**

## 7. Lo que queda abierto
- **El paper de la ventana** ya tiene el Resultado 5 con lo mecanicista; falta agregarle **este
  negativo conductual**, que es lo que lo vuelve honesto.
- **La familia `muylejos` (d=9) no se corrió**, y ahora tiene sentido correrla: si a d=5 las 24 capas
  alcanzan a pagar el impuesto, la pregunta es dónde dejan de alcanzar. Está declarado en la enmienda
  como plan B y **se diría que se corrió después de ver la campaña**.
