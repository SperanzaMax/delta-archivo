# PREREG · El monitor de desacuerdo interno · v2

Congelado el 2026-08-20 antes de correr. Reemplaza a `PREREG_MONITOR_DESACUERDO.md` (SHA
`b259fd0d…`), **anulado por instrumento vacío** — ver D-1 de `DESVIACIONES_MONITOR.md`: permutar el
archivo no cambia la salida porque la atención es equivariante, así que el estadístico valía 1 por
identidad algebraica.

El §1 del v1 se mantiene íntegro y es la razón de todo esto: **al logit le falta escala absoluta**
—`a = 0,3` no significa nada sin etiquetas, y por eso el corte del 20-ago necesitaba calibrarse—
mientras que **«las K pasadas coincidieron» es una afirmación con sentido sin ninguna calibración**.

---

## 1 · La perturbación, ahora una que sí perturba

**Cada una de las K = 16 pasadas ve el archivo con una fracción `f = 0,25` de sus entradas vivas
tapadas** (`mask` a 0), sorteada de forma independiente por pasada y por muestra.

Rompe la equivarianza porque **cambia el conjunto**, no el orden: quitar entradas distintas da
lecturas distintas, y ninguna identidad algebraica lo anula.

**Y tiene una predicción cuantitativa a priori, que es lo que la hace útil.** Si una respuesta está
anclada en **una** entrada del archivo, sobrevive exactamente cuando esa entrada no fue tapada, o sea
en una fracción **1 − f = 0,75** de las pasadas. Si la respuesta no está anclada en ninguna entrada
en particular, no hay razón para que se repita.

## 2 · El estadístico y el corte

`consistencia` = fracción de las K pasadas que coinciden con la respuesta modal (argmax de
vocabulario con `NOSE` excluido, como en toda la campaña).

**Corte estructural: abstenerse si `consistencia < 1 − f`, o sea < 0,75.** `f` es un parámetro del
**método**, elegido a priori y anunciado acá, **no un umbral ajustado contra etiquetas** — que es la
diferencia con `a*`, y es toda la cuestión. No se prueban otros valores de `f` en este experimento.

## 3 · Predicciones

Sobre las 8 unidades de la familia `c` a 14000 pasos, rng de prueba 77000+semilla, **reportando por
unidad**:

- **M-1 (¿discrimina?).** AUC del desacuerdo `1 − consistencia` separando preguntas sin respuesta de
  preguntas con respuesta **≥ 0,70 en ≥ 6 de 8**.
- **M-2 (la principal).** El corte estructural `consistencia < 0,75` pasa la compuerta
  (`falsa_abst ≤ 0,10` y `nose ≥ 0,50`) en **≥ 6 de 8**. Referencias del mismo día sobre las mismas
  unidades: **U-1 = 2/8 · σ>0,5 = 6/8 · U-2 = 7/8**. Para que esto valga como avance tiene que llegar
  a 6/8 **sin ajustar nada**.
- **M-3 (control fuerte, y puede fallar).** Usando `con_origen=True`, se tapan **exactamente** las
  entradas del archivo que originaron el hecho preguntado. La respuesta tiene que cambiar en **≥ 0,50**
  de las preguntas que sí tenían respuesta y que el modelo acertaba. Si no cambia, el modelo no está
  leyendo esas entradas y nada de M-1/M-2 se puede interpretar como anclaje en evidencia.
- **M-4 (nulo).** Con `f = 0` la consistencia debe dar **1,000 exacto** y la AUC **0,500**. Verifica
  que todo el efecto viene de la perturbación.

## 4 · Desenlaces, comprometidos por adelantado

- **M-1, M-2 y M-3 cumplen** → el «no sé» tiene un correlato mecánico y además da el corte sin
  etiquetas que quedó abierto hoy. Es lo que pedía el §3 del plan.
- **M-1 cumple y M-2 no** → la señal existe, el corte estructural no alcanza: sirve como evidencia,
  no como decisión, y se dice con esas palabras.
- **M-1 falla** → el desacuerdo bajo sub-muestreo no distingue. **Se cierra la línea del monitor por
  esta vía y no se prueba una tercera perturbación.** Esta vez la regla sí aplica: sería un resultado,
  no un instrumento vacío.
- **M-3 falla** → se archiva sin interpretar, cualquiera sean M-1 y M-2.

## 5 · Lo que no puede decir

- Monitor **de inferencia** sobre modelos ya entrenados; no dice nada sobre entrenar contra el
  desacuerdo.
- **Tapar entradas cambia el problema, no sólo la evidencia**: el modelo ve un archivo más chico y eso
  puede correrle el comportamiento global. Es el precio de romper la equivarianza y queda declarado
  como limitación, no descubierto después.
- 16 pasadas cuestan 16× la inferencia. Se mide si la señal existe, no si el costo se justifica.
- Idioma cerrado de 242 tokens, `p_nose = 0,4`.

## 6 · Procedimiento

CPU, cero GPU, sobre los checkpoints existentes. **Smoke de una unidad antes de las ocho** — es lo
que cazó el error del v1 por 64 muestras. Y ninguna unidad que este experimento lea puede estar
entrenándose: no se corre sobre `c4_s0` ni `c4_s1` hasta que la réplica cierre.
