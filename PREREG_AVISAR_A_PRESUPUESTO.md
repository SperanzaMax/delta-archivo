# ¿La cabeza aprende a AVISAR si se le da el presupuesto? · congelado ANTES de correr

**2026-09-01.** Pedido de Maxi, textual: *«tenemos que descubrir la forma de que aprenda la cabeza a
avisar cuando no encuentra la información antes de dar una información equivocada»*.

## 1. Por qué esta combinación y no otra

Tres piezas medidas que **nunca estuvieron juntas**:

1. **`blanco=error` es lo que cierra el aviso.** Las unidades que sobreviven con ese blanco llegan a
   **`nose` 1,0000 con `falsa_abst` 0,0000** (`b3_s0`, `b3_s1`, exactitud global **1,0000**), contra
   **`nose` 0,78-0,83** de las mejores con `blanco=ausencia` (`v3_s0/s1`, `w3_s0/s1`, que también
   tienen `vigente` 1,0000 y `falsa_abst` 0,0000). La diferencia no está en responder ni en callarse
   de más: está **en avisar**.
2. **Su costo es el atractor mudo.** De las 8 unidades `b3`, **4 quedan mudas**. Y el predictor es
   perfecto sobre **76 unidades del banco**: las que en el paso ~2500 emiten **0** respuestas de 512
   terminan mudas **4/4**; las que emiten ≥1 terminan mudas **0/72**. Las `b3` son las únicas del
   banco que arrancan casi mudas (0-8 de 512); todas las demás familias arrancan locuaces (350-512).
3. **`balance` y `ranking` ya demostraron que sacan del atractor:** `INFORME_PERDIDA_CABEZA_20260829.md`,
   **P-1 CUMPLE 4 de 4 mudas emiten respuestas**, con las dos condiciones.

**Lo que faltó fue presupuesto, y el informe del 29 lo dice de sí mismo:** *«P-4 no era decidible a
3000 pasos, y se escribió igual»*. El riesgo que se disparó —salir del silencio **inventando**— se
juzgó con RECUP **0,04-0,22**, o sea con el modelo todavía sin saber nada: ahí **toda** respuesta es
invención por construcción. Sería el **sexto negativo por impaciencia** del proyecto si se dejara así.

Las unidades están **reanudables en disco a 3000 pasos con `horizonte=26000`**, así que extender **no
toca la curva de `lr`**: es la misma corrida, no otra.

## 2. Diseño

**Tratamiento (6):** `rk3_s3` `rk3_s6` `rk3_s7` `rk3_s8` (ranking) · `bl3_s3` `bl3_s6` (balance),
de 3000 → **26000** pasos. Todas con `blanco=error`, `abst=cabeza`, nivel 3, `p_nose` 0,4.

**Control, ya medido y NO se re-corre:** `b3_s3` `b3_s6` `b3_s7` `b3_s8` — las cuatro mudas, con
`vigente` 0,0000 · `falsa_abst` 1,0000 · exactitud **0,4065** = el piso trivial. **0 de 4 útiles.**

**Referencia superior, para saber qué es alcanzable:** `b3_s0`/`b3_s1`, exactitud **1,0000**.

## 3. Criterios

- **R-0 · BLOQUEANTE, no-daño.** Al menos 4 de 6 llegan a 26000 sin quedar en abstención total
  (`falsa_abst` < 0,90). Si el remedio no aguanta el presupuesto, lo demás no se lee.
- **R-1 · PRINCIPAL.** **≥3 de 6 quedan ÚTILES**, y útil se define entero y por adelantado:
  `vigente` ≥ 0,60 **y** `falsa_abst` ≤ 0,10 **y** `nose` ≥ 0,90.
  Las tres juntas son la frase de Maxi: **avisa cuando no encuentra, sin callarse cuando sí sabe, y
  respondiendo bien lo que sabe.** El control da **0 de 4**.
- **R-2 · EL RIESGO DEL 29, AHORA SÍ DECIDIBLE.** En las unidades que cumplan R-1, `invento` ≤ 0,10.
  A 3000 pasos esto no era una pregunta legítima; a 26000 con RECUP alto, sí. **Si R-1 cumple y R-2
  se dispara, el veredicto es el mismo del 29 —se cambió mudez por invención— pero recién ahí estará
  fundado.**
- **R-3 · contra el piso.** Exactitud global `(acierto + acierto_nose)/n` ≥ **0,5565** (el piso
  trivial 0,4065 + 0,15) en ≥3 unidades.
- **Secundario, NO adjudica:** `ranking` contra `balance`. Con 4 y 2 unidades no hay potencia para
  contrastarlos; se informa descriptivo.

**Riesgo de legibilidad declarado, y protege a R-1, R-2 y R-3:** si por disponibilidad de Colab
llegan a 26000 **menos de 4** unidades, se lee sobre las que llegaron **diciéndolo**; con **menos de
3**, el resultado es **NO EVALUABLE** y no se lee ninguno de los tres. (Regla O-6 del 31-ago, tercera
aplicación.)

## 4. Lo que este experimento NO puede decir

- Es **una** tarea (nivel 3), un `p_nose`, una arquitectura.
- Las semillas **no son nuevas**: son las mismas que el control dejó mudas, elegidas justamente por
  eso. Eso hace el contraste fuerte contra la mudez y **no** dice nada sobre la tasa base de éxito en
  semillas frescas.
- **No separa `blanco=error` de la pérdida de cabeza**: las seis llevan las dos cosas. Si funciona,
  cuál de las dos es necesaria queda abierto y necesita su propia corrida.
