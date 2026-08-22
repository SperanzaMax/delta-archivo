# Estado al cierre del 2026-08-21 — para retomar mañana

## Nada quedó corriendo

Verificado: sin procesos locales (rotador, tramos, sondas, watchdogs), y las dos cuentas de Colab que
se usaron —**H** y **K**— dan «No active sessions». Working tree limpio, tres commits pusheados
(`e8fadff`, `e7bc326`, `4a251d0`). PC a 36 °C.

## Lo que cerró hoy, en una línea cada uno

**Tres experimentos cerraron y los tres son negativos con mecanismo identificado.** El único positivo
del día es P-3.

### 1. Campaña de presupuesto — CERRADA 4/4 (`INFORME_PRESUPUESTO_TOKEN.md`)

`t4_s2` extendida a 20000: `falsa_abst` 0,1942 → **0,0458**, pasa.

- P-1 **5/5 y 5/5** (Spearman −0,44 a −0,86) · P-2 **4 de 5 pasan** · P-4 **5/5**, con `vigente`
  subiendo en las cinco (hasta +0,32): no hubo intercambio.
- **P-3 CUMPLE 3 de 3** y es lo que le pone el límite al acotamiento: a igual presupuesto `cabeza`
  sigue por debajo de `token` en las tres semillas (0,0633/0,1296 · 0,0421/0,0713 · 0,0166/0,0458),
  razón 2× a 2,8×.
- **La frase que resume: el presupuesto explica el CRUCE DE LA COMPUERTA, no la VENTAJA.** Un umbral
  convierte una diferencia continua en un sí/no; la diferencia sigue ahí con el mismo signo.
- Lateral: el instrumento es determinista, `c4_s0` dio 0,0633 los dos días bit a bit.

### 2. Empate de clave — CUARTA VÍA CERRADA (`INFORME_EMPATE_CLAVE_20260821.md`)

Prereg SHA `b78b2141`, congelado antes de la sonda.

- **E-1 6/8 CUMPLE · E-2 6/8** (sacar los revisados **sube** el efecto) · **E-3 bloqueante limpio:
  los dos nulos pasan 0/8** · **E-4 3/8 NO CUMPLE** · E-5 0/8 con el signo invertido.
- **E-6 CUMPLE 3/3 y es el mejor control del experimento:** el AUC baja a 20000 (0,6057→0,5467 ·
  0,6416→0,6297 · 0,6412→0,5717), justo donde el round-trip midió que la colisión se disuelve. **No
  prueba que el detector sirva, prueba que mide lo que dice medir.**
- Las 2 celdas donde E-1 falla son `c1_s0` y `c2_s0`, las 2 donde `err_identidad` vale 0,007-0,009.
- **Por qué no alcanza, y es estructural: con dos entradas empatadas el modelo acierta la mitad de
  las veces, así que el empate predice el RIESGO, no el ERROR.**
- Se aplicó el §5: **cierra la línea de detectar la abstención desde una señal interna**. Las cuatro
  vías (logit · densidad · desacuerdo · empate) fallan en el mismo punto: separan estados del modelo,
  no aciertos de errores.

### 3. E-I4c — LA VÍA DEL ENVEJECIMIENTO SE CIERRA ENTERA (`INFORME_EI4C_20260821.md`)

Prereg SHA `8e051d74`. P-1 bloqueante NO CUMPLE (cos 0,8531 contra ≤0,70), P-2 no evaluable por
tercera vez, P-3 pasa por cuatro milésimas y sin efecto. Tres formas de empujar el marco: **0,9374 ·
0,7804 · 0,8531**.

### El hallazgo mecánico del día (`SMOKE_EMPATE_20260821.md`)

Salió de un instrumento mal apuntado. En `modelo.tronco` la lectura se inyecta en el bloque 0 sobre
`h = emb[x]`, **antes** de la conv y del mixer, así que la query es `ln(emb[token]) @ qr`: **función
pura del token de su posición**. → **el modelo no puede formar una query conjunta entidad × relación;
consulta token por token y resuelve la conjunción aguas abajo.** Eso *deriva* el atajo de la relación
del 20-ago en vez de constatarlo, y explica por qué la lectura del archivo es casi uniforme.

## Las dos hipótesis abiertas que dejó hoy, y valen más que los tres cierres

**(a) El régimen de deriva puede no existir.** R6 midió afuera que la deriva catastrófica es del
aprendizaje inicial, no de la vida útil. E-I4, E-I4b y E-I4c serían R6 medido desde adentro tres
veces: un modelo convergido no baja de ~0,78 ni con cambio de distribución. Si es así, **P-2 no quedó
sin contestar: su régimen no ocurre**. No está declarado como resultado —falta descartar tamaño,
tarea y harness— y la prueba que lo separaría está escrita en el informe: si es propiedad del objeto,
tampoco debería alcanzarse en un modelo más grande; si es del harness, ahí sí debería caer.

**(b) El coseno no resume el daño.** A coseno comparable, dos derivas hacen daño distinto: E-I4b a
cos 0,9067 deja las revisadas en 0,9870, E-I4c a cos 0,9021 —algo peor— en **0,9974**. R5.1, R7.1,
E-I4 y E-I4b venían suponiendo lo contrario sin discutirlo.

## Infra

`rotar_abst.sh` pasa al **watchdog v2** (escrito el 20-ago y sin cablear hasta hoy), que identifica al
tramo por parentesco en vez de `pgrep | head -1`. Probado: sale limpio cuando muere su rotador.
**Editado por rename atómico porque el rotador estaba corriendo** — bash relee el script por offset y
una reescritura in-place lo puede romper.

Herramientas nuevas reusables: `micro_lm/analizar_presupuesto.py`, `micro_lm/smoke_empate.py`,
`micro_lm/sonda_empate.py`, `interno/ei4c_distribucion.py`.

## Para arrancar mañana

1. **La decisión de fondo: qué sigue ahora que la abstención sin etiquetas está cerrada por cuatro
   vías.** El mejor resultado de la línea sigue siendo la cabeza del 18-ago, y es **supervisada**.
2. **La política de escritura** — la eviction sorpresa-gated de VIGÍA-03. Es lo único grande que queda
   del brazo interno y nunca se corrió. `modelo.py` declara la política actual («un vector por
   enunciado») y deja explícito que la pregunta de *qué* conviene guardar quedó para después.
3. Banco ECO: lo que falta es decisión de alcance, no técnica. Su §11 ya dejó resuelta la compuerta de
   sujeto (extracción ≥ 0,90) y la brecha 1,000 vs 0,550 con `qwen2.5-coder`.
4. **Dos publicaciones esperando decisión de Maxi**, ninguna la mando yo: el resumen a **UFLO** —
   plazo **1-SEP**, paquete completo en `definitiva/` incluido el deck— y el preprint a Research
   Square.
