# PRE-REGISTRO · EL SLOT NULO — la abstención como entrada de la memoria

Escrito el 2026-08-24 **después del chequeo de instrumento y antes de entrenar una sola unidad**. Se
congela y se hashea. Diseño en `DISENO_ATRIBUCION.md`.

## 1. La pregunta

Maxi decidió que **la supervisión vale**. Entonces la pregunta no es si el modelo puede descubrir
solo que no sabe, sino:

> **¿La abstención se aprende mejor cuando vive en la MEMORIA que cuando vive en el vocabulario o en
> una cabeza de salida?**

Es una pregunta de **dónde**, no de **si**. Y completa un trípode que hoy tiene dos patas medidas:
`token` (= `[IDK]`), `cabeza` (= SelectiveNet) y **`slot`** (= pointer sentinel dentro del archivo).

## 2. El chequeo de instrumento, CORRIDO antes de escribir esto

`chequeo_slot.py`. CPU, segundos, sin entrenar nada.

| | qué afirma | medido | |
|---|---|---|---|
| **A-1** | el slot compite y no se come todo | masa 0,0920 contra 1/41 = 0,0244 | **CUMPLE** |
| **A-2** | con `abst ≠ slot` nada cambia | logits `0,000e+00` · cabeza `0,000e+00` | **CUMPLE** |
| **A-4** | el gradiente llega a `k_nulo` | `1,105e-01` | **CUMPLE** |

A-2 es la que protege los controles ya corridos (`token`, `escala`, `cabeza`): el árbol crece en 256
parámetros pero las otras condiciones dan **exactamente** lo mismo que antes.

### 2.1 A-3 dio positivo y su control lo dio vuelta

**A-3** medía si la masa del nulo sube al tapar del archivo la entrada del hecho preguntado, sobre un
checkpoint entrenado (`p3_s0`). **Subió: +0,02779.** Con la lectura declarada de antemano, eso era
«una sorpresa fuerte».

**A-3b, el control, la desarmó.** Al tapar *cualquier* entrada su masa se reparte entre las que
quedan —el slot incluido—, así que el nulo sube **por construcción del softmax**. Tapando una entrada
**irrelevante**:

| | Δ masa del nulo |
|---|---:|
| tapando la entrada **del hecho** | **+0,02779** |
| tapando una **irrelevante** (control) | **+0,03128** |
| **efecto específico** | **−0,00350** |

**El efecto de A-3 era redistribución, no detección.** Sin ese control se habría anotado un falso
positivo en este documento.

**Consecuencia, y por eso el control se corrió antes:** en un modelo que no lo entrenó, el slot **no
detecta ausencia**. El mecanismo depende **enteramente de la supervisión**, y las predicciones de
abajo están escritas sabiendo eso. Nada acá espera que el slot «descubra» nada.

## 3. Lo que se corre

Condición `slot`: `--abst slot`. Todo lo demás **idéntico** al control: nivel 3, `d=128`, `capas=4`,
`batch=64`, `lr=1e-3`, `p_vieja=0.35`, `p_nose=0.4`, `donde=pre`, `idioma=2`, **26000 pasos con
horizonte 26000**, sin siembra, semillas 0/1/2. Familia `n3_s*`.

**El control se REUSA:** `p3_s0/s1/s2` son exactamente `pre` + `--abst cabeza` a 26000 pasos, del
mismo día y el mismo generador. Con A-2 verificado, el contraste es pareado de verdad. **La campaña
cuesta 3 unidades.**

Parámetros: `slot` estrena 256 (0,030 %), `cabeza` estrena 129 (0,015 %). **La asimetría se declara
acá**, y si `slot` gana, «ganó porque tiene más parámetros» es una objeción que hay que poder
contestar — el brazo con `d` reducido se decide **sólo si hace falta**, no ahora.

## 4. Predicciones

Instrumentos ya usados: `ser.py` (n=2048, semilla 54321) y `score_archivo.py`.

- **S-0 · BLOQUEANTE, y va primero.** `slot` **aprende la tarea**: acierto ≥ 0,70 en al menos 2 de 3
  semillas. Es la compuerta que `post` no pasó. Hay una razón concreta para temerla acá: el slot
  compite por masa con las entradas reales, y el 16-ago quedó medido que **el modelo acierta
  integrando varias entradas, no seleccionando una**. Si el nulo le roba masa a esa integración, la
  tarea se puede romper aunque la abstención mejore.

- **S-1 · PRINCIPAL.** `slot` pasa la **compuerta de abstención** —`nose` ≥ 0,50 y `falsa_abst` ≤
  0,10— en al menos 2 de 3 semillas. Es la misma compuerta que `cabeza` pasa en 4 de 5 unidades y que
  `token` y `escala` fallan en 5 de 5.

- **S-2 · MECANICISTA, y es la que hace valer todo lo demás.** El **score del archivo** sube del
  azar: AUC de `s_max` para separar «hay respuesta» de «no hay respuesta` **≥ 0,60** en al menos 2 de
  3 semillas. La línea base es **0,4984 / 0,5022**, medida el 16-ago sobre modelos sin slot.
  Si S-1 pasa y S-2 falla, la abstención mejoró **sin** que la memoria haya aprendido pertenencia, y
  entonces el slot es una cabeza con otro nombre.

- **S-3 · PAREADA contra `cabeza`.** `slot` no es peor que su gemela `pre`+`cabeza` en la compuerta:
  `nose` no cae más de 0,05 y `falsa_abst` no sube más de 0,05, semilla contra semilla.

- **S-4 · NO-INTERCAMBIO, completo.** `vigente` no cae más de 0,02 respecto de su gemela, y se
  reportan **`anterior` y `nose_rel` explícitamente**. Van dentro del criterio y no como observación:
  es la lección del hueco de W-4, que se declaró el 22-ago, se confirmó el 24 y **no se repite tres
  veces**.

## 5. Regla de decisión, comprometida por adelantado

- **S-0 falla** → el slot rompe la tarea. Se archiva sin interpretar y se declara que en esta
  arquitectura **la abstención no puede vivir dentro de la memoria sin costarle la lectura**. Sería
  un resultado, y encajaría con lo medido el 16-ago sobre integración.
- **S-1 falla con S-0 pasando** → la memoria **no** es mejor lugar que la cabeza. El trípode queda
  cerrado con `cabeza` ganando, que ya es publicable, y **esta línea no se reintenta con una cuarta
  forma de slot**.
- **S-1 pasa y S-2 falla** → mejora sin mecanismo. Se reporta así, sin adjudicarle a la memoria algo
  que no se midió. Igual que se hizo con W-1/W-2 el 24-ago.
- **S-1 y S-2 pasan** → es el mejor caso: la abstención se aprende mejor en la memoria **y** el
  archivo aprendió pertenencia. Recién ahí tiene sentido la atribución positiva (supervisar a qué
  entrada apunta), que hoy **no** se toca.

## 6. Riesgos declarados

- **El slot le roba masa a la integración.** Es el riesgo de S-0 y está fundado en medición propia,
  no en intuición.
- **Tres semillas y bimodalidad conocida.** Todo pareado por semilla, nunca por media.
- **La asimetría de parámetros** (256 contra 129), ya declarada en el §3.
- **Los dos gradientes de `v_nulo`, que no son el mismo.** A-4 midió `|grad v_nulo| = 0`, pero A-4
  deriva **sólo la BCE de abstención**, y por ahí `v_nulo` efectivamente no entra: la decisión sale
  de la *masa* del slot, no de su valor. En el entrenamiento real la pérdida también tiene la CE del
  valor, y por ahí sí entra — verificado en el smoke de 20 pasos: `v_nulo` se movió **0,0064** desde
  su cero inicial.

  La distinción se reporta al cerrar y **no es criterio**: si `v_nulo` queda chico, el slot funcionó
  como **compuerta pura** (restarle masa a la lectura); si crece, el modelo lo usa además como un
  valor leído, o sea aprendió a representar «nada» y no sólo a descontarlo.
- **`p_nose = 0,4` es supervisión densa.** No se está midiendo si el modelo sabe que no sabe: se está
  midiendo dónde conviene ponerle la decisión. El §8 del diseño ya lo dice y se repite acá para que
  no se lea de más en el informe.
