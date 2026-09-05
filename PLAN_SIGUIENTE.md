# Para retomar · escrito la noche del 3-sep con todo apagado

## Estado al cierre

**Nada quedó corriendo.** Colab apagado, locks liberados, rotadores muertos, repo pusheado
(48 commits, hasta `deb2d98`).

| | |
|---|---|
| **el correo institucional** | ✅ `mrsperanza@frba.utn.edu.ar`, verificado en uso |
| **Google Scholar** | ✅ verificado, el perfil ya aparece en las búsquedas |
| **OpenReview** | ✅ **adentro**, después de tres rechazos en agosto |
| **arXiv** | ⏳ cuenta creada, **falta el endorsement**. Código `QPVI87` |
| el paper de la ventana | listo, 10 páginas, con Resultado 7 nuevo |
| el experimento en modelo real | **cerrado**, con negativo |

## 1. LO PRIMERO DE MAÑANA · tres cosas que no dependen de nadie

1. **Subir la ventana a Research Square.** Todo resuelto en
   `preprint/ventana/METADATOS_RESEARCH_SQUARE.md`, con el abstract ya en texto plano.
   Inglés principal + castellano de suplemento, un solo registro. Ojo con el 2FA, que expira en
   10 minutos.
2. **Mandar el envío a TMLR.** `tmlr/ventana_tmlr.pdf` + `suplementario_anonimo.zip`, paso a paso en
   `ENVIO_TMLR_PASO_A_PASO.md`. **El PDF que va es el de `tmlr/`, no `ventana_en.pdf`**, que lleva
   nombre y ORCID.
3. **Mandar el tercer pedido de endorsement**, el de la UTN en castellano
   (`endorsement/3_utn_en_espanol.txt`). Es el que más chance tiene y el único que falta: los de
   Oymak y Yang salieron hoy 17:52 y 17:53.

Cuando salga el DOI de Research Square, salen los mails a la facultad
(`endorsement/4_facultad_con_doi.txt` largo, `5_facultad_corto.txt` corto). Van **uno por persona**.

## 2. El atractor, el quinto paper, y lo único que le falta

**No está publicado** — verificado en el panel, la lista tiene tres. Su tabla principal está
verificada a dos tercios:

| columna | estado |
|---|---|
| RECUP | **9 de 9 exactos** contra `micro_lm/ckpts_traza/_recup_8000.json` |
| `a > 0` | **9 de 9 exactos** (campo `abst` del mismo archivo) |
| **AUC(a)** | **0 de 9 rastreados** — la medición no se guardó |

Es el mismo patrón que el trípode: una columna derivada que se computó, se escribió y no se archivó.
**Se puede recalcular**: están los 17 checkpoints en `ckpts_traza/`, 170 MB. Hay que correr el
instrumento sobre ellos y guardar la salida, igual que se hizo hoy con
`compuerta_tripode_20260903.json`.

**Y le falta la versión en castellano**, que los otros cuatro tienen.

## 3. Lo que quedó abierto en la investigación

- **Ligamento E2**, el único frente sin cerrar del proyecto grande.
- **La familia `muylejos` (d=9)** del modelo real: declarada como plan B en la enmienda y no corrida.
  Ahora tiene sentido, porque con d=5 las 24 capas alcanzan a pagar el impuesto y la pregunta es
  dónde dejan de alcanzar.
- **El techo de la evidencia, 0,7003**: cinco lectores y ninguno pasa de ahí. Sin explicación.
- **La constante `q` ≈ 0,50**: mudo, locuaz y medio son la misma patología con distinto valor. Nunca
  se explicó.
- **El Resultado 7 en más corpus.** Hoy dio 0,90-1,00 en cuatro. Sería barato agregar corpus en otros
  idiomas, que es donde la geometría podría cambiar de verdad.

## 4. Las reglas que dejó el día, y son cuatro

1. **Cuando una biblioteca avisa que cayó a un camino lento, leer el mensaje entero.** `mambapy`
   estaba nombrado en el propio error de HuggingFace y se pagaron dos días de lentitud por no leerlo.
2. **Antes de aceptar un efecto techo, verificar que la condición ciega sea ciega de verdad**, por
   intervención y no por aritmética. El «techo» del 2-sep era el montaje.
3. **Una guarda que se imprime pero no filtra es decorativa**, y una guarda estadística hay que
   probarla en el régimen degenerado que pretende cubrir. Las dos fallaron hoy en el mismo juez.
4. **Para el estado operativo de algo —una plataforma, un envío— la fuente es el sistema, no la
   nota.** Hoy fallé dos veces por lo mismo: recomendé TechRxiv estando cerrado, y di el trípode por
   no enviado estando publicado hacía una semana. Las dos las destapó Maxi abriendo la página.

## 5. Operativo

- `/home/maxi/.venv-ligamento/bin/python` para el micro-LM, JAX.
- `/home/maxi/.venv_datasets_pandas/bin/python` para los modelos reales, torch 2.9 y transformers 4.57.
- **`pip install mambapy`** en toda VM nueva: es el scan paralelo, 8,7× en T4, y sin él el
  experimento del modelo real no entra en una sesión de Colab.
- `modelo_real/campana_distancia.sh <cuenta> <pasos> <etiqueta> <cond:sem>...` corre varias unidades
  en una sesión y **baja cada JSON apenas aparece**, porque las sesiones se mueren a los ~60 min.
- Los tres preprints publicados: `rs-10669947` (gemación), `rs-10839567` (trípode),
  `rs-10896018` (sello de orden).

---

## ✅ 2026-09-04 · LA COLUMNA `AUC(a)` DEL ATRACTOR, VERIFICADA 9 DE 9

**Diferencia máxima contra lo publicado: `0.0000`.** La tabla principal del atractor queda verificada
entera: RECUP 9/9, `a > 0` 9/9 y ahora **AUC(a) 9/9**.

**El blanco era el ERROR, no la ausencia**, y no fue una interpretación: el `config` de las nueve
unidades dice `blanco='error'`, y `entrenar.py:290` lo construye como `(lg_arg != tgt)`, o sea «¿el
argmax del modelo, ignorando la cabeza, difiere del target?». Por eso el paper habla de AUC «on their
own target».

Instrumento en `micro_lm/auc_atractor.py`, salida en `micro_lm/auc_atractor_8000.json`, **con el
vector de 8.000 logits por unidad guardado** para no volver a pasar por los checkpoints. Ese archivado
es exactamente lo que faltó la primera vez y dejó la columna sin rastrear. 47 minutos de CPU con
`taskset -c 0-1`.

### ✖ EL LATERAL SE CAYÓ EN LOS CONTROLES, el mismo día

Se había reportado `r = −0,9892` entre la tasa de error y el AUC como posible explicación mecánica de
la degeneración de la cabeza. **No aguanta, y los dos controles son concluyentes.**

**Control 1 · no era información nueva.** `r(RECUP, tasa de error) = −1,0000` **exacto**: la tasa de
error es una transformación algebraica de RECUP,
`err = p_nose + (1 − p_nose)(1 − RECUP)`, con diferencias de ±0,0006 en las nueve unidades. Así que
`r(error, AUC)` **es** `r(RECUP, AUC) = +0,9892`, que la tabla del paper ya mostraba. Se reescribió una
columna al revés.

**Control 2 · el mecanismo propuesto es falso.** Forzando la tasa de error de la unidad buena de
`0.4074` a `0.8201` por submuestreo, sin tocar un solo score, **el AUC se queda en `0.9997` en los tres
casos**. El AUC es invariante a la prevalencia, que es una propiedad conocida de la métrica. Entonces
«el blanco se satura y por eso el AUC baja» **no puede ser cierto**.

**Lo que sí queda vivo:** la hipótesis *dinámica* —blanco saturado → poco gradiente → la cabeza nunca
aprende a discriminar— sigue en pie, pero la correlación **no es evidencia a su favor** y no se decide
midiendo. Hay que **entrenar** con la recuperación fijada y distinta saturación del blanco, y eso va a
Colab.

> **Séptimo veredicto que se da vuelta al correrle el control.** Costó veinte minutos en vez de una
> campaña, que es exactamente para lo que sirve la regla.

**Lo que le queda al atractor para poder enviarse:** la versión en castellano. Los números ya están.

---

## ✅ 2026-09-04 · CONTROL DE ACCESO GLOBAL · la atención completa MATA el corte

Nivel 1 del híbrido, hecho y medido. Instrumento en `micro_lm/control_attn.py`, salida en
`micro_lm/control_attn.json` y `micro_lm/salidas/control_attn.log`.

**Qué se agregó.** Un `donde="attn"` en `modelo.py` que arma la query de lectura con **atención causal
completa** en vez de la conv corta. **Cero parámetros nuevos** (atención por similitud, `q=k=v=x`),
porque con proyecciones propias harían falta 3·D² = 49.152 params, el 5,7 % del modelo, y la
comparación contra `lat2` dejaría de ser a igual tamaño. No se reusaron `wq`/`wk` del bloque, que es
el acoplamiento que `DIAGNOSTICO_CONV_COMPARTIDA_20260822.md` ya diagnosticó como defecto en `lat`.

**Guarda de identidad, verificada.** 865.651 params y 70 hojas antes y después, pesos iniciales
idénticos bit a bit, y los cuatro `donde` existentes dan **0.000e+00** de diferencia contra una copia
del modelo previo.

### El resultado, sobre checkpoints ENTRENADOS

| | `lat2` (conv corta) | `attn` (acceso global) |
|---|---|---|
| `v3_s0`, kernel 3 (alcance 2) | se mueve en d=1,2 · **cero exacto en d=3..6, 0 de 120** | **se mueve en las seis, 120/120** |
| `kq3_s0`, kernel 5 (alcance 4) | se mueve en d=1..4 · **cero exacto en d=5,6, 0 de 120** | **se mueve en las seis, 120/120** |

**Dos cosas quedan medidas.** El corte cae exactamente en el alcance y se corre con el kernel, lo que
replica el Resultado 1 con un instrumento distinto y datos aleatorios en vez del generador del idioma.
Y **la atención completa elimina el corte**, que es la afirmación que se venía sosteniendo por
argumento y no estaba medida en ningún lado.

> **No es un descubrimiento, es un control.** El resultado es el esperado. Su valor es que convierte
> una frase de razonamiento en una medición por intervención, y le contesta por adelantado a un
> revisor la pregunta de si la causa es la ventana o algo más del montaje.

### ⚠ El error de la primera versión, que vale como lección

Se corrió primero **sin entrenar**, razonando que la ley es arquitectónica. Dio cero exacto también en
d=1 y d=2, donde tenía que moverse. La causa es de diseño y está en el propio código: **`convq`
arranca en `[1,0,...,0]`** para que `lat2` contenga a `pre` como caso particular, así que sin entrenar
la conv es la identidad, `lat2` es `pre` y la ventana efectiva es 0.

> **La ventana no es puramente arquitectónica. Existe sólo si el entrenamiento abrió los taps.** En
> `v3_s0` los abrió, con max|peso| por tap `0.718 · 0.223 · 0.469`. Cualquier medición de la ventana
> tiene que hacerse sobre checkpoints entrenados.

### La instrucción de diseño que sale, que es lo que el proyecto persigue

> Si un modelo consulta una memoria desde una capa temprana, esa capa necesita **acceso global** a la
> secuencia. Una convolución corta vuelve invisible una parte de la consulta, el fallo es **silencioso**,
> y se corrige poniendo atención completa exactamente ahí.

Accionable por quien entrena un modelo grande, sin necesidad de reproducir nada de este banco.
