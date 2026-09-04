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
