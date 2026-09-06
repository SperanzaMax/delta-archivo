# PRE-REGISTRO · sello RELATIVO + bit de pertenencia · 2026-09-06

Se congela antes de gastar una sola VM. Código ya implementado y con su compuerta pasada
(`micro_lm/chequeo_sello_rel.py`); lo que falta es el presupuesto de T4, que decide Maxi.

## 1. De dónde sale

El 6-sep quedó medido, **por conducta y por pesos**, que el sello de orden es una bandera:

- Conducta (`aabb4b20`): con el orden relativo intacto y el episodio corrido dos lugares, la RECUP
  cae de 1,0000 a 0,50-0,57 del lado que cruza el umbral y el acierto de 1,00 a **0,02-0,08**.
- Pesos (`bedac0b5`): `ord` termina con **una** marca de «ajeno» copiada en las 24 filas del bloque
  bajo (coseno 0,86-0,90, norma doble, dirección opuesta), ausente en los controles entrenados con
  archivo corto.
- Y al descartar la alternativa apareció que **`ord` sostiene dos funciones entrelazadas**: el bit de
  pertenencia (escalón, r 0,86-0,88 contra el indicador de bloque) y un gradiente de recencia
  *dentro* del episodio (r −0,68 a −0,70), que es el sello con DOI `rs-10896018`.

Por eso el tope de 64 filas no rompe una cosa: rompe las dos a la vez. Y por eso agrandar la tabla
no era el arreglo.

## 2. Qué se cambia, y por qué no puede ser peor por construcción

- **`--sello rel`**: la entrada se sella con `turno_actual − turno_entrada` en vez del turno
  absoluto. `turno_actual` se deriva del propio archivo (el siguiente al último escrito). Ninguna
  fila puede significar «ajeno», porque el mismo turno absoluto cae en filas distintas según cuándo
  se pregunte. Las distancias mayores que la tabla **saturan** en la última fila —«más viejo que la
  ventana»—, decisión declarada y no el clamp silencioso que fue un bug hasta el 5-sep.
  **Consecuencia: el tope de 64 deja de ser un techo del archivo y pasa a ser una ventana de
  recencia de 64 turnos.**
- **`--pert`**: un vector aprendido que se suma a la clave de las entradas de la conversación en
  curso. Hoy el modelo infiere eso de en qué fila cayó la entrada; dárselo explícito separa «de
  quién es» de «cuándo fue».

`pert` **arranca en cero exacto** (128 params sobre 866.803 = 0,0148 %), con el mismo criterio que
`convq` en [1,0,0] y `v_nulo` en cero: la condición nueva **contiene a la vieja como caso
particular**, así que no puede ser estructuralmente peor y todo lo que aparezca lo fue a buscar el
gradiente.

**Compuerta ya pasada** (`chequeo_sello_rel.py`, cuatro comprobaciones que podían fallar):
`escribir`, `responder` y `responder_con_abst` dan **0,000000e+00** contra el `modelo.py` de HEAD con
la configuración de siempre; `init_params` idéntico fuera de `pert`; `pert` en cero exacto; y el
sello relativo es **invariante al corrimiento (0,0 exacto)** mientras el absoluto se mueve 2,49.
Guarda de identidad ampliada: reanudar un checkpoint `rel` sin el flag **aborta**, verificado.

## 3. Diseño

Seis unidades nuevas, 2000 pasos, sembradas desde `kq3_s0/s1/s2` igual que la campaña del archivo
largo, con `--ses-extra 26` y `--d 128 --capas 4 --kernel-q 5 --donde lat2`:

| condición | flags | qué aísla |
|---|---|---|
| `rp3_s0/s1/s2` | `--sello rel --pert` | **la principal**: las dos piezas juntas |
| `rr3_s0/s1/s2` | `--sello rel` | cuánto aporta el sello relativo **solo** |
| `lg3_s0/s1/s2` | (ya entrenadas) | el control `abs`, sin gastar GPU |

El control ya está corrido, así que la campaña son **seis unidades**, no nueve.

## 4. Criterios, comprometidos por adelantado

- **R-0 (bloqueante).** Las tres `rp3` alcanzan en su propio régimen una exactitud ≥ 0,90 con archivo
  largo. Si el cambio rompe lo que ya funcionaba, no hay nada que comparar y se para.
- **R-1 (principal).** En `reloj_o_bandera.py`, la condición `corrido` deja de romperse: |ΔRECUP|
  respecto de `real` ≤ 0,05 **y** |Δacierto| ≤ 0,10, en las tres semillas. Hoy con `abs` la RECUP
  cae 1,0000 → 0,64-0,78 y el acierto 1,00 → 0,02-0,08.
- **R-2 (no romper).** La exactitud de `rp3` en el régimen de entrenamiento no queda más de 0,05 por
  debajo de la de `lg3` (0,9872-1,0000).
- **R-3 (geometría).** En `geometria_ord.py`, la brecha G-1 cae a ≤ 0,05 (hoy 0,216-0,248) **y** el
  gradiente intra-episodio se conserva, |r| ≥ 0,40 (hoy −0,68 a −0,70). Es la predicción más
  específica: **perder el escalón conservando la rampa**.
- **R-4 (el bit se usa).** ‖`pert`‖ termina por encima de 0,1, y ponerlo a cero en evaluación baja la
  exactitud al menos 0,05. Si `pert` queda en cero, el bit no hizo falta y lo que arregla es el sello
  solo — que es justo lo que `rr3` mide.
- **R-5 (el que decide si sirve para el objetivo).** Con un archivo que necesita **más de 64 turnos**,
  `rel` se sostiene y `abs` colapsa. Requiere extender el generador más allá de `TURNO_BASE = 24`,
  que **hoy no está hecho**: es la Fase 2 y sólo se corre si R-1 y R-3 cumplen.

## 5. Qué se hace con cada resultado

- **R-1 y R-3 cumplen:** el sello deja de ser una bandera y el tope de 64 se convierte en ventana de
  recencia. Ahí sí tiene sentido la Fase 2 (archivo con más de 64 turnos) y con ella el régimen de
  3280 entradas, que sigue abierto desde anoche.
- **R-1 cumple y R-3 no:** funciona por otra razón que la declarada. Se reporta así y se investiga
  antes de escribir nada.
- **R-0 falla:** el cambio rompe lo que andaba. Se reporta el negativo con su curva y no se insiste
  con una tercera variante en esta campaña.

## 6. Presupuesto y límites

- Seis unidades × 2000 pasos, con micro-lotes por el OOM ya conocido (`--micro-batch 8`,
  `--batch-eval 8`). La campaña del 5-sep con seis unidades salió ~5 h de T4; ésta debería ser
  comparable, **pero es una estimación y no una medición**.
- No dice nada sobre 3280 entradas ni sobre modelos que no sean este micro-LM.
- `pert` se deriva de la construcción del lote (las primeras `n_sesiones × E_MAX` entradas son del
  episodio). En este banco eso coincide con la posición en el tensor, así que **la campaña no puede
  distinguir «usa el bit» de «usa la posición»**. El control que lo separa es barajar el archivo, y
  queda declarado como pendiente antes de sacar conclusiones fuertes sobre `pert`. `rr3` —sello
  relativo sin bit— no tiene ese problema.
