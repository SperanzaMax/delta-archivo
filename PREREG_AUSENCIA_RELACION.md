# PRE-REGISTRO · la ausencia de la RELACIÓN, controlada por entidad

**Escrito el 2026-08-27**, con A5 todavía corriendo y sin haber mirado un solo número de esta
pregunta. La Fase 0 no toca GPU y puede correr apenas se congele esto; la condición, si abre, espera
a que A5 libere las cuentas.

---

## 1. De dónde sale

De la conversación de hoy con Maxi sobre dónde queda margen. El recorrido, en orden:

1. `err_identidad` —contestar el valor de otra entidad— **ya está en 0,0000** en las tres semillas
   desde `lat2` (`INFORME_LAT2_20260825.md`). No hay nada que arreglar ahí.
2. `lat2` compró eso **pagando con la relación**: `nose_rel` cayó de 0,9235 a 0,5842 en s0, y su V-3
   falló 1/3 por ese motivo. El informe lo leyó así: *con la relación ausente, media query sigue
   coincidiendo con una entrada real y el modelo se ancla ahí.*
3. El trípode cerró lo más general: **en una memoria co-entrenada leída por softmax, la ausencia no
   tiene representación.** El slot nulo convergió al prior (0,4074 / 0,4046 / 0,4020 contra tasa base
   0,4048) en vez de aprender pertenencia.

Las tres cosas apuntan al mismo agujero, y es el único grande que queda abierto en la línea.

---

## 2. Por qué el negativo del 16-ago NO cierra esta pregunta

`INFORME_SCORE_ARCHIVO_20260816.md` midió el score del archivo y dio **azar exacto**: `s_max` con AUC
0,4984 y 0,5022. Parecería que ya está contestado. **No lo está, y el motivo es preciso.**

Ese eje era **grueso**: positivos = «con respuesta en el archivo», negativos = «sin respuesta». Y
«sin respuesta» mezcla **dos poblaciones que no son la misma cosa**, según `idioma.py:222-223`:

- `nose_ent` — la entidad **nunca se nombró**. Basta con no encontrarla.
- `nose_rel` — la entidad **sí está**, con otra relación. Hay que encontrar la entidad y además
  darse cuenta de que la relación pedida no es la suya.

Al mezclarlas, el eje no controla por entidad: los `nose_ent` no tienen entrada que matchear y los
`nose_rel` sí. **Dos poblaciones con mecanismos distintos promediadas en un solo AUC pueden dar 0,50
aunque una de las dos tenga señal.**

**Esta Fase 0 controla por entidad**: positivos y negativos tienen los dos la entidad presente en el
archivo. Es el mismo instrumento sobre un eje que nunca se midió.

**Y se dice por adelantado lo incómodo:** puede volver a dar azar, y en ese caso el negativo del
16-ago se refuerza en vez de matizarse. Eso también sería un resultado.

---

## 3. FASE 0 · sin GPU, sobre checkpoints ya entrenados

**El corte, que el código ya expone.** `datos.py:42` define `TIPOS = {"vigente": 0, "anterior": 1,
"nose_ent": 2, "nose_rel": 3}`, y `tipo` viaja en el lote. Entonces:

- **Positivos**: `tipo == 0` (vigente). La entidad está y la relación pedida está.
- **Negativos**: `tipo == 3` (nose_rel). La entidad está y la relación pedida **no**.

`tipo == 2` (`nose_ent`) **se excluye**, y ésa es toda la diferencia con el 16-ago.

**Señales sondeadas, fijadas acá para no ir a pescar después.** Todas del momento de la consulta,
sobre `modelo.responder`:

| señal | qué es |
|---|---|
| `s_max` | máximo del score de similitud contra las claves, antes del softmax |
| `s_margen` | máximo menos segundo, que mide si hay un ganador claro |
| `s_ent` | entropía del softmax de lectura, o sea qué tan repartida quedó la atención |
| `leido` | el vector leído del archivo (D=128) |
| `estado` | el estado final antes de la cabeza |

Se reporta cada una **por separado** y el bloque completo. Nada de elegir la mejor y presentarla
sola: la que decide E-1 es la del bloque completo, declarada acá.

**Unidades**: `v3_s0/s1/s2` (`lat2`, 26000), que son las que tienen el problema. Se añade `p3_s0`
como referencia, porque su `nose_rel` es 0,9235 y sirve de contraste alto.

**Instrumento**: `sonda()` y `auc()` de `sonda_dos_detectores.py`, con su chequeo corrido antes de
leer nada, igual que en la Fase 0 de `escriba`.

### Predicciones de la Fase 0

- **R-0 · BLOQUEANTE.** Etiquetas permutadas en **AUC ≤ 0,55**. Si falla, se arregla el instrumento y
  no se lee nada más.
- **R-1 · LA QUE DECIDE.** El bloque completo alcanza **AUC ≥ 0,65** en al menos 2 de 3 semillas de
  `v3_*`. Mismo umbral que en `escriba`, y por la misma razón: las siete vías del
  `PLAN_FOCO_20260824.md` aterrizaron entre 0,50 y 0,67, y algo dentro de esa banda sería la misma
  nada con otro nombre.
- **R-2 · DESCRIPTIVA, sin criterio de éxito.** Se reporta el AUC de cada señal por separado. Sirve
  para saber **dónde** vive la señal si existe, y no para decidir si existe.

---

## 4. La condición, si y sólo si la Fase 0 abre

Una **cabeza de relación**: salida binaria propia que aprende «¿la relación pedida está en el
archivo?», con etiqueta supervisada del generador (`tipo == 3` contra `tipo == 0`), entrenada junto
al modelo y **sin modificar la lectura**.

Es el mismo movimiento que ganó en el trípode —sacar una decisión del softmax compartido— apuntado
esta vez a la mitad que `lat2` rompió. Y hereda de la Fase 0 de `escriba` la lección de por qué esto
puede funcionar donde las siete vías fallaron: **tiene etiqueta**.

Control pareado: `v3_s0/s1/s2`, idénticas salvo la cabeza.

---

## 5. Predicciones de la condición

Instrumentos: `ser.py` (n=2048, semilla 54321) y la eval de `entrenar.py`, que ya desagrega
`nose_ent` y `nose_rel` (`entrenar.py:81-82`).

- **C-0 · BLOQUEANTE.** `vigente` ≥ 0,70 en al menos 2 de 3.
- **C-1 · LA PRINCIPAL.** `nose_rel` sube **≥ 0,10** contra su gemela `v3`, en al menos 2 de 3. El
  umbral es alto a propósito: `lat2` perdió 0,3393 en s0, y recuperar menos de 0,10 no justifica una
  cabeza nueva.
- **C-2 · CONSERVACIÓN.** `nose_ent` no cae más de 0,05, y `err_identidad` **se mantiene en 0,0000**.
  Lo que `lat2` ganó no se toca.
- **C-3 · NO-INTERCAMBIO.** `falsa_abst` ≤ 0,10 en las tres y `acierto` no cae más de 0,05. Sin esto,
  una suba de `nose_rel` puede ser sólo el modelo abstiniéndose más.

---

## 6. Regla de decisión, comprometida por adelantado

- **R-1 falla** → **no se entrena nada.** La ausencia de la relación tampoco tiene señal decodificable
  aun controlando por entidad, y el negativo del 16-ago queda **reforzado y precisado**: no era un
  artefacto de mezclar poblaciones. Se reporta y la línea se cierra.
- **R-1 pasa y C-1 falla** → la señal existe pero no se convierte en decisión. Es el mismo techo de
  calibración que A5 está mostrando, y **eso sería el hallazgo**: dos vías distintas chocando contra
  la misma pared vale más que cualquiera de las dos sola.
- **C-1 pasa y C-2 falla** → la cabeza compra la relación rompiendo lo que `lat2` arregló. Se reporta
  el intercambio y **no se adopta**.
- **C-1, C-2 y C-3 pasan** → es la continuación natural del trípode y va a paper.

---

## 7. Riesgos declarados

**El prior, otra vez.** `nose_rel` tiene una tasa base propia y una cabeza puede pegarse a ella igual
que el slot nulo. Se reporta **la tasa base junto a cada AUC**, y en la condición la media y el desvío
del logit. Si el desvío es < 0,1 y la media está pegada a `logit(tasa base)`, es colapso al prior y
no «la señal no sirve». Son cosas distintas.

**Fuga por longitud.** Un episodio con `nose_rel` podría tener una estructura sistemáticamente
distinta (más relaciones por entidad, otra longitud). Se corre la **sonda ciega** —sólo metadatos de
posición y tamaño, sin activaciones— igual que en `escriba`, y si alcanza el umbral, R-1 no es
interpretable.

**Desbalance de clases.** Con `p_nose = 0,4`, los `nose_rel` son aproximadamente la mitad de los
`nose`. Se reporta el reparto y **se aborta la lectura si alguna clase queda por debajo de 30 casos**,
que es la guarda que hoy evitó leer una AUC sobre 4 errores en `lat2`.

---

## 8. Lo que este pre-registro NO autoriza

- No autoriza correr la condición sin que la Fase 0 abra.
- No autoriza tocar `lat2`, que es la base adoptada.
- No autoriza elegir, después de ver R-2, la señal que mejor quede y reportarla como si fuera R-1.
- No autoriza lanzar nada mientras A5 tenga las cuentas ocupadas.
