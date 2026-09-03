# ¿La distancia relación↔entidad gobierna la abstención en un modelo REAL? · congelado ANTES de correr

**2026-09-03.** Reemplaza a `PREREG_MODELO_REAL_DIVERSIDAD.md`, que quedó ilegible por una geometría
mal contada. Acá está por qué, y qué se corrige.

## 1. El defecto del montaje anterior, medido hoy

El 2-sep la condición `una` —la que **tenía** que ser ciega, porque entrena sólo con la forma donde
la relación cae afuera de la ventana— llegó a `vigente` 1,0000 y `nose_rel` 0,95-1,00 **en 100
pasos**. Se leyó como efecto techo por tarea fácil y la salida propuesta era subir de 4 a 16 hechos.

**No era efecto techo. El montaje no tenía condición ciega.**

El prereg midió la distancia de cada pieza a la **posición de lectura** (el `?`). Ésa es la distancia
correcta en el micro-LM, donde la query se forma en el último token y va a un softmax sobre un
archivo **externo**: lo que no entra ahí es invisible para esa lectura, y punto.

En Mamba no hay archivo externo. La memoria es el estado, se actualiza en **cada** posición, y cada
token tiene su propio turno de condicionar la búsqueda cuando entra. Para responder hay que
**combinar** entidad y relación, y esa combinación puede ocurrir en cualquier posición donde las dos
estén disponibles a la vez.

> **La distancia que decide es la que separa la RELACIÓN de la ENTIDAD, no la que las separa del `?`.**

Contado en tokens del BPE (`geometria_formas.py`) y **verificado por intervención** en
`state-spaces/mamba-130m-hf` (`sonda_combinacion.py`, `sonda_combinacion.log`): cambiando el token de
la relación y mirando dónde se mueve `conv1d`, en la posición de la **entidad**, capa 0.

| forma | texto | d(rel↔ent) | conv1d @ entidad, capa 0 |
|---|---|---:|---:|
| `directa` | What is the {r} of {e}? | **2** | **4,77e−01** |
| `lejana` | What is the {r} that {e} has? | **2** | **4,77e−01** |
| `invertida` | For {e}, what is the {r}? | 5 | 0,0 exacto ⚠ |
| `d5` | What is the {r} of the person named {e}? | 5 | **0,0 exacto** |

⚠ el cero de `invertida` **no adjudica**: ahí la entidad viene **antes** que la relación, así que el
cero es causalidad y no ventana. Se anota para no contarlo como evidencia.

**Las tres formas del montaje anterior tenían d(rel↔ent) = 2 o eran causalmente ciegas.** No había
condición donde la relación fuera invisible **y** posterior a la entidad. Subir a 16 hechos habría
gastado ~30 h de GPU midiendo un contraste que no existe.

## 2. Lo que sí quedó medido, y acota la predicción

`escalera_atenuacion.py`, promedio de 6 contextos, 24 capas. La ventana **no bloquea en un modelo
profundo: atenúa.** Mediana de la atenuación sobre las capas 1-23, contra `d2`:

| d(rel↔ent) | capa 0 | capa 1 | mediana capas 1-23 |
|---:|---:|---:|---:|
| 2 | 4,89e−01 | 5,24e−01 | ×1,00 |
| 3 | **0,0 exacto** | 2,95e−01 | ×1,65 |
| 4 | **0,0 exacto** | 1,21e−01 | ×1,78 |
| 5 | **0,0 exacto** | 1,12e−01 | ×2,49 |
| 7 | **0,0 exacto** | 5,52e−02 | ×2,21 |

En la capa 0 el corte es exacto y cae justo en el alcance medido. En la capa 1 ya no es cero —la
recurrencia transportó— pero la señal es entre 1,8 y 9,5 veces más débil, y en las capas altas la
brecha se cierra casi del todo (×1-2 desde la capa 16).

**Consecuencia para la predicción, y hay que decirla antes:** con 24 capas el modelo tiene margen de
sobra para pagar el impuesto. El efecto conductual, si existe, va a ser **de grado y no de escalón**.
Un `lejos` que no falle no refuta la ley: refuta que la ley tenga consecuencia conductual *en un
modelo de esta profundidad*, que es una afirmación mucho más chica y también hay que poder decirla.

**Control declarado nulo:** el «control de largo» de la primera versión (`d2` con cola más larga)
salió idéntico a `d2` **hasta el último decimal en las 24 capas**, y tenía que salir así: todo lo que
va después de la entidad no puede afectar su propia posición. Es un control degenerado por
causalidad, no un resultado. Se reemplaza por `d5b`, que mantiene la distancia y cambia el relleno.

## 3. Montaje

`state-spaces/mamba-130m-hf`, 129.135.360 parámetros, fine-tune completo, T4.
**Scan paralelo `mambapy` activado**, verificado equivalente al camino secuencial de HF en CPU
(logits 3,3e−6 relativo, gradientes 9,1e−6) y en T4 (3,5e−6), y 8,7× más rápido allá
(25,77 → 2,98 s/paso). Sin eso esta campaña no entra en sesiones de Colab.

4 hechos, `p_nose` 0,4, largo 64, batch 8, lr 3e−5. La respuesta es **un token**, así que la métrica
es exacta sin juez ni parser.

| condición | entrena con | evalúa en | la relación entra en la ventana de la entidad |
|---|---|---|---|
| `cerca` | `d2` | `d2` | **siempre** |
| `lejos` | `d5` | `d5` | **nunca** (capa 0) |
| `lejos_dos` | `d5` + `d2` | `d5` | **a veces** |
| `lejos_relleno` | `d5` + `d5b` | `d5` | **nunca**, y con diversidad |

Es el diseño del micro-LM (`v3` / `cl3` / `cf3`) traducido con la geometría bien contada.
Tres semillas por condición.

## 4. Criterios, escritos antes del dato

Escritos preguntándose primero **si la intervención funcionara perfecto, ¿esta métrica se mueve?**

- **G-0 · BLOQUEANTE.** `vigente` ≥ **0,90** en las cuatro condiciones. Si el modelo no aprendió a
  contestar lo que sí está, nada de lo demás se lee.
- **G-L · LEGIBILIDAD, y es la que faltó el 2-sep.** Si `lejos` da `nose_rel` ≥ **0,95**, **G-1 y G-2
  son NO EVALUABLES por techo** y no se leen como negativos. La respuesta en ese caso es subir
  `--n-hechos`, que ahora cuesta cuatro veces menos, y volver a correr. **El juez tiene que imprimir
  NO EVALUABLE, no un número.**
- **G-1 · PRINCIPAL.** `nose_rel`: **`cerca` − `lejos` ≥ 0,10** en **≥2 de 3** semillas.
  Es la existencia del efecto de la distancia en un modelo real.
- **G-2 · EL REMEDIO DE DATOS.** `nose_rel`: **`lejos_dos` − `lejos` ≥ 0,10** en **≥2 de 3**.
  Es la pregunta original del prereg del 2-sep, que es la más accionable: no le pide a nadie que
  cambie su arquitectura.
- **G-3 · ADJUDICA.** `lejos_relleno` contra las otras dos:
  - ≈ `lejos_dos` → gana **la diversidad sola**;
  - ≈ `lejos` → gana **que la relación entre al menos a veces**.
- **G-4 · NO DAÑO.** `falsa_abst` ≤ **0,10** en `lejos_dos`, en ≥2 de 3. Sin esto G-2 se podría
  cumplir sólo porque el modelo se volvió más callado.
- **BASELINE.** Se mide en el paso 0 y se informa; lo que el modelo preentrenado ya sabía no nos lo
  podemos atribuir.

## 5. Lo que este experimento NO puede decir

- **Es recuerdo en contexto, no memoria persistente entre secuencias.**
- **Un solo modelo y una sola profundidad.** Y la profundidad es justo la variable que la §2 señala
  como decisiva, así que un negativo acá **no** se extiende a modelos poco profundos ni a memoria
  consultada desde capas tempranas, que es donde el micro-LM midió el efecto duro.
- **No mide cuánta exposición hace falta** si gana `lejos_dos`.
- El vocabulario son apellidos ingleses reales; el modelo tiene estadística sobre ellos, lo que no
  puede tener es el **hecho** inventado.
