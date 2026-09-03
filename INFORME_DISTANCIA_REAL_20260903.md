# Las dos paredes del modelo real cayeron, y ninguna por donde estaba escrito

**2026-09-03.** El plan de anoche decía que lo primero era destrabar el experimento en modelo real,
atrapado entre dos paredes medidas: con 4 hechos la tarea satura y con 16 el paso cuesta 9,7 s en T4.
El remedio previsto era compilar `mamba-ssm` (quince a treinta minutos, con riesgo de no compilar) y
subir a 16 hechos (~30 h de GPU). **Ninguna de las dos hizo falta.**

---

## 1. El costo · el tercer backend estaba en el propio mensaje de error

Lo que HF imprimía cada vez decía, entre medio, algo que no se había leído:

> *«Falling back to the sequential implementation of Mamba, **as use_mambapy is set to False**.»*

`transformers` 4.57 tiene **tres** caminos para el scan de Mamba, no dos: los kernels CUDA, el loop
en Python, y **`mambapy`, un scan asociativo en PyTorch puro** (`modeling_mamba.py:418`). Es un
paquete de **40 kB** que se instala con pip **sin compilar nada**.

**Se verificó la equivalencia antes de mirar la velocidad**, porque si no coincide no sirve:

| | logits, error relativo | gradientes | loss |
|---|---:|---:|---|
| CPU | 3,335e−06 | 9,113e−06 | 12,86536598 vs 12,86537170 |
| T4 | 3,466e−06 | 9,421e−06 | 12,865364 vs 12,865360 |

Es ruido de fp32, no otro modelo. Y la velocidad, con el montaje exacto de la campaña
(batch 8, largo 192, 16 hechos, gradient checkpointing):

| | secuencial | `mambapy` | |
|---|---:|---:|---:|
| esta PC (3 hilos) | 279,22 s/paso | **29,21** | **9,6×** |
| T4 | 25,77 s/paso | **2,977** | **8,7×** |

1200 pasos pasan de 8,59 h a **0,99 h** por unidad. Pico de memoria 5,08 GiB de los 15 de la T4.

**La lección, y es barata de aplicar en cualquier lado:** cuando una biblioteca avisa que cayó a un
camino lento, conviene leer el mensaje entero. Nombraba la salida y decía cómo activarla.

---

## 2. La saturación · el montaje NO TENÍA condición ciega

Ésta es la parte que importa, porque el remedio previsto —subir a 16 hechos— habría gastado ~30 h de
GPU midiendo un contraste que no existe.

### 2.1 La distancia estaba contada desde el lugar equivocado

`PREREG_MODELO_REAL_DIVERSIDAD.md` midió la distancia de cada pieza a la **posición de lectura**, el
`?`. Ésa es la distancia correcta **en el micro-LM**, donde la query se forma en el último token y va
a un softmax sobre un archivo **externo**: lo que no entra ahí es invisible para esa lectura, y punto.

En Mamba no hay archivo externo. La memoria es el estado, se actualiza en **cada** posición, y cada
token tiene su propio turno de condicionar la búsqueda cuando entra. Para responder hay que
**combinar** entidad y relación, y esa combinación ocurre en cualquier posición donde las dos estén
disponibles a la vez.

> **La distancia que decide es la que separa la RELACIÓN de la ENTIDAD, no la que las separa del `?`.**

### 2.2 Verificado por intervención, no derivado

`sonda_combinacion.py`: se cambia el token de la relación y se mira dónde se mueve `conv1d`.

| forma | d(rel↔ent) | `conv1d` @ **entidad**, capa 0 | `conv1d` @ `?`, capa 0 |
|---|---:|---:|---:|
| `directa` What is the {r} of {e}? | 2 | **4,767e−01** | 0,0 exacto |
| `lejana` What is the {r} that {e} has? | 2 | **4,767e−01** | 0,0 exacto |
| `invertida` For {e}, what is the {r}? | 5 | 0,0 exacto ⚠ | 6,830e−01 |
| `d5` What is the {r} of the person named {e}? | 5 | **0,0 exacto** | 0,0 exacto |

⚠ el cero de `invertida` **no adjudica**: ahí la entidad viene antes que la relación, así que es
causalidad y no ventana. Se anota para no contarlo de evidencia.

**Las tres formas del montaje eran o bien no-ciegas (d=2) o bien causalmente ciegas.** La condición
`una`, la que *tenía* que fallar, veía la relación perfectamente. Por eso llegó a `vigente` 1,0000 en
100 pasos: no porque la tarea fuera fácil, sino porque no era el experimento que se creía.

El contraste (la salida de la capa en el final se mueve para todas, 1,3e−02 a 5,9e−02) hace legibles
los ceros: no es que el modelo ignore el token.

---

## 3. Lo que salió de mirar la profundidad, y corrige el encuadre del preprint

La objeción obvia a todo esto es *«con 24 capas la información se transporta igual»*. Se midió.
`escalera_v2.py`: 15 formas, 6 distancias, 8 contextos, las 24 capas.

**Atenuación de la señal de la relación en la posición de la entidad, capa 1, contra el promedio de
las formas d=2:**

| d(rel↔ent) | capa 0 | **capa 1** |
|---:|---:|---:|
| 2 | 4,89e−01 | ×1,10 |
| 3 | **0,0 exacto** | ×2,31 |
| 4 | **0,0 exacto** | ×3,06 |
| 5 | **0,0 exacto** | ×4,64 |
| 6 | **0,0 exacto** | ×4,65 |
| 7 | **0,0 exacto** | ×6,92 |

**Ajuste lineal 1,077 por token, r = 0,9784.**

> **En un modelo recurrente profundo la ventana no BLOQUEA: ATENÚA. La forma dura de la ley —cero
> exacto— vale en la capa 0 y cae justo en el alcance medido. Desde la capa 1 la recurrencia
> transporta, pero deja la señal más débil, y cada token de distancia cuesta otro ×1,08.**

### 3.1 El control que sí adjudica

El «control de largo» de la primera versión —`d2` con más tokens **después** de la entidad— salió
**idéntico a `d2` hasta el último decimal en las 24 capas**, y tenía que salir así: nada posterior a
una posición puede afectarla. **Era causalidad disfrazada de control.** Se reemplazó por variar el
**relleno** manteniendo la distancia. Con d=5 se probaron cuatro rellenos distintos (*of the person
named · of that other person · in the file for · of our dear friend*): dispersión entre ellos ×1,98
contra un rango de ×1,10 a ×6,92 entre distancias. **La atenuación depende de la distancia, no de qué
palabras hay en el medio.**

### 3.2 Un titular que NO aguantó, y por qué se descarta

Con umbral 1,5 la «capa donde se cierra la brecha» daba 0 · 9 · 9 · 13 · 19 · 21, o sea ~4 capas por
token, r = 0,971. Muy vendible. **Se barrió el umbral de 1,2 a 2,5 y la pendiente salta de 3,97 a
1,54 y el orden se desarma** (con umbral 2,0, d=7 da 7 y d=6 da 11). Es un artefacto de la elección
del umbral. Lo de la capa 1 no depende de ningún umbral y es lo que se reporta.

---

## 4. La compuerta conductual · NO EVALUABLE, y lo que sí dejó

`ENMIENDA_DISTANCIA_REAL.md`. Una semilla, 400 pasos, 4 hechos.

| paso | `cerca` (d2) | `lejos` (d5) |
|---:|---:|---:|
| 100 | 0,9500 | 0,8000 |
| 200 | 0,9500 | 0,8000 |
| 300 | 0,9500 | 0,8000 |
| **400** | **0,9500** | **1,0000** |

**Con 4 hechos y la geometría bien contada YA NO SATURA**, que era lo que bloqueaba todo. `vigente`
1,0000, así que el bloqueante pasa.

**Y no se lee, por decisión declarada antes de mirar más:** la evaluación tenía ~40 ejemplos de
`nose_rel`, error típico 0,063, y los valores se mueven de a 1/40. Que `cerca` diera 0,9500 exacto
cuatro veces y `lejos` 0,8000 exacto tres es la firma de eso: 38/40 y 32/40, siempre los mismos.
Se subió a **512 ejemplos** por evaluación y se corre la campaña.

El salto de `lejos` a 1,0000 en el paso 400 apunta a algo que la §3 anticipaba mecánicamente y el
prereg no: con 24 capas hay margen de sobra para pagar el impuesto.

> **Hipótesis nueva, y es POST-HOC: en un modelo profundo la ventana no decide QUÉ se puede aprender
> sino CUÁNTO CUESTA aprenderlo.**

Encaja con el arco del micro-LM sin contradecirlo: allá el vehículo tiene 2 bloques y no hay capas
donde pagar, así que el efecto se ve como techo; acá hay 24 y se vería como demora. Se le escribió
criterio propio (**G-1v**, sobre el área bajo la curva) **antes** de la campaña, en vez de
reinterpretar G-1 después.

---

## 5. Estado y lo que sigue

**Corriendo:** campaña de 12 unidades (4 condiciones × 3 semillas, 800 pasos, eval cada 100 sobre 512
ejemplos), una semilla por cuenta. Y `cl3_s2` del micro-LM, la semilla que faltaba para que la tabla
del cierre del 2-sep sea evaluable.

**Cerrado sin correr nada:** `k73` (kernel 7) **ya estaba 3 de 3 en disco** —`nose_rel` 1,0000 ·
1,0000 · 0,9761, solapado con kernel 5—. El informe de anoche pedía una tercera semilla que ya
existía.

**Pendiente, y es lo que ligaría las dos mitades:** medir la sensibilidad de `conv1d` en el modelo
**después** del fine-tune y guardarla junto a las métricas. Hoy lo mecanicista se mide en el modelo
preentrenado y lo conductual en el ajustado, y son dos objetos distintos.

---

## 6. Las reglas que deja el día

1. **Cuando una biblioteca avisa que cayó a un camino lento, leer el mensaje entero.** Nombraba la
   salida y decía cómo activarla, y se pagaron dos días de lentitud por no leerlo.
2. **Antes de aceptar un efecto techo, preguntarse si la condición ciega es ciega de verdad**, y
   verificarlo por intervención. Acá el «techo» era el montaje.
3. **Una distancia hay que contarla hasta donde la información se COMBINA, no hasta donde se lee.**
   En el micro-LM coinciden porque la query se arma en el último token; en un recurrente profundo no.
4. **Un control que no puede dar otra cosa no es un control.** El de largo era aritmética causal.
   Se detecta preguntando: *si la hipótesis fuera falsa, ¿este control podría dar distinto?*
5. **Barrer el umbral antes de titular.** El resultado lindo de «~4 capas por token» no sobrevivió; el
   de la capa 1, que no depende de umbral, sí.
