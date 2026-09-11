# Pre-registro · LA LECTURA TOP-K ENTRENADA, con el archivo largo adentro

Fecha 2026-09-11, escrito con el barrido de K en inferencia cerrado (`dilucion_topk.py`,
`dilucion_topk_*.json`, commit `aad3fc1`) y **antes de correr una sola unidad con `--topk`**.
El código que lo corre es `entrenar.py --topk --relleno --fotos` (commit de este mismo prereg).

## 1. Qué dejó el 10-sep

Sobre checkpoints entrenados con softmax completo y archivo de 40 entradas, **leer** con top-2
recupera la exactitud con 3.240 entradas de relleno sin tocar un peso (kq3_s0 ruido 0,2441 → 0,9961;
v3_s0 disjunto 0,2324 → 1,0000), y transfiere a embeddings reales de castellano. Pero el top-k en
inferencia **no rescata** el caso `real`, que es colisión genuina (misma entidad y relación en otras
conversaciones): 0,0039 con K=N y 0,0352 con K=2. Y con los distractores puestos en turnos
ANTERIORES (`dilucion_real_viejo.json`, que es literalmente «lo dicho antes»), el modelo denso da
**0,0059** con 3.240: tiene un sello de orden que en principio podría descartarlos y no lo usa,
porque nunca entrenó con un archivo así.

Nadie entrenó todavía con la lectura top-k **y** el archivo largo a la vez. E-I1 midió que con
inyección temprana densa y top-k empatan (1,0000 contra 0,9998), pero con archivo de 40.

## 2. La pregunta

**¿Un modelo entrenado desde cero con la lectura top-2 y 3.240 entradas viejas de otras
conversaciones en el archivo aprende a contestar en ese archivo, o sea a descartar lo viejo y a
encontrar lo propio entre miles?** Es el escenario de [[objetivo]]: la memoria de una persona que
crece conversación tras conversación sobre las mismas entidades.

## 3. Hipótesis

**H1 · la principal.** `tp3` (top-2 desde el paso 0, relleno 3.240 real viejo) termina con
`vigente` **≥ 0,90** en archivo largo al paso 26.000 en **al menos 2 de 3 semillas**. El punto de
comparación es 0,0059 (denso corto, mismo banco) y 0,0352 (top-2 sólo en inferencia).

**H2 · qué aporta el top-k y qué aporta el relleno.** `dp3` (idéntico, softmax completo) termina en
`vigente` largo **≥ 0,10 por debajo** de `tp3` en la media de las tres semillas. Si empatan, lo que
enseñó fue entrenar con archivo largo y no la selección; es un resultado igual de informativo y se
reporta así.

**H3 · el costo en archivo corto (riesgo).** `archivo_corto.vigente` de `tp3` no queda más de
**0,05** por debajo de `dp3`. Se mide en cada evaluación, sale gratis.

**H4 · el riesgo del gradiente, con detector.** E-I1 dice que con inyección temprana el top-k duro
entrena bien; con archivo largo la entrada correcta puede no caer entre las 2 al arrancar y entonces
no hay gradiente que la levante. Se predice que **no colapsa**: `nose ≥ 0,20 en el paso 2.500`
(el detector del 8-sep, 9 de 9) en las tres semillas. **Plan B declarado:** si colapsan 2 de 3, se
corre `tp3` con `--topk-desde 4000` (currículo de un escalón, primero denso) como bifurcación con
su propio prefijo, y se reporta el colapso.

## 4. Diseño, congelado

Base = `kq3` (lat2, kernel 5, cabeza, p_nose 0,4, d 128, capas 4, lr 1e-3, idioma 2, 26.000 pasos,
horizonte 26.000) **más** las dos piezas del 6-sep, `--sello rel --pert`, porque «viejo» sólo es
descartable si el sello es relativo y la lectura sabe qué es propio. Desde cero (`SEMBRAR=0`).

    familia  topk  relleno                       semillas
    tp3      2     3240 real viejo (pool 4096, refresco cada 1000)   0 1 2
    dp3      0     3240 real viejo (idem)                           0 1 2

Las seis unidades comparten TODO salvo `TOPK`. Prioridad si falta GPU: tp3_s0, dp3_s0, tp3_s1,
tp3_s2, dp3_s1, dp3_s2.

Métricas primarias, en el JSON de cada unidad: `vigente`, `anterior`, `nose` (archivo largo, es lo
que rige `m`), `archivo_corto` (cruce), `topk` (qué lectura rigió en cada evaluación). Se reporta la
corrida entera y no el mejor checkpoint. Al cerrar, `dilucion_topk.py` sobre los seis checkpoints
en las tres distribuciones y los dos modos de turnos.

**La película.** `--fotos 100` → 260 cuadros por unidad, con la lectura capturada desde adentro de
`responder` (top-k y relleno incluidos) sobre una muestra fija. Es para mirar, no para decidir.

    PREFIJO=tp SELLO=rel PERT=1 KERNEL_Q=5 DONDE=lat2 P_NOSE=0.4 ABST=cabeza SEMBRAR=0 \
      TOPK=2 RELLENO=3240 RELLENO_DIST=real RELLENO_TURNOS=viejo FOTOS=100 \
      ./rotar_abst3.sh 3:0,3:1,3:2 26000 2000 1000 <cuentas>
    (dp3: igual con PREFIJO=dp TOPK=0)

## 5. Verificaciones hechas antes de congelar esto

- Con `--topk 0 --relleno 0 --fotos 0` la versión nueva de `entrenar.py` reproduce la de ayer
  **bit a bit** (dos evaluaciones, todas las métricas iguales; única clave nueva `topk`).
- La máscara top-k es idéntica al gather de `dilucion_topk.py` (diferencia 0,0; masa fuera del
  top-k 0,0) y el gradiente llega sólo a las K elegidas (verificado con `jax.grad`).
- El cambio de K en medio de una corrida recompila (`jax.clear_caches()`); sin eso el jit sigue con
  el K viejo (verificado, era el bug que había que no tener).
- Tramos: reanudar con el mismo K pasa, con otro K aborta por la guarda de identidad; la película
  continúa entre tramos.
- Lo que NO está verificado y se mide en la primera unidad: el s/paso en T4 con 3.240 entradas en
  la lectura. Si supera 0,9 s/paso el tramo baja a 1.000 pasos para que el polling no corte.

## 6. Enmiendas DESPUÉS del lanzamiento (09:05, escritas a las 08:20 hora local)

- **E-1 · el detector de H4 se lee en el paso 3.000, no en el 2.500.** La campaña corre con
  `--cada 1000` (evaluaciones en 1.000, 2.000, 3.000…), y el 2.500 del detector del 8-sep venía de
  corridas con `--cada 500`. Se toma la evaluación de 3.000 con el mismo umbral `nose ≥ 0,20`, y se
  declara el sesgo: a 3.000 pasos es MÁS fácil superar el umbral, así que un «no colapsa» acá es
  menos exigente que el del 8-sep. Si una unidad colapsa en 3.000, colapsó.
- **E-2 · ritmo medido: 0,62-0,64 s/paso** (500 pasos en 311-320 s con compilación y pool). El tramo
  de 2.000 entra en el presupuesto. `tramo_abst.sh` subió el tope de polling a 3x y baja la
  película aunque no haya checkpoint todavía; reemplazado por `mv` atómico, las seis instancias en
  vuelo siguen con el inode anterior.
- **E-3 · tp3_s0 y dp3_s0 se relanzaron a las 09:20** desde cero como rotadores de una unidad
  (los dos rotadores secuenciales de la primera tanda se habrían pisado con los de la segunda).
  Costó ~20 min de T4 y nada de datos: no había checkpoint todavía.

## 7. RESULTADO DEL HITO 3.000 y PLAN B AMPLIADO (08:50 hora local, con las evaluaciones a la vista)

**Seis de seis en el piso.** `vigente` en archivo largo 0,004-0,016 en el paso 3.000 (y 0,012-0,031
en el 2.000), en archivo corto 0,000-0,012, en `tp3` Y en `dp3`. `nose` en 3.000: tp3 0,000 /
0,249 / 0,000 · dp3 0,000 / 0,535 / 0,000 → el detector marca colapso en 2 de 3 en las DOS
familias, pero lo que está midiendo no es el pozo del 8-sep sino que **nadie despegó**: la
referencia desde cero con archivo corto (`kq3_s0`) ya estaba en 0,39 en el paso 1.000 y 0,61 en
el 2.000. **El relleno de 3.240 entradas desde el paso 0 alarga la fase plana en las dos familias**,
no sólo en la top-k: con el softmax denso la señal de la entrada correcta es ~1/3.280, y con top-2
la correcta casi nunca cae entre las dos elegidas. H4 se activa, y el plan B del §3 (sólo
`--topk-desde`) no alcanza porque el control denso tampoco arranca.

Se paró la campaña a las 08:50 (los seis JSON con 3 evaluaciones y las películas de 20-30 cuadros
quedan en `corridas_20260911/` y `ckpts/` como registro del arranque en frío).

**Plan B ampliado, declarado antes de correrlo.** Se siembra desde el modelo denso con archivo
corto ya entrenado (`kq3_s0-2`, 26.000 pasos, `sembrar.py`, Adam de cero, `sembrado_de`
declarado), que es lo que hizo la campaña del sello relativo (`rp3` desde `kq3`), y se continúa
**8.000 pasos** con el archivo largo:

    ts3_sX  sembrado de kq3_sX · top-2 · relleno 3240 real viejo · sello rel · pert   (principal)
    ds3_sX  sembrado de kq3_sX · softmax completo · el mismo relleno                (control)

`--cada 500` (el detector vuelve al paso 2.500), `--fotos 25` (320 cuadros), tramos de 2.000.
Las hipótesis H1-H3 se leen igual sobre `ts3`/`ds3` al paso 8.000. **Lo que cambia y queda dicho:**
ya no es «desde cero con el top-k en el ADN» sino «un modelo que sabe leer un archivo corto
aprende a leer uno largo con top-k». El desde-cero queda como pregunta abierta con su propio
hallazgo: en frío no arranca en 3.000 pasos con ninguna de las dos lecturas.
- **E-5 (09:40) · el primer intento del plan B murió de OOM en la GPU** al arrancar: las VMs
  reusadas todavía corrían el `entrenar.py` de la corrida en frío (matar el tramo en la PC no
  mata el proceso remoto) y JAX preasigna el 75 % de la memoria. Sin keep-alive, las seis
  sesiones se desasignaron después. Se relanza con el rotador normal (un rotador por unidad,
  `lanzar_planb_0911.sh segundo`), con los checkpoints sembrados restaurados y sin cuadros
  heredados. Nada de lo medido cambia.

## 8. ★ LA MÉTRICA INTERNA ESTABA INFLADA: el modelo aprende la ANTIGÜEDAD DE ESCRITURA del relleno (09:50)

**Lo que se vio.** Paso 500 del plan B: las seis unidades dan `vigente` 0,905-1,000 en archivo
largo (métrica interna, con el pool del entrenamiento). Demasiado bueno para 500 pasos, así que
antes de anunciarlo se corrió el control (`controles_20260911/`, checkpoints `ts3_s0` y `ds3_s0`
del paso 500, banco `dilucion.py` real + viejo, 256 muestras, X = 3.240):

| pool escrito con | bit de pertenencia | ts3_s0 (top-2) | ds3_s0 (denso) |
|---|---|---|---|
| los pesos del propio checkpoint (fresco, = despliegue) | no | **0,0703** | **0,0234** |
| los pesos del propio checkpoint (fresco) | sí, como en el entrenamiento | **0,0703** | **0,0234** |
| los pesos del paso 0 (= lo que vio el entrenamiento) | sí | **0,9883** | **0,9922** |

Idénticos al cuarto decimal con y sin el bit de pertenencia; **la única variable que mueve el
resultado de 0,02 a 0,99 es con qué pesos se escribió el relleno.** El RECUP acompaña
(0,035 → 0,957): la discriminación está en las claves. El modelo aprendió que «escrito con pesos
viejos» significa «ajeno», que es una señal que existe sólo porque el pool se refresca cada 1.000
pasos mientras los pesos siguen cambiando. **En despliegue el modelo está congelado y todo se
escribe con los mismos pesos, así que esa señal no existe: la métrica interna no mide nada real.**

**Consecuencias.**
1. H1-H3 NO se pueden leer sobre la métrica interna. La métrica confirmatoria pasa a ser el banco
   con pool fresco (+ bit de pertenencia, que sí es información de un sistema real).
2. La campaña se recorta: `s1` y `s2` de las dos familias se pararon a las 09:45 (paso 2.000);
   `ts3_s0` y `ds3_s0` siguen hasta el 8.000 sólo para completar la película (320 cuadros), que
   sigue siendo el retrato real de una dinámica real, aunque parte de lo que aprende sea esto.
3. **La trampa es general y va como hallazgo:** un archivo persistente co-entrenado cuyas entradas
   viejas se escribieron con pesos anteriores le enseña al modelo a fechar las entradas por la
   firma de los pesos. Cualquier diseño con memoria escrita por el propio modelo tiene que
   controlar esto (evaluar siempre con archivo reescrito con los pesos actuales).
4. Rediseño del relleno, a pre-registrar antes de correr: el relleno se escribe **con los pesos
   actuales en cada paso** (sesiones extra sin gradiente, `stop_gradient`), aunque sea más corto
   (p. ej. 400 entradas), y la generalización a 3.240 se mide en el banco con pool fresco. Antes,
   una medida barata que decide si hace falta: `rp3_s0` (6-sep, sello relativo + pertenencia,
   entrenado con 161 entradas frescas con gradiente) sobre el mismo banco.

## 9. EL RELLENO FRESCO SÍ ENSEÑA, y la campaña que sigue (10:15, antes de correrla)

**`rp3_s0` (6-sep: sello relativo + pertenencia, entrenado 2.000 pasos con 161 entradas viejas
escritas con los pesos actuales, con gradiente) en el banco de pool FRESCO, real + viejo, PERT=1**
(`controles_20260911/rp3_s0_pool_fresco_pert1.log`):

    archivo    40  →  0,9688
    archivo   400  →  0,8867      (kq3 sin ese entrenamiento: 0,0586)
    archivo 3.280  →  0,1016      (kq3: 0,0059)

O sea: entrenar con entradas viejas **frescas** enseña de verdad a descartarlas (0,06 → 0,89 con
400, y el banco no comparte nada con el entrenamiento), y lo que falta para 3.240 es **escala**,
no mecanismo. Por eso el relleno del pool se abandona y se vuelve a `--ses-extra`, que escribe
las entradas viejas con los pesos actuales en cada paso, con un agregado que lo hace barato:
`--ses-extra-sin-grad` (las sesiones extra pasan por el tronco bajo `stop_gradient`; verificado:
gradiente 0,0 por ellas y valores escritos idénticos).

**Campaña `tf3`/`df3`, congelada:** sembrado de `kq3_sX`, `--sello rel --pert --kernel-q 5 --donde
lat2 --abst cabeza --p-nose 0.4`, **`--ses-extra 36 --ses-extra-sin-grad`** (400 entradas viejas
frescas por paso), 8.000 pasos, `--cada 500`, `--fotos 25`.

    tf3_sX   --topk 2          (la principal)
    df3_sX   softmax completo  (el control)

Se lanzan `s0` de las dos primero, para medir s/paso y memoria con 40 sesiones por paso; `s1` y `s2`
después. **Hipótesis, leídas sobre el banco de pool fresco (real, viejo, PERT=1) al paso 8.000:**

- **H1'** `tf3` ≥ 0,90 con archivo 400 (donde `rp3` llegó a 0,89 con 161 de entrenamiento) en 2 de 3.
- **H2'** con archivo 3.280, `tf3` supera a `df3` por ≥ 0,10 en la media: es donde la selección
  debería pagar, porque ahí la lectura densa diluye (5-sep) y la top-2 no.
- **H3'** la generalización: `tf3` con 3.280 ≥ 0,50 (entrenado con 400). Si queda por debajo, el
  siguiente paso es más sesiones extra, no otro mecanismo.
- **H4'** detector: `nose ≥ 0,20` en el paso 2.500 en la métrica interna (que acá SÍ es fresca).
La métrica interna (`evaluar` con `n_ses_extra=36`) se reporta, pero no decide.
