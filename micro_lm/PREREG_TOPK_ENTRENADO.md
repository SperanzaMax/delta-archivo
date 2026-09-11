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
