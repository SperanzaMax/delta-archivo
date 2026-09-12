# Pre-registro · HECHOS ENCADENADOS: ¿el modelo puede RAZONAR con el archivo? · 2026-09-11

Escrito a las 18:50, con el código listo (`idioma.pregunta_compuesta`, `datos.lote(p_compuesta)`,
`modelo.tronco` con lectura en varios bloques, `entrenar.py --p-compuesta --bloques-lectura`) y
**antes de correr una sola unidad**. Sale de la pregunta 1 de Maxi: un modelo de frontera con el
archivo, congelado el 1 de enero y actualizado sólo por el archivo, ¿puede razonar con lo que está
en el archivo, o sólo recuperarlo? El banco de hoy responde un token de un solo hecho. Esto mide el
paso mínimo de razonamiento: **dos hechos encadenados**.

## 1. La tarea

Un hecho personal da un nombre; con `--p-compuesta` el episodio agrega, a veces, un hecho SOBRE ese
nombre (`altura` o `clave`, las mismas palabras de siempre; `del` ya estaba en el vocabulario, V no
cambia) y la pregunta compuesta:

    «el director de barrio es yamil»  +  «la altura de yamil es 48»
    → «cual es la altura del director de barrio ?»  → 48        tipo `compuesta`

La mitad de las veces el segundo hecho NO se dice, y la respuesta correcta es NOSE (tipo
`nose_comp`). Las preguntas simples, las versiones y los NOSE de siempre siguen igual; con
`p_compuesta = 0` el generador es bit a bit el de antes (verificado). La compuesta se elige la mitad
de las veces que existe.

## 2. Las hipótesis, y la primera es de ARQUITECTURA

**H0 · con lectura en un solo bloque NO se puede encadenar.** Hoy el archivo se lee sólo en el
bloque 0 y la query de esa lectura se forma sobre el TEXTO (la conv corta sobre `ln1(emb)`): en el
token «?» puede leer «director de barrio → yamil», pero para la segunda lectura necesitaría «altura
de yamil», y «yamil» no está en el texto, está en lo que el archivo acaba de devolver. Ese vector
sigue por el tronco, pero ninguna query posterior lo ve. **Predicción:** `--bloques-lectura 0`
termina con `compuesta` ≤ 0,50 en las tres semillas aunque `vigente` ≥ 0,95, y `nose_comp` alto
(le queda decir «no sé»). Un resultado ≥ 0,80 aquí sería evidencia CONTRA cómo entiendo la
arquitectura, y habría que buscar por dónde pasa la información.

**H1 · con lectura en los bloques 0 y 2, sí.** La misma lectura (mismas claves, mismos valores) se
consulta también en el bloque 2, con una query formada sobre `h`, que ya lleva lo leído en el
bloque 0 propagado por los mixers de los bloques 0 y 1. **Predicción:** `compuesta` ≥ 0,85 en 2 de 3
semillas, `vigente` ≥ 0,95, `nose_comp` ≥ 0,85, al paso 26.000.

**H2 · la ventana ya no manda.** En «cual es la altura del director de barrio ?» la relación
`altura` está a distancia 5 del «?», fuera del alcance 4 del kernel 5. Para la primera lectura
(«director de barrio») alcanza. Para la segunda, la query del bloque 2 no es la conv sobre texto
sino sobre `h` con el mixer recurrente adentro, que lleva «altura». **Predicción:** kernel 5 con
bloques 0,2 no falla por la ventana (H1 se cumple sin cambiar el kernel). Si falla, el siguiente
brazo es `attn`.

**H3 · el costo.** Leer dos veces no debe cobrar en lo simple: `vigente` y `anterior` de `0,2` no
quedan más de 0,03 por debajo de `0`.

## 3. Diseño, congelado

Desde cero (`SEMBRAR=0`), base `kq3` (nivel 3, d 128, capas 4, lr 1e-3, idioma 2, kernel 5, lat2,
cabeza, `p_nose 0,4`, `p_vieja 0,35`), **`--p-compuesta 0.5`**, 26.000 pasos, horizonte 26.000,
`--cada 1000`, fotos 100. Dos brazos, tres semillas:

    ec3_sX   --bloques-lectura 0      (H0)
    ed3_sX   --bloques-lectura 0,2    (H1)

Métricas primarias `compuesta` y `nose_comp`; secundarias `vigente`, `anterior`, `nose`. Detector de
colapso `nose ≥ 0,20` en el paso 3.000 (evaluación cada 1.000). Se reporta la corrida entera.

    PREFIJO=ec SELLO=abs PERT=0 KERNEL_Q=5 DONDE=lat2 P_NOSE=0.4 ABST=cabeza SEMBRAR=0 \
      P_COMPUESTA=0.5 BLOQUES_LECTURA=0 HORIZONTE=26000 ./rotar_abst3.sh 3:0,... 26000 2000 1000
    (ed3: BLOQUES_LECTURA=0,2)

## 4. Verificado antes de congelar

- `datos.lote` con `p_compuesta = 0` es bit a bit el de siempre (16 lotes, todas las salidas).
- `modelo.tronco` con `bloque` entero es bit a bit el de siempre (el control de `entrenar.py` con
  todos los flags apagados reproduce la versión de la mañana).
- Con `p_compuesta = 1` aparecen los tipos 4 y 5; la pregunta compuesta entra en `T_Q = 12`
  (10 tokens); ejemplo real: «cual es la altura del guardia de fabrica ? → 69» con «fabrica tiene
  como guardia a nadia … la altura que tiene nadia es 69».

## 5. ENMIENDA E-1 (19:35) · la v1 de la tarea tenía un ATAJO, medido, y se cerró antes de seguir

La campaña `ec3`/`ed3` se lanzó a las 18:55 con la tarea del §1. En el paso 1.000 el brazo de UN
bloque (`ec3`, H0 predecía ≤ 0,50) daba `compuesta` **1,000 / 1,000 / 0,873** con `vigente` en
0,39-0,48 y `nose_comp` 0,07-0,13. Demasiado y demasiado pronto: con un solo hecho de persona por
episodio, «la altura del director de barrio» se contesta leyendo *la única entrada de altura cuyo
sujeto es un nombre*, sin saber quién es el director. No es encadenar. (Además `ed3` abortó por un
checkpoint viejo con ese prefijo; `hd3` también existe.)

**v2 de la tarea:** todos los nombres vigentes de los hechos personales del episodio reciben un hecho
con la misma relación (`altura` o `clave`), y si hay menos de tres se agregan nombres ajenos como
distractores; en las `nose_comp`, el preguntado es el único que se queda sin su hecho. Ahora la
compuesta exige la versión vigente del nombre Y la lectura de su hecho entre tres candidatos.
Ejemplo real del generador: «quien posee tienda es zoe no, celia … celia tiene la clave en 64 · gema
tiene la clave en 29 · la clave de diego es 0» → «cual es la clave del dueño de tienda ?» → 64.
Truncamiento medido 0,5 % (nivel 3, una sesión; la compuerta corta en 1 %). Con `p_compuesta = 0`
sigue siendo bit a bit el generador de siempre.

Las hipótesis del §2 quedan iguales. Los brazos pasan a `kc3` (un bloque) y `kd3` (bloques 0,2),
mismo diseño. Lo del §1 y la v1 se archivan en `corridas_20260911/encadenados_v1_atajo/`.

## 6. ENMIENDA E-2 (12-sep, 07:55) · H0 FALLÓ, y la intervención dice por dónde pasó la información

**Lo que dio la campaña v2 (`corridas_20260911/kc3_s*.json`, `kd3_s*.json`; el cierre de las 23:00
las cortó en 14.000-20.000 pasos de 26.000):**

    unidad   bloques  paso   vigente  anterior  nose   compuesta  nose_comp
    kc3_s0   0        14000  0,282    0,097     0,482  0,400      1,000      no arrancó (loss ~3)
    kc3_s1   0        14000  0,884    0,433     0,936  0,933      0,705
    kc3_s2   0        18000  0,987    1,000     0,959  0,797      1,000      (0,875 en 14.000, 0,844 en 16.000)
    kd3_s0   0,2      20000  0,258    0,200     0,433  0,393      0,083      no arrancó (loss 2,4-2,9)
    kd3_s1   0,2      18000  0,289    0,117     0,183  0,331      0,000      no arrancó
    kd3_s2   0,2      18000  0,353    0,043     0,090  0,447      0,000      no arrancó

- **H0 predecía `compuesta` ≤ 0,50 con `vigente` ≥ 0,95. FALLÓ:** `kc3_s2` da 0,80-0,88 con
  `vigente` 0,99, y `kc3_s1` 0,93 con `vigente` 0,88. La tercera semilla no aprendió nada (la trampa
  del frío, ya conocida), no es evidencia de H0.
- **H1 quedó SIN MEDIR:** las tres semillas con lectura en 0,2 no aprendieron ni las preguntas
  simples. No es «no encadena», es «no entrena». **H3 cae con ella:** la segunda lectura no cuesta
  0,03, cuesta la corrida. La hipótesis para eso (a medir aparte): la lectura del bloque 2 comparte
  `qr`/`kw`/`vw`/`wo` con la del bloque 0 y forma su query sobre `h`, con otra estadística que
  `emb`; en frío las dos lecturas se pelean por los mismos pesos y ninguna arranca (el «no
  arranca» pasa de 1 de 3 a 3 de 3).

**Por dónde pasó la información (`encadenados_mecanismo.py`, `encadenados_intervencion.py`;
resultados en `corridas_20260911/encadenados_intervencion.json`).** El prereg decía: «un resultado
≥ 0,80 aquí sería evidencia CONTRA cómo entiendo la arquitectura, y habría que buscar por dónde
pasa la información». La primera pista fue la lectura misma: en el «?» de la compuesta, `p` cae
sobre la entrada de ALTURA del nombre correcto (0,55 en s1, 0,85 en s2), no sobre la entrada de
persona (0,07 / 0,005). La entrada de altura ya «sabe» de quién es. Y la segunda, en el código:
`modelo.escribir` pasa cada sesión entera por el tronco, con el mixer recurrente SIN resetear
entre enunciados, así que el vector que se archiva para «la altura de yamil es 48» lleva adentro
lo que la sesión dijo antes, incluido «el director de barrio es yamil». A nivel 3 TODOS los
enunciados caen en la sesión 0 (`idioma.episodio`: `s = 0 if nivel < 4`) y el bloque de altura
se agrega al final, así que el hecho de persona SIEMPRE está antes y en la misma sesión.

La intervención, sobre los MISMOS 2.560 episodios por checkpoint, sin tocar un peso, con los
turnos originales de cada enunciado (el sello no cambia), moviendo sólo dónde se escribe el bloque
de altura:

    kc3_s1 (paso 14000)     compuesta  NOSE    p_alt  p_pers  p_otros | vigente  anterior
      original                0,952     0,004   0,546  0,071   0,297   | 0,911    0,274
      otra sesión (reset)     0,222     0,365   0,269  0,085   0,538   | 0,888    0,300
      invertido (antes)       0,209     0,430   0,251  0,086   0,505   | 0,863    0,269
    kc3_s2 (paso 18000)
      original                0,883     0,091   0,847  0,005   0,139   | 0,993    0,982
      otra sesión (reset)     0,170     0,400   0,307  0,016   0,641   | 0,966    0,982
      invertido (antes)       0,126     0,530   0,307  0,015   0,639   | 0,962    0,951

Con la entrada de altura escrita en otra sesión (estado reseteado) o ANTES del hecho de persona
(la recurrencia es causal), la compuesta cae de 0,95/0,88 a 0,13-0,22: azar entre los tres
candidatos cuando contesta (0,22/(1−0,365) = 0,35) o «no sé» (37-53 %). Y la lectura se reparte
pareja entre las tres entradas de altura (0,27-0,31 en la correcta, 0,54-0,64 en las otras dos).
Las preguntas simples no se mueven (−0,02/−0,03 en `vigente`; `anterior` igual): la intervención
no rompe el archivo, rompe SÓLO el encadenado.

**Veredicto.** H0 tenía razón sobre la lectura: con un solo bloque, la query no ve lo que el
archivo acaba de devolver. Falló porque había otro camino que el prereg no consideró: **el
encadenado se hace al ESCRIBIR.** El archivo no guarda hechos sueltos sino enunciados
contextualizados por lo que la sesión dijo antes, y la pregunta de dos saltos se contesta con UNA
lectura porque la entrada ya trae el primer salto adentro. Es un hallazgo sobre qué es una entrada
del archivo, y vale para el objetivo de fondo (lo que se dice en una conversación queda archivado
CON su contexto de conversación), pero **no es razonar al leer**, y la pregunta original sigue
abierta.

**Lo que sigue (v3, a congelar en enmienda E-3 antes de correr):** quitar el camino de escritura
—el bloque de altura va SIEMPRE en otra sesión que el hecho de persona, también al entrenar— y
medir ahí un bloque contra bloques 0,2. Como el brazo de dos bloques no arranca en frío, los dos
brazos se siembran del mismo checkpoint (`kc3_s1`, `kc3_s2`) y se entrenan lo mismo sobre los
mismos datos: el brazo de un bloque parte de 0,17-0,22 en esta condición y no tiene por dónde
subir; lo que suba el de dos bloques por encima es el encadenado al leer. H0' (un bloque, otra
sesión): `compuesta` ≤ 0,40 al final. H1' (bloques 0,2, sembrado): ≥ 0,80 en 2 de 2, con
`vigente` a ≤ 0,03 del brazo de un bloque.

## 7. ENMIENDA E-3 (12-sep, 08:00) · v3 congelada: el bloque de altura en OTRA sesión

Escrita con el código listo (`idioma.episodio(sesion_rel2)`, `datos.lote(sesion_rel2)`,
`entrenar.py --rel2-sesion`, `sembrar.py` con `rel2_sesion` como bifurcación) y **antes de correr**.
Verificado: con `sesion_rel2=None` el lote es bit a bit el de siempre (16 salidas, semilla 5);
con `--rel2-sesion 1` el bloque de altura cae en la sesión 2 y `kc3_s2` evaluado así da
`compuesta` 0,140 (contra 0,969 con el sorteo de siempre), lo mismo que la intervención del §6
(0,170): el flag reproduce la intervención con el generador.

**Diseño.** Cuatro unidades, sembradas (`sembrar.py --horizonte 8000`) de los dos checkpoints que
aprendieron, y entrenadas con LOS MISMOS datos (`--rel2-sesion 1`, `--p-compuesta 0.5`, todo lo
demás como `kc3`): 8.000 pasos, `--cada 500`, fotos 100.

    mc3_s1  de kc3_s1 (paso 14000)   --bloques-lectura 0     (H0')
    mc3_s2  de kc3_s2 (paso 18000)   --bloques-lectura 0     (H0')
    md3_s1  de kc3_s1 (paso 14000)   --bloques-lectura 0,2   (H1')
    md3_s2  de kc3_s2 (paso 18000)   --bloques-lectura 0,2   (H1')

Sembrar los dos brazos del mismo punto es lo que separa «no encadena» de «no entrena» (lo que
confundió a `kd3`): los cuatro parten sabiendo la tarea simple y el encadenado al escribir, y la
única diferencia entre `mc` y `md` es si hay una segunda lectura con query formada sobre `h`.

**H0' · un bloque, otra sesión: no puede.** `mc3` termina con `compuesta` ≤ 0,40 en las dos
(parte de 0,14-0,22 y no tiene por dónde subir: ni la query ve lo leído ni la entrada trae el
contexto). `nose_comp` puede subir (aprender a callarse es lo que le queda).

**H1' · bloques 0,2, sembrado: sí.** `md3` termina con `compuesta` ≥ 0,80 en las dos, y
`vigente`/`anterior` a ≤ 0,03 de `mc3`. Si `md3` sube pero se queda entre 0,40 y 0,80, se reporta
como parcial y el siguiente brazo es `attn` (H2 del §2). Si `md3` no sube de 0,40, el encadenado
al leer no está disponible para esta arquitectura con esta siembra, y la conclusión del §6 se
queda sola: el archivo encadena sólo al escribir.

**Control de identidad.** `mc3` con el flag es exactamente lo que ya se midió en el §6, así que su
paso 0 tiene que dar 0,14-0,22; si arranca más alto, el flag no está cortando el camino de
escritura y la campaña no vale.

    mc: PREFIJO=mc BLOQUES_LECTURA=0   REL2_SESION=1 P_COMPUESTA=0.5 SEMBRAR=0 HORIZONTE=8000 ./rotar_abst3.sh 3:1,3:2 8000 2000 500
    md: PREFIJO=md BLOQUES_LECTURA=0,2 (idem)

## 8. ENMIENDA E-4 (12-sep, 08:40) · la v3 tenía DOS ATAJOS; se paró a los 2.000 pasos y va la v4

**Lo que pasó.** A los 500-2.000 pasos, `mc3_s2` (UN bloque, bloque de altura en otra sesión) dio
`compuesta` 0,73 y después 0,95 (`evaluar` sobre `ckpts/mc3_s2.pkl`, paso 2000: 0,948), o sea
subió de 0,14 a 0,95 en 2.000 pasos con la lectura que, según el §6, no tiene por dónde
encadenar. Intervención sobre ese checkpoint (`encadenados_intervencion.py`, condición
`barajada`, 1.012 episodios, `corridas_20260911/intervencion_v3_parcial.json`):

    mc3_s2 (paso 2000, un bloque)   compuesta   p_alt   vigente
      original (bloque en su orden)   0,893      0,398   0,960
      barajada (otro orden, turnos por posición nueva)
                                      0,400      0,303   0,939
      misma sesión, después           0,893      0,519   0,939
      invertido                       0,907      0,400   0,918
    md3_s2 (paso 2000, bloques 0,2)   0,493 / barajada 0,413

Con el orden del bloque barajado cae a 0,40, el azar entre tres. **Atajo 1:** el generador escribe
los hechos de altura en el MISMO orden que los hechos de persona (`for nombre in nombres`, y
`nombres` sale de `personales` en orden), así que «el k-ésimo de altura es del k-ésimo de persona»
se resuelve con el sello de orden, sin saber quién es el director. El brazo de dos bloques ni
siquiera lo aprendió tan rápido (0,49). **Atajo 2:** en las `nose_comp` el preguntado se queda sin
hecho y el bloque tiene DOS entradas en vez de tres: `nose_comp` se contesta contando (0,96 en
todas las unidades desde el primer hito). Los dos estaban también en la v2; en `kc3` no hacía
falta el primero porque el camino de escritura del §6 alcanzaba, y el §6 no se toca: ahí la
intervención movía el bloque SIN barajarlo y la compuesta caía igual, o sea `kc3` no usaba el
orden.

**Y una trampa del instrumento, arreglada:** `partes_p` es `jax.jit` y lee `E._BLOQUES` al trazar;
con dos checkpoints de distinta arquitectura en la misma corrida el segundo se medía con la traza
del primero (`mc3_s2` daba `vigente` 0,42 detrás de `md3_s2`). `preparar` hace `jax.clear_caches()`.
Es la regla de `conf_ckpt`, versión jit.

**v4 (`--rel2-barajar`):** el bloque de altura se escribe en orden sorteado y siempre son tres
(en las `nose_comp` entra un nombre ajeno más). Verificado: sin el flag, bit a bit lo de antes;
con el flag, 3 o 4 entradas de altura tanto en `compuesta` como en `nose_comp`. Brazos `nc3`
(bloque 0) y `nd3` (bloques 0,2), sembrados de `kc3_s1`/`kc3_s2` como en E-3, 8.000 pasos, con
`--rel2-sesion 1 --rel2-barajar`. Las hipótesis H0'/H1' del §7 quedan iguales, con un umbral
más: **`nose_comp` ya no vale como métrica de encadenado** salvo que supere 0,80 con `compuesta`
también alta, porque el piso por contar desapareció y un modelo que dice NOSE a toda compuesta da
`nose_comp` 1,0 y `compuesta` 0,0.

## 9. RESULTADO de la v4 (12-sep, 10:40) · el encadenado al LEER existe, lo hace la segunda lectura, y arranca en 1 de 2

`corridas_20260912/n[cd]3_s*.json`, 8.000 pasos, las cuatro cerradas:

    unidad   bloques  compuesta  nose_comp  vigente  anterior  nose   falsa_abst
    nc3_s1   0        0,237      0,031      0,995    1,000     0,872  0,003
    nc3_s2   0        0,312      0,000      1,000    1,000     0,849  0,004
    nd3_s1   0,2      0,764      0,042      0,981    1,000     0,871  0,024
    nd3_s2   0,2      0,307      0,021      1,000    1,000     0,833  0,017

Curvas (`compuesta` cada 500): `nc3` oscila entre 0,19 y 0,53 los 8.000 pasos sin tendencia
(azar entre tres = 0,33); `nd3_s1` sube desde el paso 3.000 (0,53) a 0,75-0,79 entre 4.000 y
7.000 y cierra en 0,76; `nd3_s2` no despega (0,09-0,37).

- **H0' CUMPLE:** con el camino de escritura cortado y el orden barajado, un bloque no encadena
  (0,24 / 0,31, ≤ 0,40 en las dos).
- **H1' PARCIAL:** 1 de 2 llega a 0,76 (≥ 0,80 no), la otra queda en el azar. `vigente` y
  `anterior` no cuestan nada (≤ 0,02 del brazo de un bloque; H3 vale para la versión sembrada).
- **Y es la SEGUNDA lectura la que encadena** (`encadenados_mecanismo.py` sobre los checkpoints
  del 8.000, 118 compuestas, `corridas_20260912/mecanismo_v4.json`): en el «?», la lectura del
  bloque 2 de `nd3_s1` pone **0,606 en la entrada de altura del nombre correcto y 0,384 en las
  otras dos**; en `nc3_s1` (un bloque) 0,307 / 0,623 y en `nd3_s2` 0,356 / 0,644, o sea uniforme
  entre los tres. La query formada sobre `h` en el bloque 2 lleva el nombre que el bloque 0 leyó y
  selecciona con él. No queda otro camino: el bloque está en otra sesión (no hay contexto de
  escritura), barajado (no hay orden) y siempre de tres (no hay conteo).
- **`nose_comp` se cae en todas (0,00-0,04):** sin el piso por contar, el modelo contesta siempre la
  compuesta aunque el preguntado no tenga hecho. Decir «no sé» acá exige el encadenado Y notar la
  ausencia entre tres candidatos; en 8.000 pasos ninguna lo aprendió. Queda como métrica abierta.

**Veredicto de la campaña entera (v2 → v4).** El modelo encadena por DOS caminos: al escribir
(la entrada lleva el contexto de la sesión; §6, y es lo que usaba `kc3`) y al leer, si hay una
segunda lectura con query sobre `h` (§9, `nd3_s1`). El primero es gratis y robusto; el segundo
arranca en 1 de 2 semillas sembradas y no arranca en frío (3 de 3 en `kd3`). Lo que sigue, por el
§7: brazo `attn` (query global) como tercera arquitectura, más semillas de `nd`, y medir por qué
la segunda lectura no arranca (los pesos compartidos `qr/kw/vw/wo` entre las dos lecturas).

## 10. ENMIENDA E-5 (12-sep, 10:50) · tasa de arranque de la segunda lectura, y el brazo `attn`

Congelada antes de correr. Sale del §9: `nd` arranca en 1 de 2, y con dos semillas no se puede
decir si es «a veces» o «casi nunca». Dos preguntas, dos brazos, todo sembrado de `kc3_s2` (paso
18.000, la base que mejor sabía la tarea simple) con `sembrar.py --semilla` (mismos pesos, otro
orden de datos; declarado en `sembrado_de`), 8.000 pasos, `--rel2-sesion 1 --rel2-barajar`, lo
demás como la v4:

    nd3_s3, nd3_s4, nd3_s5, nd3_s6   lat2, bloques 0,2   (tasa de arranque: con nd3_s1/s2 son 6 de kc3_s*)
    ne3_s1, ne3_s2                   attn, bloques 0,2   (`sembrar.py --donde attn`: la query de las DOS
                                                          lecturas es atención causal completa; convq queda sin uso)

**P1 · tasa.** Se reporta cuántas de las 6 `nd` terminan con `compuesta` ≥ 0,60 (el criterio es
más laxo que el 0,80 del §7 porque `nd3_s1` cerró en 0,76 y la pregunta es si ARRANCA, no si
llega). Predicción: entre 2 y 4 de 6. Si 0 de 4 nuevas, `nd3_s1` fue suerte y el mecanismo no es
usable sin más cambios.
**P2 · `attn`.** Si la query global arranca en 2 de 2 (≥ 0,60), la ventana de la conv es lo que
frena la segunda lectura (H2 del §2 se invierte). Si arranca igual o peor que `nd`, la ventana no
es el cuello. Riesgo declarado: cambiar `lat2` → `attn` en un modelo sembrado le cambia también la
PRIMERA lectura (la query del bloque 0 deja de ser la conv que aprendió), así que `vigente` puede
caer al principio; si en el paso 8.000 `vigente` < 0,90, el brazo no se compara y hay que correrlo
desde cero.

## 11. E-5, estado a las 16:00 del 12-sep (parcial, la campaña sigue)

Las 12 vueltas del rotador se agotaron sin T4 a media tarde (el pool entero devolvió 503 desde
~13:30). Cerradas: `nd3_s4` 0,32 y `nd3_s6` 0,40 (8.000 pasos; no arrancan). A medias: `nd3_s5`
0,26 en 6.000, `nd3_s3` 0,24 en 2.000. **`ne3` no corrió:** la guarda de `donde` de `entrenar.py`
usaba `.get("donde", "pre")` y resucitaba `pre` sobre el checkpoint sembrado sin la clave —el
mismo defecto que las guardas de `sello`/`pert` ya tenían tapado— y abortó tres veces en T4.
Arreglado (sin `sembrado_de` sigue valiendo `pre`; sembrado y sin clave, se acepta lo pedido) y
verificado con un smoke local de 2 pasos. Las cuatro pendientes se relanzaron con 40 vueltas.

Tasa de arranque de la segunda lectura hasta acá: **1 de 4 cerradas** (`nd3_s1` 0,76; `s2`,
`s4`, `s6` en 0,31-0,40), contra la predicción de 2-4 de 6. Si `s3` y `s5` tampoco, queda 1 de 6:
el mecanismo existe pero es raro con esta siembra, y la pregunta pasa a ser qué lo dispara.

## 12. RESULTADO de E-5 (12-sep, 17:30) · 1 de 6, y la query global no ayuda · ENMIENDA E-6

Las seis cerradas en 8.000 (`corridas_20260912/n[de]3_s*.json`):

    unidad   base      donde  compuesta  vigente  anterior
    nd3_s1   kc3_s1    lat2   0,764      0,981    1,000     (v4)
    nd3_s2   kc3_s2    lat2   0,307      1,000    1,000     (v4)
    nd3_s3   kc3_s2    lat2   0,244      1,000    1,000
    nd3_s4   kc3_s2    lat2   0,323      1,000    1,000
    nd3_s5   kc3_s2    lat2   0,313      1,000    1,000
    nd3_s6   kc3_s2    lat2   0,398      1,000    1,000
    ne3_s1   kc3_s2    attn   0,101      1,000    1,000
    ne3_s2   kc3_s2    attn   0,373      1,000    1,000

- **P1 FALLÓ:** tasa de arranque 1 de 6 (predije 2-4 de 6). Con `vigente` 1,000 en todas, no es
  que no entrenen: aprenden la tarea simple perfecto y la segunda lectura queda uniforme.
- **P2: `attn` no arranca (0 de 2)** con `vigente` 1,000, así que el brazo se compara y la ventana
  de la conv NO es lo que frena la segunda lectura. H2 del §2 se sostiene por el otro lado.
- **La explicación alternativa que la tabla deja a la vista:** la única que arrancó es la única
  sembrada de `kc3_s1` (paso 14.000, `vigente` 0,88, menos convergida); las cinco de `kc3_s2`
  (18.000, `vigente` 0,99) no. Puede ser la BASE y no la semilla de datos: un modelo que ya resuelve
  la tarea simple a la perfección no tiene gradiente que lo empuje a usar la segunda lectura
  (la compuesta es 1 de cada ~4 preguntas con respuesta). Antes de decir «1 de 6» hay que separar
  base de semilla.

**E-6 (congelada, 17:35):** cuatro unidades más, **todas sembradas de `kc3_s1`** con otra semilla
de datos: `nf3_s3-s6` (`sembrar.py --semilla`), `lat2`, bloques 0,2, mismos flags, 8.000 pasos.
Predicción si es la base: ≥ 2 de 4 arrancan (≥ 0,60). Si 0 de 4, `nd3_s1` fue una sola corrida
con suerte y la tasa es ~1 de 10, y el mecanismo necesita otro diseño (p.ej. pesos propios para la
segunda lectura, o un currículo con más compuestas).

## 13. RESULTADO de E-6 (12-sep, 20:25) · ERA LA BASE: 5 de 5 desde kc3_s1, 0 de 7 desde kc3_s2

    unidad   base    semilla  paso   compuesta (final / máx)  vigente
    nd3_s1   kc3_s1  1        8000   0,76 / 0,79              0,981
    nf3_s3   kc3_s1  3        4000*  0,64 / 0,79              0,970   (*sin T4 para seguir; ya arrancó)
    nf3_s4   kc3_s1  4        8000   0,69 / 0,85              0,992
    nf3_s5   kc3_s1  5        8000   0,68 / 0,75              0,988
    nf3_s6   kc3_s1  6        8000   0,78 / 0,78              0,996
    ---
    nd3_s2-s6, ne3_s1-s2   kc3_s2   8000   0,10-0,40           1,000

La predicción de E-6 (≥ 2 de 4) se cumplió de sobra: 4 de 4. **La segunda lectura la aprende un
modelo que TODAVÍA no resuelve lo simple a la perfección** (kc3_s1: paso 14.000, `vigente` 0,88)
y no la aprende uno que ya lo resuelve (kc3_s2: 18.000, `vigente` 0,99). La hipótesis mecánica:
la compuesta es ~1 de cada 4 preguntas con respuesta; cuando la pérdida de lo simple ya es ~0, el
gradiente que queda es chico y las dos lecturas comparten `qr/kw/vw/wo`, así que moverlos para la
segunda cuesta en la primera. Con la base menos convergida, el mismo gradiente todavía mueve todo.
Es la misma familia que «en frío no arranca» (3 de 3 en `kd3`): hay una ventana de plasticidad, ni
en frío ni saturado. Lo que sigue (no congelado): pesos PROPIOS para la segunda lectura (qr2/wo2),
o subir `p_compuesta`, y medir si con eso arranca desde `kc3_s2`.

Y sigue abierto `nose_comp` (0,00-0,15 en todas): nadie dice «no sé» en la compuesta.

## 14. ENMIENDA E-7 (12-sep, 21:10) · pesos propios para la segunda lectura, y el diagnóstico de `nose_comp`

**Código.** `modelo.responder`/`responder_con_abst`: la `lectura` recibe el bloque y, si el módulo
trae `qr2`/`wo2`, la lectura que no es la primera los usa (claves y valores siguen compartidos:
son del archivo). `sembrar.py --lectura-propia` los agrega como COPIA de `qr`/`wo`, así el sembrado
arranca bit a bit igual al compartido (verificado: `ng3_s2` recién sembrado evaluado en 0,2 =
`kc3_s2` en 0,2, 0,5561 exacto) y el gradiente decide si los separa (smoke local de 4 pasos:
se mueven 2,3e-5). `tronco` mira la firma de `lectura`, así los instrumentos viejos de un
argumento siguen andando.

**E-7 (congelada antes de correr).** `ng3_s2, s3, s4`: sembradas de **`kc3_s2`** (la base
saturada, 0 de 7 con pesos compartidos), `--lectura-propia`, bloques 0,2, mismos flags que la v4,
8.000 pasos. Control: `nd3_s2-s6` (misma base, compartidos, 0,10-0,40). **Predicción:** si lo que
frena en la base saturada es que las dos lecturas se pelean por `qr/wo`, ≥ 2 de 3 arrancan
(`compuesta` ≥ 0,60). Si 0 de 3, no es la competencia de pesos sino la falta de gradiente, y el
camino es la base con plasticidad (o subir `p_compuesta`).

**Diagnóstico de `nose_comp` (medido, 16 lotes, mismos datos para los tres).** La cabeza «¿no sé?»
separa perfecto en las simples y NADA en las compuestas:

    unidad   AUC nose_comp vs compuesta   AUC nose_rel vs vigente   AUC nose_ent vs vigente
    nd3_s1   0,588                        1,000                     0,992
    nf3_s6   0,556                        1,000                     0,989
    nc3_s2   0,559                        1,000                     0,992

Y el logit medio es el mismo en las dos (−0,6 y −0,5): la cabeza no tiene un rasgo que diga
«el nombre no tiene hecho». No es un problema de umbral, es que la señal no existe en `h` al
final del tronco. En las simples la señal es que la lectura no encuentra nada que matchee (y el
logit sube a 11); en la compuesta la segunda lectura reparte entre los dos candidatos que quedan
y nada marca la ausencia del tercero. Candidato para la próxima: el slot nulo (`--abst slot`) en
la segunda lectura, que da un lugar explícito a «nada matchea». No se corre hoy.
