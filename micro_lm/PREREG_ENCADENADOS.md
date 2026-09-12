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

## 6. ENMIENDA E-2 (12-sep, 08:40) · H0 FALLÓ, y la intervención dice por dónde pasó la información

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
