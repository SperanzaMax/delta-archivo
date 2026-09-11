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
