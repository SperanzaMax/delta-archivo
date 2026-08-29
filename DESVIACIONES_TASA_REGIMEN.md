# DESVIACIONES · `PREREG_TASA_REGIMEN.md` (SHA `dc62ecae`)

**2026-08-29.** Se escriben antes de juzgar nada, como corresponde.

---

## D-1 · `b3_s7` y `b3_s8` NO se corren hasta 26000

**Lo que el pre-registro pedía:** seis unidades `b3_s3` … `b3_s8` a 26000 pasos, juzgadas con
`ser_cobertura.py` campo `propio`, n=4000, semilla 54321.

**Lo que se hizo:** `s3` y `s6` se terminaron a 26000. `s4` y `s5` quedaron donde estaban (19500 y
6000). **`s7` (13500) y `s8` (8000) se detuvieron y no se completan.** Son 30500 pasos no corridos de
los 63000 que faltaban.

**Por qué.** Los dos criterios que el pre-registro fija están **fallados por aritmética antes de
correr un paso más**, y no por pronóstico sobre cómo evolucionarían:

- **T-0** pide `vigente` ≥ 0,70 en **≥ 4 de 6**. Cuatro unidades (`s3 s6 s7 s8`) están en abstención
  total con `vigente` 0,0000, y su recuperación medida sin la cabeza es 0,31–0,40, o sea que **el
  techo de `vigente` si la cabeza abriera entera seguiría por debajo de 0,70**. Quedan dos unidades
  para cuatro plazas.
- **T-1** pide el régimen (`nose` ≥ 0,99 **y** `falsa_abst` ≤ 0,01) en **≥ 3 de 6**. Las cuatro mudas
  tienen `falsa_abst` **1,0000**, así que fallan el segundo término aunque su `nose` valga 1,0000.
  Quedan dos unidades para tres plazas.

Terminar `s7` y `s8` no puede mover ninguno de los dos. **Es cómputo sin información**, y son 30500
pasos de cuota de cuentas prestadas.

**Qué se pierde, dicho sin adornos.** Con `s7` y `s8` a 26000 se tendrían dos confirmaciones más del
predictor del paso 2500 a horizonte completo. Se acepta el costo porque `s3` y `s6` **sí** se llevaron
a 26000 y son las mudas **más viejas** de las cuatro, o sea el test más duro disponible: si el
predictor se cae, se cae ahí antes que en `s7` o `s8`.

**Qué NO se hace con esto.** No se rescata ningún criterio ni se re-lee T-1 con las unidades que
quedaron. **El veredicto de `PREREG_TASA_REGIMEN` es «T-0 falla»**, y la celda que el propio §5 del
pre-registro escribió antes dice qué significa: *el régimen de entrenamiento no es reproducible, no se
lee nada más y se investiga eso primero*. Eso es exactamente lo que hace
`PREREG_ATRACTOR_MUDO.md`.

---

## D-2 · `s4` y `s5` quedan sin terminar

Misma razón y menos consecuencia: con T-0 fallado no hay lectura que hacer con ellas. `s4` llevaba
desde el paso 17000 oscilando alrededor de 0,66 —el perfil con el que `s2` terminó en 0,6503— así que
tampoco había margen para que cambiara el conteo.

Se anota que **`s5` es la única unidad viva de la campaña que quedó incompleta** (6000 de 26000). Si
alguna vez hace falta una unidad viva más de esta familia, ese checkpoint está en disco y se retoma
con los mismos flags, pero **no cuenta para este pre-registro**.

---

## D-3 · el comando del §6 del ESTADO del 28 no lanzaba

No es una desviación del diseño sino de la ejecución, y se anota porque volvería a morder.

El comando escrito para relanzar usa `env $COM ... LOG_ROTADOR=$PWD/rot_....log` **sin comillas**, y
esta ruta tiene un espacio (`Nuevo Transformer`). `env` recibe la ruta partida en dos, no encuentra el
segundo pedazo y sale con **exit 127 sin lanzar nada** — y como el redirect ya había creado el `.log`,
el rastro parece el de un rotador que arrancó. Se reemplazó por `lanzar_fase1_0829.sh`, que entrecomilla.

Es la misma familia de error que `pc-termica-cargas-largas` ya tenía anotada para los heredocs.
