# ENMIENDA · `PREREG_ARCHIVO_LARGO` (SHA `c769a4ef`) · 2026-09-05, tarde

**Se escribe DESPUÉS de un piloto en T4 y ANTES de abrir la campaña.** El prereg original queda
intacto; esto se le adjunta, siguiendo el precedente de `ENMIENDA_FILTRADO_PREVIO.md` y
`ENMIENDA_DISTANCIA_REAL.md`. **Lo único que cambia del diseño es el presupuesto de pasos, y cambia
por una velocidad medida, no por un resultado.**

## 1. El piloto, y por qué existió

Hoy dos scripts de campaña habrían abortado en su primera VM, así que antes de abrir seis unidades se
corrió **un tramo de 250 pasos de `lg3_s0` en una T4 real**. Encontró un tercer problema, y éste no
era de script.

**OOM.** Con `--ses-extra 26` el archivo pasa de 4 a 40 sesiones y el forward procesa `B × S`
secuencias: a batch 64 son 2560 en vez de 256. La T4 muere con
`RESOURCE_EXHAUSTED: Out of memory while trying to allocate 63.73GiB [executable_name='jit_predecir']`.

**La campaña no puede correr a batch 64 en T4**, y bajar el batch efectivo cambiaría la optimización y
rompería la comparación contra el control, que hereda su historia de `kq3_sX` entrenado a 64.

## 2. Cambio 1 · micro-lotes, para no tocar el batch efectivo

Se agrega `--micro-batch`: el paso se parte en trozos y **se promedian los gradientes**. Como la
pérdida es una media y los trozos son del mismo tamaño, el resultado es el mismo salvo redondeo.

**Verificado antes de usarlo**, misma semilla y 6 pasos, batch 32 entero contra micro-lotes de 8:
**máx |diferencia| de los pesos 1,49e−08 relativo**, y las cuatro métricas idénticas en todos sus
dígitos. Y `--batch-eval`, porque el OOM fue en la evaluación: baja el batch de evaluación y sube `n`
para que la cantidad de preguntas evaluadas no cambie (8 × 64 = 64 × 8 = 512).

La campaña corre con **`--micro-batch 8 --batch-eval 8`** en las dos condiciones. **Esto no es una
desviación del diseño**: el batch efectivo sigue siendo 64, que es lo que el prereg fija.

## 3. Cambio 2 · el presupuesto baja de 6000 a 2000 pasos

**Esto sí es una desviación, y se declara.** El presupuesto del prereg salió de un ratio medido en
CPU (6,96×) aplicado a 0,22 s/paso, y daba ~1,53 s/paso y ~2,5 h por unidad tratada. **En la T4 real,
con micro-lotes, el tramo de 250 pasos tardó ~25 min de reloj**, o sea **~5 s/paso**: los ocho
micro-pasos se serializan y la GPU deja de estar saturada. A esa tasa, 6000 pasos son **8-10 h por
unidad** y la campaña entera no entra en un día de pool.

Se baja a **2000 pasos**, con **`--horizonte 6000` intacto**, que es exactamente para lo que el
horizonte existe: «hasta dónde corro hoy» separado de «sobre cuántos pasos decae la lr». Parar en 2000
y extender después no cambia la curva — verificado bit a bit el 14-ago.

## 4. Lo que el piloto midió, dicho como corresponde

El tramo de 250 pasos **se leyó**, así que no se puede presentar como ciego. Sus números:

| | archivo largo (300 casilleros) | archivo corto (40) |
|---|---:|---:|
| vigente | **0,6705** | 0,9676 |
| anterior | 0,8384 | 1,0000 |

Contra la línea de partida de la misma unidad, medida antes de entrenar: **0,1820** en largo y 1,0000
en corto. El piso trivial es 0,4065.

**Con 250 pasos, L-1 y L-3 ya cumplirían.** Eso es una señal fuerte de que el colapso del archivo
largo es de **transferencia** y no un techo del mecanismo — pero **no es la campaña**: es una semilla,
un tramo, y sobre todo **sin el control**, que es lo único que separa «entrenar con archivo largo» de
«entrenar 250 pasos más». **L-2 sigue siendo bloqueante.**

**El piloto NO se descarta ni se re-corre:** `lg3_s0` continúa desde el paso 250 con la misma
configuración, y esos 250 pasos son su primer tramo. Lo que se declara es que **sus números fueron
vistos antes de fijar el presupuesto de 2000**. La decisión de bajar el presupuesto se justifica por
la velocidad medida y se habría tomado igual con el piloto en cero; queda escrito acá para que quien
lea pueda descontarlo.

## 5. Lo que NO cambia

- **Ningún criterio ni umbral.** L-1 a L-5 quedan como están, incluido el 0,20 de brecha real−barajado
  de L-4, que es la pregunta de fondo y sigue sin tocarse.
- **El diseño:** 3 semillas × 2 condiciones, siembra desde `kq3_s0/s1/s2`, `--ses-extra 26` contra 0.
- **La cláusula de abandono del §5** y **el §6 de lo que la campaña no puede decidir.**
