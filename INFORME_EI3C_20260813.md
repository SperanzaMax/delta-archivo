# E-I3c: el sello aporta el orden, y un sello mentiroso es peor que ninguno

**2026-08-13.** Pre-registro en el docstring de `interno/ei3c_orden_limpio.py`, hash fijado antes de
correr (`interno/PREREG_EI3C_HASH.txt`, sha256 `1e52ae91…`). Semillas extra en
`ei3c_semillas_extra.py`, con su criterio también escrito antes.

Este experimento existe porque **E-I3b se rompió solo**. Para tener tres versiones de una clave, ahí
las dos revisiones se escribían en la misma secuencia, y el estado que el modelo produce en la
posición de v3 arrastra el rastro de v2: el orden quedaba escrito en el **contenido** del vector, sin
que ningún metadato lo pusiera. A 4000 pasos no se veía (nadie resolvía la pregunta difícil); con
12000, la condición **sin sello** llegaba a 0,97 y delataba la fuga.

Acá cada versión se escribe en su **propia secuencia** —S1 v1, S2 v2, S3 v3, S4 la consulta— con el
estado reseteado entre ellas. Para el modelo, v2 y v3 son «primera y única mención de su secuencia»,
igual que v1: sus estados son indistinguibles en cuanto a recencia. Lo único en todo el sistema que
dice cuál vino antes es el sello.

---

## 1. Resultado (3 semillas, 12000 pasos, inyección temprana)

| condición | vigente | **ANTERIOR** | una sola versión |
|---|---|---|---|
| `ninguno` | 0,3090 | **0,2917** ± 0,0207 | 0,9436 |
| `sello` | 0,9774 | **0,8316** ± 0,2399 | 0,9974 |
| `barajado` | 0,2682 | **0,2804** ± 0,0159 | 0,8802 |

| predicción | veredicto |
|---|---|
| **P-1** (bloqueante) ANTERIOR(ninguno) ≤ 0,40 — la fuga está tapada | **CUMPLE** — 0,2917 |
| **P-2** ANTERIOR(sello) ≥ 0,80 | **CUMPLE con 3 semillas** — 0,8316 · **ver §3** |
| **P-3** sello − barajado ≥ +0,30 | **CUMPLE** — **+0,5512** |

## 2. La fuga está tapada, y se nota en el número exacto

`ninguno` da 0,3090 en la versión vigente y 0,2917 en la anterior: **1/3**, el azar entre las tres
versiones archivadas. Y al mismo tiempo identifica bien el ítem cuando no hay conflicto (0,9436 en
claves con una sola versión).

Es el mismo modo de falla de R1/R4 y de E-I2, ahora con tres versiones en vez de dos: **la similitud
encuentra el hecho y no sabe cuál versión rige**. Comparado con el 0,97 que daba esta celda en E-I3b,
la diferencia mide exactamente cuánta información de orden se colaba por el contenido.

## 3. El número de `sello` depende del presupuesto, y hay que decirlo así

Con 3 semillas, ANTERIOR(sello) = 0,8316 y P-2 cumple. **Con 5 semillas baja a 0,7667 (sd 0,2340) y
P-2 no cumple**: 0,9766 · 0,9635 · 0,5547 · 0,8594 · 0,4792.

La causa no es que el mecanismo falle, sino que **a 12000 pasos dos de cada cinco semillas todavía no
convergieron**. Se probó directamente: la semilla 2, extendida a 24000 pasos, pasó de **0,5547 a
0,9531**.

> El veredicto honesto es condicional: **el sello permite responder por la versión anterior, con un
> presupuesto que 12000 pasos no siempre alcanzan.** El número limpio exige correr las cinco semillas
> a 24000, y hasta entonces P-2 queda declarado como «cumple con presupuesto suficiente, no cumple al
> presupuesto corrido».

Es la sexta vez en el programa que el presupuesto decide el signo de un resultado. Ya no es un
accidente: en este brazo, **el tiempo de convergencia es parte del fenómeno**, no un detalle de
implementación.

## 4. El hallazgo lateral: un sello mentiroso es peor que no tener sello

`barajado` lleva los **mismos valores de turno**, desordenados. No sólo no ayuda:

- deja la versión vigente en **0,2682**, *por debajo* del 0,3090 de no tener sello;
- y degrada la identificación del ítem, **0,8802 contra 0,9974**, que es la única función que
  funcionaba bien en todas las condiciones.

O sea: el lector **se apoya** en el sello, no lo ignora. Cuando el metadato miente, arrastra consigo
la parte que la geometría sí resolvía. Para un sistema desplegado esto es una advertencia concreta:
**un timestamp corrupto no es información faltante, es información dañina**, y es peor que no tener
ninguno.

## 5. Lo que este experimento NO puede afirmar

En E-I3c los turnos son **fijos por rol**: v1 se lleva siempre los turnos 0-5, v2 los 6-8, v3 los
9-11. Al modelo le alcanza con aprender una tabla «los embeddings 9,10,11 son la vigente; 6,7,8 la
anterior», sin ninguna noción de que 9 viene *después* de 6. Y `barajado` **no separa** las dos
lecturas: al aleatorizar el sello rompe la tabla exactamente igual que rompe el orden.

Por eso la afirmación que sostiene este informe es **«el sello aporta la información que falta, y no
es capacidad extra»** — no «el lector usa el orden como orden». Eso lo mide E-I3d, con turnos que se
mueven de muestra en muestra.

## 6. Costo

9 corridas de 12000 pasos + 2 extra + 1 de 24000 ≈ 4 h de CPU local, PC entre 45 °C y 47 °C.
