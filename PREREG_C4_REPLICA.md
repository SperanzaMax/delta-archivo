# PREREG · La réplica de `c4_s2` en `c4_s0` y `c4_s1`

Congelado el 2026-08-20 antes de lanzar los tramos, después de cerrar
`INFORME_C4S2_20260820.md`. Es una **réplica declarada**: mismo tratamiento, mismos criterios, dos
unidades que no se eligieron por sus resultados sino por ser las que faltan del nivel 4.

---

## 1 · Por qué

`c4_s2` pasó de fallar la compuerta a 14000 pasos (`falsa_abst` 0,1927) a pasarla holgada a 20000
(0,0170), sin tocar nada más que el presupuesto. Eso deja una pregunta que el informe de hoy declaró
explícitamente y no puede responder con una unidad:

**¿cuánto de «la cabeza no alcanza en tarea difícil» era en realidad «la unidad no había terminado de
aprender»?**

De la respuesta dependen dos cosas ya escritas: D-3 del `INFORME_CELDA_DIFICIL_20260819.md` («primera
unidad de la serie donde `cabeza` pierde») y el confound abierto del §4 del
`INFORME_FRONTERA_20260819.md`, donde vía y margen están confundidos porque **todos** los puntos del
hueco son sub-entrenados.

## 2 · Diseño

Idéntico al de `PREREG_C4S2_PRESUPUESTO.md` (SHA `8446a27e…`), aplicado a **`c4_s0` y `c4_s1`**:
reanudar desde su checkpoint de 14000 hasta **20000 pasos**, `--cada 250` → 24 puntos nuevos por
unidad. Config heredada de cada corrida, con `horizonte 20000` ya fijado desde el origen, así que la
curva de lr no cambia.

**Los checkpoints de 14000 se copian a `.p14000` ANTES de lanzar** — el tramo sobrescribe el `.pkl` y
sin la copia no hay con qué comparar. Esta vez la copia se hace de entrada y no como rescate, que es
la lección D-1 del día.

**Y ninguna sonda va a leer estos checkpoints mientras se entrenan.** Es la otra mitad de D-1: hoy un
análisis midió una unidad que se estaba entrenando al mismo tiempo. Mientras estos tramos corran, no
se corre ningún análisis sobre la familia `c` de nivel 4.

## 3 · Predicciones

Las mismas cuatro, por unidad y **reportadas por unidad, nunca la media**:

- **R-1.** `falsa_abst` medida con **2048 muestras** (rng 77000+semilla) baja de 14000 a 20000 en
  **ambas** unidades. Ésta es la predicción principal y es la que replica lo de `c4_s2`.
- **R-2.** Ninguna de las dos muestra tendencia creciente en la serie de 24 puntos
  (Spearman < +0,41 o p ≥ 0,05, con rangos promediados).
- **R-3.** Alguna de las dos **pasa la compuerta a 20000** (`falsa_abst` ≤ 0,10 y `nose` ≥ 0,50)
  habiéndola fallado a 14000.
- **R-4 (control de sanidad, puede fallar).** `vigente` no cae más de 0,10 en ninguna.

## 4 · Desenlaces, comprometidos por adelantado

- **R-1 y R-3 cumplen en las dos** → lo de `c4_s2` no era una unidad rara: **en nivel 4 la compuerta
  se falla por presupuesto**. Hay que revisar D-3 de la celda difícil y declarar que el confound del
  §4 de la frontera es peor de lo que decía ese informe, porque el sub-entrenamiento **produce** los
  fallos que se atribuían a la dificultad.
- **R-1 cumple en una sola** → mejora sin ser regla; se reporta como tal y `c4_s2` queda como caso, no
  como ley.
- **R-1 falla en las dos** → `c4_s2` era la excepción, las conclusiones del 19-ago quedan **en pie**, y
  lo de hoy se reduce a una unidad que necesitaba más pasos.

## 5 · Alcance

Dos unidades, una semilla cada una, misma tarea y misma condición. Sigue sin decir nada sobre `token`
ni `escala`: si el presupuesto explica los fallos de `cabeza`, la comparación pareada entre
condiciones **también** habría que rehacerla con todas entrenadas a fondo, y eso es una campaña
aparte que este prereg no autoriza.
