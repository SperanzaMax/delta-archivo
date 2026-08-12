# La curva del eje `m`: el fallo no es leer, y no crece con la ambigüedad

**2026-08-12.** Primera medición del eje de ambigüedad referencial sobre un sujeto que **pasa la
compuerta de sanidad** de §11 (`qwen2.5-coder`, 7B, respuesta por nombre, `d=5`, 20 casos por celda,
mismo material en las dos tareas de cada `m`).

---

> ## ⚠ CORRECCIÓN DEL MISMO DÍA — §2.2, §2.3 y §3.4 NO SE SOSTIENEN
>
> Al verificar el **texto crudo** de las respuestas contadas como abstención (`verificar_abstenciones.py`,
> el pendiente que §4 declaraba), resultó que **10 de 11 no eran abstenciones**: el modelo respondía
> con el nombre de la **persona** (el valor nuevo: `Rosa Belmonte`, `Nadir Haq`, `Elsa Moray`) en vez
> del de la **organización**. El parser buscaba nombres de entidad, no encontraba ninguno, y lo
> contaba como rechazo. Una sola respuesta era una abstención genuina:
> *«No correction needed for any of the entities listed…»*.
>
> **Causa:** la pregunta estaba mal formulada. *«Which one is being corrected?»* admite leerse como
> «¿qué cosa se está corrigiendo?», y bajo esa lectura «Rosa Belmonte» es una respuesta correcta.
> El defecto es del instrumento, no del sujeto.
>
> **Qué se cae:** todo lo que este informe dice sobre abstención — que a `m=1` «todo el fallo es
> negarse a contestar», que a `m=8` el modelo «se abstiene más y se equivoca menos», y que la
> abstención es señal medible de «sé que no sé». Nada de eso está sostenido.
>
> **Qué sobrevive, y es más específico:** el modelo **procesa la corrección** (identifica sin
> problema el valor nuevo) pero **no la liga a un sujeto**. Preguntado por a quién se refiere,
> devuelve el contenido de la corrección en lugar de su objetivo. Eso sigue siendo el fenómeno que el
> banco quiere medir, ahora mejor descrito.
>
> **Y un segundo problema, independiente:** una réplica accidental de la celda `m=4` (mismo modelo,
> misma tarea, otra semilla de material) dio **0,600** contra el **0,350** de este informe. Con 20
> casos el error estándar es ~0,11 y la diferencia no es significativa (p ≈ 0,11) — pero significa
> que **el ruido es del tamaño de los efectos leídos acá**, y que el «escalón entre `m=2` y `m=4`»
> de §2.3 puede no existir.
>
> Re-medición en curso con la pregunta reparada, `NONE` ofrecido explícitamente (para que la
> abstención sea una respuesta registrable y no una inferencia del parser), clasificación en cuatro
> categorías y 3 semillas por celda: `curva_m2.py`.
>
> **Quinta vez en el programa que un número limpio esconde un artefacto — y la primera en que el
> artefacto es del instrumento propio, no del sujeto ni del análisis.**

---

## 1. La curva

| `m` | extracción (compuerta) | resolución | abstenciones | **errores reales** | azar |
|---|---|---|---|---|---|
| 1 | 1,000 | 0,650 | 7/20 | **0** | 1,000 |
| 2 | 1,000 | 0,700 | 1/20 | 5 | 0,500 |
| 4 | 1,000 | 0,350 | 5/20 | 8 | 0,250 |
| 8 | 1,000 | 0,350 | 8/20 | 5 | 0,125 |

## 2. Tres cosas que la curva dice y que el diseño no anticipaba

### 2.1 La carga de lectura no interviene: la compuerta da 1,000 en todo el eje

Con **ocho** entidades activas y cinco turnos de relleno —trece hechos en la ventana— el modelo sigue
identificando sin un solo error cuál tiene un director mencionado. La extracción no se degrada nada
en todo el rango.

Eso deja el eje limpio: **lo que cae al subir `m` no es la capacidad de leer la ventana**, es
específicamente ligar una corrección sin sujeto a un hecho que el modelo localiza sin esfuerzo. Es la
atribución que el chequeo del 11-ago no podía hacer, y ahora está hecha en los cuatro puntos.

### 2.2 `m = 1` no es el caso fácil: es el caso que revela el rechazo

Con una sola candidata en la lista **equivocarse de entidad es imposible**. Y aun así el modelo saca
0,650. Los 13 aciertos y las 7 abstenciones cierran los 20 exactos: **cero errores, todo el fallo es
negarse a contestar**.

Con la única opción posible delante, el modelo se rehúsa a ligar «no, it's X» al único hecho que hay.
Eso no es ambigüedad —no hay nada que desambiguar—: es que **la forma elíptica por sí sola ya bloquea
la resolución**, incluso sin competencia. El eje `e` (grado de elipsis) tiene efecto propio,
independiente del eje `m`, y este es el primer dato que lo separa.

### 2.3 El eje satura entre `m=4` y `m=8`, y lo que cambia es el MODO DE FALLO

`m=4` y `m=8` dan la **misma** resolución (0,350). Duplicar las entidades competidoras no agrega
dificultad. Lo que cambia es cómo se falla:

| | abstenciones | errores reales |
|---|---|---|
| `m=4` | 5 | **8** |
| `m=8` | **8** | 5 |

Con más entidades en juego el modelo **se abstiene más y se equivoca menos**. Es contraintuitivo y es
la mejor noticia del día para el diseño del banco: el error silencioso —responder con confianza una
entidad equivocada— **no es monótono en la ambigüedad**, tiene su máximo en el medio del eje.

Un banco que reportara sólo `recall` vería una meseta plana entre `m=4` y `m=8` y concluiría que el
eje se agotó. Al desagregar aparece que las dos celdas son cualitativamente distintas: una envenena
la memoria y la otra no. **Es el argumento de `SER` medido, no argumentado.**

Nota sobre el azar: como el azar cae de 0,250 a 0,125 mientras la resolución se queda en 0,350, la
ventaja relativa **crece** (1,4× a `m=4`, 2,8× a `m=8`). Otra razón para no leer la meseta como
agotamiento del eje.

## 3. Consecuencias para ECO

1. **El rango útil del eje `m` está entre 1 y 4.** `m=8` no agrega dificultad de recall; se conserva
   porque cambia la composición del error, que es justamente lo que el banco quiere medir. Si hubiera
   que recortar por costo, se recorta `m=8` **antes** que `m=2`.
2. **`SER` deja de ser una métrica auxiliar y pasa a ser la principal.** Es lo único que distingue
   `m=4` de `m=8`. Refuerza la reformulación del barrido de literatura: la desagregación por tipo de
   error es el aporte, no la idea de penalizar el error.
3. **Los ejes `e` y `m` son separables y hay que cruzarlos.** El 0,650 de `m=1` prueba que la elipsis
   tiene efecto sin competencia alguna. El diseño ya los tenía como ejes distintos; ahora hay
   evidencia de que no son redundantes.
4. **La abstención es señal, no ruido.** Va como categoría propia en el reporte, nunca sumada a los
   errores. Puente con el plan del modelo que sabe que no sabe: acá hay un sujeto que **sabe cuándo
   no puede**, y lo hace más a menudo cuanto más ambigua es la situación.

## 4. Límites de esta medición

- **Un solo sujeto y 20 casos por celda.** Sirve para dimensionar el eje, no para publicar. La
  campaña necesita semillas y más de un modelo.
- **`d = 5` fijo.** El eje de distancia no se tocó.
- **Un solo tipo de elipsis** (`e4`, total). Los niveles intermedios del eje `e` siguen sin medir.
- Las abstenciones se cuentan como respuesta no interpretable; **no se verificó el texto crudo** de
  esas 21 respuestas para confirmar que son rechazos genuinos y no fallos de formato. Pendiente
  barato y necesario antes de construir `SER` sobre ellas.

**Costo:** 320 consultas, ~1 h 20 de CPU local, máx 57 °C.
