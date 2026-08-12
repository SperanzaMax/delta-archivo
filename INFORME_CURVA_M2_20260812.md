# La curva del eje `m`, con el instrumento reparado: dos causas, no una

**2026-08-12, segunda medición.** Reemplaza la lectura de `INFORME_CURVA_M_20260812.md`, cuyo
análisis de abstención resultó ser un artefacto del parser y de una pregunta mal formulada (ver la
corrección al principio de aquel informe).

Sujeto `qwen2.5-coder` (7B), `d=5`, **20 casos × 3 semillas** por celda, pregunta que nombra el tipo
de respuesta esperado, **`NONE` ofrecido explícitamente**, y clasificación en cuatro categorías en
vez de dos. Script: `curva_m2.py`.

---

## 1. La curva

| `m` | acierto (media ± sd entre semillas) | **errores /60** | **abstenciones /60** | fuera de dominio /60 |
|---|---|---|---|---|
| 1 | **0,667 ± 0,076** | **0** | 19 (31,7 %) | 1 |
| 4 | **0,450 ± 0,100** | 11 (18,3 %) | 19 (31,7 %) | 3 |
| 8 | **0,250 ± 0,087** | 17 (28,3 %) | 25 (41,7 %) | 3 |

Monótona, con desviaciones entre semillas de ~0,08-0,10. La separación `m=1` vs `m=8` es de **0,417**,
grande frente a esa dispersión. En la v1 las celdas `m=4` y `m=8` daban idéntico 0,350: era el
instrumento, no el fenómeno.

## 2. El resultado: el recall cae por dos causas distintas, y sólo se ven desagregadas

### 2.1 La elipsis produce abstención, y no le importa cuántas entidades haya

`m=1` tiene **una sola candidata en la lista**: equivocarse de entidad es imposible por construcción.
Y aun así el acierto es 0,667. Los 0 errores en 60 casos lo dicen entero: **todo el déficit es el
modelo escribiendo `NONE`**.

La tasa de abstención es 31,7 % a `m=1`, 31,7 % a `m=4` y 41,7 % a `m=8`. La suba del último punto
son ~10 puntos porcentuales con un error estándar de ~6: **plana dentro del ruido**.

> Con un único hecho posible en la ventana, el modelo se niega a atribuirle la corrección un tercio
> de las veces. La forma elíptica bloquea la resolución **sin ninguna ambigüedad que la ayude**.

Eso vuelve independiente al eje `e` del diseño: no es un caso particular del eje `m` con pocas
entidades. Tiene efecto propio y medible en el punto donde el otro eje está anulado.

### 2.2 La ambigüedad produce error silencioso, y ahí sí escala

Los errores —elegir una organización de la lista, con confianza, y que sea la equivocada— van
**0 → 11 → 17** de 60. Es el único componente que crece con `m`.

Esa es la métrica que justifica todo el argumento de `SER`: un sistema que se abstiene es manejable;
uno que atribuye una corrección a la entidad equivocada **planta un hecho falso en la memoria** y
todo lo que venga después hereda el error. Con 8 entidades activas, casi un tercio de las
correcciones termina así.

### 2.3 Por qué esto obliga a desagregar

Un banco que reporte sólo `recall` ve una única curva que baja de 0,667 a 0,250 y concluye «se pone
más difícil». Desagregado se ve que **son dos fenómenos con causas distintas**: uno constante en `m`
(la elipsis) y otro creciente (la competencia). Un sistema podría mejorar el recall reduciendo
abstenciones y **empeorar** en lo que importa, aumentando los hechos falsos.

Es el argumento de la §4.3 del diseño, ahora medido en vez de argumentado.

## 3. Un hallazgo lateral que hay que confirmar bien

Entre v1 y v2, los errores de la celda `m=4` pasan de **8/20** a **1-7/20** (11/60). La diferencia
apunta a que **ofrecer explícitamente la opción de abstenerse reduce el error silencioso**: dado un
camino de salida, el modelo lo toma en vez de arriesgar una atribución.

Si se confirma, es directamente accionable para sistemas de memoria: **un índice al que se le permite
decir «no sé» envenena mucho menos la memoria que uno forzado a elegir.**

**Pero no está aislado:** entre v1 y v2 cambiaron dos cosas a la vez —la redacción de la pregunta y
la disponibilidad de `NONE`—. El contraste limpio es pregunta reparada **sin** `NONE` contra pregunta
reparada **con** `NONE`, mismo material. Es barato y queda como el próximo experimento.

## 4. Lo que esta corrida deja para el diseño de ECO

1. **Las cuatro categorías reemplazan a las dos.** `acierto / error / abstención explícita / fuera de
   dominio`. La cuarta no es cosmética: es la que detectó el artefacto de la v1.
2. **Ofrecer `NONE` es parte del protocolo**, no una opción. Sin él, la abstención hay que inferirla
   del fracaso del parser — y ahí fue donde nos equivocamos.
3. **3 semillas es el mínimo.** Con una sola, la misma celda dio 0,350 y 0,600 en dos corridas.
4. **`m=8` se conserva.** En la v1 parecía redundante con `m=4` (mismo 0,350); con el instrumento
   bueno los separa 0,200 en acierto y 6 errores.

## 5. Límites

- **Un solo sujeto.** Todo esto es `qwen2.5-coder` 7B, el único de cinco candidatos locales que pasa
  la compuerta de extracción. El corte de admisión es abrupto (los otros cuatro quedan en el azar),
  así que no hay forma barata de replicar con un segundo modelo.
- **`d = 5` y `e = e4` fijos.** Los ejes de distancia y de grado de elipsis siguen sin medir.
- 60 casos por punto siguen siendo pocos para diferencias menores a ~0,15.

**Costo:** 180 consultas, ~1 h 40 de CPU local.
