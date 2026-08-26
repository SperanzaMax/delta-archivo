# INFORME · calibrar (A3) y ensamble de semillas (A4)

**EXPLORATORIO Y DECLARADO COMO TAL.** Sin pre-registro, igual que `sonda_umbral.py` del 18-ago.
**No confirma nada.** Son los dos contrastes que le ponen piso y techo a cualquier detector propio y
que este proyecto nunca había corrido. Tres unidades `p3_*`, n = 6000 de ajuste + 6000 de prueba,
CPU, sin tocar el pool.

---

## A3 · Calibrar cierra tres cuartos de la brecha, y es gratis

El umbral se elige **en la muestra de ajuste** pidiendo margen (`falsa_abst` ≤ 0,07) y se juzga en la
de prueba con el criterio real (≤ 0,10). Es la lección del 19-ago aplicada: *el óptimo pegado al
borde del criterio no generaliza*.

| unidad | `nose` con σ>0,5 (lo que se usa hoy) | **calibrado** | oráculo (mira las etiquetas de prueba) | brecha cerrada |
|---|---:|---:|---:|---:|
| `p3_s0` | 0,9156 | **0,9509** | 0,9609 | **78 %** |
| `p3_s1` | 0,5366 | **0,5928** | 0,6136 | **73 %** |
| `p3_s2` | 0,7080 | **0,7471** | 0,7596 | **76 %** |

`falsa_abst` queda en 0,068 · 0,071 · 0,080, o sea dentro del criterio en las tres.

> **Calibrar sube la detección entre +0,035 y +0,056 sin costar nada**, y captura ~75 % de lo que
> alcanza un oráculo con acceso a las etiquetas de prueba. Confirma lo que el techo del 18-ago ya
> decía —AUC 0,777-0,998, o sea la información está y el corte está mal puesto— y lo convierte en
> una mejora concreta sobre checkpoints que ya existen.

No cambia ningún veredicto de la compuerta: las tres ya pasaban con σ>0,5. Lo que cambia es cuánto
detecta el modelo al mismo costo de falsas abstenciones.

---

## A4 · El ensamble gana donde el modelo es malo, y pierde donde es bueno

El acuerdo se mide sobre el **argmax** de cada unidad, no sobre su predicción final: `pred` ya
incorpora la decisión de la cabeza, y meterla ahí haría que el ensamble midiera en parte su propio
detector. Blanco limpio (`argmax != tgt`, sin mirar la cabeza).

| unidad de referencia | su acierto | **acuerdo (ensamble)** | confianza | cabeza |
|---|---:|---:|---:|---:|
| `p3_s0` (la mejor) | 0,5862 | 0,7464 | 0,9072 | **0,9612** |
| `p3_s1` (la peor) | 0,4890 | **0,8339** | 0,7795 | 0,7128 |
| `p3_s2` | 0,5402 | 0,7876 | 0,7733 | **0,8148** |

**El ensamble gana en la semilla peor por +0,1211 sobre la cabeza, queda segundo por 0,0272 en la
del medio, y pierde por 0,2148 en la mejor.** Es el mismo patrón que la ganancia de sumar la
confianza: **rinde donde el detector propio es flojo y sobra donde ya es bueno.**

### Una corrección propia, y es la segunda vez hoy que el blanco decide el signo

El primer cálculo de A4 usó el blanco **contaminado** de `clasificar` (el mismo defecto D-D3 del otro
informe) y daba lo contrario: acuerdo 0,6737 contra cabeza 0,4056, o sea «el ensamble gana cómodo».
Con el blanco limpio la cabeza pasa de 0,4056 a 0,9612 en la misma unidad. **La contaminación no
atenuaba el resultado: lo invertía.**

Y hubo un segundo sesgo, propio y distinto: el primer análisis comparó el ensamble **sólo contra
`u0`**, que resultó ser la mejor semilla, y de ahí salía «el ensamble pierde claramente». Evaluado
contra las tres, no pierde claramente: depende de cuál sea la referencia.

### Lo que el ensamble NO sirve para hacer

**Su moda acierta menos que la mejor unidad sola: 0,5622 contra 0,5862.** Votar entre semillas
**arrastra la buena hacia abajo**, que es exactamente lo que la advertencia declarada por adelantado
anticipaba: en este banco las semillas difieren en **capacidad** y no sólo en ruido (bimodalidad
medida desde E-I3c). Sirve como detector, no como método de respuesta.

### Límite del instrumento, no del método

Con tres unidades el acuerdo toma **sólo tres valores** (0,333 · 0,667 · 1,000), así que su AUC está
topeado por granularidad. Un ensamble más grande tendría más resolución. **El número de acá es un
piso, no el techo del método** — y aun así triplica el costo de inferencia, que es lo que lo hace
poco atractivo salvo como contraste.

---

## Lectura conjunta

Las dos vías apuntan al mismo lugar y ninguna necesita arquitectura nueva:

1. **La información para detectar está**, y buena parte de lo que falta es **poner bien el corte**
   (A3, +0,04 a +0,06 gratis).
2. **Lo que se le agrega al detector rinde en proporción a lo malo que sea** — la confianza de salida
   (+0,016 a +0,109) y el ensamble (−0,215 a +0,121) tienen los dos ese patrón.

Y las dos correcciones propias de hoy dejan una regla que vale más que los números: **en este banco,
un blanco mal definido no atenúa el efecto, le cambia el signo.** Pasó dos veces en la misma jornada,
en dos experimentos distintos.
