# ENMIENDA · `PREREG_ABSTENCION.md` · 2026-09-10

Se escribe **antes** de lanzar la campaña de hoy. Ninguna hipótesis cambia. Cambian el instrumento
con que se miden los controles, la cantidad de semillas y una celda que faltaba.

## E-1. EL INSTRUMENTO. Los controles se venían midiendo sobre CUATRO muestras

Los cuatro brazos de control (`con archivo`, `BARAJADO`, `VACIO`, `lectura APAGADA`) se corrían
sobre el **último lote de entrenamiento**, que con `--batch 4` son cuatro muestras. Verificado en las
siete unidades que tienen controles guardados, todas dicen `n=4`. La consecuencia está a la vista en
los datos ya cerrados:

| unidad | BARAJADO |
|---|---|
| `ver_s0` | 0,2500 |
| `verlargo_s0` | 0,0000 |
| `verlargo_s1` | 0,0000 |
| `verlargo_s2` | 0,0000 |

**La diferencia entre un control en cero y uno en 0,2500 es UNA muestra.** Un control observado
0 de 4 no distingue el cero verdadero de una tasa del 25 %, y con tres semillas son 12 muestras, que
al 95 % siguen siendo compatibles con hasta un 26 %. La frase «controles en 0,0000 exacto en las
tres» describe bien lo que se observó y **no** alcanza para afirmar que el brazo barajado no lee
nada.

Lo mismo explica los `NaN` que aparecen por todas las curvas: con `p_nose_rel` 0,20 y cuatro
muestras por hito, la mitad de los hitos no tiene ni un caso de la clase.

**El arreglo, que es barato.** Al terminar la corrida se sortean `--eval-n` muestras frescas (512
por defecto) con un generador **propio**, y los cuatro brazos se evalúan sobre **las mismas
muestras**. Cuesta menos de un minuto al final de una corrida de una hora y sube el `n` de los
controles de 4 a 512.

**Verificado antes de lanzar, y era la parte que podía arruinar la campaña:**

1. El generador del entrenamiento no se toca. Dos corridas de seis pasos con la misma semilla, una
   con el script de ayer y otra con el de hoy, dan `hist` **idéntico campo por campo**. Lo único que
   difiere es `s_paso`, que es el reloj, y los `NaN`, que no son iguales a sí mismos por definición.
2. El bloque nuevo va dentro de un `try`. Una evaluación que explote no puede llevarse por delante
   el volcado de una corrida de setenta minutos que ya terminó.

**Alcance retroactivo.** Lo que sale de las **curvas** promediadas (el `vigente` 0,9399 ± 0,0057 de
las versiones, la ablación en 0,5585, el acierto 0,68 de la abstención) se apoya en cientos de
muestras y no se toca. Lo que sale de los **controles** de fin de corrida se apoya en cuatro.

## E-2. Doce unidades, tres semillas por brazo, un solo instrumento

Ninguna unidad de ayer se sobrescribe, y ningún brazo mezcla instrumentos: los tres puntos de cada
brazo se corren hoy y con el mismo script.

| brazo | calentamiento | pasos | post-calentamiento | semillas |
|---|---|---|---|---|
| `abst3s` | ninguno | 16.000 | 16.000 | 0, 1, 2 |
| `abstcal3s` | 3.000 | 16.000 | 13.000 | 0, 1, 2 |
| `abstcal6p` | 6.000 | **19.000** | 13.000 | 0, 1, 2 |
| `abstlargo` | 3.000 | **22.000** | 19.000 | 0, 1, 2 |

## E-3. La comparación de calentamientos NO estaba mal, respondía otra pregunta

Ayer quedó anotado que `abstcal6` contra `abstcal` era una comparación confundida porque el brazo de
6.000 tenía 3.000 pasos menos después del calentamiento. Corresponde precisar, porque la precisión
decide qué celda hay que correr. Son **dos preguntas distintas y las dos son legítimas**, lo que no
se puede es contestar una y leer la otra:

- **A presupuesto TOTAL igualado** (`abstcal` contra `abstcal6`, los dos a 16.000). Pregunta si,
  con el cómputo que hay, conviene gastarlo calentando. Es la que se corrió.
- **A presupuesto POST-calentamiento igualado** (`abstcal` contra `abstcal6p`, 13.000 pasos con
  abstención en los dos). Pregunta si el calentamiento más largo deja al modelo en mejor lugar. Es
  la que falta y es la que se corre hoy.

Con las dos celdas, el efecto del calentamiento y el del presupuesto quedan separados.

## E-4. La desestabilización tardía, con el criterio declarado ANTES de mirar

Ayer aparecieron 3 corridas de 8 que se desestabilizan tarde. El fenómeno no estaba previsto y
todavía no tiene informe. Para que no se convierta en un grado de libertad del analista, el criterio
se fija acá y no se toca después:

    pico   maximo de la media movil de 8 hitos de `acierto` sobre toda la curva
    DESESTABILIZADA  si la media movil sobre el ultimo 10 % de los hitos cae por debajo
                     de 0,80 x pico

**Regla de manejo, y es la que importa: se reportan las tres semillas sin descartar ninguna.** La
cantidad de corridas desestabilizadas y el paso en que ocurre se informan aparte, como resultado
propio. Ninguna hipótesis se evalúa sobre un subconjunto elegido después de ver los datos.

## E-5. Lo que no cambia

La vara sigue siendo `GLOBAL = (acierto + acierto_nose)/n` contra el piso del mudo 0,4000. H1 sigue
exigiendo las dos condiciones juntas, `GLOBAL` sobre el piso **y** `acierto` ≥ 0,80. H2 y H3 quedan
como estaban.
