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

---

# CORRECCIÓN de E-4 · el mismo día, con las corridas de hoy todavía sin resultado

El texto de arriba queda como se congeló (SHA `f89071b6`). Esto se agrega después, y se agrega
porque **el criterio que escribí en E-4 tiene exactamente el defecto que E-1 denuncia**. Se corrige
ahora, con las doce corridas de hoy en marcha y ninguna con datos todavía, así que no hay forma de
que la elección del criterio esté contaminada por el resultado que va a producir.

## Lo que estaba mal, en dos pasos

**1er intento, el de E-4.** `pico` = máximo de la media móvil, desestabilizada si el mínimo de la
cola cae bajo 0,80 × pico. Aplicado a las 28 unidades guardadas del banco da **15 de 28**, y marca
corridas que nunca aprendieron nada (`arch12_s0` con pico 0,2188). Compara un **máximo** contra un
**mínimo** sobre hitos de cuatro muestras, o sea dos extremos de ruido, que es la misma familia de
error que ayer nos costó seis conclusiones.

**2do intento.** Promedios contra promedios con el mismo umbral 0,80. Da **0 de 13** y también está
mal, ahora en la otra dirección. `ent15_s0` tiene el tramo medio en **1,0000 exacto sobre 800
muestras** y la cola con hitos en 0,00 y 0,50. Un modelo al 100 % no produce un hito de cuatro de
cuatro mal. La caída es real y el umbral proporcional no la ve.

## El criterio que queda

La cola contra el tramo medio como **dos proporciones**, con el n real de muestras detrás de cada
tramo, y sin ningún umbral proporcional:

    medio    hitos del 50 % al 75 % de la curva
    cola     hitos del ultimo 10 %
    aplica solo si `medio` >= 0,80  (una corrida que nunca llego al techo no se puede desestabilizar)
    DESESTABILIZADA  si la caida es GRANDE (>= 0,05 absoluto) Y CLARA (z >= 3)

Implementado en `desestabiliza.py`. La regla de manejo de E-4 no cambia: **se reportan las tres
semillas sin descartar ninguna**, y la cantidad de corridas desestabilizadas se informa aparte.

## Lo que da sobre las corridas de ayer, y cambia la lectura

**1 de 13**, no 3 de 8. La única es `ent15_s0`, con una caída de +0,0926 y z = 8,72. Todas las demás
que llegaron al techo terminan igual o mejor de lo que estaban, y cuatro de ellas **mejoran** en la
cola. Lo que ayer se leyó como un fenómeno de 3 de 8 era, en su mayor parte, el ruido de una métrica
de cuatro muestras leído como degradación, que es lo que
`REGLA: cómo leer las curvas` ya decía y no aplicamos.

**El fenómeno igual existe y hay que estudiarlo**, porque `ent15_s0` es genuino y no se explica por
muestreo. Pero es raro, no frecuente, y esa diferencia decide si hace falta un remedio.
