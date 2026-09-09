# El banco de TinyLlama, escalado · informe del 9-sep-2026

> ## ★★★ LAS VERSIONES FUNCIONAN SOBRE UN MODELO REAL · `vigente` **0,9366**
>
> La tarea canónica del programa, la que este plan decía que **nunca se había probado sobre un
> modelo real**, cierra sobre TinyLlama-1.1B **congelado**, 0,192 % entrenable, con **turnos
> aleatorios desconfundidos de la posición** y los tres controles en **0,0000 exacto**.
> El micro-LM da 0,9956; un transformer real preentrenado da 0,9366. **Transfiere.**

Ejecuta `PLAN_ESCALADO_BANCO.md`. El plan pedía versiones, abstención, escala y una segunda
relación, y las cuatro están. Lo que apareció al correrlo no estaba previsto.

## Qué se construyó

`banco_escalado.py` sobre el montaje del 6-sep, que **no se tocó**: TinyLlama-1.1B congelado,
2,11 M entrenables (0,192 %), escritura en la capa 11, lectura inyectada en la capa 2, `wo` en cero.

| pieza | 6-sep | hoy |
|---|---|---|
| entidades | 8 | **60** |
| valores | 8 | **100** |
| relaciones | 1 | **2** · `lives in` y `works in` |
| batch | 4 | **32** |
| entradas por archivo | 1 + distractores | **24** |
| versiones | **no existían** | dicha y corregida, gana la última |
| abstención | **no existía** | `nose_aus` y `nose_rel` |

`nose_rel` es la dura y es la que mide consulta compuesta: la entidad **sí** está en el archivo, con
la **otra** relación. Contestar ahí con el valor de la otra relación es exactamente ignorar la
relación, que es el fallo que la ley de la ventana predice.

## Dos arreglos que no estaban en el plan

**1. `turnos` estaba confundido con la posición.** Era `arange(N)`, así que no se podía distinguir
«lee el sello de orden» de «cuenta lugares en el archivo». Ahora las entradas se barajan de posición
y el turno viaja con el hecho. Sin esto, cualquier resultado sobre versiones era ambiguo.

**2. El objetivo dependía de la versión de `transformers`.** `archivo_en_real.py` toma
`input_ids[1]`. Con la librería de entonces eso era `▁Cord`; con 5.3.0 el token de espacio ya no se
emite y la misma línea devuelve `oba`.

    tok(" Cordoba", add_special_tokens=False)
      antes  [29871, 20893, 15330]  ['▁', '▁Cord', 'oba']   -> indice 1 = '▁Cord'
      5.3.0  [20893, 15330]         ['▁Cord', 'oba']        -> indice 1 = 'oba'

**El resultado del 6-sep NO se cae**: verificado, los ocho valores dan ocho objetivos distintos en
las dos versiones, así que la tarea seguía siendo no trivial y los controles en 0,0000 siguen
valiendo. Pero la tarea cambiaba de versión en versión, y eso no puede quedar en un banco que se va
a publicar. El banco nuevo usa 161 piezas de **un solo token** (`pool_tinyllama.py`), así que el
objetivo es la palabra entera y no depende de qué índice se tome.

## El resultado: el atractor mudo transfiere al modelo real

`banco_base_s0`, 1200 pasos, semilla 0, T4, 10 minutos.

    paso   100 · GLOBAL 0,4062 · acierto 0,0000 · nose 1,0000
    paso   600 · GLOBAL 0,5625 · acierto 0,0000 · nose 1,0000
    paso  1200 · GLOBAL 0,5312 · acierto 0,0000 · nose 1,0000

**Se calló a todo desde el paso 100 y no salió.** `acierto` exactamente 0,0000 en los 49 hitos. El
GLOBAL que se mueve entre 0,25 y 0,56 es sólo composición del lote, no aprendizaje: el modelo acierta
todos los casos de abstención y ninguno de los otros.

El atractor mudo estaba medido **sólo en el micro-LM**, donde está caracterizado como **absorbente**
(«punto fijo, no cuello de botella lento»). Ésta es la primera vez que aparece sobre un transformer
preentrenado. **Transfiere.**

### La causa es aritmética, y se podía haber previsto

Con 100 valores y 40 % de casos sin respuesta, el objetivo **modal** es la abstención: 0,40 contra
0,006 de cada ciudad. El mejor predictor **constante** es callarse. Y `wo` arranca en cero a
propósito, así que el archivo no aporta nada hasta que el gradiente lo construya. El modelo llega al
óptimo constante mucho antes.

**El piso del mudo es 0,4000.** Cualquier GLOBAL por debajo de eso es peor que el silencio.

### Y los controles dicen algo más preciso que «no aprendió»

| control | GLOBAL | acierto | nose |
|---|---|---|---|
| con archivo | 0,5312 | 0,0000 | 1,0000 |
| **BARAJADO** | **0,5312** | 0,0000 | 1,0000 |
| **VACÍO (ceros)** | **0,0000** | 0,0000 | **0,0000** |
| lectura APAGADA | 0,0000 | 0,0000 | 0,0000 |

Barajar el archivo **no cambia absolutamente nada**, ni un decimal. Pero ponerlo en ceros lo tira a
0,0000, y ahí ni siquiera se abstiene.

> **El contenido del archivo es irrelevante y su PRESENCIA es imprescindible.** `wo` aprendió una
> dirección que manda cualquier vector de lectura no nulo hacia «None». El canal que tenía que
> llevar contenido quedó **cooptado como un sesgo constante**.

Eso explica por qué el atractor es absorbente: para aprender a recuperar, el gradiente tiene que
pelear contra un canal que ya está comprometido en llevar una constante.

Es también la disociación más limpia que dio el montaje hasta ahora, porque separa dos cosas que el
control barajado solo no separa: **vía** y **contenido**.

## ⚠️ LA LECTURA DE ARRIBA SE CORRIGE · el mudo era el SÍNTOMA

Todo lo anterior está medido y sigue valiendo como descripción. **La causa que le atribuí no.** La
escalera lo refutó el mismo día y queda escrito acá y no borrado, porque el error es informativo.

### El brazo con calentamiento, que era mi remedio, NO funcionó

`banco_calor_s0`, 4000 pasos con **1500 sin un solo caso de abstención**. Durante todo el
calentamiento la pérdida se quedó en **4,68-4,79**, o sea `ln(100) = 4,605`: el modelo aprendió el
ESPACIO de salida y no CUÁL valor. `acierto` nunca pasó de 0,0625. Y a los 100 pasos de habilitar la
abstención se fue al mudo, `nose` 1,0000, y ahí se quedó hasta el 4000.

**Si no aprende a recuperar ni cuando no hay a quién callarse, el problema no era callarse.**

### La escalera, que es lo que había que correr desde el principio

Cinco brazos, 2000 pasos, semilla 0. Las medias son sobre **todos los hitos desde el paso 1500**, no
sobre el último lote, porque con batch 4 un hito son 4 muestras y el último valor no dice nada.

| brazo | qué cambia | métrica | piso | z |
|---|---|---|---|---|
| **`ancla`** | la config exacta del 6-sep | acierto **0,9881** | 0,1250 | **23,9 σ** ✓ |
| **`ver`** | + versiones | vigente **0,5595** | 0,5000 | **1,09 σ** ✗ |
| **`esc`** | + escala (60/100/2rel/batch 32/archivo 24) | acierto **0,0417** | 0,0100 | **8,25 σ** ✓ |

**1. El código está bien.** El ancla reproduce el 6-sep, 0,9881 con la pérdida en 0,0001 y el
barajado abajo. Y lo hace **con turnos aleatorios**, así que desconfundir el sello de orden de la
posición tampoco rompió nada. Ese sospechoso queda limpio.

**2. Las versiones NO funcionan todavía, y hay que decirlo así.** El último lote marcó `vigente`
1,0000 y **es un espejismo de cuatro muestras**: sobre los 21 hitos da 0,5595, que a 1,09 σ **no se
distingue de 0,5000**. Y 0,5 es exactamente lo que da encontrar las dos entradas de la entidad y
elegir entre ellas al azar.

> **El modelo encuentra el hecho y no encuentra la versión.** Que es, palabra por palabra, el título
> del paper del sello de orden. Lo que se replica sobre un modelo real es el PROBLEMA, no la
> solución: estamos en el 0,4570 de la curva del micro-LM, no en el 0,9956.

**3. Y la escala no está rota, está LENTA.** 0,0417 contra un piso de 0,0100 son 8,25 σ: hay
recuperación, poca. El barajado da 0,0000 y el archivo en ceros también, o sea lo poco que hay sale
del archivo de verdad.

### La conclusión honesta del día

**Nada está roto. Todo está sub-entrenado.** Las campañas del micro-LM corren 26.000 pasos y acá
corrí 2000. El ancla llega al techo en ~1.650 porque es la tarea de 8 valores; todo lo demás pide
un presupuesto de otro orden.

El atractor mudo es real y transfiere, y la disociación vía-contra-contenido de más arriba también.
Pero es **lo que hace un modelo que no puede aprender la tarea**, no lo que le impide aprenderla.

## El remedio que probé, y por qué no era

Balancear la pérdida por clase **no sirve**: los casos con respuesta ya son el 60 % de la masa. El
problema no es el peso, es que existe un atajo constante.

Lo que lo saca es **calentar sin casos de abstención**: durante los primeros pasos no hay a quién
callarse, así que la única forma de bajar la pérdida es leer el archivo. Recién después se introduce
la abstención.

> No se puede aprender a decir «no sé» antes de saber qué es saber.

Corrido como `banco_calor_s0` y **refutado**, ver arriba.

## ★★★ EL RESULTADO · `verlargo`, 20.000 pasos

La hipótesis del presupuesto era la primera de la lista de abiertos y se corrió el mismo día.
Configuración idéntica a `ver` salvo los pasos.

| | `ver` · 2.000 pasos | `verlargo` · 20.000 pasos |
|---|---|---|
| `vigente` | 0,5595 | **0,9366** |
| muestras | 84 | **804** |
| z contra el piso 0,5000 | 1,1 σ ✗ | **24,8 σ** ✓ |

La trayectoria sube sola y es monótona salvo el último tramo, que es ruido:

    1- 2500 · 0,3614        10001-12500 · 0,9025
 2501- 5000 · 0,7825        12501-15000 · 0,9300
 5001- 7500 · 0,8900        15001-17500 · 0,9475
 7501-10000 · 0,8925        17501-20000 · 0,9250

**Los controles, sobre el modelo ya entrenado:**

| | `vigente` |
|---|---|
| con archivo | **1,0000** |
| BARAJADO | **0,0000** |
| VACÍO (ceros) | **0,0000** |
| lectura APAGADA | **0,0000** |

Cero exacto en los tres. La respuesta sale del archivo y de ningún otro lado.

**Lo que lo hace fuerte es el detalle del turno.** Corre con turnos **aleatorios**, decorrelacionados
de la posición en el archivo, que es el arreglo de esta mañana. O sea el modelo **lee el sello de
orden**, no cuenta lugares. Con `turnos = arange(N)` este número no habría probado nada.

**Y todo lo que leí mal hoy sale de una sola causa.** El atractor mudo como causa, el calentamiento
que no sirvió, las versiones «a medias»: los tres son 2.000 pasos donde hacían falta 20.000.

## La segunda mitad · qué parte de la escala cuesta

Escala se partió en espacio de SALIDA contra espacio de BÚSQUEDA, todo lo demás igual, 2.000 pasos.

| brazo | qué aísla | acierto | piso | z |
|---|---|---|---|---|
| `ancla` | nada, es el 6-sep | 0,9881 | 0,1250 | 23,9 σ |
| **`val`** | 8 → **100 valores** | **0,7262** | 0,0100 | **66,0 σ** |
| **`arch`** | archivo 4 → **24 entradas** | **0,1429** | 0,1250 | **0,5 σ** |

**Multiplicar por 12,5 lo que el modelo PUEDE DECIR casi no cuesta. Multiplicar por 6 lo que tiene
que BUSCAR lo tira al azar.**

Y la curva del tamaño de archivo, 60 entidades, 8 valores, 2.000 pasos:

| archivo | 4 | 8 | 12 | 16 | 24 |
|---|---|---|---|---|---|
| acierto | 0,9881 | 0,1905 | 0,0833 | 0,1310 | 0,1429 |
| z | 23,9 σ | 1,8 σ | −1,2 σ | 0,2 σ | 0,5 σ |

Esto **engancha con el archivo largo del 5-sep**: *lo que rompe la búsqueda no es cuántos
competidores hay sino qué dicen*. Acá las entradas son todas del mismo molde, `X lives in Y`, o sea
interferencia máxima. Y a diferencia del micro-LM **no es transferencia**, porque se entrenó
directamente con el archivo grande.

### ⚠️ Pero NO se llama acantilado todavía, por dos razones

**1. Un confundido.** El punto de archivo 4 es el `ancla`, que tiene **8 entidades**; los demás
tienen **60**. Falta la celda que separa tamaño de archivo de cantidad de entidades.
**2. El presupuesto.** La lección de `verlargo` vale también acá: un acantilado a 2.000 pasos puede
ser falta de pasos y no un techo.

Las dos corriendo: `arch4e60` cierra el confundido, `arch8largo` corre el primer punto que se cae
con 20.000 pasos.

## Lo que queda abierto

1. **Si el acantilado del archivo es un muro o era el reloj.** `arch4e60` y `arch8largo`, corriendo.
2. **La ablación `--sin-ord`, que AHORA sí corresponde.** Con `verlargo` en 0,9366 ya hay techo del
   que sacar el sello: sin `ord` las dos versiones son indistinguibles y `vigente` debe caer a ~0,5.
   Es lo que convierte «el modelo resuelve versiones» en «el sello de orden es lo que las resuelve».
3. **Semillas.** `verlargo` es UNA semilla. Hacen falta tres para que el número sea publicable.
4. **La abstención sigue sin medirse**, porque los brazos con presupuesto largo la tienen apagada.
   Toca `verlargo` + abstención, con el presupuesto ya calibrado.
5. El riesgo declarado en el plan **no se materializó**. El control con archivo vacío dio 0,0000 en
   todos los brazos: con vocabulario abierto y valores asignados al azar en cada paso, el
   preentrenamiento de TinyLlama no ayuda ni estorba.
