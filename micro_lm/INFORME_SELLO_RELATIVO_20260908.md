# El sello relativo, medido · R-1 y R-3 sobre las seis unidades

2026-09-08. Cierra la campaña `rp3_s0/s1/s2` (con bit de pertenencia) y `rr3_s0/s1/s2` (sin bit),
2000 pasos las seis, `sello=rel`, `kernel_q=5`, `ses_extra=26`. Preregs `aabb4b20` (R-1) y
`bedac0b5` (R-3).

## 0. Antes de los números: dos bugs de instrumento, y los dos cambiaban la lectura

Se anotan primero porque la primera corrida de los dos instrumentos dio resultados que se leían como
hallazgos y no lo eran.

**(a) El signo de la primera componente principal es ARBITRARIO.** `np.linalg.svd` no fija ninguna
convención: `Vt[0]` y `-Vt[0]` son la misma componente. La primera medición de R-3 dio rampa
**positiva en las tres `rp` y negativa en las tres `rr`**, un 3/3 contra 3/3 perfecto que se leía
como que el bit invierte el gradiente de recencia. En valor absoluto las seis dan 0,091-0,133, o sea
lo mismo. Arreglado con convención por la componente de mayor magnitud, y se devuelve `rampa_abs_G4`.

**(b) `reloj_o_bandera.py` medía las tres `rp3` con el bit de pertenencia APAGADO.** Su función
`leer()` reimplementa `modelo.responder` a mano para poder guardar la distribución de lectura, y al
copiarla se quedó sin `marca_pert`. Las unidades entrenadas CON el bit se estaban midiendo sin él.

La firma del bug, y conviene reconocerla: **la recuperación aguantaba y la respuesta se caía.**
Acierto 0,2969-0,5781 en la sonda contra `vigente` 0,9725-0,9824 en el propio entrenamiento de esas
mismas unidades, con las curvas de las seis indistinguibles entre sí. Eso no es un modelo que
aprendió peor, es un modelo al que le sacaron una entrada de la que aprendió a depender.

**Control del arreglo:** las tres `rr3` (`pert=False`) no pueden moverse, porque para ellas
`marca_pert` devuelve 0,0. Movieron **0 de 54 celdas**, bit a bit. El arreglo toca sólo lo que tenía
que tocar.

**Y la regla que sale, que ensancha el §3 del RETOMAR del 6-sep:** todo instrumento que carga un
checkpoint tiene que reproducir de su config todo lo que decide la arquitectura, **sean globals del
módulo o argumentos de llamada**. `conf_ckpt.aplicar` cubre los primeros y no puede cubrir los
segundos; para eso está ahora `conf_ckpt.pertenece_de(cfg, mask)`. El corolario es el que muerde:
**una función que reimplementa `responder` no hereda sus defaults**, así que cada vez que `responder`
gana un argumento, esa copia queda vieja y en silencio. `auditar_instrumentos.py` ahora marca
`PERT(arg)` a quien arma su propia clave sin sumar el bit.

## 1. R-1 · el sello es una BANDERA, no un reloj — y se contesta en las `rr3`

`recup` por condición, las tres semillas sin bit:

| condición | rr3_s0 | rr3_s1 | rr3_s2 | qué mueve |
|---|---|---|---|---|
| `real`           | 1,0000 | 1,0000 | 1,0000 | el régimen de entrenamiento |
| **`corrido`**    | **1,0000** | **1,0000** | **1,0000** | orden relativo intacto, frontera dos lugares |
| `extra_corridas` | 1,0000 | 1,0000 | 1,0000 | idem, sobre las ajenas |
| `comprimido`     | 1,0000 | 1,0000 | 0,9688 | sin orden ENTRE las viejas |
| `barajado`       | 0,3125 | 0,4531 | 0,4375 | control ya publicado |
| `invertido`      | 0,3594 | 0,5000 | 0,5156 | la correcta en turno BAJO |

**Mover la frontera manteniendo el orden relativo no cuesta nada** (1,0000 exacto, 3/3), y romper el
orden o invertir la pertenencia cuesta la mitad. Es un **escalón**, no una pendiente, y confirma con
`sello=rel` entrenado lo que el 6-sep quedó medido sobre el absoluto. La consecuencia práctica sigue
siendo la misma: **agrandar `ord` no sirve**, porque la tabla no está funcionando como reloj.

## 2. R-1 sobre las `rp3` · el bit de pertenencia compra INMUNIDAD al orden

Las mismas seis condiciones, con el bit bien pasado:

| condición | rp3_s0 | rp3_s1 | rp3_s2 | contra rr3 |
|---|---|---|---|---|
| `real`           | 1,0000 | 1,0000 | 1,0000 | = |
| `corrido`        | 1,0000 | 1,0000 | 1,0000 | = |
| `extra_corridas` | 1,0000 | 1,0000 | 1,0000 | = |
| `comprimido`     | 1,0000 | 1,0000 | 1,0000 | = |
| **`barajado`**   | **0,9688** | **1,0000** | **0,9844** | contra 0,31-0,45 |
| **`invertido`**  | **1,0000** | **1,0000** | **1,0000** | contra 0,36-0,52 |

**Con el bit, ninguna perturbación del sello rompe la recuperación.** El índice de masa acompaña:
donde `rr3` sube a 0,77-0,82 en barajado e invertido —o sea deja de descartar lo viejo— `rp3` se
queda en 0,39-0,44. El bit hace exactamente lo que se diseñó que hiciera el 6-sep: **desacopla «de
quién es» de «cuándo fue»**, y por eso mover los turnos deja de importar.

Hay que declarar el corolario, porque cambia cómo se lee este banco: **sobre `pert=True` las
condiciones de R-1 dejan de ser un test del sello.** No es que la bandera desaparezca; es que el
modelo tiene un tercer camino y ya no necesita la tabla para esto. La pregunta «reloj o bandera» se
contesta en las `rr3`.

El acierto no llega a inmunizarse del todo (`barajado` 0,61-0,78, `invertido` 0,77-0,86 contra 0,08
sin bit): **recuperar y contestar siguen disociados**, que es la misma disociación que mostró la
película larga del 5-sep.

## 3. R-3 · la contraparte de peso, con el signo arreglado

| unidad | brechaG1 | r_distG2 | \|rampa\|G4 | varPC1 | \|n\| bajo | \|n\| alto |
|---|---|---|---|---|---|---|
| rp3_s0 | −0,0346 | −0,5172 | 0,0907 | 0,5911 | 2,1665 | 1,2475 |
| rp3_s1 | −0,0329 | −0,5313 | 0,1044 | 0,6576 | 2,3678 | 1,3027 |
| rp3_s2 | −0,0365 | −0,5271 | 0,1300 | 0,5933 | 2,3407 | 1,2599 |
| rr3_s0 | −0,0460 | −0,5149 | 0,0970 | 0,7806 | 3,1407 | 1,4892 |
| rr3_s1 | −0,0408 | −0,5253 | 0,1047 | 0,8261 | 3,4853 | 1,5669 |
| rr3_s2 | −0,0438 | −0,5276 | 0,1327 | 0,7877 | 3,4166 | 1,5562 |

Lo que **no** separa a las ramas: la rampa (0,091-0,133 en las seis, solapadas) y la estructura de
orden `r_distG2` (−0,515 a −0,531 en las seis, notablemente estable).

Lo que **sí** separa, 3/3 contra 3/3 y sin solapamiento:

- **la norma de las filas de `ord`**: rp 2,17-2,37 contra rr 3,14-3,49 en el bloque bajo, y 1,25-1,30
  contra 1,49-1,57 en el alto. **Un 30 % menos de señal en la tabla.**
- **`var_pc1`**: rp 0,59-0,66 contra rr 0,78-0,83. Con el bit, la varianza deja de concentrarse en
  una sola dirección.

**El bit DESCARGA la tabla.** Es la contraparte geométrica exacta de lo conductual: al darle por
fuera una de las dos funciones que `ord` sostenía, la tabla necesita menos norma y menos
unidimensionalidad para hacer la que le queda. Dos instrumentos independientes, la misma conclusión.

Nota de alcance, y va acá porque nadie la declaró al diseñar: `UMBRAL = 24` está pensado para
`sello=abs`, donde la fila es el turno. En `sello=rel` la fila es la **distancia**, así que «bloque
bajo» significa «lo reciente» y no «lo ajeno». Los contrastes de bloque (`brechaG1`, `cos_*`) hay que
leerlos con ese cambio de significado; las normas y `var_pc1`, que es lo que separa, no dependen de
dónde caiga la frontera.

## 4. Qué queda

1. **Fase 2**, archivo de más de 64 turnos: R-1 y R-3 cerraron, así que está habilitada.
2. El corolario del §2 conviene volcarlo al diseño: si se quiere seguir midiendo el sello, hay que
   hacerlo sobre unidades **sin** bit, o inventar una perturbación que el bit no cubra.
3. Los 32 instrumentos que el auditor sigue marcando, ahora también por `PERT(arg)`.

## 5. Archivos

`reloj_o_bandera_20260908.json` (la corrida con el bug, se conserva como control),
`reloj_o_bandera_20260908_conbit.json` (la buena), `geometria_ord_20260908.json`.
