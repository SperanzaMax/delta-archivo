# El monitor de desacuerdo interno — NEGATIVO, y se cierra la vía

`PREREG_MONITOR_DESACUERDO_V2.md` (SHA `b8e187b4…`) · anulación del v1 en `DESVIACIONES_MONITOR.md`
· datos en `desacuerdo_20260820.json`.

8 unidades de la familia `c` **a 14000 pasos** (las tres de nivel 4 leídas de sus copias `.p14000`),
K = 16 pasadas, `f = 0,25` de las entradas tapadas, 512 muestras por unidad.

---

## 1 · El resultado

| unidad | consistencia media | AUC | `falsa_abst` con el corte estructural | M-3 |
|---|---:|---:|---:|---:|
| `c1_s0` | 0,6466 | 0,669 | 0,5901 | 0,997 |
| `c2_s0` | 0,7057 | 0,656 | 0,4300 | 0,983 |
| `c3_s0` | 0,6641 | 0,552 | 0,5973 | 0,987 |
| `c3_s1` | 0,6589 | 0,583 | 0,5946 | 0,979 |
| `c3_s2` | 0,6841 | 0,558 | 0,5480 | 0,977 |
| `c4_s0` | 0,6970 | 0,509 | 0,5625 | 0,992 |
| `c4_s1` | 0,6887 | 0,502 | 0,5975 | 0,992 |
| `c4_s2` | 0,6920 | 0,586 | 0,5440 | 0,984 |

- **M-1 NO CUMPLE: 0 / 8.** Ninguna unidad llega a AUC 0,70; el rango es **0,502 a 0,669** y las tres
  de nivel 4 están prácticamente en el azar.
- **M-2 NO CUMPLE: 0 / 8.** El corte estructural abstiene en el **43-60 %** de las preguntas que sí
  tenían respuesta. Referencias del mismo día: U-1 = 2/8, σ>0,5 = 6/8, U-2 = 7/8.
- **M-3 CUMPLE 8 / 8**, y con margen enorme: **0,977 a 0,997**.
- **M-4 OK 8 / 8**: el nulo con `f = 0` da 1,000 exacto en todas.

**Según el §4, comprometido antes de correr: M-1 falla ⇒ se cierra la línea del monitor por esta vía
y NO se prueba una tercera perturbación.** Esta vez la regla aplica de verdad: hay un resultado, no un
instrumento vacío como el v1.

## 2 · Los dos controles que hacen que el negativo signifique algo

**M-3 dice que el modelo SÍ lee el archivo.** Tapando exactamente las entradas que originaron el
hecho preguntado, la respuesta cambia en el **98-99 %** de los casos en que antes acertaba. No hay
ninguna duda de que la respuesta depende de esa evidencia, ni de que la perturbación funciona.

**M-4 dice que todo el efecto viene de la perturbación**, no de ruido numérico.

O sea: **el instrumento anda, el modelo lee evidencia, y aun así el desacuerdo no distingue.** Eso es
mucho más fuerte que un negativo sin controles.

## 3 · Por qué no distingue — y encaja con algo ya medido

La premisa cuantitativa del v2 era que una respuesta anclada en **una** entrada sobrevive en `1 − f =
0,75` de las pasadas. **La consistencia media observada es 0,65-0,71, por debajo de 0,75 en las ocho
unidades.** Si se la lee como `(1−f)^k`, da **k ≈ 1,3** entradas de las que depende la respuesta: algo
más que una, coherente con que el hecho y su revisión sean dos entradas distintas. La premisa era
optimista.

Pero eso explica el corte, no la falta de discriminación. **Lo que explica que la AUC sea ~0,5 ya
estaba medido el 15-ago:** `err_fuera = 0,0000` en los cuatro niveles — **el modelo nunca inventa
contenido; toda respuesta errada es un valor REAL del archivo puesto en la entidad equivocada.**

Entonces, cuando la pregunta **no** tiene respuesta en el archivo, el modelo no se pone a adivinar
en el vacío: **se agarra de otra entrada real**. Su respuesta está tan anclada —y es tan estable ante
tapar entradas al azar— como cuando la pregunta sí tenía respuesta. **El desacuerdo mide si la
respuesta se apoya en el archivo, no si se apoya en la entrada CORRECTA**, y esas dos cosas sólo se
distinguen con la etiqueta, que es exactamente lo que no tenemos.

## 4 · Lo que esto deja para el objetivo

Dos vías cerradas en un día, y las dos por el mismo motivo de fondo, que conviene dejar escrito:

- **El corte sin etiquetas sobre el logit** falló porque al logit le falta escala absoluta y la
  información no está en forma de valle.
- **El monitor de desacuerdo** falló porque la señal que sí tiene un cero absoluto —el acuerdo entre
  pasadas— **no es específica**: responde «esta respuesta viene del archivo», que es cierto también
  cuando la respuesta es incorrecta.

**Lo que queda en pie es lo de siempre y ahora está mejor delimitado:** la información para abstenerse
existe en el modelo (AUC 0,825 del logit entrenado, y U-2 pasando 7/8), pero **todo lo que la vuelve
utilizable pasa hoy por etiquetas**. Antes de intentar una tercera vía conviene tener claro qué
tendría que medir: algo que separe *anclado en la entrada correcta* de *anclado en cualquier entrada*,
que es la distinción que ni el logit ni el desacuerdo hacen.

## 5 · Alcance

Ocho unidades, un modelo, idioma cerrado de 242 tokens, `p_nose = 0,4`, monitor **de inferencia**.
No dice nada sobre entrenar contra el desacuerdo, que sigue sin probarse y necesita GPU.
