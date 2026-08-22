# INFORME · E-I4c — LA DERIVA POR CAMBIO DE DISTRIBUCIÓN TAMPOCO LLEGA

Pre-registro `PREREG_EI4C_DISTRIBUCION.md`, SHA `8e051d74…`, congelado antes de escribir el script.
`interno/ei4c_distribucion.py`, resultados en `interno/resultados_ei4c.json`. Fase A 6000 pasos con
claves de `[0,64)`, fase B con claves de `[64,128)`, edades 0/500/2000/6000, 3 semillas, ~15 min de
CPU por semilla.

## Resultado

| edad | cos | revisadas | no revisadas |
|---:|---:|---:|---:|
| 0 | 1,0000 | 0,9970 | 0,9970 |
| 500 | 0,9217 | 0,9978 | 0,9991 |
| 2000 | 0,9021 | 0,9974 | 0,9978 |
| 6000 | **0,8531** | 0,9922 | 0,9913 |

Por semilla en la edad máxima: s0 cos 0,8680 · s1 0,8322 · s2 0,8590. Consistentes entre sí.

- **P-1 (bloqueante) NO CUMPLE: cos 0,8531 contra el ≤ 0,70 exigido.**
- **P-2 NO EVALUABLE**, por tercera vez.
- **P-3 «cumple» y no vale nada: 0,9021 contra 0,9067 de E-I4b.** La diferencia es 0,0046. El
  criterio era «< 0,9067» y el dato lo cruza por cuatro milésimas, así que técnicamente pasa, pero
  **el cambio de distribución NO mueve el marco más rápido que la antigüedad pura** — que era la razón
  entera para elegir esta vía. Se reporta como cumplido y sin efecto, no como éxito.

## El veredicto que corresponde por el §5

> Si P-1 falla, se cierra la vía del envejecimiento **entera** —las dos formas de producir deriva ya
> habrían fallado en producirla— y se reporta como límite del harness, no del mecanismo.

**Se cierra.** Tres experimentos, tres formas de empujar el marco, y las tres se quedan arriba:

| | cómo empuja | cos mínimo alcanzado |
|---|---|---:|
| E-I4 | edades hasta 400 | 0,9374 |
| E-I4b | edades hasta 8000, 12000 pasos | 0,7804 |
| E-I4c | cambio de distribución | 0,8531 |

La pregunta de si el índice co-entrenado tolera lo que mata al no paramétrico **queda sin responder**,
y ya no por falta de intentos.

## La lectura que el §5 no anticipó, y que hay que poner con su etiqueta

El prereg mandaba reportar esto como límite del harness. Pero tres fracasos en producir el estímulo
empiezan a ser un dato sobre el objeto y no sobre el instrumento, y hay una predicción previa que lo
esperaba: **R6** midió afuera que la deriva catastrófica (cos 1,000 → 0,207 en 400 pasos) es un
fenómeno del **aprendizaje inicial**, no de la vida útil del modelo, y que sobre un modelo preentrenado
que se afina el coseno se queda en 0,882.

E-I4, E-I4b y E-I4c son R6 medido desde adentro, tres veces: **un modelo convergido no mueve su marco
por debajo de ~0,78 ni cuando se le cambia la distribución de entrada**. Si eso es así, P-2 no es una
pregunta que quedó sin contestar: es una pregunta cuyo régimen **no ocurre** en un co-entrenado
convergido, y el umbral de 0,70 de R5.1 describiría una zona a la que este tipo de sistema no llega.

**No lo declaro como resultado** —haría falta descartar que sea el tamaño del modelo, la tarea o el
harness, y con tres corridas del mismo banco no se puede—. Lo dejo escrito como la hipótesis que
ordena lo que hay, y con la prueba que la separaría: si la zona < 0,70 no es alcanzable, tampoco
debería serlo en un modelo más grande entrenado más tiempo; si es del harness, ahí sí debería caer.

## Un dato lateral que contradice el supuesto de fondo de toda la línea

A coseno comparable, **el daño no es el mismo según cómo se produjo la deriva**:

| | cos | revisadas |
|---|---:|---:|
| E-I4b (antigüedad), edad 2000 | 0,9067 | 0,9870 |
| E-I4c (distribución), edad 2000 | 0,9021 | **0,9974** |

Mismo coseno —0,9067 contra 0,9021, y el de E-I4c es incluso algo peor— y sin embargo E-I4c conserva
más accuracy. El supuesto que atraviesa R5.1, R7.1, E-I4 y E-I4b es que **el coseno del marco resume
el daño**; acá dos derivas con el mismo coseno hacen daño distinto.

Encaja con la observación entre semillas de E-I4b («la relación coseno→daño no es una función
universal: depende de cómo quedó organizado el espacio de esa corrida») y la extiende: no depende sólo
de la corrida, depende del **tipo** de deriva. Con 3 semillas y una sola comparación pareada es una
señal, no un hallazgo — pero toca un supuesto que se venía usando sin discutir.

## Lo que no dice

- Un solo tipo de cambio de distribución, y de los más suaves: cambian las claves, no la estructura de
  la tarea. Un cambio de estructura podría mover el marco más.
- La fase A converge a 1,0000, o sea el modelo está en el régimen donde R6 predice estabilidad. No se
  probó el caso intermedio, un modelo a medio entrenar.
- La accuracy de la fase B se mide sobre hechos de la distribución A. El control de eso es la edad 0
  (mismo desajuste, cero deriva) y da 0,9970: el desajuste solo no explica nada de la caída, que es de
  todos modos ínfima.
