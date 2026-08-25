# NOTA DE MÉTODO · el mismo arreglo, dos veces, con siete días de diferencia

**2026-08-25.** Octava vez en el proyecto que un número limpio esconde un artefacto, y la segunda en
que el artefacto es del instrumento propio. Esta merece registro aparte porque **el error ya se había
cometido, diagnosticado y arreglado**, y volvió a entrar por la puerta de al lado.

## Qué pasó

El cierre automático del 24-ago evaluó las tres unidades del slot nulo y devolvió `nose = 0,0000` y
`falsa_abst = 0,0000`, exactos, en las tres semillas.

`ser.py:89` decía:

```python
usa_cabeza = cfg.get("abst", "token") == "cabeza"
predecir = E.predecir_cabeza if usa_cabeza else E.predecir
```

Con `--abst slot`, el entrenamiento usa el camino binario de `cabeza` —`entrenar.py:337`,
`_bin = a.abst in ("cabeza", "slot")`— y pone `NOSE` en −1e9 dentro del softmax de valores. Medir esas
unidades con el argmax plano es preguntarle por una salida que el entrenamiento cerró: **no puede
emitir `NOSE`**. El cero no era una propiedad del modelo, era la única respuesta que el instrumento
admitía.

Hacían falta las dos mitades del arreglo, y la segunda es menos obvia: `ser.py` fijaba `E._DONDE` pero
nunca `E._ABST`, así que aun llamando a `predecir_cabeza` el logit binario habría salido de la cabeza
lineal, que en estas unidades **nunca se entrenó**. Un arreglo a medias habría producido otro número
limpio y también falso.

## Por qué duele

El comentario de `ser.py:83-87` documenta este mismo error, para `cabeza`, el 18-ago:

> `ser.py` es del 15-ago y la cabeza es del 18: la incompatibilidad venía de la diferencia de fechas.

El arreglo de entonces se escribió como `== "cabeza"`. Seis días después nació una tercera condición
que comparte exactamente ese camino, y la comparación por igualdad la dejó afuera en silencio. **El
arreglo anterior estaba escrito de una forma que no podía sobrevivir a una condición nueva.**

## Lo que se lleva

- **Un `==` contra un valor de configuración es una bomba de tiempo cuando el conjunto de valores
  crece.** La forma correcta es `in (...)`, y mejor todavía derivarlo de una sola fuente: hoy
  `entrenar.py` define el conjunto binario en `_bin` y `ser.py` lo repite a mano. Repetir la regla es
  lo que permitió que se desincronizaran.
- **Cuando se agrega una condición, el instrumento de medición es parte de la condición.** El chequeo
  A-1/A-2/A-4 del prereg del slot verificó el modelo con enorme cuidado —incluso cazó un falso
  positivo con A-3b— y nadie verificó que `ser.py` supiera leerlo. La compuerta miraba el sujeto y no
  el termómetro.
- **La regresión sigue siendo barata y sigue pagando:** `p3_s0` con el código nuevo reproduce las seis
  métricas hasta el último dígito. Verificar eso costó dos minutos y es lo que permite usar el control
  reusado sin dudar.
- **El arreglo favoreció al slot** (acierto 0,7572 → 0,8991 en s0) y S-1 y S-2 fallaron igual. Un
  artefacto que se corrige a favor del resultado que uno esperaba refutar es la mejor posición
  posible para reportar el negativo.

## Regla operativa

Antes de leer la primera métrica de una campaña con una condición NUEVA, correr el evaluador sobre una
unidad VIEJA y confirmar que reproduce su número publicado, **y** confirmar que el evaluador conoce la
condición nueva por su nombre. Las dos cosas, no una.
