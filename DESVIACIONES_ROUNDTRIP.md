# Desviaciones · PREREG_ROUNDTRIP.md (SHA `55ba857a…`)

## D-1 · `s(E')` es el logit NORMALIZADO, no el crudo

El §1 dice «`s(E') =` logit del token `X` en esa consulta». La sonda usa
**`log_softmax(logits)[X]`**, o sea `log P(X | consulta con E')`.

**Por qué, y por qué el prereg estaba mal escrito:** la vuelta compara el mismo token `X` entre
**consultas distintas**, y el logit crudo no tiene escala común entre ellas —cada consulta puede
tener todos sus logits corridos por un offset que no significa nada—. La inversión por Bayes con
prior uniforme sobre las entidades es `P(E'|X) ∝ P(X|E')`, y `P(X|E')` exige la normalización sobre
el vocabulario. Con logits crudos, la softmax sobre `C` del §2 **no es una posterior**: es una
función de offsets arbitrarios.

Es una corrección de la formalización, no un cambio del método ni del criterio: ninguna predicción
se reescribe y `cierra`, `p_E` y los cinco criterios quedan iguales.

**Verificado que no cambia nada:** la sonda calcula las dos versiones en el mismo forward y guarda la
comparación. Concuerdan en 0,87-0,97 de las muestras y **ninguna de las dos pasa la compuerta en
ninguna unidad**.

## D-2 · Desglose de RT-3, agregado después del smoke — y la sospecha que lo motivó era FALSA

El smoke de `c1_s0` dio **RT-3 = 0,859** con 64 muestras, contra el ≥ 0,95 declarado. Supuse que el
nulo mezclaba dos poblaciones: las preguntas **sin** respuesta, donde el valor emitido `X` no es de
la entidad preguntada y no hay razón para que ésta gane. Agregué dos columnas informativas —RT-3 en
aciertos y RT-3 en preguntas sin respuesta— **sin tocar el criterio**.

**El desglose desmiente la sospecha.** En `c3_s0`: 0,204 en los aciertos contra 0,223 en las sin
respuesta; en las ocho unidades las dos columnas quedan a menos de 0,03 una de otra. **No era la
especificación del nulo.** Queda registrado como lo que fue: una hipótesis mía sobre por qué fallaba
un control, refutada por el dato que agregué para verificarla.

## D-3 · El análisis se contaminó a sí mismo: `import sonda_roundtrip` EJECUTABA la sonda

`diag_roundtrip.py` y `diag_relacion.py` importan la sonda para reusar `candidatas()` y
`variantes()`. La sonda tenía **todo su código de corrida a nivel de módulo**, así que el import
corría el experimento entero con los argumentos del diagnóstico (`argparse` lee el mismo `sys.argv`)
y **sobrescribía `roundtrip_20260820.json`** con una corrida de 512 muestras, donde el prereg declara
2048.

**Qué se salvó y qué no:**
- La tabla de la corrida declarada (2048 muestras) **se había leído y verificado antes** de que el
  primer diagnóstico corriera —`n = 2048` en las ocho unidades—, así que los números existen y están
  asentados.
- El **archivo** quedó pisado. Se corrigió la sonda (todo bajo `main()` con
  `if __name__ == "__main__"`) y **se volvió a correr la corrida declarada** para que el registro en
  disco sea el del prereg y no el del accidente.
- La corrida chica, involuntaria, **funciona como réplica**: `c1_s0` AUC 0,979 contra 0,975 · `c3_s0`
  0,497 contra 0,502 · `c3_s1` 0,499 contra 0,483. Ninguna conclusión depende de cuál se lea.

**Es la misma familia que la D-1 del día** —dos cosas escribiendo el mismo archivo— pero del lado del
proceso en vez del checkpoint, y esta vez el que contaminó el dato fue el análisis que venía a
explicarlo. La regla del 20-ago se extiende: *un análisis no puede escribir en la entrada de otro
análisis, y un script que se importa no puede tener efectos al importarse.*
