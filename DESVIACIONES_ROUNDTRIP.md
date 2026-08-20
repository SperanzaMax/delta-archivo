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

**Se reporta el crudo al lado.** La sonda calcula las dos versiones en el mismo forward (cuesta cero)
y guarda la comparación, así que si la conclusión dependiera de esta elección se puede ver.

## D-2 · Desglose de RT-3, agregado después del smoke y sin tocar el criterio

El smoke de `c1_s0` dio **RT-3 = 0,859** con 64 muestras, contra el ≥ 0,95 declarado. Al mirar de
dónde salía apareció algo que el §3 no había previsto: **el nulo se computa sobre todas las
muestras, incluidas las preguntas SIN respuesta**, y ahí el valor emitido `X` no es de la entidad
preguntada —no es de nadie en particular—, así que no hay ninguna razón a priori para que la
preguntada le gane a una rival ausente. El nulo, tal como está escrito, mezcla dos poblaciones.

**El criterio declarado NO se cambia** (sigue siendo el global ≥ 0,95). Se agregan dos columnas
informativas —RT-3 en aciertos y RT-3 en preguntas sin respuesta— para poder interpretar el número
en vez de leerlo a ciegas. Si RT-3 falla globalmente pero da ≈ 1,000 en los aciertos, lo que falla es
la especificación del nulo y así hay que escribirlo, sin rescatar nada por la ventana.
