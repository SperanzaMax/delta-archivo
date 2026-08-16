# Mitigar la alucinación sin reentrenar: funciona para media enfermedad

**2026-08-15, noche** · `mitigar.py`, sobre checkpoints ya entrenados, sin GPU

## La pregunta

La campaña de abstención va a *enseñarle* al modelo a decir `NOSE`. Antes de gastar GPU en eso vale
preguntar algo más barato: **el modelo que ya tenemos, ¿sabe internamente cuándo está por
equivocarse?** Si la confianza de su salida separa aciertos de errores, la abstención no hay que
enseñarla — se lee.

Tres señales de la misma distribución de salida: probabilidad del token elegido, margen entre el
primero y el segundo, y entropía. La técnica no es nueva (*selective prediction*, umbral de
confianza); lo que se mide acá es si sirve **en un modelo con archivo persistente y para los tipos
de error que este proyecto separa**.

## Compuerta: ¿la confianza discrimina?

| checkpoint | AUC prob | AUC margen | AUC entropía |
|---|---:|---:|---:|
| n4_s0 (N4) | **0,8631** | 0,8626 | 0,8602 |
| n3_s2 (N3) | 0,8688 | **0,8688** | 0,8623 |

Sí, y con margen. **El modelo acierta con más confianza de la que falla.** Las tres señales dan
prácticamente lo mismo, lo que sugiere que capturan la misma información.

## Cuánto se puede apagar

Umbral **calibrado en una mitad de los datos y medido en la otra** — elegirlo sobre el mismo
conjunto que se evalúa es un oráculo y sobreestima. Comparado contra abstenerse **al azar** en la
misma proporción, que es el piso que hay que superar para que la señal aporte algo.

`n4_s0`, SER base 0,2283:

| cobertura | acierto | SER | SER evitado | vs azar |
|---:|---:|---:|---:|---:|
| 1,00 | 0,7704 | 0,2287 | 1,2 % | 1,01× |
| 0,90 | 0,8077 | 0,1727 | 25,4 % | 1,20× |
| **0,78** | **0,8624** | **0,1073** | **53,6 %** | **1,68×** |
| 0,59 | 0,9493 | 0,0300 | 87,0 % | 4,56× |
| 0,51 | 0,9844 | 0,0080 | 96,5 % | 14,81× |

Respondiendo el 78 % de las preguntas se elimina **más de la mitad de los errores silenciosos**, y
el acierto sobre lo que contesta sube de 0,77 a 0,86. `n3_s2` replica el patrón (47,4 % a cobertura
0,81). A cobertura 0,80 se apaga el **48 %** de los errores de identidad.

## El límite, que es lo importante

Los checkpoints se entrenaron con `p_nose = 0`: **nunca vieron una pregunta sin respuesta**.
Evaluándolos con `p_nose = 0,4` inventan por fuerza — la pregunta es si lo hacen con menos
confianza.

| régimen de evaluación | AUC | invento apagado (cob. 0,80) | vs azar |
|---|---:|---:|---:|
| sin preguntas sin respuesta | 0,8631 | — | 1,68× |
| **con preguntas sin respuesta** | **0,7397** | **28,8 %** | **1,16×** |

**La señal se degrada justo donde más falta hace.** El modelo detecta razonablemente bien cuándo
*atribuyó mal* un dato (35,9 % de los errores de identidad se apagan) y bastante peor cuándo *el
dato no existía* (28,8 %). La ventaja sobre el azar cae de 1,68× a 1,16×.

Dicho de otro modo: **confía casi igual inventando que acertando.** Tiene sentido mecánicamente —
nada en su entrenamiento le pidió distinguir «está y es X» de «no está», así que la ausencia no
tiene representación propia; el lector devuelve su mejor candidato y el candidato es igual de
confiable en los dos casos.

## Consecuencias para el plan

1. **La compuerta de mañana sigue siendo necesaria.** La mitigación gratis no cubre el caso central.
2. **Ahora hay un baseline concreto**: la campaña de abstención entrenada tiene que superar el
   **28,8 % de invento apagado** que se consigue sin entrenar nada. Antes no teníamos con qué
   comparar, y cualquier número hubiera parecido bueno.
3. **El umbral de confianza es complementario, no alternativo**: aunque el modelo aprenda a
   abstenerse, aplicarle el umbral encima debería seguir bajando el SER. Vale medir las dos cosas
   juntas cuando haya un checkpoint entrenado con `p_nose > 0`.
4. Para un sistema desplegado, la curva riesgo-cobertura ya es utilizable: **contestar el 78 % y
   callarse el 22 % elimina la mitad de los errores silenciosos**, sin tocar el modelo.

## Lo que este informe NO dice

- No es una técnica nueva; es umbral de confianza de toda la vida. Lo medido es su alcance en este
  problema y su **quiebre en el caso de ausencia**.
- Una sola semilla por nivel (N3 y N4 todavía no tienen las tres), así que los valores absolutos son
  provisorios. El contraste entre regímenes —0,86 contra 0,74— es la parte robusta.
- No se probó sobre un modelo entrenado con `p_nose > 0`, que es justamente el que puede tener una
  representación de la ausencia. Esa medición es la que sigue.
