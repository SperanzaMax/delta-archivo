# E-I4b: ahora sí hay deriva, y el daño empieza antes del umbral de afuera

**2026-08-13.** Pre-registro en el docstring de `interno/ei4b_estres.py`, hash fijado antes de correr.
Continuación directa de `INFORME_EI4_20260813.md`, que midió edades de 0 a 400 pasos y no encontró
degradación **porque el coseno nunca bajó de 0,9374** — la zona donde R5.1 predice que no debe pasar
nada. Acá se empuja: entrenamiento de 12000 pasos y edades de hasta 8000.

---

## 1. Resultado (3 semillas, 12000 pasos)

| edad de la escritura | coseno del marco | revisadas | una sola versión |
|---|---|---|---|
| 0 (control) | 1,0000 | 0,9987 ± 0,0013 | 0,9991 |
| 400 | 0,9548 | 0,9983 ± 0,0015 | 0,9983 |
| 2000 | 0,9067 | 0,9870 ± 0,0065 | 0,9944 |
| **8000** | **0,7804** | **0,9115** ± 0,0184 | 0,9245 |

| predicción | veredicto |
|---|---|
| **P-1** (bloqueante) cos(8000) ≤ 0,80 — hay deriva que interpretar | **CUMPLE** — 0,7804 |
| **P-2** si cos < 0,7, la accuracy cae ≥ 0,10 | **NO EVALUABLE** — el coseno no cruzó 0,70 |
| **P-3** el daño pega primero en las revisadas | **CUMPLE, pero apenas** — −0,0872 vs −0,0747 |

## 2. Lo que E-I4b agrega sobre E-I4

E-I4 no podía distinguir «el mecanismo aguanta» de «no lo empujamos». Ahora sí hay señal: con el
marco movido a 0,7804 aparece una degradación clara de **8,7 puntos** en las claves revisadas. La
curva completa muestra que **el daño es gradual y empieza antes del umbral de 0,7**:

| coseno | caída respecto del control |
|---|---|
| 0,9548 | −0,0004 (nada) |
| 0,9067 | −0,0117 |
| 0,7804 | −0,0872 |

No hay un acantilado en 0,7: hay una pendiente que se empina. Es compatible con R5.1 —que describía
«funciona ≳0,7, degrada 0,7→0,4»— pero **la degradación arranca antes de lo que ese umbral sugiere**,
sólo que tan despacio que a 0,90 todavía es invisible en la práctica (1,2 puntos).

## 3. Lo que sigue sin poder decirse

**El coseno no cruzó 0,70**, así que la comparación que motivaba el experimento —¿el índice
co-entrenado aguanta por debajo del umbral que mata al no paramétrico?— sigue abierta. A 0,78 el
modelo conserva el 91 % de su rendimiento, lo cual es *consistente* con tolerancia, pero la prueba
exige llegar a la zona 0,7-0,4.

Para llegar hacen falta edades de 16000-32000 pasos (el marco se mueve cada vez más despacio a medida
que el modelo converge) o forzar la deriva con **cambio de distribución**, que es lo que hizo R6
afuera. La segunda vía es más barata y probablemente más realista: un modelo desplegado no envejece
por pasos de gradiente, envejece porque lo siguen afinando en datos nuevos.

## 4. Una predicción mecánica que casi no se cumple

P-3 decía que el daño debía pegar primero en las claves revisadas —distinguir dos versiones del mismo
hecho exige más precisión que distinguir hechos distintos—. Se cumple direccionalmente (−0,0872 vs
−0,0747) pero **la diferencia es de sólo 1,2 puntos**, mucho menos de lo que el argumento sugería.

La lectura honesta: cuando el marco se mueve, **el daño es bastante parejo** entre identificar el ítem
y elegir su versión. La intuición de que el envejecimiento «se come primero lo fino» no está
sostenida por estos datos; a lo sumo, insinuada.

## 5. Y una observación entre semillas

El punto donde duele no coincide entre semillas: s0 llegó a cos 0,7514 con 6,6 puntos de caída y s1 a
0,8375 con 10,2. O sea que **la relación coseno→daño no es una función universal**: depende de cómo
quedó organizado el espacio de esa corrida en particular. Con 3 semillas no da para más que anotarlo.

**Costo:** 3 corridas de 12000 pasos + 12 evaluaciones ≈ 45 min de CPU local.
