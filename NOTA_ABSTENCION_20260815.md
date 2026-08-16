# La abstención tiene un óptimo local, y hay que rodearlo

**2026-08-15, noche** · pruebas en CPU con modelo reducido, sin gastar GPU

## El problema

Con `p_nose = 0` —lo que corrió todo el proyecto hasta hoy— la métrica de abstención sale `NaN`:
no hay preguntas sin respuesta, así que `NOSE` no es una opción que el modelo pueda tomar y **todos
los errores son silenciosos por construcción**. Para medir abstención hay que subir `p_nose`, y ahí
aparecen dos atajos, uno en cada extremo.

Medido sobre el generador real:

| `p_nose` | % sin respuesta | acierto de NO abstenerse nunca | acierto de abstenerse siempre |
|---:|---:|---:|---:|
| 0,1 | 0,1062 | 0,8938 | 0,1062 |
| 0,2 | 0,2047 | **0,7953** | 0,2047 |
| 0,3 | 0,2898 | 0,7102 | 0,2898 |
| 0,4 | 0,4094 | 0,5906 | **0,4094** |

Con `p_nose = 0,2`, no abstenerse nunca vale **0,7953** — más que los 0,7598 de la mejor corrida de
nivel 4 que tenemos. El gradiente no tendría ningún motivo para aprender a abstenerse. Por eso la
campaña se cambió a 0,4 antes de gastar la primera GPU.

## Lo que pasó al probarlo (modelo reducido, d=32, capas=2, 45 298 params)

| régimen | 3000 pasos | lectura |
|---|---|---|
| `p_nose = 0,0` (control) | vigente **0,1296**, en subida | aprende, lento, lejos de saturar |
| `p_nose = 0,2` | vigente 0,0030 · falsa_abst 0,9953 | colapsa a abstenerse de todo |
| `p_nose = 0,4` | vigente 0,0000 · falsa_abst 1,0000 | colapsa más fuerte |

**El control es lo que hace legible el resultado.** Sin él, el colapso se leería como incapacidad
del proxy. Con él se ve que el modelo sí aprende cuando no hay preguntas sin respuesta: el colapso
no es incapacidad, es que **abstenerse paga más que un mecanismo a medio aprender**. Con el
mecanismo rindiendo 0,13, decir `NOSE` siempre rinde 0,20 o 0,41.

Es la misma estructura del atajo de la recencia de E-I3d, donde dos de tres semillas convergieron a
«gana el turno más alto» y fallaban la versión anterior por debajo del azar. El gradiente prefiere
la solución barata **mientras la cara no rinda todavía**.

## El currículum, probado y mal probado

Se reanudó el control (que venía de 0,1296 con `p_nose = 0`) introduciendo `p_nose = 0,4`. Colapsó a
0,0037 en 500 pasos.

**Eso no refuta el currículum: lo probó en el punto equivocado.** La premisa era introducir las
preguntas sin respuesta *después* de que el mecanismo sature, y 0,1296 no satura nada — abstenerse
seguía pagando 0,41 contra 0,13. El modelo hizo lo racional.

## Criterio operativo que sale de acá

> **Introducir `NOSE` sólo cuando `vigente` supere la tasa de preguntas sin respuesta.**

Es el punto donde el mecanismo rinde más que el atajo: con `p_nose = 0,4` el umbral es 0,41, con 0,2
es 0,20. En el modelo real (863 730 params) `vigente` llega a 1,0000 en nivel 1 hacia los 3000–4000
pasos, así que el currículum es viable — y **no cuesta nada de infraestructura**: la guarda de
identidad del checkpoint compara `nivel, semilla, lr, idioma, d, capas` y **no** `p_nose`, así que
una corrida se puede reanudar cambiando el régimen. Queda registrado por evaluación en la historia,
para que un salto de métrica no se lea como aprendizaje.

## Límite declarado

Todo esto es un **proxy de 45 298 parámetros contra los 863 730 reales**, con 3000 pasos contra
12 000. El mecanismo del óptimo local es genérico y ya se observó en E-I3d, pero **la dinámica a
escala real puede ser distinta y esto no la prueba**. Lo que sí queda establecido sin depender del
proxy es la aritmética de la tabla de arriba: los atajos existen y valen lo que valen.

## Estado del instrumento (esto sí está cerrado)

- `SER` desagregado en versión / identidad / fuera, más la categoría nueva **`invento`** —contestar
  un valor cuando la respuesta correcta era `NOSE`—, que es la alucinación pura y no existía en el
  código porque con `p_nose = 0` el caso no podía darse.
- Métricas probadas contra cuatro modelos falsos (oráculo, nunca abstiene, siempre abstiene, azar):
  **10 comprobaciones, todas capaces de fallar, todas pasan**. Lo importante: un modelo que dice
  `NOSE` a todo saca `nose = 1,0000` y `SER = 0,0000`; lo delatan `falsa_abst = 1` y `acierto = 0`.
  Ninguna de las dos métricas sirve sola.
- Reproducibilidad desde la semilla verificada en 4 niveles × 2 regímenes, **ensuciando a propósito
  el RNG global de numpy**, que era el bug del 14-ago.
