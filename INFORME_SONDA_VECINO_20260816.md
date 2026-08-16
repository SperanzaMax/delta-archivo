# Sonda del vecino · el archivo no está corrupto: el error es de lectura

**2026-08-16** · `sonda_vecino.py` · CPU, checkpoints existentes, cero GPU
Pre-registro `PREREG_SONDA_VECINO.md`, SHA `faebb671…`, congelado 20:58 UTC **antes** de correr.

## Qué decidía

El error de identidad es el 93 % de lo que falla. La revisión externa lo partió en dos destinos y sólo
uno es atacable: si la corrección elíptica se ligó al vecino **al escribir**, el archivo contiene un
hecho falso que al leerse es **mecánicamente idéntico** a uno verdadero —matcheo alto, margen alto,
entropía baja— y **ninguna abstención lo detecta, por diseño**. Ese residuo pondría un piso
infranqueable al criterio de éxito de la campaña.

## Resultado, n = 4000 por checkpoint

| | `n3_s2` (N3) | `n4_s0` (N4) |
|---|---:|---:|
| **casos de `err_identidad`** | ~870 | 877 |
| vecino **intacto** (devuelve lo suyo) | — | **0,8301** |
| vecino **corrupto** (tiene el valor ajeno) | **0,0482** | **0,1186** |
| vecino devuelve otra cosa | — | 0,0513 |
| *control · corrupto entre los aciertos* (P-3) | *0,0928* ✓ | *0,0704* ✓ |

| predicción | `n3_s2` | `n4_s0` |
|---|---|---|
| P-1 rescate ≥ 0,30 | 0,0518 ✗ | 0,1049 ✗ |
| P-2 vecino corrupto ≥ 0,25 | 0,0482 ✗ | 0,1186 ✗ |
| P-3 control ≤ 0,10 | 0,0928 ✓ | 0,0704 ✓ |

## Lectura

**1. La clase 3 no es el mecanismo dominante.** El vecino corrupto explica entre el 5 % y el 12 % de
los errores de identidad, no la mitad que la revisión externa conjeturaba a partir del «48 % apagado».
En la enorme mayoría de los casos **el archivo está intacto y el error aparece al recuperar** → es
clase 2, **convertible en abstención**. Es la buena noticia para la campaña: no hay un piso oculto que
ninguna compuerta pueda bajar.

**2. Y hay que compararlo contra la tasa de fondo, que es lo que casi me hace sobre-interpretar.**
Entre los **aciertos** el vecino también sale «corrupto» un 7-9 % de las veces. Contra ese fondo:

- en `n4_s0` los errores tienen 0,1186 contra 0,0704 → **1,68×**, señal débil pero presente;
- en `n3_s2` los errores tienen 0,0482 contra 0,0928 → **va al revés**: hay *menos* corrupción entre
  los errores que entre los aciertos.

**Los dos checkpoints apuntan en direcciones opuestas.** Con eso, la afirmación honesta no es «la
escritura contribuye poco» sino **«no hay evidencia de que la corrupción de escritura tenga relación
con el error de identidad»**. Un efecto que cambia de signo entre dos checkpoints no es un efecto.

**3. El hallazgo lateral, y el más sólido: el error es determinista, no ruido.** La consulta
reformulada acierta el **0,958 / 0,978** en los casos donde el modelo ya acertaba, y sólo **0,052 /
0,105** en los que fallaba. La sonda funciona —el contraste es enorme— y lo que muestra es que
**cuando el modelo se equivoca en un episodio, se equivoca de forma estable**: no es que a veces
confunda al dueño, es que para ciertos episodios lo confunde siempre.

Eso tiene consecuencia directa sobre la abstención: un error sistemático puede venir con **alta
confianza**, porque el modelo no está dudando entre dos opciones sino comprometido con una asociación
equivocada. Encaja con que el umbral de confianza apagara sólo el 48 % de estos errores.

## Dos límites del instrumento, declarados

- **P-1 es inválida, no refutada.** La «consulta no ambigua» que construyo es `pregunta(rel, ent)`,
  que en N3 **es la misma forma que la consulta original**: lo elíptico es la *corrección* dentro de
  la sesión, no la *pregunta*. Así que re-pregunté lo mismo. Esa predicción no midió lo que decía
  medir y se retira.
- **La comparación errores-vs-aciertos no es simétrica.** En los errores el vecino es *el hecho que
  contiene el valor contestado*; en los aciertos no existe tal hecho y se toma `otros[0]`,
  arbitrario. Sirve como piso de referencia, pero no es un control pareado y no se debe leer como si
  lo fuera.

## Lo que queda abierto, y que ninguna de las dos hipótesis cubría

Vecino intacto (83 %) **y** rescate nulo (10 %) a la vez admiten una tercera historia que no estaba ni
en nuestra tabla ni en la de la revisión externa: **que la corrección elíptica no se ligue a nadie —
que se pierda al escribir sin corromper a ningún vecino.** El hecho propio quedaría con su versión
vieja y el modelo contestaría el valor de otra entidad simplemente porque el suyo nuevo no está.

Se testea preguntando por la versión **anterior** del hecho propio: si devuelve bien la v1 pero nunca
la v2, la corrección se perdió en la escritura. Es barato y es el siguiente paso natural.
