# E-I3d: sí compara turnos — pero sólo sabe buscar el máximo

**2026-08-13.** Pre-registro en el docstring de `interno/ei3d_turnos_moviles.py`, hash fijado antes de
correr. Cierra el límite declarado en §5 de `INFORME_EI3C_20260813.md`.

En E-I3c los turnos eran **fijos por rol** —v1 siempre 0-5, v2 siempre 6-8, v3 siempre 9-11—, así que
al modelo le alcanzaba con aprender la tabla «los embeddings 9,10,11 son la vigente». Eso es una
correspondencia slot→rol, no un orden, y el control `barajado` no las separa: aleatorizar el sello
rompe la tabla igual que rompe el orden.

Acá los turnos **se mueven**: por muestra se sortean 12 turnos distintos de un rango de 32, se ordenan
y se reparten respetando el orden real de escritura. El turno 9 puede ser una primera versión en una
muestra y una tercera en la siguiente. Ninguna tabla fija resuelve la tarea; sólo comparar.

---

## 1. Resultado (3 semillas, 12000 pasos)

| condición | **vigente** | **ANTERIOR** | una sola versión |
|---|---|---|---|
| `ninguno` | 0,3247 | 0,2977 ± 0,0326 | 0,9115 |
| `sello` | **0,9644** | **0,3142** ± **0,5330** | 0,9957 |
| `barajado` | 0,2804 | 0,2925 ± 0,0266 | 0,9175 |

| predicción | veredicto |
|---|---|
| **P-1** ANTERIOR(sello) ≥ 0,80 | **NO CUMPLE** — 0,3142 · **pero ver §3** |
| **P-2** VIGENTE(sello) ≥ 0,80 | **CUMPLE** — 0,9644 |
| **P-3** sello − barajado ≥ +0,30 en ANTERIOR | **NO CUMPLE** — +0,0217 |

## 2. La conclusión automática del script es incorrecta, y la corrección importa

El script imprime «NO CUMPLE → era tabla de slots». **No lo era, y el propio resultado lo desmiente:
si el modelo hubiera aprendido una tabla slot→rol, la versión VIGENTE también se habría caído al
mover los turnos.** No se cayó: 0,9644, contra 0,3247 sin sello y 0,2804 con el sello barajado.

Con los turnos moviéndose de muestra en muestra, la única forma de acertar la vigente es **comparar**
los turnos de las entradas que responden a esa clave y quedarse con el mayor. Eso es usar el orden
como orden, y está medido.

El pre-registro anticipó exactamente esta posibilidad: *«es posible que "la más nueva" sobreviva a los
turnos móviles (comparar un máximo es más fácil) y que se caiga sólo la anterior. Ese resultado
partido sería informativo, no un empate.»* Es el resultado partido.

## 3. La distribución de ANTERIOR es bimodal: el promedio no describe a nadie

Por semilla: **0,0052 · 0,9297 · 0,0078**. No es una media con dispersión, son **dos poblaciones**:

- dos semillas caen en el **atajo de la recencia**: resuelven la vigente en 0,96 y contestan la
  anterior *por debajo del azar* (1/64 = 0,0156) — no es ruido, es error sistemático: devuelven la
  vigente cuando se les pide la anterior;
- una semilla aprende la operación completa y llega a **0,9297**, mientras `barajado` y `ninguno` no
  pasan de 0,33 en ninguna de sus seis corridas.

**Que una semilla llegue a 0,93 con el sello real, y ninguna sin él, muestra que la operación
"penúltimo" es alcanzable y que la información viene del sello.** Lo que falla no es la posibilidad,
es la convergencia.

Reportar el 0,3142 como «el resultado» sería la sexta vez que en este programa una media esconde su
distribución (D-012 en E3, la meseta falsa de E1, la `τ` bimodal de la elíptica).

## 4. Lo que queda afirmado, con precisión

1. **Comparar turnos para hallar el máximo: sí, y es robusto.** 0,9644 con turnos móviles, en las
   tres semillas. Descarta la tabla de slots y valida hacia atrás la lectura de E-I3c.
2. **Ordenar para hallar el penúltimo: alcanzable, pero raro.** 1 de 3 semillas. El gradiente prefiere
   el atajo porque resuelve perfecto la pregunta frecuente.
3. **La información viene del sello real.** `barajado`, con los mismos valores desordenados, no llega
   a 0,33 en ninguna corrida, y además hunde la vigente a 0,2804 — por debajo de no tener sello.

En una frase: **el lector aprende a leer el reloj para saber qué hora es ahora, no para reconstruir la
secuencia de lo que pasó.**

## 5. Consecuencia práctica (ya incorporada a `DISENO_MICRO_LM.md` §7.bis)

Para el micro-LM no alcanza con poner el archivo en la arquitectura y entrenar: hay que **sesgar la
búsqueda** hacia la solución que compara —curriculum con preguntas por versiones no vigentes desde el
principio, balance de tipos de pregunta, y posiblemente una pérdida auxiliar sobre el orden—, y
**reportar por semilla, nunca sólo la media**.

## 6. Límites

3 semillas y 12000 pasos, cuando E-I3c mostró que a ese presupuesto 2 de cada 5 semillas no habían
convergido y que duplicarlo llevó una de 0,5547 a 0,9531. **Es posible que parte del 0,005 sea otra
vez presupuesto**: la prueba directa es correr las semillas del atajo a 24000-36000 pasos y ver si
saltan de modo. Mientras no se haga, la afirmación «el gradiente prefiere el atajo» vale para este
presupuesto, no para el mecanismo en general.
