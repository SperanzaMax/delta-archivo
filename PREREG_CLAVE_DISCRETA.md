# PRE-REGISTRO · la clave del archivo como SÍMBOLO, no como dirección

**2026-08-30.** Se congela **antes de escribir el instrumento** y antes de correr nada.

Sale de una idea de Maxi, textual:

> «si en el mar de embedin de los pesos del modelo los embedin de la memoria es con letras?»

y de su premisa, que es la que hay que corregir primero:

> «cuando crese se corre los pesos y se pierde la ubicacion de la informacion»

---

## 1. La premisa está medida, y NO es la justificación

Hay que decirlo antes que nada porque cambia el motivo del experimento: **la deriva no muerde donde
Maxi teme que muerda.** R5.2 la midió brutal al arrancar (coseno 1,000 → 0,727 en 25 pasos), pero R6
mostró que es un fenómeno del **aprendizaje inicial**: sobre un modelo entrenado que se afina queda
en 0,882. Y después se intentó forzarla **tres veces** —E-I4 (0,9374), E-I4b (0,7804), E-I4c
(0,8531)— y nunca cruzó el umbral 0,70 ni cambiando la distribución.

> **Si este experimento se justificara sólo por inmunidad a la deriva, no habría que correrlo.**

Y hay una segunda razón para no correrlo por ahí: el chequeo del 30-ago
(`chequeo_clave_escalar.py`) midió tres codificaciones —escalar, denso+sello, discreta— y **dan el
mismo número dígito por dígito** en todos los niveles de ruido (1,0000 / 0,9993 / 0,8643 / 0,2957).
El cuello de botella es **inferir el tema**, y la codificación de la clave no lo toca.

## 2. La justificación que SÍ se sostiene, y sale de un negativo propio

`modelo.py:238` — la lectura es

```
sim = q · ak / sqrt(d) + penal        →        softmax(sim, -1)
```

**El softmax suma 1 SIEMPRE.** El modelo está obligado a leer algo aunque nada coincida, y eso no es
interpretación: es la razón mecánica que el `INFORME_SCORE_ARCHIVO_20260816` dejó anotada para su
propio negativo, que fue **`s_max` = 0,4984 y 0,5022, el azar exacto**, separando con-respuesta de
sin-respuesta. Enunciado del 16-ago, textual: *«el modelo selecciona con fuerza, y la fuerza de la
selección no codifica si lo que buscaba está — encuentra con la misma convicción cuando no hay nada
que encontrar»*.

> **Lo que una clave discreta cambia no es la robustez: es que «ninguna coincidencia» pasa a ser un
> evento OBSERVABLE.** Con direcciones continuas ese evento no existe por construcción, porque el
> vecino más cercano siempre existe. Con símbolos, «no matchea ninguno» es una condición exacta.

Y ahí conecta con el problema central del proyecto: hoy la ausencia se detecta por **calibración**
(la cabeza, supervisada, y el cierre del 27: «sabe y no lo convierte en decisión»). Si la ausencia
tuviera **firma mecánica**, dejaría de ser calibración y pasaría a ser lookup.

**Esa es la única razón por la que este experimento vale, y queda escrita para poder juzgarla.**

## 3. Fase 0 · BARATA, EN CPU, SOBRE CHECKPOINTS QUE YA EXISTEN

El patrón que ahorró dos campañas enteras este mes (`escriba` y el score del archivo): la pregunta
cara se contesta primero con lo que ya está en disco.

Se cuantiza **post-hoc** la clave `ak` de las unidades `v3_s*` (`lat2`, las mejores del proyecto:
acierto 0,9984-0,9992, `err_identidad` 0,0000) y se mide qué pasa. **No se entrena nada.**

- **Q-0 · BLOQUEANTE, y puede fallar.** Cuantizar no puede destruir la memoria: acierto con claves
  cuantizadas ≥ 0,90 del acierto continuo, en 2 de 3 semillas. **Si falla, la idea muere acá y no se
  prueba una segunda cuantización.** Se barre el número de símbolos por código y se reporta la curva
  entera, no un punto.
- **Q-1 · PRINCIPAL.** Con claves discretas, el estadístico «¿coincide algún símbolo?» separa
  con-respuesta de sin-respuesta con **AUC ≥ 0,70** en 2 de 3. La referencia es el **0,4984 / 0,5022
  del continuo**, que es el azar exacto, así que cualquier cosa por encima de 0,60 ya sería señal
  donde no había ninguna.
- **Q-2 · CONTROL QUE PUEDE FALLAR, y es el que hace válido a Q-1.** El mismo estadístico sobre
  claves cuantizadas **con símbolos barajados** (misma cantidad de símbolos, misma distribución
  marginal, asignación rota) tiene que quedar en 0,45-0,55. Si el barajado también separa, lo que
  mide es la ESCALA y no la coincidencia. Es la lección del 22-ago: *el nulo no se le exige un valor,
  se compara contra él.*
- **Q-3 · ESPECIFICIDAD.** La separación tiene que ser mayor en `nose_ent` (falta la entidad, así que
  no debería coincidir nada) que en `nose_rel` (la entidad está archivada). Si son iguales, el
  estadístico no está midiendo coincidencia. Predicción derivada del mecanismo, y es la tensión que
  quedó abierta desde el dictamen del 16-ago sin resolverse.

## 4. Antes de cualquier fase de entrenamiento · la regla nueva de HOY

`PRECISION_RECOMPENSA_L_CE.md` (4b61894e) dejó escrito hoy, y se aplica acá por adelantado:

> **Antes de contrastar dos condiciones, medir cuánto gradiente mueve el cambio contra el resto de la
> pérdida. Un contraste sobre el 3 % de la pérdida no es un contraste.**

Así que la Fase 1 **no se lanza sin** medir primero el gradiente que llega a las claves discretas
contra el que llega al resto. Va como chequeo bloqueante y no como riesgo a vigilar — la otra regla
de esta semana.

## 5. El obstáculo real, declarado por adelantado

**Elegir un símbolo es discreto y el gradiente no pasa.** Es el mismo obstáculo (1) que el §
«PRECISIÓN DECISIVA» del 8-ago identificó para todo el brazo, y no se resuelve con voluntarismo. Hay
tres soluciones publicadas y **se elige UNA, declarada acá antes de ver nada**: *straight-through* con
pérdida de compromiso, como VQ-VAE. Las otras dos (Gumbel-softmax, product-key) **no se prueban** si
ésta falla — misma cláusula que cerró la gemación y el trípode.

Antecedentes que hay que citar y no redescubrir: VQ-VAE (van den Oord 2017), *Product-key Memory*
(Lample 2019), *Memory Layers at Scale* (Meta). **Ninguno usa el código discreto para detectar
ausencia**, que es el corte de acá; lo usan para capacidad y para ruteo.

## 6. Cómo se lee cada desenlace, escrito ANTES

| celda | lectura | qué se hace |
|---|---|---|
| **Q-0 falla** | la memoria no tolera el código discreto | **muere acá**, cero GPU gastada |
| **Q-1 y Q-2 cumplen** | la ausencia tiene firma mecánica donde el continuo daba azar | se diseña la Fase 1 con el chequeo de gradiente del §4 |
| **Q-1 sí, Q-2 no** | mide la escala, no la coincidencia | negativo, y explica por qué |
| **Q-1 falla con Q-0 bien** | ni discretizando aparece la firma | **cierra la vía**, y refuerza el cierre del 21-ago: la ausencia no está en la interfaz de memoria por ninguna representación |
| **Q-3 falla con Q-1 bien** | separa, pero no por coincidencia | se reporta como correlación, no como mecanismo |

## 7. Criterio de abandono

> Si **Q-0 o Q-1 fallan**, la clave discreta se cierra y no se prueba una segunda forma de
> cuantización. Sería la sexta representación probada sobre la misma pregunta (geometría, sello,
> escalar, slot, cabeza, discreta) y el patrón del proyecto es que la representación no es la palanca.

## 8. Lo que NO contesta

- **No prueba que escale.** 863.859 params, 242 tokens, archivo de ~40 entradas.
- **No dice que el modelo sepa cuándo no sabe.** Si Q-1 cumple, dice que la ausencia es *detectable
  desde la interfaz*, que es una condición previa y no el objetivo.
- **No toca la deriva**, por el §1: su régimen no ocurre en un modelo convergido.
- **Y no arregla el cuello de botella medido hoy**, que es inferir el tema. Q-1 es sobre detección de
  ausencia, no sobre recuperación.
