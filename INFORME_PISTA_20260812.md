# La pista no era el problema: el sujeto no leía

**2026-08-12.** Resultado de `PREREG_PISTA.md` (SHA-256 `fef8aa5a…`, script `787bc657…`, congelados
antes del dato) más dos controles que el prereg no anticipaba y que dieron vuelta el diagnóstico.

**Titular: el chequeo bloqueante del 11-ago queda INVALIDADO. Y el banco ECO sobrevive igual, por una
razón mejor que la que teníamos.**

---

## 1. Lo pre-registrado

`m = 4` entidades activas, `d ∈ {0,5}`, 20 casos por celda, opciones barajadas, `albert:v4.0` a
temperatura 0 — el mismo sujeto del 11-ago.

| condición | acc | azar |
|---|---|---|
| `desnuda` (réplica del 11-ago) | 0,250 | 0,250 |
| `recencia` (marcador explícito) | 0,350 | 0,250 |
| `tipada` (referencia **objetivamente** recuperable) | **0,250** | 0,250 |

| predicción | veredicto |
|---|---|
| **P1** replicación en [0,05 · 0,45] | **CUMPLE** (0,250) |
| **P2** `recencia` − `desnuda` ≥ +0,25 | **NO CUMPLE** (+0,100) |
| **P3** `tipada` ≥ 0,80 | **NO CUMPLE** (0,250 = el azar exacto) |
| **P4** modo en primera mención ≥ 0,40 | **NO CUMPLE** (modo sí en posición 1, pero 0,375) |

P4 quedó a 0,025 del umbral. Según §5 del prereg se lee tal cual: **no cumple**, y no se busca una
segunda lectura. La distribución (0,375 / 0,200 / 0,175 / 0,250) se aparta del uniforme en la
dirección que predice *centering*, pero con este sujeto el dato no significa nada — ver §3.

## 2. Lo que P3 obligó a hacer

`tipada` está construida para que la respuesta sea verificable **sin ninguna convención**: de los
cuatro hechos, uno solo es de tipo `director`; los otros tres son ciudades y empresas; el valor nuevo
es un nombre de persona. La respuesta es única y no depende de recencia, saliencia ni foco. Inspección
manual de casos generados: correctos, con la respuesta entre las opciones.

Que un sujeto dé **el azar exacto** en una tarea así no es un dato sobre la tarea. Y tres celdas
clavadas en 0,250 son exactamente el tipo de número limpio que en este proyecto ya resultó ser un
artefacto dos veces. Se descompuso en dos preguntas sobre **el mismo material**:

| pregunta | qué mide |
|---|---|
| «¿cuál de estas entidades tiene un **director** mencionado?» | extracción — ni siquiera participa la corrección |
| «¿a cuál se refiere la corrección?» | resolución de la referencia |

Y se controló el confound de instrumento: responder exige mapear la entidad a su número en una lista
barajada, que es indexación posicional y no es lo que se quiere medir. Se repitió todo pidiendo el
**nombre** en vez del número.

### albert:v4.0 — las cuatro celdas

| | número | nombre |
|---|---|---|
| extracción | 0,200 | 0,250 |
| resolución | 0,200 | 0,200 |

**No es el formato. No es la convención. No es la elipsis.** El sujeto no identifica cuál de cuatro
entidades tiene un director mencionado en un texto de nueve líneas donde una sola línea lo dice.

### qwen2.5-coder (7B) — el mismo material

| | número | nombre |
|---|---|---|
| extracción | 0,950 | **1,000** |
| resolución | 0,300 | **0,550** |

## 3. Consecuencia 1: el chequeo del 11-ago no midió lo que dijo medir

`DISENO_BANCO_ELIPTICO.md` §10 concluyó **«el banco DISCRIMINA»** porque la exactitud caía a
0,150-0,400 con `m > 1`. Con lo de hoy, esa caída es **incapacidad del sujeto**, no dificultad de la
tarea: el mismo modelo da el azar cuando la respuesta es objetivamente única, y falla la extracción
más simple posible. Un instrumento que devuelve azar ante una tarea trivial no puede sostener ningún
veredicto sobre una difícil.

**El error de método, que es el hallazgo transferible.** El chequeo se validó con `m = 1 → 1,000` y
eso se leyó como «el modelo entiende la consigna y la tarea es resoluble». Pero con `m = 1` la lista
de candidatas tiene **un solo elemento**: responder «1» acierta siempre **sin leer la conversación**.
El control de sanidad estaba vacío. Y como estaba vacío, las celdas `m > 1` no tenían contra qué
contrastarse, y su caída admitía las dos lecturas —tarea difícil o sujeto incapaz— sin poder
separarlas.

> Un control de sanidad tiene que poder **fallar** si el sujeto no sirve. `m = 1` no podía.

Cuarta vez en el programa que un número limpio esconde un artefacto: la meseta de E1 (27-jul), el
D-012 de E3, el cero por posición del 11-ago, y este.

## 4. Consecuencia 2: ECO sobrevive, y ahora sobre evidencia real

Con un sujeto que **sí lee**, sobre material idéntico y en el mejor formato para él:

> **encontrar el hecho: 1,000 · ligarle la corrección elíptica: 0,550**

Una brecha de **0,450** entre dos preguntas sobre el mismo texto, donde la única diferencia es que
una exige resolver a qué se refiere «no, it's Kira Osei». Eso es lo que ECO quería medir, y ahora
está medido con el control de sanidad que ayer faltaba: **la extracción al 1,000 prueba que el sujeto
lee, y por eso el 0,550 sí se puede atribuir a la elipsis.**

El eje se corrige, además, en su interpretación. No es «cuántas entidades hay» lo que rompe: es
**ligar una corrección sin sujeto explícito a un hecho que el modelo ya localizó sin esfuerzo**.

## 5. Dos hallazgos laterales que valen aparte

**(a) El formato de respuesta subestima, y sólo en las tareas difíciles.** En qwen, pedir el número
en vez del nombre cuesta **0,050** en extracción (0,950 vs 1,000) y **0,250** en resolución (0,300 vs
0,550). El costo de la indexación posicional **no es constante**: aparece cuando la tarea ya es
exigente. Consecuencia práctica para cualquier evaluación con opción múltiple —incluidos los bancos
de §5 de ECO— y consecuencia inmediata para el banco: **se responde con nombres**.

**(b) Abstención espontánea.** En las dos celdas de resolución de qwen hubo **4 de 20 respuestas
ilegibles** (20 %), contra **0 de 20** en las dos de extracción. El modelo no emite un candidato
cuando no puede resolver la referencia — y no lo hace nunca cuando la tarea es de extracción. Eso es
materia prima directa para `SER`: hay una señal de «no sé» que el banco puede medir en vez de forzar
una elección. Conecta con el plan del modelo que sabe que no sabe.

## 6. Qué se hace con esto

1. **Anular §10 de `DISENO_BANCO_ELIPTICO.md`** y reemplazar su veredicto por el de hoy.
2. **Sujeto mínimo declarado**: ningún modelo entra al banco sin pasar la **extracción ≥ 0,90** sobre
   el mismo material. Es la compuerta que faltaba, y es la que separa incapacidad de dificultad.
3. **Formato de respuesta por nombre**, no por número.
4. **Registrar las abstenciones** como categoría propia, no como ruido.
5. Pendiente inmediato: repetir con `m ∈ {1,2,4,8}` sobre qwen para tener **la curva** del eje `m`,
   ahora que hay un sujeto que puede. El `m = 1` deja de ser control (no puede fallar) y pasa a ser
   sólo el extremo fácil de la curva.

**Costo del día:** 6 corridas locales, ~1 h 40 de CPU, máx 57 °C.
