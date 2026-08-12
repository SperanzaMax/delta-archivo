# ECO — un banco para correcciones elípticas

**Elliptical Correction Benchmark.** Diseño, 2026-08-11. **No es un pre-registro**: es la
especificación de un instrumento. Los experimentos que se corran sobre él llevarán su propio prereg.

---

## 1. Por qué hace falta, dicho con lo que ya está medido

El experimento de hoy (`INFORME_ELIPTICA.md`) dejó un número que no es un matiz: una corrección
conversacional que **no nombra a su sujeto** obtiene **0,0000 de recuperación en las diez semillas**.
No baja: nunca entra. Y el fallo es **silencioso** — el índice devuelve cinco candidatos ordenados con
confianza, ninguno correcto (el bueno a 0,4237, el de otra entidad a 0,4064).

Eso no es un hallazgo sobre la gemación. Es un **régimen donde todos los sistemas de la capa de
aplicación sacan cero**, y ninguno lo reporta porque ningún banco se lo pregunta:

- **LongMemEval** mide *knowledge updates* sobre sistemas completos, con sesiones cuyo texto es
  naturalmente informativo. No aísla la indexación ni controla el grado de elipsis.
- Los sistemas desplegados (Zep/Graphiti, REMem, Mem0, Letta) resuelven conflictos en la capa
  simbólica **suponiendo que el hecho llegó identificado**. La pregunta de este banco es
  precisamente qué pasa cuando no llega así.

**Lo que ECO aporta y no existe:** un eje de dificultad *controlable* sobre el problema real —
cuánta información de identidad conserva una corrección — con la indexación aislada y todo lo demás
fijo.

## 2. La limitación del experimento de hoy, que el banco tiene que reparar

Sería deshonesto vender lo de hoy como banco tal cual está. Tiene tres defectos concretos:

1. **Sólo cuatro formulaciones elípticas fijas.** Con 60 valores únicos eso da **240 cadenas
   posibles** para 24 000 entradas. Dos entidades corregidas al mismo valor producen vectores
   idénticos bit a bit.
2. **La elipsis es binaria**: o el texto trae la entidad o no la trae. En una conversación real hay
   gradaciones (pronombre, hiperónimo, mención parcial, elipsis total).
3. **Una sola entidad en juego.** Si sólo se habla de una cosa, resolver la referencia es trivial.
   El caso difícil —y el realista— es cuando hay varias entidades activas y la corrección es
   **ambigua entre ellas**.

El punto 3 es el que convierte esto en un banco con identidad propia, y es donde apunta §3.2.

## 3. Diseño

### 3.1 Eje A — grado de elipsis (`e`)

Cinco niveles, de más a menos información de identidad en el texto de la corrección:

| nivel | forma | ejemplo |
|---|---|---|
| `e0` | auto-contenida | «The director of Helios Laboratory is Beto.» |
| `e1` | entidad parcial | «The Helios director is Beto.» |
| `e2` | hiperónimo | «The lab's director is Beto.» |
| `e3` | pronominal | «Its director is Beto.» |
| `e4` | elipsis total | «No, it's Beto.» |

`e0` y `e4` son los dos extremos ya medidos (0,8541 y 0,4247 de coseno con la consulta). Los tres
niveles intermedios son nuevos y son los que dan la **curva**, que es lo que hace útil a un banco:
no un veredicto binario sino una función de degradación.

### 3.2 Eje B — ambigüedad referencial (`m`) ← el eje que no existe en ningún banco

Número de entidades **activas** en la ventana conversacional cuando llega la corrección:
`m ∈ {1, 2, 4, 8}`.

Con `m = 1` resolver la referencia es trivial. Con `m = 8`, «no, es Beto» es ambiguo **incluso para
un humano** sin más contexto, y un hidratador puede resolverlo **con la entidad equivocada** — que es
peor que no resolverlo, porque planta un hecho falso en la memoria.

Ese caso —hidratación incorrecta— es la limitación que el paper de hoy declara y no modela. Acá se
mide de frente.

### 3.3 Eje C — profundidad de revisión (`K`)

`K ∈ {1, 2, 4, 8}` revisiones sucesivas del mismo hecho, heredado del harness actual.

### 3.4 Eje D — distancia (`d`)

Turnos intermedios entre el hecho original y su corrección: `d ∈ {0, 5, 20}`. Modela que la
corrección puede llegar mucho después, que es el caso que define la memoria entre sesiones.

### 3.5 Generación

Ampliar el espacio de textos para eliminar la degeneración: plantillas parametrizadas por
formulación × registro × puntuación, con un objetivo declarado de **≥ 5000 cadenas únicas** por
nivel de elipsis. El censo de vectores únicos (C1 de la compuerta de encoders) es condición de
admisión: si un nivel no lo pasa, no se corre.

## 4. Métricas

Las dos primeras son las del harness actual. La tercera es nueva y es la que hace al banco.

1. **`recall@k` de la versión vigente.** ¿Está la corrección al día entre las $k$ recuperadas?
2. **Cobertura.** ¿Están la vigente y la anterior a la vez? (para tareas que necesitan el historial).
3. **Tasa de error silencioso (`SER`).** Fracción de consultas donde el sistema devuelve un candidato
   **con confianza alta** y es **incorrecto** — sea de otra entidad, sea una versión superada.

`SER` es la métrica que faltaba. Un sistema que dice «no sé» es manejable; uno que responde con
seguridad un dato equivocado envenena todo lo que venga después. Hoy medimos que el fallo es
silencioso; `SER` lo convierte en número comparable entre sistemas.

**Reportar `SER` junto a `recall` es la propuesta metodológica del banco**, del mismo modo que en
detección no se reporta precisión sin recall.

## 5. Baselines obligatorios

Un banco sin líneas de base fuertes no mide nada. Cinco, en orden creciente de exigencia:

| baseline | qué es |
|---|---|
| `sin` | sin archivo — piso |
| `crudo` | archiva el texto de la corrección tal cual (lo que hace un sistema ingenuo) |
| `hidratado_oráculo` | co-referencia resuelta **perfectamente** (techo) |
| `hidratado_τ` | hidratación con tasa de error τ, con **error de dos tipos**: no hidratar, o hidratar con la entidad equivocada |
| `hidratado_LLM` | co-referencia resuelta por un modelo real, en la ventana de `m` entidades |

El quinto es el que ancla el banco a la realidad: sitúa a los sistemas actuales **en el eje**, en
lugar de dejar τ como un parámetro imaginario. Es la respuesta a la limitación «τ es impuesto, no
medido» que declara el paper.

## 6. Qué tiene que producir el banco para valer

- **Una curva**, no un punto: `recall` y `SER` como función de `e` y de `m`.
- **La frontera de utilidad**: el par $(e, m)$ a partir del cual todo baseline cae por debajo de un
  umbral usable. Ese es el enunciado que un ingeniero puede llevarse.
- **La ubicación de los sistemas reales sobre esa curva**, vía `hidratado_LLM`.

## 7. Riesgos declarados antes de invertir

1. **Que sea fácil.** Si `hidratado_LLM` resuelve todo hasta `m = 8`, el banco no discrimina y hay
   que decirlo. Chequeo barato antes de construir nada: medir la exactitud de co-referencia de un
   modelo chico con `m = 8`. **Si supera el 95 %, el banco no vale la pena y esto se archiva.**
2. **Que sea artificialmente difícil.** Si la degeneración de textos vuelve imposible la tarea por
   colisiones y no por elipsis, se mide un artefacto. El censo de únicos (§3.5) es la guarda.
3. **Que ya exista.** Antes de nombrarlo hay que barrer si algún banco reciente de memoria
   conversacional incluye un eje de elipsis o de ambigüedad referencial. Si existe, esto pasa a ser
   una extensión y se declara como tal — no se reinventa.
4. **El costo real es la validación humana.** Un banco sintético sin ninguna ancla en correcciones
   reales es criticable con razón. La versión mínima defendible incluye un conjunto chico de
   correcciones humanas anotadas.

## 8. Costo estimado

| etapa | costo |
|---|---|
| Chequeo del riesgo 1 (¿discrimina?) | **~1 hora**, local. **Bloqueante** |
| Barrido de literatura (riesgo 3) | ~2 horas |
| Generador con los 5 niveles + censo | ~medio día |
| Campaña completa (5 `e` × 4 `m` × 4 `K` × 10 semillas) | embeddings locales, unas pocas horas de CPU |
| `hidratado_LLM` | depende del modelo; con uno local es gratis en dinero y caro en tiempo |

**Nada de esto se construye antes del chequeo del riesgo 1.** Es la lección de hoy: verificar que el
contraste tenga rango antes de invertir en el instrumento.

## 9. Lo que este banco NO va a resolver

Sigue siendo indexación sobre un codificador congelado. No dice nada del índice co-entrenado, que es
el hueco grande. Su valor es distinto: **da una vara comparable** para el problema que sí está
ocupado por sistemas desplegados, y ubica a cada uno en un eje que hoy nadie mide.
