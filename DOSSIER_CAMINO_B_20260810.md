# Camino B — qué hay en el mundo, y qué queda para nosotros

**2026-08-10.** Barrido del estado del arte de **memoria persistente entre sesiones** (el camino B de
`DOS_CAMINOS.md`), hecho *antes* de invertir en la línea. Sirve de **base** en lo que ya está resuelto
y de **pared** en lo que ya está ocupado.

---

## 1. La pared: el benchmark ya existe y mide exactamente nuestra tarea

**LongMemEval** (Wu et al., ICLR 2025) es el estándar del problema. Define cinco habilidades:
extracción de información, razonamiento multi-sesión, razonamiento temporal, **knowledge updates** y
abstención.

El tipo de pregunta **knowledge updates** es literalmente nuestra tarea de hechos versionados: *«un
hecho enunciado en una sesión se cambia en una posterior»*. Escala: 500 preguntas curadas;
`LongMemEval_S` con historiales de ~115 000 tokens sobre ~40 sesiones por usuario, y `LongMemEval_M`
llevándolo a ~500 sesiones.

**Consecuencia directa para nosotros:** no podemos presentar la tarea de hechos como una tarea nueva.
Es una instancia sintética y controlada de una categoría que ya tiene benchmark, y hay que decirlo así.
Lo que sí es defendible es *por qué* usar la versión sintética: LongMemEval mide sistemas completos
(LLM + memoria + prompt), donde no se puede aislar de dónde viene el efecto. Nuestro harness aísla
**el mecanismo de indexación** con todo lo demás fijo. Son niveles de análisis distintos, no
competidores — el mismo argumento que sostiene a Ligamento frente a los modelos de frontera.

## 2. La pared, parte 2: alguien ya publicó nuestra conclusión de diseño

**«Don't Ask the LLM to Track Freshness: A Deterministic Recipe for Memory Conflict Resolution»**
(arXiv 2606.01435). El título es la tesis: para resolver conflictos de memoria, **no le pidas al LLM
que rastree qué es lo más reciente — usá una regla determinista**.

Eso es, en otras palabras, nuestro dictamen de R1+R4: **geometría para agrupar, metadato para
ordenar**. Llegamos por vías independientes y a niveles distintos (ellos en capa de aplicación,
nosotros midiendo la geometría del índice), pero la conclusión de diseño es la misma.

**Cómo se usa esto a favor y no en contra:** nuestro **P1 negativo** deja de ser sólo un negativo. P1
predecía que la geometría de la gemación mejoraría la cobertura sobre guardar ambas versiones sin
estructura, y **no lo hizo**. Eso es una **confirmación independiente y cuantificada** de la tesis
determinista, obtenida en un banco donde las variables están aisladas. La estructura geométrica no
compra el orden; el metadato sí.

**Y acota la ambición de la gemación**, que es sano: su valor —si lo tiene— no está en ordenar
versiones, sino en **agrupar** el clúster de un recuerdo cuando el índice se puebla. Eso es lo que P4
está midiendo.

## 3. Lo que ya está ocupado (capa de aplicación)

| sistema | qué hace con las contradicciones |
|---|---|
| **Graphiti / Zep** | Cuando un hecho nuevo contradice a uno viejo, **invalida el viejo pero conserva el registro histórico**, con ventana de validez («X fue verdadero de T1 a T2»). |
| **REMem** (ICLR 2026) | Grafo de memoria híbrido con *gists* y hechos con marca temporal. +3,4 % y +13,4 % absolutos sobre Mem0 y HippoRAG 2 en recolección episódica y razonamiento. |
| **NeuSymMS** (2605.17596) | Memoria neuro-simbólica auto-curada para agentes persistentes. |
| **Supersede** (2606.27472) | Diagnostica y **entrena** el «memory-update gap»: la brecha específica en actualizar memoria. |
| **Mem0 / Letta** | Memoria como servicio, capa de aplicación. |

Sobre las cifras, con cuidado: en LongMemEval con GPT-4o se reporta **Zep 63,8 % vs Mem0 49,0 %**,
pero Mem0 reporta **94,4 %** con otra configuración. La discrepancia es de metodología, no de
capacidad — un recordatorio de que en este subcampo los números publicados no son comparables entre sí
sin leer el protocolo. Es exactamente el problema que motiva medir en un banco controlado.

**Todos operan en la capa de aplicación**: grafos, metadatos, punteros explícitos, invalidación
simbólica. Ninguno mete el índice **adentro de la red**.

## 4. El hueco que queda

Juntando esto con lo que ya sabíamos (`DOSSIER_LITERATURA_20260808.md`, sobre el scoop parcial
intra-secuencia de HOLA/HAM/Tensor Cache):

1. **Índice persistente entre sesiones + regla delta, co-entrenados.** Los competidores
   intra-secuencia mueren en el forward; los de capa de aplicación no tocan la red. El cruce sigue sin
   hacerse, y sus dos obstáculos siguen siendo los mismos: **el gradiente no fluye por la selección
   top-k**, y el **stale index** (los embeddings se mueven mientras el índice está congelado).
2. **Capacidad medida, no rendimiento en benchmark.** Todos reportan exactitud de sistema completo.
   Nadie reporta **cuántos hechos versionados entran antes de que el mecanismo falle**, con N
   controlado, semillas y potencia estadística. Es el mismo hueco que E1 llenó para atención híbrida.
3. **La pregunta que sale de cruzar los dos caminos, y que es la más nuestra.** E2-b midió que **el
   contexto es precondición del cómputo, no corrección**: el mismo acceso vale 0,9998 en el primer
   bloque y 0,4990 en el último. **Si eso vale también para un archivo persistente, entonces la
   respuesta recuperada del índice no puede inyectarse al final** — y eso es exactamente lo que hace
   todo pipeline RAG y todo sistema de la tabla de arriba.

   Formulable, falsable, y nadie la puede formular sin haber medido antes lo de E2-b.

## 5. Qué se hace con esto

- **Base:** adoptar la tesis determinista (metadato para el orden) en vez de pelearla; ya está en la
  enmienda E2 y P1 la corrobora desde otro ángulo.
- **Pared:** no presentar la tarea de hechos como novedosa, sino como el **instrumento aislado** de una
  categoría que LongMemEval mide a nivel de sistema.
- **Puerta:** el punto 3 — «¿dónde tiene que entrar lo recuperado?» — es la pregunta que hereda el
  camino B del camino A, y la única de las tres que no está ocupada por nadie.
