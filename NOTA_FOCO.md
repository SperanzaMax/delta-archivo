# La identidad se captura al escribir, no se reconstruye al leer

**2026-08-12, nota de razonamiento.** No es un pre-registro. Es la hipótesis que sale de mirar el
0,0000 del régimen elíptico desde el objetivo de fondo en vez de desde el mecanismo.

## 1. El desvío que todos los sistemas comparten

Cuando alguien te dice «no, es Beto», no rastreás la conversación hacia atrás buscando de quién se
hablaba. Lo sabés porque **estabas ahí**, con el foco puesto en algo. La referencia nunca fue un
problema a resolver: fue un estado que ya tenías.

Los sistemas desplegados hacen exactamente lo contrario. Ingieren **texto ya cerrado** —chunks,
sesiones, transcripts— y le piden a un LLM que **reconstruya** la identidad después. Zep/Graphiti
arman su grafo temporal en una pasada posterior; Mem0 y Letta extraen hechos de turnos ya
consumidos. Para cuando corre el extractor, el estado que resolvía la referencia **ya se tiró**.

Eso no es un detalle de implementación. Es el mismo error que E2-b midió en otro plano:

> **E2-b:** un acceso al contexto en el primer bloque da 0,9998; el mismo acceso en el último, 0,4990.
> No es cuántas veces se accede: es **antes o después del cómputo**.

Traducido a memoria: **la identidad tiene que entrar en el momento de la escritura**. Reconstruirla
al leer es acceso tardío, y el 0,0000 de las correcciones elípticas crudas es su caso extremo.

## 2. Qué se propone, exactamente

Tres formas de archivar la misma corrección elíptica:

| condición | qué se guarda | costo por corrección |
|---|---|---|
| `crudo` | `emb("no, it's Beto")` | 1 embedding. **Medido: 0,0000** |
| `hidratado_LLM` | un LLM resuelve la referencia y se archiva el hecho completo | **1 llamada a un LLM.** Medido hoy con m=4: apenas por encima del azar |
| `foco` | el hecho crudo **más el puntero de foco que el estado conversacional ya tenía** | **O(1), sin modelo** |

El punto no es que `foco` sea más listo. Es que **no necesita ser listo**: la información que el
hidratador intenta recuperar con un modelo de 2,5 B de parámetros estaba disponible gratis un turno
antes, y el sistema la descartó.

## 3. Dónde esto se puede caer, dicho antes de medirlo

**(a) Que la regla de foco sea trivialmente correcta por construcción.** Es el riesgo serio. En un
generador sintético donde la corrección apunta siempre al último hecho, «última entidad mencionada»
acierta el 100 % sin saber nada. La guarda: la verdad de base debe fijarse por un criterio
**independiente** de la regla de foco, y toda regla de foco debe ser **ciega al futuro** (mira sólo
los turnos anteriores a la corrección). Con turnos de relleno sobre otras entidades, la regla ingenua
falla por construcción — y ahí empieza a ser un experimento y no una demostración.

**(b) Que el foco no sea único.** Tres candidatas, de menos a más sofisticada:
`F1` última entidad mencionada · `F2` última mencionada **compatible con el tipo** del valor nuevo ·
`F3` centro de atención con herencia entre turnos (*centering*). Cada una tiene su propia tasa de
error, y medirla **es** el experimento. Si todas fallan parejo, la tesis se cae.

**(c) Que el caso real no dé acceso al flujo.** Si a un sistema le entregan un transcript cerrado,
el foco ya se perdió y esto no aplica. Lejos de ser una objeción, es el enunciado: **es un argumento
sobre la arquitectura de ingesta, no sobre el algoritmo de recuperación.** Quien controla el momento
de la escritura —un asistente que conversa— tiene la información; quien indexa a posteriori, no.

**(d) Que ya esté hecho.** El estado de diálogo (*dialogue state tracking*) es un campo entero, y los
sistemas orientados a tarea llevan slots con foco desde hace años. Lo que no se hizo, hasta donde
llega el barrido, es **usar ese estado como parte de la clave de indexación de una memoria
persistente entre sesiones**. Hay que verificarlo antes de reclamar nada.

## 4. Por qué esto sí apunta al objetivo de fondo

La línea de la gemación cerró en negativo tres veces, y cerró bien: el problema no era dónde poner el
vector. Pero las tres corridas comparten un supuesto que nunca se puso a prueba — que lo único
disponible al archivar es **el texto**. Un humano no archiva texto: archiva texto **con su situación**.

Si la tesis se sostiene, el enunciado que queda es corto y ejecutable:

> No le pidas a un LLM que reconstruya de qué se estaba hablando. Guardá de qué se estaba hablando.

Y es el mismo dictamen determinista que el programa ya confirmó para la frescura (2606.01435),
extendido de la **versión** a la **identidad**.
