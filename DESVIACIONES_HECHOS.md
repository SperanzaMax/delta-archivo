# Desviaciones del pre-registro — tarea de hechos versionados

Registradas **antes** de observar cualquier resultado del experimento.

## D1 — Semillas de la parte generativa (2026-08-09, antes de generar embeddings)

**Prereg §6:** "10 semillas para la asignación entidad/atributo/valor y para los ejes `t̂`".

**Lo que se hace:** un único conjunto de 3.000 entidades generado con semilla 0. Las 10 semillas
del análisis controlan el submuestreo (1.000 entidades por semilla) y los ejes `t̂`.

**Motivo:** cumplir la letra del prereg exigiría 10 × 9.000 = 90.000 embeddings, unas 4 h de T4.
Los embeddings dependen únicamente del texto, así que la variabilidad que importa para los IC
—qué entidades entran en cada réplica y con qué ejes— se preserva.

**Qué se pierde, dicho explícitamente:** la variabilidad de la *asignación* entidad/atributo/valor
no entra en los intervalos. Los IC reportados cubren muestreo y ejes, no la generación del corpus.
Si el efecto dependiera fuertemente de qué atributo le tocó a cada entidad, estos IC lo
subestimarían.

**Alcance:** afecta la amplitud de los IC, no el signo ni la dirección de P1–P4.
