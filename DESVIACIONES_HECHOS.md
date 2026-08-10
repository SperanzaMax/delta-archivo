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

## D2 — El primer intento de P4 fue INVÁLIDO por un error mío de implementación (2026-08-10)

Registrado **con los números a la vista y descartándolos**, no antes: es un error de código detectado
al leer el resultado, y queda acá porque el resultado descartado es parte del registro.

**Lo que salió:** ceros exactos (0,0000) en `duplicados` desde K = 2 y en **las dos** condiciones a
K = 4 y K = 8, con P4 declarando **CONFIRMA** porque ANTERIOR quedaba «por debajo de 0,5». Una
confirmación construida sobre un cero es una confirmación falsa.

**Las dos causas, verificadas:**

1. **La revisión r-ésima de `gemacion` la puse como rayo lineal** `E1 + r·ε·eje` normalizado. Eso aleja
   sin cota: cos con `E1` cae 0,958 → 0,858 → 0,640 → 0,385 para r = 1, 2, 4, 8. A r ≥ 4 la entrada
   queda casi **ortogonal** a la consulta y no entra nunca al top-k. La implementación de R13
   (`exp_gemacion.py:106`) hace otra cosa: da el paso **desde la última entrada** y **re-tangenta cada
   vez** — una caminata sobre la esfera con paso ε, no un rayo.
2. **Las revisiones de `duplicados` las puse en `tangente(aleatorio, E1)`**, que es un vector
   **ortogonal** a `E1` — no una entrada nueva plausible. Por eso `duplicados` daba 0 ya en K = 2.

**Qué se hace:** el resultado se **descarta entero** (borrados `INFORME_P4.md` y `resultados_p4.json`),
y P4 se vuelve a correr **generando los textos reales de cada revisión y embebiéndolos**, en vez de
simular la geometría de las revisiones. Es más caro y es lo correcto: la posición de `emb(v_r)` es un
dato del encoder, no algo que yo deba modelar.

**Alcance:** no afecta a P1, P2 ni P3, que corrieron sobre el archivo de K = 1 construido por
`correr_hechos.py` y no usan este código.

**Es el tercer fallo silencioso de este proyecto** (R13 con un umbral que nunca disparaba; la compuerta
que miraba AUC y no discriminación; y ahora esto). Los tres se detectaron mirando si los números eran
*posibles*, no si eran favorables. Que P4 «confirmara» es precisamente lo que lo delató.
