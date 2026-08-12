# Brazo interno — un índice persistente dentro de la red

**2026-08-12.** Diseño, no pre-registro. Cada etapa llevará el suyo, congelado antes de su dato.

Esto ataca el objetivo de fondo —**«que un LLM no olvide lo que le dije»**— por la única vía que el
programa dejó abierta y que nadie ocupó: **memoria persistente entre secuencias, co-entrenada dentro
de la arquitectura**. Todo lo medido hasta hoy en la línea de memoria fue índice **no paramétrico**
sobre **encoder congelado**, y esa limitación está declarada en el paper que sale a Research Square.

---

## 1. Por qué ahora, y no antes

Tres resultados propios lo destraban, y ninguno estaba disponible cuando la línea empezó:

| resultado | qué destraba |
|---|---|
| **E2-b**: acceso al contexto en el primer bloque 0,9998, en el último 0,4990 | **dónde** inyectar lo recuperado. Todo pipeline RAG concatena tarde; esto dice que hay que entrar temprano |
| **R6**: deriva del encoder 0,882 preentrenado vs 0,207 desde cero, umbral de tolerancia ~0,7 | el *stale index* **no** es fatal: es un fenómeno del aprendizaje inicial, no de la vida útil |
| **E1/E-004**: una sola cabeza softmax de cuatro restituye el techo de capacidad | el presupuesto de atención para leer el índice es barato |

Y el canal de contexto de E2 ya está implementado y validado en el harness (`src/contexto.py`,
`k_ctx = 8` posiciones que llegan por un canal separable del tronco). **La lectura del índice es
estructuralmente ese canal**, con el contenido recuperado en vez de dado.

## 2. La tarea: cross-secuencia, con el estado reseteado

La tarea canónica del programa, partida en tres secuencias con el estado recurrente **reseteado**
entre ellas:

```
  S1   escribe   (k, v1)          → [reset del estado]
  S2   revisa    (k, v2)          → [reset del estado]
  S3   consulta   k               → debe responder v2
```

**El piso es el azar por construcción.** Delta puro y softmax puro no pueden: la información de S1 y
S2 no sobrevive al reset, y no está en la secuencia de entrada. Cualquier acierto por encima del azar
viene del archivo, y de ningún otro lado.

Es la estructura de R13, que ya se corrió con archivo no paramétrico y modelo congelado. Lo nuevo es
que acá **el índice se entrena con el modelo**.

## 3. Los dos obstáculos, declarados antes de empezar

**(a) El gradiente no fluye por la selección top-k.** Es no diferenciable. La salida que se adopta:
el gradiente fluye por los **pesos de atención sobre lo ya recuperado**, no por *qué* se recuperó.
Eso deja una pregunta genuinamente abierta, y es la principal del brazo:

> ¿Aprende el modelo a **formar consultas útiles** cuando el gradiente nunca le dice que recuperó lo
> equivocado?

**(b) *Stale index*.** Los vectores archivados envejecen mientras el encoder se mueve. Mitigación con
evidencia propia: **preentrenar y luego afinar** (R6), en vez de escribir desde cero. Queda medible
cuánta deriva tolera antes de romperse.

## 4. Etapas

Cada una es barata y decide si la siguiente vale la pena. Ninguna se salta.

### E-I0 — el piso (bloqueante, y va primero por la lección de hoy)

Verificar que delta puro y softmax puro dan **el azar** en la tarea cross-secuencia, y que un
control **con la información en la secuencia** la resuelve. Sin las dos mitades, cualquier resultado
posterior es ilegible.

> Es exactamente el error que costó el veredicto del 11-ago: un control de sanidad tiene que poder
> **fallar**. Acá el control es *«¿resuelve la tarea cuando la información SÍ está?»*, y si no la
> resuelve, no se mide nada más.

### E-I1 — lectura aprendida, escritura oráculo

El archivo se llena por fuera con lo correcto; el modelo sólo aprende a **consultarlo** y a usar lo
recuperado. Aísla el obstáculo (a) sin mezclarlo con qué archivar.

**Predicción central**, derivada de E2-b y contraria a lo que hace todo pipeline RAG: la inyección
en el **primer** bloque supera a la del último por un margen grande, y la brecha **crece** con la
profundidad de la consulta.

### E-I2 — escritura también aprendida

El modelo decide **qué** archivar. Se conecta con la dirección de VIGÍA-03: escritura *gateada por
sorpresa*, no por todo lo desalojado.

### E-I3 — la deriva, de frente

Con el índice escrito por un encoder que sigue entrenando: medir la degradación real, comparar
preentrenado contra desde cero (predicción de R6), y cuantificar cuánto compra un reindexado
periódico contra su costo.

## 5. Qué contaría como éxito, y qué como fracaso

- **Éxito:** exactitud sustancialmente sobre el azar en S3 con el estado reseteado, con el piso
  verificado en E-I0. Con eso, la afirmación «el modelo no olvidó lo que se le dijo en otra sesión»
  es literal y medida, no analógica.
- **Fracaso informativo:** que el modelo no aprenda a formar consultas útiles sin gradiente por la
  selección. Sería la respuesta a la pregunta que el campo no contestó, y explicaría por qué todos se
  quedaron en caches intra-secuencia.
- **Fracaso no informativo, a evitar:** que la tarea sea resoluble sin el archivo. De ahí que E-I0 sea
  bloqueante.

## 6. Costo y hardware

El harness entrena modelos de d=64 / 4 bloques en la PC (4 núcleos, sin GPU). Las corridas de E1
fueron de miles de pasos y se hicieron localmente y en T4. **Acá sí conviene reservar Colab con GPU**:
es entrenamiento con gradientes, no inferencia, que es donde la GPU rinde de verdad — y el
razonamiento de hoy sobre el costo de inferencia no aplica a este brazo.
