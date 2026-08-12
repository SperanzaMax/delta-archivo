# Consulta a Fable5 — panorama y dirección del programa de memoria

**2026-08-11.** Todo el estado en un documento, para responder en **una sola pasada**. Las preguntas
concretas están en §6; lo anterior es el material para contestarlas.

---

## 1. El objetivo, en una frase

> **«Que un LLM no olvide nunca lo que le dije.»**

El énfasis está en **dije**. No es contexto largo ni RAG sobre documentos: es memoria **episódica de
la interacción con una persona**, que sobrevive al cierre de la sesión, e incluye que esa persona
**corrija** algo dicho antes y el modelo sepa cuál versión rige.

La tarea canónica es un hecho versionado: «el director de X es Ana» → «no, es Beto» → «¿quién dirige
X?».

## 2. Lo medido, con números

Todo lo que sigue está pre-registrado con hash SHA-256 congelado **antes** de generar el dato,
analizado por scripts congelados junto al prereg, 8-10 semillas, IC95.

### 2.1 Infraestructura (dónde tiene que estar el mecanismo)

| exp | resultado |
|---|---|
| **E1** | El techo de capacidad del estado recurrente es real y la hibridación con softmax lo restituye: **+0,0792** IC95 [+0,0747, +0,0838], 4× el margen. **Una sola** cabeza softmax de cuatro ya alcanza |
| **E2** | El contexto que llega tarde no sirve; la maquinaria de cross-attention no hace falta en este régimen |
| **E2-b** | **El contexto es PRECONDICIÓN del cómputo, no corrección.** Un único acceso en el primer bloque da **0,9998**; el mismo acceso en el último, **0,4990** (el piso exacto). No es el número de accesos ni el espaciado: es antes-o-después |
| **E3** | Una compuerta de cómputo aprendida **se estrangula**: cierra a 1 bloque de 4 y no modula por nada. Firma exacta 0,500 en 8/8. Y en la condición donde el contexto llega **denso**, igual da el piso → **impide usar contexto que sí está llegando**. Costo aislado **+0,4860** [+0,4691, +0,4985] contra +0,0896 la de contexto |
| **E4 / E-006** | Compartir el FFN **perjudica** (F−S = +0,0669). **Fusión funcional con CERO unidades multimodales** (F: 0,606 con 0,000 unidades; S: 0,539 con 0,980). Y la frontera estaba al revés: **unificar temprano, especializar tarde**, con FLOPs y parámetros idénticos (+0,0786), replicado out-of-sample en 8 semillas nuevas |

### 2.2 El mecanismo propio, y su cierre

**Gemación** (idea propia): al revisar un recuerdo no se sobrescribe; se deposita la versión nueva
**cerca**, para que la geometría reemplace a la matriz de enlaces del DNC — O(1) por escritura en vez
de O(N²). Linaje: SDM de Kanerva.

**Cerrada en tres regímenes, cada uno con mecanismo identificado:**

1. **Paso fijo** → caminata no acotada. Coseno 0,811 → 0,324 (r=4) → −0,139 (r=6). Cobertura 0,000
   desde K≥4.
2. **Acotada** → la reparación **funciona** (pendiente +0,0000 vs −0,1410) **y pierde igual**:
   −0,0864 [−0,1014, −0,0714]. Peaje constante de 0,036, porque `emb(v_r)` ya está óptimamente
   colocado: contiene la entidad que la consulta menciona.
3. **Elíptica** (hoy) → test adversarial de nuestra propia conclusión: en el régimen donde el texto
   **no** lleva la clave, la geometría debería ser la única fuente. Contra la línea de base honesta
   (hidratación por co-referencia, barrida por su tasa de error τ): gana sólo por encima de
   **τ\* ≈ 0,45**, o sea con un resolutor peor que una moneda. **El negativo se extiende.**

**Hallazgo independiente y utilizable:** las correcciones elípticas crudas obtienen **0,0000 de
recuperación en 10/10 semillas** — el 100 % de las correcciones se pierde, y **en silencio**: el
índice devuelve cinco candidatos confiados, ninguno correcto.

### 2.3 Convergencias que valen

- **E2-b (0,4990) y E3 (0,500) dan el mismo número por dos vías distintas**: acceso tardío, y acceso
  disponible pero sin cómputo detrás.
- Nuestros negativos **confirman de forma independiente y cuantificada** la tesis determinista
  publicada («no le pidas al LLM que rastree la frescura; usá una regla»), desde el nivel de la
  geometría en vez del de la aplicación.

## 3. Lo que está ocupado (para no reinventar)

- **LongMemEval** (ICLR 2025) ya tiene la categoría *knowledge updates* = nuestra tarea, a nivel de
  sistema completo.
- **Tesis determinista** (arXiv 2606.01435) ya publicó nuestro dictamen de diseño.
- Capa de aplicación: Zep/Graphiti, REMem (ICLR 2026), NeuSymMS, Supersede, Mem0, Letta. **Ninguno
  mete el índice adentro de la red.**
- Intra-secuencia: HOLA (2607.02303), HAM (2603.22325), Tensor Cache (2605.22884), Memory Caching
  (2602.24281). **Todos mueren en el forward.**

## 4. Los cuatro caminos abiertos, como los vemos

1. **Índice co-entrenado, con lectura temprana.** El hueco declarado. Sus dos obstáculos ya no son
   ciegos: para el gradiente por top-k hay salidas conocidas, y para el *stale index* **ya medimos**
   que la deriva catastrófica es del aprendizaje inicial, no de la vida útil (coseno 0,882
   preentrenado vs 0,207 desde cero, umbral de tolerancia ~0,7). Predicción propia disponible, que
   sale de E2-b: *lo recuperado tiene que entrar temprano, y la brecha crece con la profundidad de la
   consulta* — lo contrario de lo que hace todo pipeline RAG.
2. **Capacidad medida.** Nadie reporta cuántos hechos versionados entran antes de que el mecanismo
   falle, con N controlado y potencia. Es el hueco que E1 llenó para atención híbrida.
3. **Banco ECO** (diseño en `DISENO_BANCO_ELIPTICO.md`): benchmark de correcciones elípticas, con dos
   ejes que no existen en ningún banco — **grado de elipsis** (5 niveles) y **ambigüedad referencial**
   (número de entidades activas) — y una métrica nueva, la **tasa de error silencioso**.
4. **Eviction sorpresa-gated** (qué merece archivarse, no sólo adónde va).

## 5. Restricciones reales del proyecto

- **Autor único, sin financiamiento, sin institución.** Firma como investigador independiente.
- **Hardware:** una PC de 4 núcleos sin GPU, más Colab (que últimamente también da CPU). Todo lo
  medido acá corrió en ese hardware.
- **Publicación:** cinco rechazos en Preprints.org (2 por duplicado con Zenodo, 3 por scope). Plan
  actual: TMLR (cuota de **2 envíos/año** para autor único, los desk rejects cuentan) + Research
  Square para lo que no entra ahí. arXiv trabado por falta de endorsement.
- **Método:** todo pre-registrado con hash antes del dato. De los últimos diez veredictos
  registrados, **nueve no se cumplieron** — y los dos hallazgos más interesantes salieron de un
  contraste secundario y de una condición exploratoria.

## 6. Las preguntas, en orden de importancia

1. **Encuadre.** ¿Se sostiene académicamente leer todo el programa como *«medir dónde tiene que estar
   el mecanismo de memoria y cuánto cómputo hace falta detrás»*? Un lector externo, con el material
   completo delante, describió el trabajo como «combinar mecanismos de atención para mejor balance
   entre memoria y costo» — que describe la infraestructura y **no llega al objetivo**. ¿El problema
   es el framing, la escritura, o que el objetivo todavía no está realmente atacado por lo medido?

2. **Prioridad.** De los cuatro caminos de §4, ¿cuál tiene mejor relación valor/costo para un autor
   único con este hardware? Nos interesa especialmente si el **1** es abordable en este régimen o si
   es una trampa de recursos disfrazada de hueco.

3. **El banco.** ¿Los dos ejes de ECO (grado de elipsis, ambigüedad referencial) ya existen en algún
   benchmark de memoria conversacional? Y si no existen: ¿es un aporte suficiente para sostener un
   paper, o es un instrumento que sólo vale acompañado de resultados sobre sistemas reales?

4. **Qué se nos escapa.** Con §2 y §3 delante: ¿hay literatura relevante que no estemos viendo, o
   alguna conexión teórica que no estemos explotando? Nos interesa particularmente si la
   convergencia E2-b ≈ E3 (el mismo 0,50 por dos vías) conecta con algún resultado conocido.

5. **Publicación.** Con la cuota de TMLR (2/año) y un preprint ya enviado a Research Square, ¿cómo
   repartirías lo que hay? Hoy existen tres cuerpos: E1 (listo para TMLR), E3+E4 (redactado hoy),
   y la línea de memoria (enviada a Research Square).

6. **Lo incómodo.** ¿Hay algo en §2 que consideres **sobre-interpretado**? Preferimos que lo señales
   ahora y no que lo señale un revisor. En particular: la convergencia E2-b ≈ E3 nos parece fuerte,
   pero somos nosotros los que la encontramos y los que la queremos.
