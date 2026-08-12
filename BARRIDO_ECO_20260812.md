# Barrido de literatura para ECO — riesgo 3 del diseño

**2026-08-12.** Responde al riesgo 3 declarado en `DISENO_BANCO_ELIPTICO.md` §7: *«¿ya existe?»*.
Método: búsqueda dirigida sobre bancos de memoria conversacional 2025-2026 y sobre la línea
lingüística de elipsis/co-referencia. **Veredicto: ECO sigue en pie, pero acotado — de las tres cosas
que el diseño reclamaba como propias, una sobrevive entera, una hay que reformularla y una hay que
citarla como antecedente directo.**

---

## 1. Lo que se buscó y con qué se chocó

| pieza reclamada en el diseño | estado | quién la ocupa |
|---|---|---|
| Eje B — **ambigüedad referencial** (`m` entidades activas) | **LIBRE** | nadie lo controla como eje |
| Eje A — **grado de elipsis** graduado | **PARCIAL** | el fenómeno está muy trabajado, pero del lado de la *consulta* |
| Métrica **SER** (error silencioso) | **OCUPADA en su idea** | FAMA, de Memora (arXiv 2604.20006) |

## 2. El hallazgo que obliga a reformular: FAMA ya mide el error silencioso

**Memora — «From Recall to Forgetting: Benchmarking Long-Term Memory for Personalized Agents»**
(arXiv 2604.20006) introduce **FAMA** (*Forgetting-Aware Memory Accuracy*), que empareja cada pregunta
con dos criterios: **presencia de memoria** (¿está lo válido?) y **ausencia por olvido** (¿quedó
excluido lo invalidado?). Penaliza explícitamente *«erroneous reuse of obsolete memory that standard
accuracy metrics do not capture»*, se computa **a nivel de criterio** con tres jueces LLM, y valida
con anotadores humanos (acuerdo 88,3 %).

Eso es, en sustancia, la mitad de nuestro `SER`. **No se puede presentar SER como métrica nueva.**

Lo que sí queda diferenciado, y hay que escribirlo así:

1. **Nivel de medición.** FAMA evalúa la **respuesta** del sistema completo con jueces LLM. `SER`
   evalúa la **indexación**: candidato devuelto con confianza alta e incorrecto, sin modelo generador
   en el medio y sin jueces. Es determinista y reproducible en CPU — que es exactamente nuestra
   restricción de hardware.
2. **Tipo de error cubierto.** FAMA mira una sola clase: reutilizar lo invalidado (versión superada).
   Nuestro dato del 11-ago tiene la otra clase, que FAMA no separa: el candidato de **otra entidad**
   a 0,4064 contra el bueno a 0,4237. Con la referencia sin resolver, el error no es de recencia sino
   **de identidad**, y ese es el que ECO puede aislar.

**Consecuencia:** `SER` se redefine como *variante a nivel de índice de FAMA, con desagregación por
tipo de error (versión vs. identidad)*, y Memora pasa a ser cita obligada, no literatura de fondo.

## 3. La elipsis está ocupada, pero del lado equivocado del sistema

Existe una línea entera y madura, y ninguno de sus trabajos toca memoria:

- **CANARD** — 40.527 pares de reescritura de preguntas, sobre QuAC, para co-referencia y elipsis.
- **QReCC** — 14K conversaciones con el mismo objetivo.
- **GECOR** (EMNLP 2019) — el más cercano en nombre: dataset sobre CamRest676 con **1.174 versiones
  con elipsis y 1.209 con co-referencia** anotadas, y modelo generativo que las resuelve.
- **MuDoCo, InCar** — el resto del conjunto estándar de *query rewriting*.

Todos comparten una forma: la elipsis está en **la consulta del usuario**, y la tarea es reescribirla
a una forma auto-contenida usando el contexto del diálogo, que está **disponible y completo**.

**El corte que deja lugar a ECO:** nosotros ponemos la elipsis en la **escritura** — en el hecho que
se archiva — y medimos qué pasa **después**, cuando el contexto que la resolvía ya no está porque la
sesión cerró. El *query rewriting* nunca enfrenta ese caso: reescribe con el diálogo delante. Es la
diferencia entre resolver una referencia y **haber perdido la posibilidad de resolverla**.

Ese enunciado hay que ponerlo en el paper como delimitación explícita, con las cuatro citas. Sin eso,
un revisor con CANARD en la cabeza cierra la lectura en el abstract.

## 4. El eje que sobrevive entero: ambigüedad referencial

Ningún banco de memoria controla **cuántas entidades están activas** cuando llega la actualización:

- **MemoryAgentBench** — la resolución de conflictos se arma con *«concatenated edit pairs»* y se pide
  responder según el hecho más reciente. El hecho **llega identificado**; no hay elipsis ni entidades
  competidoras. Reporta que el multi-hop conflict resolution falla en todos los paradigmas, pero por
  otra razón.
- **Memora** — no controla la forma lingüística de la corrección ni las entidades competidoras; genera
  diálogo natural con prompting multi-agente y verifica alineación con la traza de memoria.
- **Mem2ActBench** — sí tiene *underspecification*, pero de **parámetros de llamadas a herramientas**
  (explícito / inferido / por defecto), y el paper **declara que no controla el conteo de entidades
  como eje independiente**: agrupa por atributos ligados a entidad justamente para *evitar* la mezcla.
- **LongMemEval / LongMemEval-V2, LoCoMo, BEAM, MemoryArena** — *knowledge updates* a nivel de sistema,
  sin aislar la indexación ni graduar la elipsis.

Convergencia útil: **MemoryAgentBench identifica la resolución de conflictos como el cuello de botella
del campo**, y lo hace en el régimen fácil (hecho identificado). Nuestro 0,0000 del 11-ago muestra que
un escalón antes —la identificación misma— ya está roto. Eso ubica a ECO como el instrumento que mide
el escalón que el banco más reciente da por resuelto.

## 5. Veredicto y qué cambia en el diseño

**ECO no está hecho. No se archiva.** Pero deja de poder presentarse como «dos ejes nuevos y una
métrica nueva». La versión defendible es:

> Un banco que aísla la **indexación** de correcciones conversacionales y gradúa dos cosas que los
> bancos de memoria no gradúan —cuánta identidad conserva el texto de la corrección, y cuántas
> entidades compiten por ella— reportando el error desagregado por tipo (versión vs. identidad) sobre
> la métrica de olvido que Memora ya estableció.

Cambios concretos que esto impone sobre `DISENO_BANCO_ELIPTICO.md`:

1. **§4.3:** `SER` se reescribe como variante de FAMA a nivel de índice, con desagregación por tipo de
   error. Se cita Memora. La novedad se reclama sobre la desagregación y el nivel, no sobre la idea.
2. **§3.1:** el eje de elipsis se declara **importado** de la línea de *query rewriting*
   (CANARD/QReCC/GECOR) y **trasladado al lado de la escritura**. Se cita.
3. **§3.2:** el eje `m` se mantiene como el aporte principal — es el único que nadie controla.
4. **§5:** agregar a Memora y MemoryAgentBench como sistemas a ubicar sobre la curva, si el costo lo
   permite.

## 6. Lo que este barrido NO resuelve

Sigue en pie el problema **(2) del §10** del diseño, y ahora con más peso: nuestro propio chequeo
bloqueante dio **por debajo del azar** en dos celdas, lo que dice que la verdad de base por recencia
es una convención y no una señal recuperable. La literatura de *query rewriting* lo confirma desde el
otro lado: ahí la elipsis siempre es resoluble **porque el contexto está delante**. Si en ECO la
referencia es objetivamente irrecuperable, el eje `m` no mide dificultad sino arbitrariedad — y
entonces el aporte principal de §5 se cae solo.

**Orden obligado antes de construir nada:** primero hacer la referencia objetivamente recuperable
(marcador de recencia explícito, o tipo de valor que sólo encaje con una entidad activa), re-correr el
chequeo, y recién ahí el generador.

---

### Fuentes

- Memora / FAMA — arXiv [2604.20006](https://arxiv.org/abs/2604.20006)
- MemoryAgentBench — [resumen](https://www.emergentmind.com/topics/memoryagentbench)
- Mem2ActBench — arXiv [2601.19935](https://arxiv.org/html/2601.19935)
- LongMemEval-V2 — arXiv [2605.12493](https://arxiv.org/html/2605.12493v1)
- Reflective memory / Connecting the Dots — arXiv [2606.01223](https://arxiv.org/pdf/2606.01223)
- CANARD — [QANTA](https://qanta-org.github.io/research/projects/canard.html)
- GECOR — arXiv [1909.12086](https://arxiv.org/abs/1909.12086)
