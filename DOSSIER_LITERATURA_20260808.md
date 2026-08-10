# Deltas + archivo recuperable dentro del modelo — dossier de literatura y crítica

Fecha: 2026-08-08. Estado: **investigación previa, sin protocolo ni pre-registro.**
Origen: idea de Maxi (2026-08-07), precisada 2026-08-08 — "no artefactos separados que se unen
para un fin, sino todo dentro del modelo".

---

## 1. Veredicto corto

**La idea, tal como está formulada, ya está ocupada.** Entre marzo y julio de 2026 salieron al
menos cuatro trabajos que implementan exactamente "estado delta + memoria recuperable, dentro de
la arquitectura, co-entrenada". No es un caso de "alguien rozó el tema": es el mismo diseño.

El más cercano —**HOLA**, julio 2026— usa como criterio de escritura al archivo la cantidad
`β_t·‖e_t‖`, es decir **el residuo de predicción de la regla delta**. Esa es, literalmente, la
señal de sorpresa que Maxi publicó en CENTINELA-01 (Zenodo 21385806, julio 2026) como predictor
temprano de colapso de memoria. HOLA la usa como criterio de archivado sin citar ese trabajo.

Esto no cierra la puerta. La reubica.

---

## 2. El estado del arte, ordenado por cercanía a la idea

### 2.1 HOLA — "A Hippocampus for Linear Attention" (arXiv 2607.02303, 2026-07-02)
Wanyun Cui, Shanghai University of Finance and Economics.

- Mantiene el estado delta-rule (GDN) como memoria compresiva **y** agrega un KV cache exacto
  acotado. Lo llama "memoria semiparamétrica de test-time". Inspiración declarada: Complementary
  Learning Systems (hipocampo/neocórtex).
- **Escritura:** sin módulo de eviction aprendido. Conserva los top-`w` tokens por
  `m_t = β_t·‖e_t‖` (con `‖k_t‖=1` por normalización L2), es decir el residuo efectivamente
  comprometido al estado. `w = 64`.
- **Lectura:** *siempre*, no gateada. Softmax sobre el cache con un RMSNorm-γ desacoplado
  (para lograr recuperación filosa en vez de promedio blando), combinada aditivamente con la
  salida del estado.
- **Diferenciable y co-entrenado**, casi iso-paramétrico (~0.004 % de params extra).
- **Resultados (340M, 15B tokens SlimPajama):** WikiText PPL 27.32 → 22.92 (−16.1 %), por debajo
  de Transformer++ (26.88); LAMBADA 30.95 → 30.26; FDA 11.7 → 20.1; SWDE 29.0 → 35.9;
  RULER S-NIAH-1@32k 0.14 → 0.58.
- **Limitaciones que declara el propio autor:** resultados a escala principal son **single-seed**;
  **no hay comparación entrenada contra módulos de eviction aprendida**; brecha grande contra
  atención completa en tareas token-exactas (FDA 20.1 vs 46.1).
- **No evalúa MQAR.** Su evidencia de memoria es PPL + NIAH + retrieval in-context.

### 2.2 HAM — "Hybrid Associative Memories" (arXiv 2603.22325, 2026-03)
Leon Lufkin et al.

- Gated DeltaNet (comprime *todos* los tokens) + KV memory softmax sobre un subconjunto
  seleccionado.
- **Escritura aprendida:** un router produce scores por cabeza que se comparan contra un
  **umbral aprendido**; solo los tokens que lo superan entran al KV cache.
- Crecimiento del cache data-dependiente, controlable con un único umbral continuo.
- Es el competidor directo de "criterio de escritura aprendido".

### 2.3 Tensor Cache — "Eviction-conditioned Associative Memory" (arXiv 2605.22884, 2026-05-25)
Swain, Han, Weidele, Martino, Torralba.

- Los pares KV **desalojados** se escriben a una memoria asociativa aprendida en vez de tirarse.
  Eviction por menos-frecuentemente-accedido. Lectura consulta ambas. End-to-end.
- Es, casi palabra por palabra, "archivar lo desalojado" — la formulación original de Maxi.

### 2.4 Memory Caching — "RNNs with Growing Memory" (arXiv 2602.24281, 2026-03-02)
Behrouz, Li, Deng, Zhong, Razaviyayn, Mirrokni (Google Research — el equipo de Titans).

- RNN con memoria que **crece** sin cota en inferencia; lectura por similitud kNN; read/write
  aparentemente diferenciables y co-entrenados. Evalúa LongBench / LAMBADA.

### 2.5 Contexto necesario (no compiten directo, pero acotan el terreno)

| Trabajo | Qué aporta | Qué **no** hace |
|---|---|---|
| **Titans** (Behrouz et al., 2501.00663) | Memoria de largo plazo aprendida en test-time; **sorpresa como criterio** (gradiente de la pérdida asociativa) + momento + olvido adaptativo | La memoria es paramétrica, no un índice recuperable |
| **Infini-attention** (2404.07143) | Memoria compresiva + delta rule + gate que mezcla con atención local | Memoria acotada O(1), sin archivo exacto |
| **Memorizing Transformers** (ICLR 2022) | kNN sobre memoria **no diferenciable**, gate escalar aprendido por cabeza | Índice congelado — el "artefacto separado" que Maxi rechaza |
| **Memory Layers at Scale** (Meta, 2412.09764) | Memoria clave-valor entrenable, sparse, hasta 128B params de memoria | Memoria de hechos, no de contexto reciente; no interactúa con estado recurrente |
| **EDA** (2606.26560, Qwen) | Desacopla dirección de **borrado** de la de escritura en delta rule; dirección de borrado aprendida | Lo borrado **se pierde**; no hay archivo |
| **LTE** (2510.20787, He & Garner) | Eviction de tokens aprendida y contextualizada + sparse attention | Lo desalojado **se descarta** |
| **Forgetful Attention / SV-Attention** (2607.12204) | Selección certificada por SVDD, borrado exacto auditable; 7 semillas, p=0.001 en enwik8 | Se degrada al escalar a 10M params |
| **Survey "Memory for LLMs"** (2607.25380) | Taxonomía (implícita/explícita × offline/online × corto/largo plazo). Declara como problema abierto la **asignación adaptativa y control de memoria** y admite que "la evaluación multidimensional de memoria sigue fragmentada" | No propone métrica unificada de capacidad |

---

## 3. Lo que nadie hizo (el hueco real)

Los cuatro competidores miden **perplejidad** y **needle-in-a-haystack**. Ninguno mide
**capacidad** en el sentido de Ligamento/TELAR-03: número de asociaciones simultáneas retenidas,
con `N` controlado, techo saturado y semillas suficientes para tener potencia.

Tres huecos concretos, en orden de filo:

**H-A. ¿El archivo restituye el techo de capacidad, o solo mejora la perplejidad?**
E1 (Ligamento) estableció que la hibridación con softmax restituye el techo y que **una sola**
cabeza softmax de cuatro alcanza (E-004, mix13 8/8). La pregunta que sigue —y que nadie
responde— es si un archivo recuperable de presupuesto `w` sustituye a esa cabeza softmax
a menor costo. Es la continuación literal de E1, con el baseline ya medido y congelado.

**H-B. ¿Es `β‖e‖` el criterio de escritura correcto, o solo el conveniente?**
HOLA lo adopta sin justificarlo empíricamente contra alternativas aprendidas — y lo dice.
CENTINELA-01 ya estableció qué señal de la regla delta predice colapso (varianza sí,
autocorrelación no). Hay una comparación de criterios que le falta al campo y para la que
Maxi tiene el instrumento previo.

**H-C. La lectura.** HOLA lee el cache **siempre**. HAM también. Nadie probó **gatear la
lectura** por sorpresa/incertidumbre actual. Esto es la otra mitad de VIGÍA-03: si la eviction
sorpresa-gated evita gastar rango en lo predecible, la *lectura* sorpresa-gated evitaría gastar
cómputo consultando el archivo cuando el estado ya sabe la respuesta.

**H-D (metodológico, y el más defendible).** Todo este subcampo corre single-seed y sin análisis
de potencia. Maxi tiene prereg, 8 semillas, `stoppower` publicado y un incidente documentado
(2026-07-27) de un falso plateau detectado por auditoría. Es una ventaja real, no un consuelo.

---

## 4. El confound que puede matar todo antes de empezar

Si el archivo guarda el par `(k, v)` **exacto** y la consulta de MQAR pide justamente ese par,
la tarea deja de medir memoria y pasa a ser un lookup. El resultado positivo sería trivial y
sin valor.

Nótese que HOLA guarda pares exactos **y no corre MQAR**. Puede ser casualidad. Puede no serlo.

**Esto tiene que testearse primero, antes de invertir en cualquier otra cosa**, con un cache
oráculo de capacidad ilimitada: si la exactitud salta a ~1.0 sin esfuerzo, la tarea está
trivializada y hay que cambiar la métrica (o el criterio de escritura) antes de seguir.

---

## 5. Fuentes

- HOLA: https://arxiv.org/abs/2607.02303
- Hybrid Associative Memories: https://arxiv.org/abs/2603.22325
- Tensor Cache: https://arxiv.org/pdf/2605.22884
- Memory Caching (RNNs with Growing Memory): https://arxiv.org/pdf/2602.24281
- Titans: https://arxiv.org/pdf/2501.00663
- Infini-attention: https://arxiv.org/html/2404.07143v1
- Memorizing Transformers: https://arxiv.org/pdf/2203.08913
- Memory Layers at Scale: https://arxiv.org/abs/2412.09764
- Erase-then-Delta Attention: https://arxiv.org/html/2606.26560
- Contextualized Learnable Token Eviction: https://arxiv.org/pdf/2510.20787
- Forgetful Attention (SV-Attention): https://arxiv.org/html/2607.12204
- Survey "Memory for Large Language Models": https://arxiv.org/html/2607.25380v1
