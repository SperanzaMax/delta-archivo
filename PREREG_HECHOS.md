# Pre-registro — Memoria versionada de hechos sobre embeddings de un LLM

**Fecha de congelamiento:** 2026-08-09, antes de generar ningún embedding de esta tarea.
**Motivo:** en la jornada previa dos resultados se revelaron artefactos *después* de mirarlos (un
umbral de similitud que nunca disparaba; una dimensión efectiva calculada sobre covarianza
singular). Este documento fija predicciones y criterios **antes** de ver datos.

---

## 1. Pregunta

Cuando un hecho se revisa, ¿depositar la versión nueva **anclada geométricamente** a la anterior
(gemación) preserva información recuperable que se pierde con sobrescritura, **y** aporta algo por
encima de simplemente guardar ambas versiones sin estructura?

## 2. Sustrato

Embeddings de `gemma:2b` (Ollama 0.20.2, Q4_0) sobre textos completos. Elegido porque R13 mostró
que el harness de d=64 no sirve: su única representación estable entre contextos es **función pura
del token**, lo que trivializa la tarea. El embedding de oración es estable para el mismo
contenido y a la vez genuinamente contextual.

## 3. Condiciones

| condición | dirección de la revisión | guarda historia |
|---|---|---|
| `sin` | — (control, sin archivo) | no |
| `sobrescritura` | `emb(v2)`, reemplaza la entrada de v1 | **no** |
| `duplicados` | `emb(v2)`, entrada nueva independiente | sí, sin estructura |
| `gemacion` | `emb(v1) + ε·t̂`, entrada nueva anclada | sí, con estructura |

`duplicados` es la condición que aísla el aporte. **Sin ella el experimento no puede distinguir
si el mérito es de guardar la historia o de la geometría con que se guarda**, y cualquier
resultado positivo sería atribuible a lo primero.

## 4. Predicciones

**P1 (principal, falsable).** `gemacion` supera a `duplicados` en **cobertura de clúster**: la
consulta recupera *ambas* versiones entre sus top-k con mayor frecuencia, porque están ancladas
a distancia ε en vez de en direcciones independientes.
**Falsa si** la diferencia no supera el margen preespecificado (abajo).
*Esta es la predicción que decide si la geometría aporta algo. Si cae, el mecanismo se reduce a
"guardá las dos versiones", que no necesita nada de este trabajo.*

**P2 (control).** `sobrescritura` ≈ azar en ANTERIOR. No es un hallazgo: es verificación de que
la tarea mide lo que dice. Si `sobrescritura` recupera el valor anterior, hay fuga y el
experimento es inválido.

**P3.** `gemacion` ≥ `sobrescritura` − 0.02 en VIGENTE: anclar la revisión no debe costar
precisión sobre el valor al día.
**Falsa si** `gemacion` queda por debajo de ese margen.

**P4 (de R4, ley de escala).** Con K revisiones por entidad (K ∈ {2, 4, 8}), el sesgo δ necesario
para recuperar la versión vigente crece **superlinealmente** en K, y con K=8 la recuperación de
ANTERIOR cae por debajo de 0.5 a δ fijo.

**P5 (referencia externa, NO test).** VersionRAG reporta 58 % para RAG convencional en consultas
versionadas. Se reporta como contexto. **No se declarará superioridad sobre ese número**: la tarea
no es idéntica y la comparación directa sería indebida.

## 5. Métricas

- **VIGENTE**: el ítem recuperado como top-1 corresponde a la versión v2.
- **ANTERIOR**: se recupera v1 al pedir la versión previa.
- **COBERTURA**: ambas versiones aparecen en el top-k (k = 5, fijado de antemano).
- **RANGO**: posición mediana del ítem correcto (métrica de R7: el coseno solo es insensible).

## 6. Diseño y análisis

- **N = 3.000 entidades**, cada una con v1, v2 y consulta → 9.000 embeddings.
- **10 semillas** para la asignación entidad/atributo/valor y para los ejes `t̂`.
- Se reporta media ± **IC95** (t de Student, 9 gl).
- **Margen preespecificado para P1 y P3: 0.02 absoluto.** Diferencias menores se declaran nulas
  aunque el IC excluya el cero.
- ε de gemación: **0.30**, tomado de R2 sin re-ajustar. Cualquier otro valor que se pruebe se
  reporta como exploratorio y separado.

## 7. Compromisos (lo que NO se va a hacer)

- No se cambiará ε, k, ni el umbral de similitud después de ver resultados. Si algo se ajusta,
  se reporta como **exploratorio** en sección aparte y no se usa para sostener P1.
- No se descartarán entidades por "ruidosas" salvo por un criterio automático fijado acá:
  se excluyen ítems donde `emb(v1)` y `emb(v2)` tengan coseno < 0.5 (molde roto), y se reporta
  cuántos fueron.
- Si P1 cae, se reporta como negativo. El resultado publicable en ese caso es
  "guardar la historia alcanza; la geometría no agrega", que es informativo y honesto.
- Antes de cualquier análisis se corre la **compuerta de identificabilidad** de R13: la consulta
  debe separar su propia entidad del resto con AUC > 0.95. Si no separa, el experimento se
  detiene ahí — como en R13, donde correrlo igual produjo números sin sentido.

## 8. Qué haría falta para invalidar todo el trabajo previo

Que la compuerta del punto 7 falle sobre embeddings reales. Eso significaría que la identidad de
una entidad no es recuperable ni siquiera con representaciones de texto completo, y entonces la
memoria persistente por similitud no es viable en ningún sustrato accesible.
