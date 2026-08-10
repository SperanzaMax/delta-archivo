# P4 — ley de escala en K revisiones

k = 5 (top-k de lectura) · ε = 0.3 · 10 semillas × 1000 · margen 0.02. Umbrales sin cambios.

**Por qué esta tanda:** con K = 1 la tarea está en el techo (duplicados 0,9988 · gemacion 0,9928) y P1 no puede discriminar. Al poblar el clúster con más revisiones, la geometría tiene que trabajar.

## VIGENTE (recuperar la versión al día)

| K | duplicados | gemacion | gemacion − duplicados |
|---|---|---|---|
| 1 | 0.9993 | 0.9949 | **-0.0044** [-0.0053, -0.0035] |
| 2 | 0.9954 | 0.0567 | **-0.9387** [-0.9425, -0.9349] |
| 4 | 0.9715 | 0.0000 | **-0.9715** [-0.9748, -0.9682] |
| 8 | 0.5681 | 0.0000 | **-0.5681** [-0.5754, -0.5608] |

## ANTERIOR (recuperar la versión previa)

| K | duplicados | gemacion | gemacion − duplicados |
|---|---|---|---|
| 1 | 0.9991 | 0.9949 | **-0.0042** [-0.0050, -0.0034] |
| 2 | 0.9914 | 0.0567 | **-0.9347** [-0.9390, -0.9304] |
| 4 | 0.9482 | 0.0000 | **-0.9482** [-0.9537, -0.9427] |
| 8 | 0.2822 | 0.0000 | **-0.2822** [-0.2912, -0.2732] |

## COBERTURA (ambas versiones en el top-k) — la métrica de P1

| K | duplicados | gemacion | gemacion − duplicados |
|---|---|---|---|
| 1 | 0.9991 | 0.9949 | **-0.0042** [-0.0050, -0.0034] |
| 2 | 0.9914 | 0.0567 | **-0.9347** [-0.9390, -0.9304] ← supera el margen |
| 4 | 0.9482 | 0.0000 | **-0.9482** [-0.9537, -0.9427] ← supera el margen |
| 8 | 0.2822 | 0.0000 | **-0.2822** [-0.2912, -0.2732] ← supera el margen |

## Veredicto de P4

El prereg predice que **con K = 8 la recuperación de ANTERIOR cae por debajo de 0,5 a δ fijo**. Medido: **0.0000** IC95 [0.0000, 0.0000] → **CONFIRMA**.

---

## Verificación: el colapso es REAL, y tiene mecanismo exacto

Es el tercer cero de este proyecto que viene acompañado de un «CONFIRMA», así que se verificó antes de
aceptarlo (D2 documenta el caso en que el cero **sí** era un bug). Coseno de cada entrada contra la
**consulta**, entidad de ejemplo:

| r | `gemacion` (caminata, paso ε) | `duplicados` (texto real de la versión) |
|---|---|---|
| 0 | 0,8106 | — |
| 1 | 0,7777 | 0,7887 |
| 2 | 0,6792 | 0,7683 |
| 3 | 0,5234 | 0,8204 |
| 4 | **0,3235** | 0,8064 |
| 6 | **−0,1391** | 0,8187 |
| 8 | **−0,1391** | 0,8015 |

**El mecanismo:** con un paso ε fijo por revisión, la caminata **se aleja acumulativamente del ancla**.
El arco recorrido crece sin cota (~0,30 rad por paso; 2,36 rad a r = 8), así que a partir de r ≈ 4 la
entrada vigente queda **más lejos de la consulta que cualquier distractor** y deja de entrar al top-k.
En `duplicados`, en cambio, cada revisión es el embedding real de su texto y **se mantiene a ~0,80 de
la consulta para todo r**: no hay deriva porque no hay desplazamiento inducido.

No es un artefacto: es una propiedad de la gemación tal como está especificada (ε fijo, §6 del prereg,
tomado de R2 y explícitamente no re-ajustable).

## Lectura

**P4 confirma, y por el mecanismo que predecía.** §4 anticipaba que «el sesgo δ necesario para
recuperar la versión vigente crece superlinealmente en K, y con K = 8 la recuperación de ANTERIOR cae
por debajo de 0,5 a δ fijo». Ocurre, y **antes de lo previsto**: ya a K = 4 está en cero.

**Junto con P1, el cuadro para la gemación es un negativo consistente y explicado:**

- A **K = 1** no mejora la cobertura sobre guardar ambas versiones (ambas en el techo: 0,9991 vs
  0,9949) → P1 no confirma.
- A **K ≥ 2** es **catastróficamente peor** (0,0567 vs 0,9914) porque el clúster se va caminando.
- `duplicados` se degrada con elegancia (0,9991 → 0,9914 → 0,9482 → 0,2822): sólo empieza a fallar a
  K = 8, y por saturación del top-k, no por deriva.

**Converge con R4 y lo endurece.** R4 ya había medido que «δ* crece superlinealmente con K» y que «el
sesgo geométrico sirve hasta ~8 revisiones». Acá falla antes, y la razón es identificable: en R4 la
consulta era uno de los ítems almacenados, mientras que acá es un **ancla externa fija** (el embedding
de la pregunta), que no acompaña la caminata.

**Qué queda en pie de la gemación.** El mecanismo de depositar-al-lado **no sobrevive a revisiones
sucesivas con paso fijo**. Si tiene futuro, exige que el desplazamiento sea **acotado** (por ejemplo
ε decreciente, o revisiones que orbiten el ancla en vez de alejarse), y eso es un mecanismo distinto
que habría que pre-registrar aparte. Con lo medido hoy, **guardar ambas versiones sin estructura le
gana**, que es exactamente lo que el prereg §4 anticipaba como lectura si P1 caía: *«el mecanismo se
reduce a "guardá las dos versiones", que no necesita nada de este trabajo»*.
