# PRE-REGISTRO · LA QUERY CONJUNTA (2026-08-22)

Se congela y se hashea ANTES de lanzar la campania. El chequeo de instrumento ya corrio (§3) porque
es lo que decide si vale la pena escribir predicciones; su resultado esta transcripto aca y su script
esta commiteado.

## 1. De donde sale

El 20-ago el round-trip encontro la causa de `err_identidad`, el error dominante del micro-LM
(0,19-0,21 global) y el unico que todavia impide decir que el modelo no alucina: no es
marginalizacion sobre la entidad, es **colision de clave**. Con relacion unica en el episodio el
acierto es 0,94-0,99 y `err_identidad` 0,005-0,014; con relacion repetida el acierto cae a 0,45-0,58
y el error sube a 0,38-0,54, que es **el azar entre las dos entradas que empatan**. Y `0,42 x 0,44`
reconstruye el 0,19-0,21 global.

El 21-ago, un smoke mal apuntado dio el mecanismo (`SMOKE_EMPATE_20260821.md`): en `modelo.tronco` la
lectura del archivo se inyecta en el bloque 0 **antes** de la conv y del mixer, sobre `h = emb[x]`.
La query que consulta el archivo es por lo tanto `ln(emb[token]) @ qr`, **funcion pura del token de su
posicion**. De ahi se DERIVA el atajo de la relacion en vez de solo constatarlo: en la posicion del
token de la relacion, la query matchea por igual a todas las entradas que comparten esa relacion, y el
empate entre ellas esta garantizado por construccion. El modelo no puede formar una query conjunta
entidad x relacion; consulta token por token y resuelve la conjuncion aguas abajo, integrando —lo que
tambien explica el hallazgo del 16-ago (`INFORME_RANK_HECHO`), que acierta sin que la entrada correcta
gane—.

**La pregunta de este experimento es causal y tiene una sola variable:** si la colision de clave viene
de que la query no puede ser conjunta, permitir que lo sea tiene que disolverla.

## 2. Las dos condiciones

Identicas en todo salvo en donde entra la lectura dentro del bloque 0 (`modelo.tronco`, flag
`--donde`):

| | posicion de la lectura | la query es |
|---|---|---|
| `pre` (control) | antes de la conv y del mixer, sobre `ln1(h)` | `ln(emb[token]) @ qr` — funcion pura del token |
| `post` (tratamiento) | despues del mixer del MISMO bloque, sobre `ln2(h)` | funcion del token **y de su pasado causal** |

Cada condicion reusa el LayerNorm que ya precede a la operacion siguiente, asi que **ninguna estrena
parametros que la otra no tenga**: verificado, las dos dan 863.859 parametros exactos. La inyeccion
sigue siendo temprana en las dos —quedan 3,5 bloques de computo aguas abajo—, con lo cual el contraste
no se confunde con el de E-I1/E-I2, que penalizaba inyectar en capas **profundas**.

Config, igual a la de la campania de abstencion (`c3_s*`): nivel 3, `d=128`, `capas=4`, `batch=64`,
`lr=1e-3`, `p_vieja=0.35`, `p_nose=0.4`, `--abst cabeza`, `idioma=2`, **20000 pasos con horizonte
20000**, semillas 0/1/2. Se eligio nivel 3 porque es el de la correccion eliptica, donde
`err_identidad` domina.

**El presupuesto se fija en 20000 y no en 14000 antes de mirar nada.** La campania de presupuesto
(21-ago) cerro que 14000 subestima, y la impaciencia ya costo cuatro correcciones en este proyecto. Un
negativo a 14000 no seria un negativo.

Las seis unidades se entrenan frescas. Reusar `c3_s*` como brazo `pre` ahorraria la mitad de la GPU,
pero fueron entrenadas con el `entrenar.py` del 18-ago y el contraste dejaria de ser pareado por una
razon que despues no se puede descartar. Es la clase de ahorro que este proyecto ya pago caro.

## 3. Chequeo de instrumento, CORRIDO ANTES de escribir las predicciones

Regla heredada del monitor v1 (20-ago), que perturbaba permutando el archivo, era invariante a eso por
algebra, y quedo **vacio** sin que nada lo avisara hasta el smoke. Script
`micro_lm/chequeo_query_conjunta.py`, pesos al azar, CPU, sin entrenar. Resultado:

| | delta de la query por cambiar el contexto | delta entre dos posiciones con el mismo token |
|---|---:|---:|
| `pre` | **0,00000000** | **0,00000000** |
| `post` | 0,65232694 | 0,98664361 |

- **C-1 CUMPLE.** El cero es exacto, no chico. La afirmacion del 21-ago estaba **leida del codigo y
  nunca medida**; ahora esta medida, y es independiente de que el experimento de hoy salga como sea.
- **C-2 CUMPLE.** El arreglo no es un instrumento vacio.
- **C-3 CUMPLE**, y es el que conecta con la colision: en `pre`, dos posiciones con el mismo token
  tienen la **misma** query aunque el contexto difiera.

## 4. Predicciones

Instrumentos: `ser.py` (n=2048, semilla 54321) y `diag_relacion.py` (2048 muestras, solo preguntas con
respuesta). Los dos leen `donde` y la regla de decision **del checkpoint**, no de un flag.

- **P-1 · PRINCIPAL.** `err_identidad` es menor en `post` que en `pre` en **al menos 2 de 3 semillas**,
  y la mediana de las tres baja **>= 0,03 absoluto**.

- **P-2 · MECANICISTA, el eslabon que hace causal al resultado.** La brecha
  `acierto(relacion unica) - acierto(relacion repetida)` en `post` es **a lo sumo la mitad** de la de
  `pre`, en al menos 2 de 3 semillas. Es lo que separa «bajo el error» de «bajo el error POR ESTO».

- **P-3 · NO-INTERCAMBIO.** `falsa_abst` en `post` **<= 0,10** (la compuerta de la campania) en las
  tres semillas, y la mediana de `vigente` no baja mas de **0,02** respecto de `pre`. Si el error de
  identidad baja porque el modelo se abstiene mas, no sirve: es exactamente el modo de falla con el
  que el empate de clave se cerro ayer.

- **P-4 · ESPECIFICIDAD.** `err_identidad` con relacion **unica** en `post` no supera **0,03** (en
  `pre` vale 0,005-0,014). Si `post` mejora parejo en todos lados, lo que hubo fue mas capacidad, no
  disolucion de la colision.

- **P-5 · LA SENAL DIRECTA.** La razon top2/top1 de la distribucion de lectura en la posicion de foco
  baja en `post` (`pre` medido en ~0,92 con ~6 entradas validas; predicho `post` < 0,80). Una query
  conjunta **selecciona** donde la pura **integra**.

## 5. Regla de decision, comprometida por adelantado

Si **P-1 no cumple** con el instrumento sano (C-1..C-3 ya cumplen), entonces la forma de la query
**no es la causa** de la colision de clave: el mecanismo del 21-ago queda como correlacion y no como
explicacion, y **no se prueba una tercera posicion de inyeccion** en esta linea. Se escribe el negativo
con su diagnostico y se pasa a la politica de escritura.

Si P-1 cumple y **P-2 no**, el efecto es real pero la explicacion no es la colision: queda como mejora
sin mecanismo y se reporta asi, sin adjudicarsela al hallazgo del round-trip.

## 6. Riesgo declarado antes de correr

`post` tiene **menos profundidad de computo por delante** que `pre` (3,5 bloques contra 4). Si `post`
sale peor en TODO —P-4 incluido—, la lectura correcta es perdida de computo aguas abajo, que es el
hallazgo de E2-b («el contexto es precondicion, no correccion»: un acceso en el primer bloque da
0,9998 y en el ultimo 0,4990), y **no** un fracaso de la query conjunta. P-4 es lo que separa las dos
lecturas, y por eso esta escrito antes.

Segundo riesgo, de operacion: la campania corre **por tramos entre cuentas de Colab**, y reanudar un
tramo `post` sin pasar el flag lo continuaria como `pre` en silencio, cambiando la arquitectura a
mitad de corrida. Ya esta tapado en el codigo —`donde` entra en el chequeo de identidad del
checkpoint y aborta— y se verifica en el smoke antes de lanzar.
