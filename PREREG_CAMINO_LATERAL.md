# PRE-REGISTRO · LA QUERY CONJUNTA POR CAMINO LATERAL (2026-08-22, tarde)

Se congela y se hashea antes de lanzar. Sale del §final de `INFORME_QUERY_CONJUNTA_20260822.md`, del
mismo dia.

## 1. Por que este experimento existe

La campania de la mañana fallo las cuatro predicciones, y su §6 —escrito antes de correr— explica por
que el resultado **no responde la pregunta**: la condicion `post` movio **dos cosas a la vez**, la
forma de la query y el punto donde la lectura entra al computo. La segunda resulto devastadora
(acierto 0,97 -> 0,39, plano desde el paso 4000, y falla tambien con relacion UNICA donde no hay
ninguna colision que disolver), asi que se come cualquier efecto de la primera. La hipotesis de la
query conjunta quedo **sin probar**.

Y dejo un hallazgo mecanico que es el que dicta este diseño: **la ventana de inyeccion util no es
«temprano», es ANTES DEL PRIMER MIXER.** Media capa mas tarde hace casi todo el daño que E2-b medía a
cinco bloques de distancia. De ahi el trade-off: una query conjunta necesita contexto ya computado, y
la lectura util necesita entrar antes de que el computo ocurra.

**Este experimento rompe el trade-off por el unico lado que queda: darle contexto a la query SIN
mover el punto de inyeccion.**

## 2. La condicion `lat`

En `modelo.tronco`, la inyeccion queda **exactamente donde `pre` la tiene** —sumada a `h`, antes de
la conv y del mixer— y lo unico que cambia es de que se forma la query:

| | punto de inyeccion | la query se forma sobre |
|---|---|---|
| `pre` (control) | antes del mixer | `ln1(h)` — funcion pura del token |
| `lat` (tratamiento) | **el mismo** | `conv3(ln1(h))` — el token y los **dos anteriores** |

La conv es la **misma del bloque**, asi que no estrena parametros: las tres condiciones tienen los
mismos 863.859. Y el contexto que aporta es **local**: la conv de kernel 3 no ve mas alla de dos
tokens atras, con lo cual no reintroduce la dependencia global que en `post` venia del mixer.

Que dos tokens alcancen no es un supuesto comodo: en la forma canonica del idioma la entidad y la
relacion caen **a distancia 2** (`el director de museo es X` — `director` en la posicion del sustantivo,
`museo` dos mas adelante). Es exactamente el alcance que hace falta para formar una query conjunta
entidad x relacion.

## 3. Chequeo de instrumento, CORRIDO antes de escribir las predicciones

`chequeo_query_conjunta.py`, pesos al azar, sin entrenar. Delta relativo de la query al intervenir
un solo token:

| | contexto lejano (mitad de la secuencia) | vecino `p−1` | lejano `p−5` | mismo token, otro lugar |
|---|---:|---:|---:|---:|
| `pre` | 0,00000000 | 0,00000000 | 0,00000000 | 0,00000000 |
| `post` | 0,65232694 | 0,81207407 | **0,17371441** | 0,98664361 |
| `lat` | 0,00000000 | **0,75329703** | **0,00000000** | 1,21521115 |

- **L-1 CUMPLE** — `lat` depende del vecino (0,753): la query puede ser conjunta.
- **L-2 CUMPLE** — `lat` **no** depende del lejano (0,000 exacto), mientras `post` si (0,174). La
  separacion entre «contexto local» y «contexto global» es limpia, no gradual.
- **L-3 CUMPLE** — `pre` sigue siendo funcion pura del token.

## 4. El control se REUSA, y esta verificado bit a bit

Las tres unidades `pre` de esta mañana (`p3_s0/s1/s2`, 26000 pasos) son el control. Se verifico que
el codigo con `lat` agregado produce en `pre` **exactamente los mismos numeros** que antes del cambio,
sobre el mismo checkpoint y la misma semilla de evaluacion: `acierto 0.970467596390484`,
`err_identidad 0.01220703125`, `nose 0.9119420989143546`, `falsa_abst 0.008203445447087777` — las
ocho metricas identicas hasta el ultimo digito.

Esto no es una comodidad: es lo que hace que el contraste sea **pareado de verdad**. Mismo generador,
mismo `entrenar.py`, mismo dia, mismo presupuesto. La campania de la mañana no pudo reusar nada y por
eso costo seis unidades; esta cuesta tres.

Config de `lat`, identica a la de `pre`: nivel 3, `d=128`, `capas=4`, `batch=64`, `lr=1e-3`,
`p_vieja=0.35`, `p_nose=0.4`, `--abst cabeza`, `idioma=2`, **26000 pasos con horizonte 26000**, sin
siembra, semillas 0/1/2. Familia `w3_s*`.

## 5. Predicciones

Instrumentos, los mismos y ya usados: `ser.py` (n=2048, semilla 54321) y `diag_relacion.py` (2048
muestras), los dos leyendo `donde` y la regla de decision **del checkpoint**.

- **W-0 · BLOQUEANTE, y va primero.** `lat` **aprende la tarea**: acierto >= 0,70 en al menos 2 de 3
  semillas. Es la compuerta que `post` no paso (0,37-0,40). Si falla, mover la formacion de la query
  rompe el modelo aunque la inyeccion no se mueva, y **todo lo demas queda no evaluable** — como paso
  hoy a la mañana. Se declara primero a proposito: es la leccion de que `post` invalido su propio
  experimento.

- **W-1 · PRINCIPAL.** `err_identidad` con **relacion repetida** (`ident_rep` de `diag_relacion`) es
  menor en `lat` que en `pre` en al menos 2 de 3 semillas. Es donde vive la colision de clave.

- **W-2 · MECANICISTA.** La brecha `acierto(relacion unica) − acierto(relacion repetida)` en `lat` es
  a lo sumo **la mitad** de la de `pre`, en al menos 2 de 3 semillas. Separa «bajo el error» de «bajo
  por haber disuelto la colision».

- **W-3 · ESPECIFICIDAD, y es la que ordena la lectura si algo sale raro.** Con relacion **unica**,
  `lat` no empeora: `ident_unica <= 0,03` en las tres (`pre` esta en 0,000-0,003). Si `lat` empeora
  tambien ahi, lo que hubo fue daño general y no disolucion de la colision, y se lee como el §6 de la
  mañana.

- **W-4 · NO-INTERCAMBIO.** `falsa_abst <= 0,10` en las tres unidades `lat` (la compuerta de la
  campania de abstencion, que las tres `pre` pasan y las tres `post` fallan), y `nose` no cae mas de
  0,05 respecto de su gemela `pre`.

## 6. Regla de decision, comprometida por adelantado

- Si **W-0 falla**, se archiva sin interpretar y se declara que en esta arquitectura **la query no se
  puede tocar sin romper el modelo**, ni siquiera dejando la inyeccion en su lugar. Eso convertiria
  el trade-off del informe de la mañana en algo mas fuerte que un trade-off, y seria un resultado.
- Si **W-0 pasa y W-1 falla**, entonces la forma de la query **no es la causa** de la colision de
  clave, esta vez con un experimento que si aisla el factor. El mecanismo del 21-ago queda como
  correlacion y la linea se cierra: no se prueba una tercera forma de query.
- Si **W-1 pasa y W-2 no**, hay mejora sin mecanismo y se reporta asi, sin adjudicarsela al
  round-trip.

## 7. Riesgo declarado

La bimodalidad entre semillas esta medida y es grande: en `pre` a 26000 pasos, `ident_rep` vale
**0,0564 · 0,4683 · 0,2529**. Con tres semillas, un W-1 que pase 2 de 3 puede ser bimodalidad y no
efecto. Por eso W-1 se reporta **pareado por semilla** —`lat` s0 contra `pre` s0, y asi— y no por
media, que es lo que la regla de E-I3c dejo escrito («no reportar media de un nivel sin sus semillas»,
y aca ademas las semillas estan apareadas por construccion).
