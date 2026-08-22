# PREREG · EL EMPATE DE CLAVE COMO DETECTOR SIN ETIQUETAS

**CONGELADO el 2026-08-21 antes de escribir la sonda.** El SHA queda registrado en
`SHA_EMPATE_CLAVE.txt`, calculado sobre este archivo tal como está. Lo que se corra después se mide
contra esto; cualquier cambio posterior va como desviación declarada, no como edición.

Antecedentes: `INFORME_ROUNDTRIP_20260820.md` (la causa), `SMOKE_EMPATE_20260821.md` (el
instrumento), `INFORME_SIN_ETIQUETAS_20260820.md` y `INFORME_MONITOR_20260820.md` (las dos vías
cerradas que esto intenta suceder).

## §1 · Qué se prueba y por qué es distinto de lo ya cerrado

Tres vías cerradas buscaron la señal en la **salida**: el logit (la información está pero no en forma
de valle), su densidad (la mezcla de gaussianas es peor que no hacer nada) y su estabilidad bajo
perturbación (el desacuerdo mide si la respuesta viene del archivo, no si viene de la entrada
correcta). Ésta la busca en la **entrada**, y ataca el error dominante: `err_identidad` ≈ 0,19-0,23,
que el round-trip mostró que vive entero en la colisión de relación.

Lo que el `INFORME_MONITOR` pedía y no encontraba era algo que separe «anclado en la entrada
correcta» de «anclado en cualquier entrada». El empate de clave es esa distinción, del lado de la
entrada y sin etiquetas: cuando dos entradas del archivo compiten a la par, el modelo elige entre
ellas al azar y el `0,42 × 0,44 ≈ 0,185` del round-trip reconstruye el error global.

**Mecanismo, derivado del código y no supuesto** (`modelo.tronco`, verificado en el smoke): la
lectura se inyecta en el bloque 0 sobre `h = emb[x]`, antes de la conv y del mixer, así que la query
es `ln(emb[token]) @ qr` — función pura del token de su posición. **El modelo no puede formar una
query conjunta entidad × relación**; consulta token por token y la conjunción la resuelve aguas
abajo. De ahí que la relación funcione como atajo, y de ahí que el empate sea observable posición por
posición.

## §2 · Instrumento, fijado antes de correr

Sobre los scores crudos `sim` del bloque 0, por posición de la consulta y sólo sobre entradas válidas:

- `z_foco` = `(s1-s2)/std` en la posición de máximo matcheo.
- `z_min` = el menor `(s1-s2)/std` entre las posiciones hasta la de la respuesta.
- `consenso` = solapamiento entre los top-2 de las dos posiciones de mayor matcheo. Sale del
  mecanismo del §1: si la posición-relación apunta a un conjunto de entradas y la posición-entidad a
  otro, la conjunción es ambigua. Es la predicción más específica del diseño y la que ninguna de las
  tres vías cerradas podía formular.

Queda **excluida** `r21 = p2/p1`: el smoke la midió y está saturada contra 1 porque la lectura es
casi uniforme (~0,92 en las ocho celdas, con ~6 entradas válidas). Se declara para que no vuelva a
entrar por la ventana.

**Unidades y presupuesto.** Verificado en disco antes de congelar: las 8 unidades de la familia `c`
existen a **14000**, y las 3 de nivel 4 tienen además **20000**. El borrador de esta mañana pedía las
8 a 20000; no existen y extenderlas costaría 5 × 6000 pasos de GPU. La resolución no es conformarse
con lo que hay, es que **el presupuesto pasa a ser un eje del diseño**:

- **Brazo principal: las 8 unidades a 14000**, todas al mismo paso, que es donde el fenómeno tiene
  varianza para ser detectado.
- **Brazo de presupuesto: las 3 de nivel 4, a 14000 y a 20000**, pareadas. Pone a prueba la
  predicción del `INFORME_ROUNDTRIP`: a 20000 la colisión baja a 0,18-0,25 con la relación única en
  0,000, así que **el detector tiene que perder poder ahí**. Si detecta *igual* a 20000, no está
  midiendo la colisión.

Esto no repite el error del 20-ago —medir todo a 14000 y sacar una conclusión general— porque acá el
paso no es un supuesto de fondo sino una variable declarada, y la predicción sobre él va con signo
comprometido (E-6).

Se reporta **por unidad, nunca sólo la media** (E-I3d: la bimodalidad entre semillas es parte del
fenómeno).

**Lectura desde copia congelada**, con checkpoint y paso impresos por unidad (regla D-1 del 20-ago:
una unidad que entra en un análisis no puede estar entrenándose al mismo tiempo).

## §3 · El nulo, declarado antes de mirar

Es lo que dio el veredicto en el corte sin etiquetas, donde U-1 «pasaba» 2/8 y eran exactamente las 2
donde el nulo pasaba 99-100/100.

**Nulo N-1:** reemplazar los scores del archivo por gaussianas de igual μ y σ por episodio,
conservando el número de entradas válidas. Si `z_foco` discrimina igual sobre el nulo, la señal es
estructura del estadístico y no del modelo.

**Nulo N-2, específico de esta vía:** barajar qué entrada del archivo corresponde a qué hecho,
dejando los scores intactos. Rompe la relación entre el empate y la colisión sin tocar la
distribución. Un detector que sobreviva a N-2 está midiendo la forma de los scores, no la colisión.

Ninguno de los dos es «permutar etiquetas»: el monitor v1 enseñó que un nulo mal elegido da limpio
sin significar nada.

## §4 · Predicciones, con sus umbrales

Se separan las dos preguntas que el smoke tiene mezcladas.

**Detección de la condición (lo que el smoke ya sugiere):**

- **E-1** AUC(`z_foco`; relación repetida vs única) ≥ 0,60 en ≥ 6 de 8 unidades.
- **E-2** E-1 se sostiene dentro de los hechos **no revisados** (descarta el confound de versiones,
  que en el smoke resultó ir en contra: sacándolo el efecto sube).
- **E-3** (nulo, bloqueante) N-1 y N-2 pasan en ≤ 2 de 8. **Si un nulo pasa donde pasa el detector, la
  unidad no cuenta como éxito**, se declare lo que se declare en E-1.

**Conversión en abstención (la pregunta que importa, y la que el smoke NO respalda):**

- **E-4** AUC(`z_foco`; `err_identidad` vs acierto) ≥ 0,65 en ≥ 5 de 8. El smoke da **0,53-0,58**, o
  sea E-4 arranca en contra. Se declara igual y con el umbral alto a propósito: detectar que dos
  entradas empatan no es todavía saber que la respuesta va a salir mal, y ésa es la distancia que
  esta vía tiene que cubrir para valer.
- **E-5** `consenso` mejora sobre `z_foco` en E-4 por ≥ 0,05. Es la predicción propia del mecanismo.

**Especificidad (el brazo de presupuesto):**

- **E-6** En las 3 unidades de nivel 4, el AUC de E-1 **baja** de 14000 a 20000 en ≥ 2 de 3. El
  fundamento es el `INFORME_ROUNDTRIP`: a 20000 la colisión cae a 0,18-0,25 y con relación única
  llega a 0,000, así que si el detector mide colisión tiene que perder poder cuando la colisión se
  va. **Un detector que rinde igual a 20000 está midiendo otra cosa** — y ésa es la falla que ni el
  logit ni el desacuerdo pudieron descartar en su momento.

## §5 · Regla de cierre, comprometida por adelantado

- Si **E-3 falla**, la vía se cierra y no se prueba un tercer estadístico sobre la misma señal.
- Si **E-1 y E-2 cumplen pero E-4 falla**, el resultado se reporta como lo que es: **la colisión es
  observable sin etiquetas, y aun así no alcanza para abstenerse**. Eso sería un negativo con
  mecanismo identificado, del mismo tipo que los tres anteriores, y **cierra la línea de detectar la
  abstención desde una señal interna** — que es la pieza que le falta al plan del modelo que sabe que
  no sabe.
- Si E-4 cumple, recién ahí tiene sentido la pregunta siguiente, que **no** es parte de esto: usar el
  detector como compuerta de abstención y medir `falsa_abst`.

## §6 · Lo que este diseño no puede contestar

- Vale para este archivo y este idioma cerrado. Que la colisión sea de *relación* es una propiedad de
  cómo el generador reparte 4 hechos sobre 6 relaciones.
- No prueba causalidad. Que el empate acompañe al error no dice que el error sea por el empate; para
  eso haría falta intervenir sobre las claves, que es otro experimento.
- Y una advertencia que el round-trip dejó escrita: a 20000 pasos la colisión baja a 0,18-0,25 con la
  relación única en 0,000. **El fenómeno se está disolviendo con el presupuesto**, así que medirlo más
  tarde puede dejar sin varianza justo lo que se quiere detectar.
