# INFORME · EL EMPATE DE CLAVE (2026-08-21)

Pre-registro `PREREG_EMPATE_CLAVE.md`, SHA `b78b2141…`, congelado antes de escribir la sonda.
Instrumento `micro_lm/sonda_empate.py`, resultados en `micro_lm/empate_20260821.json`. 1024 muestras
por celda, checkpoints de `ckpts/rt_congelados/` (nada entrenándose al mismo tiempo), sólo preguntas
con respuesta, generador de prueba 77000 + semilla.

**Veredicto en una línea: la colisión de clave SÍ es observable sin etiquetas, y aun así no alcanza
para abstenerse.** Es la cuarta vía que se cierra, y la primera que se cierra con la señal encontrada
en vez de ausente.

## Resultado

| celda | `err_id` | E-1 | E-2 | N-1 | N-2 | E-4 | E-5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `c1_s0`@14000 | 0,0068 | 0,5167 | 0,5672 | 0,5256 | 0,4968 | 0,6990 | 0,4082 |
| `c2_s0`@14000 | 0,0088 | 0,5794 | 0,5847 | 0,4880 | 0,5354 | 0,5558 | 0,4061 |
| `c3_s0`@14000 | 0,1963 | **0,6449** | 0,6358 | 0,5213 | 0,4973 | 0,5804 | 0,5042 |
| `c3_s1`@14000 | 0,2266 | **0,7062** | 0,7662 | 0,5338 | 0,5263 | 0,6696 | 0,4807 |
| `c3_s2`@14000 | 0,2158 | **0,7361** | 0,7970 | 0,5228 | 0,4948 | 0,6681 | 0,4647 |
| `c4_s0`@14000 | 0,1582 | **0,6057** | 0,6404 | 0,4944 | 0,5216 | 0,5956 | 0,4610 |
| `c4_s1`@14000 | 0,1865 | **0,6416** | 0,7219 | 0,4808 | 0,4723 | 0,5625 | 0,4694 |
| `c4_s2`@14000 | 0,1709 | **0,6412** | 0,6266 | 0,4830 | 0,5244 | 0,5757 | 0,4658 |

- **E-1 CUMPLE, 6 de 8** (pedía ≥ 6).
- **E-2 CUMPLE, 6 de 8.** Sacar los hechos revisados no baja el efecto: lo **sube** en 6 de 8 celdas.
  El confound de versiones queda descartado por el lado que me incomodaba.
- **E-3 (bloqueante) CUMPLE con margen: los dos nulos pasan 0 de 8.** N-1 (scores reemplazados por
  gaussianas de igual μ y σ) queda en 0,476-0,534; N-2 (etiqueta de colisión permutada dentro de
  estratos de tamaño de episodio y revisión) en 0,472-0,535. Ninguno se acerca a 0,60.
- **E-4 NO CUMPLE: 3 de 8** contra 5 pedidos.
- **E-5 NO CUMPLE: 0 de 8**, y con el signo dado vuelta — ver abajo.

## Lo que hace fuerte al positivo parcial: dónde NO detecta

**Las dos únicas celdas donde E-1 falla son `c1_s0` y `c2_s0`, que son exactamente las dos donde no
hay colisión que detectar**: `err_identidad` vale 0,0068 y 0,0088 ahí, contra 0,158-0,227 en las seis
de nivel 3 y 4. El round-trip ya había mostrado por qué —en los niveles fáciles el modelo sí usa la
entidad, contesta distinto a cada una— y el detector se comporta en consecuencia.

Un detector que hubiera «encontrado» empate también ahí estaría midiendo otra cosa.

## E-6 · el control de especificidad, y es el resultado más limpio del experimento

| unidad | AUC a 14000 | AUC a 20000 | |
|---|---:|---:|---|
| `c4_s0` | 0,6057 | 0,5467 | baja |
| `c4_s1` | 0,6416 | 0,6297 | baja |
| `c4_s2` | 0,6412 | 0,5717 | baja |

**CUMPLE 3 de 3.** El `INFORME_ROUNDTRIP` había medido que a 20000 la colisión se disuelve, y en
estas mismas celdas `err_identidad` cae de 0,158-0,187 a 0,075-0,109. **El detector pierde poder
exactamente donde el fenómeno que dice medir se va.**

Es el control que ninguna de las tres vías anteriores tuvo: no prueba que el detector sirva, prueba
que **mide lo que dice medir**. Un estadístico que rindiera igual a 20000 estaría leyendo estructura
del episodio.

## E-5 · la predicción propia del mecanismo falla, y falla con el signo invertido

`consenso` da 0,406-0,504: no sólo no mejora sobre `z_foco`, está **por debajo de 0,50**, o sea el
efecto va al revés del predicho. Yo esperaba que **menos** solapamiento entre las dos posiciones de
mayor matcheo significara conjunción ambigua y más error. Lo medido es lo contrario: **más
solapamiento acompaña más error** (0,46 invertido = 0,54).

La lectura que encaja con el mecanismo, y hay que decir que es post-hoc: si las dos posiciones de la
consulta apuntan al mismo conjunto de entradas, no hay dos mediciones independientes que cruzar y la
conjunción entidad × relación no se puede resolver. El solapamiento alto no indica acuerdo útil,
indica **redundancia**. Es la misma idea que la fusión de cabezas de R8 —cruzar mediciones
independientes fija la posición— vista por la cara negativa.

Queda como observación, no como hallazgo: la predicción estaba escrita con el signo opuesto y no se
puede cobrar un acierto invirtiéndola después.

## Por qué la señal no alcanza para abstenerse

E-1 detecta la **condición** (¿hay dos entradas compitiendo?) con AUC 0,61-0,74. E-4 pregunta por el
**desenlace** (¿esta respuesta va a estar mal?) y ahí el mismo estadístico da 0,55-0,70, con sólo 3 de
8 sobre el umbral.

La brecha entre las dos cosas es el resultado, y tiene una explicación estructural: cuando dos
entradas empatan, el modelo acierta **la mitad de las veces** —el round-trip lo midió: 0,45-0,58 de
acierto con relación repetida—. Así que detectar el empate perfecto identificaría un conjunto donde
la mitad de las respuestas son correctas. **El empate predice el riesgo, no el error.** Un detector
así, usado como compuerta, se llevaría puestas tantas respuestas buenas como malas.

Eso es exactamente lo que `falsa_abst` mide, y es la razón de que E-4 —y no E-1— fuera la predicción
que decidía.

## Regla de cierre (§5 del prereg), aplicada

> Si E-1 y E-2 cumplen pero E-4 falla, el resultado se reporta como lo que es: la colisión es
> observable sin etiquetas, y aun así no alcanza para abstenerse. Eso sería un negativo con mecanismo
> identificado, del mismo tipo que los tres anteriores, y **cierra la línea de detectar la abstención
> desde una señal interna**.

Se aplica tal cual. Las cuatro vías, con lo que cada una dejó:

1. **El logit** — la información está pero no en forma de valle (las poblaciones se separan 1,2 σ y
   una mezcla necesita 2 σ para tener dos modas).
2. **La densidad** — la mezcla de gaussianas es peor que no hacer nada (σ>0,5 pasa 6/8, la mezcla
   2/8).
3. **El desacuerdo** — mide si la respuesta viene del archivo, no si viene de la entrada correcta,
   porque el modelo nunca inventa: se ancla en otra entrada real.
4. **El empate de clave** — mide la condición correcta, con nulos limpios y especificidad demostrada,
   y aun así el error es sólo la mitad de los casos que marca.

## Lo que este informe no dice

- No dice que la abstención sea imposible. Dice que **cuatro señales internas distintas no la
  soportan**, y las cuatro fallan en el mismo punto: separan estados del modelo, no aciertos de
  errores. La cabeza de abstención del 18-ago sigue siendo el mejor resultado de la línea y es
  **supervisada** — aprende de etiquetas, no las descubre.
- Una semilla en `c1`/`c2`, tres en `c3` y `c4`. El patrón entre niveles es fuerte; la variación entre
  semillas dentro de un nivel no se puede separar del ruido con esta n.
- Todo vale para este archivo y este idioma cerrado, donde la colisión es de **relación** porque el
  generador reparte 4 hechos sobre 6 relaciones.
- **No prueba causalidad.** Que el empate acompañe al error no dice que el error sea por el empate;
  eso pide intervenir sobre las claves, que es otro experimento.

## Desviaciones declaradas

- **D-1** · `consenso` se implementó como suma de mínimos entre las dos distribuciones (solapamiento
  continuo) y no como conteo de coincidencias del top-2. El conteo toma tres valores y un AUC sobre
  tres valores es casi todo empates, justo lo que E-5 tenía que resolver con 0,05 de margen.
- **D-2** · el nulo N-2 se implementó como permutación de la etiqueta de colisión **dentro de estratos
  de (entradas válidas, revisado)**. Barajar la asignación entrada→hecho no habría cambiado los
  scores; lo que hay que romper es el pareo entre empate y colisión sin romper la estructura del
  episodio. Estratificar es lo que le da al nulo la posibilidad de fallar: si el detector estuviera
  leyendo el tamaño del episodio, el nulo estratificado seguiría discriminando y lo delataría.
- **D-3** · el prereg pedía las 8 unidades a 20000 y en disco existen a 14000, con sólo las 3 de nivel
  4 extendidas. En vez de gastar 5 × 6000 pasos de GPU, el presupuesto pasó a ser un eje del diseño
  (brazo principal a 14000 + brazo pareado en nivel 4), y eso **agregó** E-6, que terminó siendo el
  control más informativo. La corrección se hizo antes de congelar, no después de ver datos.
- **D-4** · 1024 muestras por celda, no 2048. El prereg no fijaba n. Con ~430 positivos y ~590
  negativos el error estándar del AUC es ≈ 0,018, y los efectos de E-1 están 6-13 σ por encima del
  nulo. Para E-4 la n de errores es menor (75-230) y ahí el error estándar sube a ≈ 0,03: **E-4 está
  medido con menos precisión que E-1**, aunque no tanta menos como para explicar una brecha de 3/8
  contra 5/8.
