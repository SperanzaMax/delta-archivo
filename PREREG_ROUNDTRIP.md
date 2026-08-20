# PREREG · Consistencia de ida y vuelta

Congelado el 2026-08-20 antes de escribir la sonda.

## 0 · De dónde sale

Dos negativos del mismo día dejaron la misma frase escrita en sus informes: hace falta **algo que
separe «anclado en la entrada CORRECTA» de «anclado en cualquier entrada»**. Ni el logit
(`INFORME_SIN_ETIQUETAS_20260820.md`) ni el desacuerdo bajo sub-muestreo
(`INFORME_MONITOR_20260820.md`) hacen esa distinción, y el control M-3 del monitor explicó por qué:
el modelo **nunca inventa contenido** (`err_fuera = 0,0000` desde el 15-ago), así que cuando la
pregunta no tiene respuesta se ancla en otra entrada **real** del archivo y queda igual de estable.
Estabilidad y corrección dejan de ser lo mismo.

Y hay una lectura del mecanismo, de Maxi, que hace la predicción concreta. Los valores del archivo
que pertenecen a entidades distintas son **eventos disjuntos**: si el valor es de `E'`, no es de `E`.
Pero la lectura del archivo es `Σₙ αₙ·vₙ` — una atención softmax **promedia**. Toma entradas
mutuamente excluyentes y las mezcla ponderadamente, o sea trata como mezcla lo que estructuralmente
es una disyunción. Al emitir sólo el valor, el modelo **marginaliza sobre la entidad de origen**:
`P(V) = Σ_E P(V|E)·P(E)`. La marginalización destruye la disyunción, y eso explica los dos números
que hasta ahora convivían raro: acierta **dentro del conjunto** (`err_fuera = 0,0000`) y pierde **de
quién era** (`err_identidad ≈ 0,23`, el error dominante en N3 y N4).

Si esa lectura es correcta, la información de identidad no se perdió: quedó **repartida entre las
entidades candidatas**. Este experimento va a buscarla ahí.

## 1 · El instrumento, y por qué no es el que parece

La formulación natural —preguntarle al modelo «¿de qué entidad es `X`?»— **no se puede usar y hay que
decirlo antes**: `idioma.pregunta()` genera un solo formato (`cual es el <rel> de <ent> ?` y su
variante `anterior`) y el modelo **nunca vio una pregunta inversa**. Preguntársela sería medir un
formato fuera de distribución, o sea la misma trampa del monitor v1, esta vez del lado de la entrada.

La vuelta se hace **con el único formato que el modelo sí conoce**, sustituyendo la entidad de la
pregunta:

1. **Ida.** Con el archivo intacto, el modelo responde `X` = argmax del vocabulario con `NOSE`
   excluido (la definición que usa toda la campaña).
2. **Vuelta.** Para cada entidad candidata `E'`, se rearma **la misma pregunta con `E'` en el lugar
   de la entidad preguntada** —un token reemplazado, mismo largo, misma relación, mismo tipo de
   pregunta— y se lee `s(E') =` logit del token `X` en esa consulta. El archivo se escribe **una sola
   vez** y no se toca: lo que cambia es a quién se le pregunta.
3. **Cierre.** `vuelta(X) = argmax_{E'∈C} s(E')`, y `cierra = [vuelta(X) == E_q]`.

`C` = las entidades del episodio (vía `con_meta`) **más** la entidad preguntada `E_q`, que en
`nose_ent` no es ninguna de ellas. `E_q` se lee del tensor de consulta como el token del bloque
`ENTIDADES`, no de la metadata: así vale igual para los cuatro tipos de pregunta.

Esto es la inversión por Bayes hecha con el forward directo: la ida da `P(X|E,R)`, la vuelta compara
ese mismo `P(X|·,R)` entre entidades. Y es exactamente la disyunción del §0 puesta a competir: las
candidatas son mutuamente excluyentes y acá se las obliga a repartirse una masa que suma 1.

## 2 · El estadístico y el corte

- **`cierra`** (binario). **Corte estructural: abstenerse si la vuelta no vuelve.** No hay umbral. No
  se ajusta contra etiquetas. Es la misma virtud que tenía «las K pasadas coincidieron» en el
  monitor —una afirmación con sentido sin calibración— pero midiendo **identidad** en vez de
  estabilidad, que es lo que faltaba.
- **`p_E`** = softmax de `s` sobre `C`, evaluada en `E_q`. Score continuo en [0,1], también sin
  etiquetas, para las AUC.

Con `|C| = 5` la banda de indiferencia de `cierra` es 1/5; no se prueban variantes de `C` ni cortes
sobre `p_E` en este experimento.

## 3 · Predicciones

Sobre las **8 unidades de la familia `c` a 14000 pasos** (las tres de nivel 4 desde su copia
`.p14000`), rng de prueba **77000 + semilla**, `p_nose = 0,4`, **2048 muestras por unidad**,
**reportando por unidad**:

- **RT-0 (smoke bloqueante, y tiene dos mitades que pueden fallar).** Sobre 64 muestras de una
  unidad, antes de las ocho:
  - **(a)** sustituir la entidad tiene que **cambiar los logits**: `max|Δlogit| > 1e-3` en ≥ 0,95 de
    las muestras. Es el chequeo que el v1 del monitor no hizo y le costó el experimento.
  - **(b)** en las respuestas que `c1_s0` (nivel 1, el fácil, donde `err_identidad` es casi nulo)
    **acierta**, la vuelta tiene que cerrar en **≥ 0,90**. Si el instrumento no cierra donde el
    modelo sabe, no mide identidad.
  - Si (a) o (b) falla, **el prereg se anula por instrumento vacío y NO cuenta como negativo** — la
    distinción que hubo que hacer explícita el 20-ago.
- **RT-1 (¿hay señal?, con etiquetas).** AUC de `p_E` separando **aciertos** de **errores de
  identidad** ≥ **0,70** en **≥ 6 de 8**.
- **RT-2 (la principal, sin etiquetas).** El corte estructural pasa la compuerta
  (`falsa_abst ≤ 0,10` **y** `nose ≥ 0,50`) en **≥ 6 de 8**, **y además domina a `σ>0,5`** (`nose`
  mayor con `falsa_abst` no peor) en **≥ 5 de 8**. Las dos mitades, porque las referencias del mismo
  día sobre las mismas unidades son **U-1 = 2/8 · σ>0,5 = 6/8 · U-2 = 7/8 · monitor = 0/8**: llegar a
  6/8 sin dominar a `σ>0,5` sería empatar con **no hacer nada**, que es literalmente lo que ese
  criterio mide.
- **RT-3 (nulo de candidatas).** Con `C` armado con entidades que **no aparecen en el episodio**,
  `cierra` debe subir a **≥ 0,95**. Si tampoco cierra contra rivales que el archivo nunca vio, `s`
  no está midiendo competencia entre entidades.
- **RT-4 (nulo de archivo).** Con el archivo **entero tapado**, la AUC de `p_E` debe caer a
  **0,45-0,55**. Verifica que la señal viene de la evidencia y no de la gramática de la pregunta.
- **RT-5 (mecanicista — la predicción del §0, y es la que ninguna vía anterior podía hacer).** En los
  **errores de identidad**, la vuelta apunta a la entidad **dueña real** del valor emitido en
  **≥ 0,50** de los casos, contra un azar de ≈ 1/3 entre las candidatas restantes. Eso es, literal,
  separar «anclado en la entrada correcta» de «anclado en cualquier entrada».
- **Secundaria, declarada y sin criterio** (se reporta pase lo que pase, no decide nada): entropía de
  la posterior sobre `C` en aciertos contra errores. Si el error de identidad es marginalización,
  tiene que ser **mayor en los errores**.

**Celda adicional declarada acá, no después:** las tres unidades de nivel 4 **a 20000 pasos**, donde
la réplica de hoy mostró que las tres pasan la compuerta. Se reporta aparte y **no entra en los
conteos de RT-1/RT-2**, que son a 14000 para ser comparables con las tres referencias.

## 4 · Desenlaces, comprometidos por adelantado

- **RT-2 y RT-5 cumplen** → hay tercera vía: corte sin etiquetas **y** mecanismo identificado. Pasa a
  informe y habilita la pregunta siguiente (usar la vuelta como pérdida auxiliar en entrenamiento),
  que **no** se prueba acá.
- **RT-1 cumple y RT-2 no** → la información de identidad está y la decisión no; se reporta como
  evidencia, no como monitor, y se dice con esas palabras.
- **RT-1 falla** → la vuelta no distingue. **Se cierra la vía y no se prueba una cuarta perturbación**
  (la misma regla del §4 del monitor). Excepción única: que RT-0 haya fallado, en cuyo caso no hubo
  resultado.
- **RT-5 falla con RT-1 cumpliendo** → el score sirve pero **la lectura de la disyunción no queda
  apoyada**: la vuelta detectaría el error sin que el valor provenga de la entidad que la vuelta
  nombra. Se reporta así, sin rescatar el marco.
- **RT-3 o RT-4 fallan** → se archiva sin interpretar, cualesquiera sean RT-1, RT-2 y RT-5.

## 5 · Lo que no puede decir

- Es un monitor **de inferencia** sobre modelos ya entrenados. No dice nada sobre entrenar contra la
  vuelta.
- **La sustitución de entidad interviene la PREGUNTA, no el archivo.** Si `E'` no aparece en el
  episodio, la consulta queda fuera de distribución — es justo lo que RT-3 explota como nulo, y es
  una limitación declarada, no un hallazgo posterior.
- Costo `|C|` forwards **de la consulta** (el archivo se escribe una vez): ~5×, contra el 16× del
  monitor. Igual se mide si la señal existe, no si el costo se justifica.
- Idioma cerrado de 242 tokens, entidades de **un solo token**. En texto natural la vuelta no es
  reemplazar un token, así que esto no prueba que el método transfiera.
- `err_identidad` se define contra la metadata del generador. Un error donde el valor emitido no
  pertenece a **ninguna** entidad del episodio no existe en la práctica (`err_fuera = 0,0000`), pero
  si apareciera se cuenta aparte y no entra en RT-5.

## 6 · Procedimiento

- CPU, cero GPU, sobre checkpoints **congelados**: se copian a `ckpts/rt_congelados/` antes de correr
  y la sonda imprime archivo y paso por unidad. Es la regla D-1 del día — una unidad que entra en un
  análisis no puede estar entrenándose al mismo tiempo. Al momento de congelar este prereg **no hay
  ningún tramo corriendo** (verificado: `colab status` sin sesiones, y el `keep-alive` huérfano de la
  cuenta L terminado).
- **Smoke de una unidad (64 muestras) antes de las ocho**, que es RT-0.
- El veredicto lo escribe una persona leyendo la tabla, no el `print` del script
  (`regla-verificar-antes-de-veredicto`, y el veredicto automático equivocado de E-I3d).
