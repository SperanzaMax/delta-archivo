# Gemación — resultados geométricos (2026-08-08)

Modelo mínimo: geometría pura, NumPy, sin red y sin entrenar. `gemacion_e1.py`.
Recuerdos en S^{d−1}; cada revisión se deposita a distancia `eps` de la versión anterior, con
componente `alpha` en una dirección temporal `t̂` y `1−alpha` aleatoria. La consulta usa la clave
original + ruido, sesgada `delta` hacia `t̂`.

- **M1** = la versión **vigente** es el top-1 (¿se recupera la información al día?)
- **M2** = el top-1 pertenece al **recuerdo correcto**, cualquier versión (¿se identifica el ítem?)

Tres orígenes de `t̂`: **global** (uno para todo el índice), **campo** (`t̂(x)` función determinista
de la posición) y **libre** (uno por recuerdo, guardado como metadato, consulta en dos saltos).

---

## R1 — La geometría agrupa perfecto pero no ordena

Sin eje temporal (revisiones en dirección aleatoria), con d ∈ {16, 64, 256} y eps ∈ [0, 1.5]:

- **M2 = 1.000 en toda la grilla.** Nunca confunde un recuerdo con otro.
- **M1 ≈ 0.000 en toda la grilla.** El top-1 es siempre la versión **más vieja**: se consulta con
  la clave semántica y la v0 sigue en el índice.

Es el modo de fallo que VersionRAG mide en RAG convencional (58 % en consultas versionadas):
se recupera contenido semánticamente similar pero **temporalmente inválido**.

**La dimensión compra radio.** Pureza del vecindario top-5 a eps = 0.8: d=16 → 0.515,
d=64 → 0.873, d=256 → **1.000**. Coherente con la concentración de la medida sobre la que
Kanerva construyó SDM: en alta dimensión sobra lugar "cerca".

## R2 — Ventana (alpha, delta), N=200 K=4 d=64, 10 semillas, IC95

Máximos por modo (M1):

| eje | mejor celda | M1 |
|---|---|---|
| **libre** | α=0.5, δ=1.2 | **0.996 ± 0.004** |
| global | α=0.5, δ=1.2 | 0.990 ± 0.005 |
| **campo** | α=0.4, δ=1.2 | **0.811 ± 0.016** |

La frontera es una diagonal: cuanto mayor `alpha`, mayor `delta` hace falta. Con α ≥ 0.7 el
mecanismo colapsa a 0.000 para todo δ probado — **el óptimo es intermedio**, ni revisiones
aleatorias ni totalmente alineadas al eje.

**El campo determinista pierde sistemáticamente.** Es un negativo informativo: al depender `t̂`
de la posición, el eje **rota mientras el recuerdo se desplaza**, la trayectoria se curva y el
sesgo calculado en `a_0` deja de apuntar a las versiones lejanas. Converge, desde otro ángulo,
con el resultado de Basu (2603.22858): *un sistema de coordenadas que se mueve rompe la memoria
persistente*. Lo que importa no es que el eje sea local — es que sea **estable a lo largo de la
trayectoria**.

## R3 — Estrés de crecimiento (5 semillas, IC95)

| N | K | entradas | M1 (libre) | M2 (libre) |
|---|---|---|---|---|
| 1 000 | 4 | 5 000 | 0.694 ± 0.031 | 1.000 |
| 10 000 | 4 | 50 000 | 0.663 ± 0.041 | 1.000 |
| **100 000** | 4 | **500 000** | **0.663 ± 0.015** | **1.000** |
| 1 000 | 16 | 17 000 | 0.000 | 1.000 |
| 1 000 | 64 | 65 000 | 0.000 | 1.000 |

**Invariante en N.** Medio millón de entradas no degrada nada: ni la identificación ni la
recuperación de versión. El costo de agregar recuerdos es nulo en calidad.

**No invariante en K.** A δ fijo, con ≥16 revisiones por recuerdo M1 colapsa — aunque M2 se
mantiene en 1.000. El agrupamiento nunca se rompe; lo que se rompe es la legibilidad de la
recencia.

## R4 — Ley de escala δ*(K), y por qué el eje por recuerdo es estructuralmente superior

N=1000, α=0.4, 5 semillas. Celda = **M1 / M2**.

**eje GLOBAL** — M1 y M2 **caen juntas** al subir δ:

| K \ δ | 1.0 | 2.0 | 3.0 | 5.0 |
|---|---|---|---|---|
| 2 | 1.000/1.000 | 0.950/0.950 | 0.140/0.140 | 0.000/0.000 |
| 8 | 0.000/0.999 | 0.349/0.866 | 0.401/0.481 | 0.098/0.101 |
| 16 | 0.000/0.999 | 0.000/0.617 | 0.000/0.208 | 0.000/0.045 |

**eje LIBRE (por recuerdo)** — **M2 = 1.000 siempre**, M1 sube monótona con δ:

| K \ δ | 1.0 | 2.0 | 3.0 | 5.0 |
|---|---|---|---|---|
| 2 | 1.000/1.000 | 1.000/1.000 | 0.998/1.000 | 0.911/1.000 |
| 4 | 0.903/1.000 | 1.000/1.000 | 1.000/1.000 | 1.000/1.000 |
| 8 | 0.000/1.000 | 0.422/1.000 | 0.877/1.000 | **0.987/1.000** |
| 16 | 0.000/1.000 | 0.019/1.000 | 0.059/1.000 | 0.136/1.000 |
| 32 | 0.000/1.000 | 0.008/1.000 | 0.023/1.000 | 0.045/1.000 |

**Mecanismo:** con eje global el sesgo apunta en la misma dirección para todos los recuerdos, así
que un δ grande arrastra la consulta a la región donde viven las versiones tardías **de todos** —
y se pierde el ítem. Con eje por recuerdo el sesgo es específico: apunta a lo largo de la
trayectoria de *ese* recuerdo y nunca cruza a otro vecindario.

**Consecuencia de diseño:** el eje por recuerdo **desacopla** identificar el ítem (M2) de
recuperar su versión vigente (M1). El eje global las acopla, y por eso tiene techo duro.

**Límite cuantificado:** δ* crece superlinealmente con K (K=2 → δ*≈1; K=4 → δ*≈2; K=8 → δ*≈5),
y con K ≥ 16 ni δ=5 alcanza. El sesgo geométrico sirve hasta ~8 revisiones por recuerdo.

---

## Lo que esto dicta para la arquitectura

**Geometría para agrupar, metadato para ordenar.** M2 = 1.000 se sostiene en todas las
condiciones probadas — hasta 500 000 entradas y hasta 64 revisiones. El agrupamiento por
proximidad es lo difícil, y funciona perfecto y gratis. El ordenamiento por recencia es trivial
y no hace falta pedírselo a la geometría: alcanza con leer top-k del clúster y desempatar por
un contador.

El sesgo por eje sigue siendo útil como recuperación en **un solo salto** cuando K es chico
(≤ 8 revisiones), y ahí conviene el eje por recuerdo con δ ≈ 3–5.

---

# R5 — Prueba de Basu: la deriva del sistema de coordenadas

**Nota de método.** El primer intento (`gemacion_deriva.py`) modeló la deriva como ruido
independiente por entrada y dio 1.000 en todas las celdas — síntoma de prueba mal especificada,
no de robustez: el ruido independiente *dispersaba* las versiones viejas y dejaba sola a la
vigente. La deriva real es una **transformación del espacio**: las representaciones de una misma
época se mueven juntas, y el daño viene de que las entradas quedan congeladas en el marco de su
época mientras la consulta se calcula en el de hoy. Corregido en `gemacion_deriva2.py` con
rotaciones, más el tiempo transcurrido desde la última escritura (`gap`).

## R5.1 — Curva de tolerancia

Deriva como rotación, eje por recuerdo, gap=8, 5 semillas:

| cos(marco de hoy, marco de escritura) | M1 gemación | M2 | M1 sobrescritura |
|---|---|---|---|
| 1.000 | 1.000 | 1.000 | 1.000 |
| 0.837 | 1.000 | 1.000 | 1.000 |
| 0.711 | 1.000 | 1.000 | 1.000 |
| 0.455 | 0.887 ± 0.026 | 0.889 | ~0.91 |
| −0.118 | 0.001 | 0.002 | 0.018 |

**Umbral: la memoria persistente funciona mientras cos ≳ 0.7, degrada entre 0.7 y 0.4, y muere
por debajo.**

**Se falsó mi predicción previa.** Yo esperaba que la gemación fuera *más* frágil que un índice
plano, por codificar información en distancias del orden de eps=0.3. No lo es: gemación y
sobrescritura caen prácticamente juntas. Y M1 ≈ M2 en toda la tabla — cuando falla, lo que se
rompe es **identificar el recuerdo**, no ordenar sus versiones. La gemación no agrega fragilidad;
el costo de la persistencia lo paga cualquier memoria.

## R5.2 — El otro lado de la desigualdad: cuánto deriva de verdad

Entrenamiento real en el harness de Ligamento (delta puro, 400 pasos, batch 16, carga 8, 30 s de
CPU). Se mide el coseno del espacio de claves `k = l2n(silu(x·W_k))` contra el del paso 0, sobre
un batch de sondeo fijo:

| paso | acc | cos vs paso 0 | cos vs checkpoint anterior |
|---|---|---|---|
| 0 | 0.016 | 1.000 | — |
| **25** | 0.008 | **0.727** | 0.727 |
| 100 | 0.016 | 0.673 | 0.995 |
| 250 | 0.094 | 0.456 | 0.945 |
| 350 | 0.391 | 0.310 | 0.899 |
| **400** | **0.844** | **0.207** | 0.928 |

**El espacio de claves sale de la zona segura (cos < 0.7) en ~25 pasos de entrenamiento.** Un
recuerdo escrito al inicio es irrecuperable hacia el paso 150–250.

**Y la deriva se acelera justo cuando el modelo aprende.** El `cos vs anterior` se mantiene en
0.99 durante la meseta (pasos 50–175, accuracy estancada) y cae a 0.87–0.93 exactamente cuando la
accuracy despega (0.39 → 0.84 entre los pasos 350 y 400). **Aprender es mover las coordenadas** —
la deriva no es ruido de fondo, es la señal misma.

## R5.3 — Veredicto

Basu se confirma, con número: el presupuesto de deriva tolerable se consume en **decenas de pasos**
de entrenamiento, mientras el entrenamiento dura miles.

**La gemación es viable como memoria de inferencia, no como memoria co-entrenada.** Con el encoder
congelado cos = 1 y todo el mecanismo funciona (R1–R4 valen enteros). Escribiendo al índice
mientras el encoder aprende, no hay política de radio ni de eje que lo salve.

Salidas posibles, en orden de costo: (a) escribir sólo con el encoder ya convergido — memoria de
inferencia; (b) coordenadas extrínsecas ancladas, la salida de Basu, estables pero ciegas a la
estructura; (c) reindexado periódico, que es lo que paga REALM y es caro; (d) que el criterio de
escritura sea invariante a la deriva — abierto.

Esto explica de paso por qué el campo entero se quedó en caches intra-secuencia: dentro de una
secuencia el encoder no se mueve.

**Limitaciones:** 1 semilla, una configuración, y un modelo entrenado desde inicialización — el
peor caso. Un encoder preentrenado que se afina derivaría bastante menos; falta medir cuánto.

---

---

# R6 — Modelo preentrenado, y la hipótesis GPS

## R6.1 — El preentrenado cambia el veredicto

Se preentrenó delta puro en carga 8 hasta converger (1500 pasos, acc 1.000, 93 s) y luego se
afinó sobre una distribución nueva (carga 16), midiendo el desplazamiento del espacio de claves
respecto del modelo ya entrenado:

| pasos de afinado | acc | cos vs modelo entrenado |
|---|---|---|
| 0 | 0.965 | 1.000 |
| 100 | 0.977 | 0.940 |
| 200 | 0.992 | 0.911 |
| **400** | **0.988** | **0.882** |

Contra el peor caso (entrenamiento desde inicialización), a los mismos 400 pasos:
**0.207 desde cero vs 0.882 preentrenado.**

**Un modelo ya entrenado mantiene sus coordenadas dentro de la zona segura durante todo un
afinado.** El veredicto de R5 se relaja: la memoria persistente es inviable durante el
entrenamiento desde cero, pero **viable sobre un modelo entrenado que se afina**. La deriva
catastrófica es un fenómeno del aprendizaje inicial, no de la vida útil del modelo.

## R6.2 — La hipótesis GPS: refutada, y no hace falta

La idea: si la deriva fuera predecible como la deformación que corrige un GPS, bastaría con
re-codificar unos pocos ítems de anclaje y aplicar la corrección estimada a todo el índice.
Se probaron cuatro correcciones, estimadas sobre `n` anclas y evaluadas en ítems **held-out**
(sonda de 1664 vectores por cabeza):

| anclas | sin corregir | ortogonal | +escala | lineal | afín |
|---|---|---|---|---|---|
| 8 | 0.880 | 0.572 | 0.572 | 0.627 | 0.606 |
| 32 | 0.880 | 0.811 | 0.811 | 0.751 | 0.748 |
| 128 | 0.879 | 0.878 | 0.878 | 0.884 | 0.884 |
| 1024 | 0.882 | 0.888 | 0.888 | 0.903 | **0.906** |

Error de Gram: **0.60** — la deriva **no preserva las distancias entre ítems**.

**Tres razones por las que la analogía no se sostiene:**

1. **La deriva no es rígida.** El GPS corrige una deformación de forma conocida; acá la métrica
   misma se deforma. No existe rotación que la deshaga — por eso "ortogonal" nunca supera
   materialmente al crudo.
2. **La parte capturable por una transformación global es marginal:** el mejor estimador (afín,
   con 1024 anclas) lleva 0.880 → 0.906. **La mayor parte de la deriva es idiosincrática por
   ítem**, no una transformación compartida.
3. **El costo de anclaje es prohibitivo.** Hacen falta más de 1000 anclas sobre 1664 vectores —
   el 62 % del índice— para ganar 0.026. A ese precio conviene re-codificar todo. Con pocas
   anclas (8–64) **la corrección empeora las cosas**.

El GPS necesita pocos satélites porque la deformación tiene pocos grados de libertad. Esta deriva
tiene tantos grados de libertad como ítems.

**Pero el punto práctico es otro: no hace falta corregir.** Con cos = 0.88 tras un afinado
completo, el sistema está cómodamente sobre el umbral de 0.7 sin corrección alguna.

## R6.3 — Limitación que queda abierta (importante)

La curva de tolerancia de R5.1 se midió con la deriva modelada como **rotación**, que preserva la
estructura relativa dentro de cada cohorte. La deriva real **no es rígida** (Gram 0.60). Por lo
tanto **el cruce "cos = 0.88 ⇒ seguro" es tentativo**: una deriva no rígida del mismo coseno puede
dañar más que una rotación equivalente. Falta medir la tolerancia con deriva no rígida antes de
dar el régimen por seguro. Es el pendiente número uno.

---

---

# R7 — Tolerancia no rígida, y la hipótesis del barrio

## R7.1 — La no rigidez no agrega daño: R6.3 queda cerrado

Deriva no rígida = rotación + estiramiento (matriz simétrica) + ruido **idiosincrático por ítem**,
que es la parte que ninguna corrección global puede capturar. Comparada con la rígida a igual
coseno:

| tipo | cos | M1 vigente | M2 |
|---|---|---|---|
| rígida | 0.829 | 1.000 | 1.000 |
| rígida | 0.604 | 0.997 ± 0.003 | 0.997 |
| rígida | 0.435 | 0.861 ± 0.039 | 0.862 |
| **no rígida** | **0.881** | **1.000** | **1.000** |
| no rígida | 0.704 | 1.000 | 1.000 |
| no rígida | 0.616 | 0.973 ± 0.012 | 0.973 |

A coseno comparable (0.604 rígida → 0.997 vs 0.616 no rígida → 0.973) el daño es equivalente.
**La advertencia de R6.3 era razonable pero infundada: el umbral de ~0.7 se sostiene**, y el
modelo preentrenado (cos 0.882) cae en zona segura.

## R7.2 — Los datos reales son mucho más duros que la simulación

Con las claves **reales** de un modelo preentrenado (índice de 1664 entradas por cabeza, consulta
con el encoder de hoy contra las posiciones donde se escribió):

| | @1 | @5 | @10 | @25 | @50 | @100 | rango mediano |
|---|---|---|---|---|---|---|---|
| cruda | 0.148 | 0.368 | 0.486 | 0.642 | 0.743 | 0.813 | 12.0 |
| **corregida (afín, 256 anclas)** | 0.192 | 0.440 | 0.568 | 0.731 | 0.825 | **0.886** | **7.0** |

**recall@1 = 0.148, no 1.000.** La simulación geométrica de R1–R7.1 usa puntos **aleatorios** en
la esfera, casi ortogonales entre sí; las claves reales están **correlacionadas** y tienen la
vecindad densamente poblada, así que una deriva de cos 0.88 te pasa por encima de varios vecinos.
**Es una limitación de todo el trabajo simulado previo, y hay que decirlo: la geometría aleatoria
es optimista.**

## R7.3 — La hipótesis del barrio funciona, y me obliga a corregir R6.2

La idea: no hace falta saber la dirección exacta de la casa, alcanza con saber por dónde queda —
la búsqueda se reduce drásticamente. Traducido: la métrica correcta no es el coseno sino el
**rango** del ítem.

Y con esa métrica, el panorama se da vuelta:

- **Sin corrección alguna, el ítem correcto está en la posición 12 de 1664** — el 0.7 % superior
  del índice. El barrio es chiquísimo.
- Buscando en el 1.5 % del índice (top-25) se lo encuentra el 64 % de las veces; en el 6 %
  (top-100), el 81 %.
- **La corrección afín baja el rango mediano de 12 a 7** y sube recall@100 de 0.813 a 0.886.

**Corrección a R6.2:** ahí declaré la hipótesis GPS refutada porque la corrección afín solo movía
el coseno de 0.880 a 0.906 — una ganancia despreciable. Medida por rango, esa misma corrección
**casi divide a la mitad el espacio de búsqueda**. La conclusión era un artefacto de la métrica:
el coseno promedio es insensible a lo que importa. La hipótesis GPS, en su forma débil —dar el
barrio, no la casa— **funciona**.

Y encaja con cómo el campo ya resuelve esto: leer top-k y dejar que la atención softmax elija
entre los candidatos es exactamente lo que hacen HOLA y HAM.

---

---

# R8 — El desempate dentro del barrio

Recurso que no se había usado: el modelo tiene **H=4 cabezas**, cada una con su propio espacio de
claves y su propia deriva. Son cuatro mediciones parcialmente independientes del mismo ítem — la
misma lógica por la que el GPS cruza varios satélites en vez de confiar en uno.

Deriva real, índice de 1664 entradas, 1408 ítems held-out:

| criterio | @1 | @5 | @10 | @25 | @100 | rango mediano |
|---|---|---|---|---|---|---|
| 1 cabeza | 0.209 | 0.462 | 0.582 | 0.755 | 0.929 | 6.0 |
| 1 cabeza + afín | 0.231 | 0.502 | 0.648 | 0.783 | 0.934 | 4.0 |
| **suma de 4 cabezas** | **0.502** | 0.819 | 0.885 | 0.925 | 0.951 | **0.0** |
| RRF (reciprocal rank fusion) | 0.424 | 0.773 | 0.849 | 0.894 | 0.956 | 1.0 |
| **suma + afín** | **0.548** | **0.862** | **0.911** | **0.950** | **0.992** | **0.0** |

*(el 0.209 de "1 cabeza" no contradice el 0.148 de R7.2: allá se promedió el recall de las cuatro
cabezas por separado, acá se reporta la cabeza 0 sola.)*

**La fusión de cabezas es el desempate.** De 0.209 a 0.548 — **2.6×** — y el rango mediano cae a
**0**: en más de la mitad de los casos el ítem correcto pasa a ser directamente el top-1. La
corrección afín agrega +0.046 encima, consistente pero secundaria.

Es la hipótesis GPS confirmada por una tercera vía: **cruzar mediciones independientes fija la
posición**, aunque ninguna de ellas por separado alcance.

**El rank mutuo no aporta nada.** Quedarse con el candidato que también rankea alto a la consulta
da un recall@1 final de 0.548 para todo k probado — idéntico a no usarlo. Negativo limpio.

## R8.1 — El burbujeo: la intuición es correcta, el ahorro está en otro lado

Medido el desorden entre el orden crudo y el orden final:

| top-k | inversiones medias | máximo posible | % de desorden |
|---|---|---|---|
| 10 | 7.3 | 45 | 16.3 % |
| 25 | 52.7 | 300 | 17.6 % |
| 50 | 216.0 | 1225 | 17.6 % |

**La lista llega ~82 % ordenada**, y el desorden es constante en k. Un burbujeo adaptativo cuesta
O(k + inversiones) ≈ 78 operaciones para k=25 en vez de las 300 del peor caso: la intuición se
confirma.

Pero conviene ser exacto sobre dónde está el costo: para k=25 el ordenamiento son decenas de
comparaciones, mientras que puntuar los candidatos son ~1600 multiplicaciones. **El sorting es
ruido; el gasto es el scoring.**

Donde la idea sí paga es en **evaluar menos candidatos**: si el desempate fuera caro (pasar cada
candidato por una red, por ejemplo), con la lista casi ordenada se puede evaluar en orden y parar
temprano. Y los números lo respaldan — con rango mediano 0, evaluando **un solo** candidato se
acierta la mitad de las veces; con cinco, el 86 %.

## R8.2 — El mecanismo completo

1. **Gemación** → deposita versiones cerca, agrupa perfecto, escala a 500 k entradas.
2. **Geometría** → entrega el barrio (top-25 con 95 % de cobertura sobre 1664).
3. **Fusión de cabezas** → fija la posición dentro del barrio (0.209 → 0.548).
4. **Corrección afín** → margen adicional (+0.046).

Estado honesto: **recall@1 = 0.548 no es una memoria exacta confiable en un solo tiro.** Lo que sí
es sólido es reducir 1664 candidatos a 25 conservando el 95 % — y a esa altura el modelo puede
atender a los 25 y resolver por contenido, que es exactamente el diseño al que llegaron HOLA y HAM
por otro camino.

---

---

# R9 — Desempate por contenido: el valor identifica mejor que la clave

Hasta R8 solo se usaron las **claves** de la capa 0. Pero la memoria guarda pares (clave, valor), y
el modelo tiene 4 bloques × 3 proyecciones × 4 cabezas = **48 mediciones** disponibles.

| conjunto | mediciones | @1 | @5 | @25 | @100 | rango med |
|---|---|---|---|---|---|---|
| L0 claves (R8) | 4 | 0.550 | 0.862 | 0.949 | 0.992 | 0.0 |
| **L0 clave+valor** | **8** | **0.908** | 0.962 | 0.975 | **1.000** | 0.0 |
| L0 clave+valor+query | 12 | 0.901 | 0.963 | 0.977 | 1.000 | 0.0 |
| claves 4 capas | 16 | 0.817 | 0.910 | 0.956 | 0.997 | 0.0 |
| valores 4 capas | 16 | 0.911 | 0.957 | 0.976 | 1.000 | 0.0 |
| **TODO** | **48** | **0.936** | 0.961 | 0.975 | 1.000 | 0.0 |

**recall@1 salta de 0.550 a 0.908 agregando solamente el valor.** Y con las 48 mediciones,
0.936. El recall@100 llega a **1.000**: el ítem correcto siempre está en el barrio.

## R9.1 — Aporte individual de cada proyección

| proyección | @1 | @25 | | proyección | @1 | @25 |
|---|---|---|---|---|---|---|
| **L0_v** | **0.780** | 0.977 | | L2_k | 0.305 | 0.597 |
| L0_k | 0.550 | 0.949 | | L2_v | 0.339 | 0.612 |
| L1_v | 0.428 | 0.717 | | L3_k | 0.291 | 0.593 |
| L0_q | 0.360 | 0.969 | | L3_v | 0.403 | 0.706 |
| L1_k | 0.344 | 0.652 | | L3_q | 0.287 | 0.593 |

**El valor solo (0.780) supera ampliamente a la clave sola (0.550).** Es el mejor identificador
individual de todo el modelo. Queda como pregunta abierta por qué: puede que la representación de
valor sea más distintiva (en MQAR el valor es el token a recuperar, menos comprimido), o que `W_v`
derive menos que `W_k`. No está medido.

**Las capas altas aportan poco.** L1–L3 rondan 0.29–0.43, muy por debajo de L0. Sus derivas están
correlacionadas con las de abajo, porque dependen de ellas.

## R9.2 — Diversidad le gana a cantidad

El resultado conceptual del bloque: **"L0 clave+valor" (8 mediciones) = 0.908 le gana a
"claves 4 capas" (16 mediciones) = 0.817.** El doble de mediciones, peor resultado.

Lo que fija la posición no es cuántos satélites hay, sino cuán **independientes** son entre sí. La
clave y el valor de la misma capa son más independientes entre sí que las claves de capas
distintas. Es la misma lección del GPS: importa la geometría de las referencias, no su número.

## R9.3 — Estado final del mecanismo

| etapa | recall@1 |
|---|---|
| 1 cabeza, sin corregir | 0.209 |
| + fusión de 4 cabezas (R8) | 0.502 |
| + corrección afín (R8) | 0.550 |
| **+ desempate por valor (R9)** | **0.908** |
| + todas las proyecciones y capas | 0.936 |

Con recall@25 = 0.975 y recall@100 = 1.000. **La memoria pasó de inutilizable a utilizable**, y el
salto grande no vino de la geometría sino de cruzar mediciones independientes del mismo ítem.

---

---

# R10 — R3 rehecho: la invariancia en N era un artefacto de la dimensión

## R10.1 — El espectro real

| espacio | dim | dim efectiva | \|cos\| medio | λ₁/λ_total |
|---|---|---|---|---|
| claves reales (4 cabezas) | 16 | 10.43 | **0.2276** | 0.188 |
| claves reales (cabeza 0) | 16 | 11.74 | 0.2538 | 0.152 |
| uniforme en S¹⁵ | 16 | 15.85 | 0.2012 | 0.074 |
| **uniforme en S⁶³ (lo que usaba R1–R4)** | **64** | 61.71 | **0.0991** | 0.022 |

## R10.2 — Corrección de diagnóstico: no era la correlación, era la dimensión

En R7.2 atribuí el optimismo de la simulación a que las claves reales están **correlacionadas**.
Los números dicen otra cosa: el \|cos\| medio real (0.2276) apenas supera al uniforme de la misma
dimensión (0.2012), y la dimensión efectiva es 10.4 sobre 16 — hay compresión, pero moderada.

**Lo determinante es que R1–R4 corrieron en d=64 cuando el espacio real de claves por cabeza es
d=16.** El solapamiento se duplica (0.099 → 0.228) solo por bajar la dimensión.

R3 rehecho, con eje por recuerdo:

| geometría | N=1 000 | N=10 000 | N=100 000 |
|---|---|---|---|
| uniforme d=64 (R3 original) | 1.000 | 1.000 | 1.000 |
| **uniforme d=16 (dim real)** | 0.555 | 0.158 | **0.015** |
| **calibrado a claves reales** | 0.581 | 0.225 | **0.037** |

El calibrado (0.037) queda casi igual al uniforme d=16 (0.015) y lejísimos del d=64 (1.000):
**la estructura de correlación aporta poco; la dimensión lo explica casi todo.**
La calibración es fiel — sintético vs real: dim efectiva 12.51 vs 11.74, \|cos\| 0.2585 vs 0.2565.

## R10.3 — La fusión restaura la escalabilidad: R3 no se cae, se corrige

Mediciones independientes de d=16, calibradas a las claves reales (3 semillas, IC95):

| mediciones | N=1 000 | N=10 000 | N=100 000 |
|---|---|---|---|
| 1 | 0.610 ± 0.044 | 0.238 ± 0.067 | 0.037 ± 0.009 |
| 2 | 0.995 ± 0.000 | 0.953 ± 0.009 | 0.802 ± 0.013 |
| **4** | **1.000** | **1.000** | **1.000** |
| 8 | 1.000 | 1.000 | 1.000 |
| 16 | 1.000 | 1.000 | 1.000 |

**Con 4 mediciones la invariancia en N vuelve, completa, hasta 100 000 recuerdos.** El umbral está
entre 2 y 4 — y el modelo tiene exactamente 4 cabezas.

Esto reinterpreta R8/R9: **la fusión no era un truco de desempate, es lo que hace que el mecanismo
escale.** Una sola medición de baja dimensión colapsa al crecer el índice; cuatro mediciones
independientes lo sostienen. La invariancia en N que R3 celebraba es real, pero no la regala la
dimensión alta: la da el cruce de referencias — otra vez, el principio del GPS.

---

---

# R11 — Validación externa con un modelo real (Albert 4.0)

800 embeddings de `albert:v4.0` (Gemma 4 E4B vía Ollama, dim 2048), generados en CPU con guardas
térmicas. Máxima alcanzada: 49 °C, sin una sola pausa térmica.

## R11.1 — El espacio de un LLM real es extremadamente anisotrópico

| espacio | d | \|cos\| medio | dim efectiva (sin centrar) | dim efectiva (centrada) |
|---|---|---|---|---|
| **Albert 4.0** | 2048 | **0.5468** | **2.9** | **16.7** |
| uniforme en S²⁰⁴⁷ | 2048 | 0.0178 | 575.3 | 575.0 |
| claves del harness | 16 | 0.2601 | 8.8 | — |
| uniforme d=64 (R1–R4) | 64 | 0.1002 | 59.3 | — |

La norma del vector medio de Albert es **0.741**, contra 0.035 del uniforme: todos los embeddings
viven en un cono estrecho alrededor de una dirección común. Dos textos cualesquiera —una receta de
pan y un párrafo sobre álgebra lineal— tienen coseno 0.55 entre sí.

La distinción entre las dos columnas importa: **sin centrar** mide concentración, que es lo que ve
la similitud coseno; **centrada** mide variabilidad real, o sea capacidad de distinguir. La señal
semántica de Albert vive en ~17 dimensiones efectivas, no en 2048.

## R11.2 — Y sin embargo escala perfecto

R3 con la geometría de Albert (generador: media + 100 componentes principales + residuo
isotrópico; fidelidad validada — \|cos\| real 0.5468 vs sintético 0.5507), **una sola medición**:

| base | N=1 000 | N=10 000 |
|---|---|---|
| **Albert (2048, realista)** | **1.000** | **1.000** |
| uniforme d=16 (harness) | 0.550 | 0.132 |

Con los 800 embeddings **reales** (sin generador) a N=800: también 1.000.

## R11.3 — Lectura: la capacidad la da la dimensión ambiente, no la de la señal

Es contraintuitivo y es el hallazgo del bloque. Albert tiene señal semántica en ~17 dimensiones
—menos que las 64 de R1–R4, comparable a las 16 del harness— y aun así **no colapsa al crecer N**.

La razón: la gemación deposita las versiones en direcciones aleatorias del espacio **ambiente**, y
ahí hay 2048 dimensiones disponibles aunque la señal ocupe 17. La anisotropía desplaza el coseno
medio a 0.55, pero lo que decide la recuperación **no es el coseno absoluto sino el margen
relativo** contra los competidores, y ese margen se sostiene.

Consecuencias:

1. **El colapso de R10 es un artefacto del harness**, no una propiedad de los LLM. Las cabezas de
   `DH=16` son chiquitas; un modelo real tiene espacio de sobra.
2. **El requisito de ≥4 mediciones (R10.3) aplica al harness, no necesariamente a un modelo real.**
   En Albert una sola medición alcanza.
3. La anisotropía extrema de los embeddings de LLM —que suele tratarse como un defecto— **no
   impide la memoria persistente**.

**Límite de la evidencia:** los 800 embeddings reales solo permiten probar N=800, donde ni siquiera
d=16 colapsa del todo (0.585), así que no es discriminante. La prueba a escala usa el generador, y
su fidelidad está validada solo en \|cos\| medio y espectro de las primeras 100 componentes. Para
cerrar esto haría falta un corpus de decenas de miles de embeddings reales.

---

---

# R12 — Cierre con 10.000 embeddings reales (Colab T4)

10.000 embeddings de `gemma:2b` vía Ollama **0.20.2** (la misma versión que corre en la PC, para
que los vectores sean directamente comparables), sobre 10.000 párrafos naturales de
`Salesforce/wikitext`. Tesla T4, 1564 s, **0 fallos**, 156 ms/embedding (3,8× la CPU local).

## R12.1 — Corrección de R11: la dimensión efectiva estaba mal estimada

| | R11 (n=800) | **R12 (n=10.000)** |
|---|---|---|
| n/d | 0.39 — **covarianza singular** | **4.88** |
| dim efectiva (muestral) | 16.7 | **99.5** |
| dim efectiva (Ledoit-Wolf, ρ=0.066) | — | **113.3** |
| uniforme (referencia) | 575 | 1700.2 |

**El 16.7 de R11 era un artefacto del sesgo, y estaba mal por un factor ~6.** Con n < d la
covarianza muestral es singular y su espectro no significa nada; era exactamente el riesgo
declarado en R11.3 y por eso valía la pena rehacerlo.

El shrinkage mueve la estimación de 99.5 a 113.3 — **14 %, moderado**: la conclusión (dimensión
efectiva ~100 sobre 2048 nominales, contra 1700 del uniforme) **no depende del estimador**.

Concentración de varianza: **54 componentes explican el 50 %**, 680 el 90 %, 1635 el 99 %.

## R12.2 — La anisotropía es un desplazamiento, no una compresión

| espacio | \|cos\| medio | sd | dim efectiva |
|---|---|---|---|
| gemma:2b **crudo** | **0.7371** | 0.0600 | 99.5 |
| gemma:2b **centrado** | **0.0198** | 0.0263 | 99.5 |
| uniforme S²⁰⁴⁷ | 0.0177 | 0.0222 | 1700.2 |

Norma del vector medio: **0.8585** (uniforme: ~0.02).

Restarle la media deja el coseno par-a-par en 0.0198, **prácticamente idéntico al del espacio
uniforme** (0.0177), mientras la dimensión efectiva no se mueve. Son dos fenómenos separables:

1. **Un desplazamiento global** — un vector medio enorme que infla el coseno crudo a 0.74. Es el
   componente que la literatura llama "rogue dimension".
2. **Una concentración de varianza** — la estructura vive en ~100 dimensiones de 2048.

Solo el segundo es una limitación real de capacidad. El primero es un corrimiento del origen que
no destruye información.

## R12.3 — R3 con embeddings reales: confirmado

| base | N=1 000 | N=5 000 | N=10 000 |
|---|---|---|---|
| **gemma REAL** | **1.000** | **1.000** | **1.000** |
| uniforme d=16 (control) | 0.544 | 0.256 | **0.179** |

**Sin generador, sin calibración, sin simulación.** El control de baja dimensión colapsa
progresivamente (0.544 → 0.179) exactamente donde el espacio real no se mueve de 1.000.

**Queda cerrada la limitación de R11.3.** La conclusión de R11 se sostiene y se refuerza: la
capacidad la da la dimensión ambiente, y un LLM real tiene margen de sobra pese a su anisotropía.

## R12.4 — Problemas de entorno resueltos (para no volver a pagarlos)

Ninguno era del experimento; los cuatro costaron una vuelta cada uno:

1. **`zstd` no viene en la VM de Colab** y el instalador de Ollama lo exige para extraer.
2. **Ollama ≥ 0.32 rechaza embeddings** en modelos que no declaran esa capability, y `gemma:2b`
   declara solo `completion`. La 0.20.2 no verifica y los sirve. El error de la versión nueva
   ("Start it with `--embeddings`") es **engañoso: ese flag no existe**.
3. **`datasets` exige `repo_id` con namespace**: `wikitext` a secas da `HfUriError` →
   `Salesforce/wikitext`.
4. **`colab exec` corta por `TimeoutError`** cuando un paso tarda sin imprimir → el trabajo va con
   `nohup` dentro de la VM y el cliente solo lee el log.

Todo consolidado en `colab_bootstrap.py`, con cortes tempranos: si el endpoint de embeddings no
responde, aborta antes de gastar GPU en lugar de reintentar 10.000 veces en silencio.

---

## Pendiente

1. Por qué el valor identifica mejor que la clave (R9.1) — sigue sin medir.
2. Rango en función de la deriva acumulada: ¿cuánto crece el barrio con el tiempo?
3. Rehacer R1–R4 con la geometría real ahora medida (dim efectiva ~100, no 64 ni 16).
2. Por qué el valor identifica mejor que la clave (¿representación más distintiva, o `W_v` deriva
   menos?). No está medido.
3. Rango en función de la deriva acumulada: ¿cuánto crece el barrio con el tiempo?
3. Deriva real con más semillas y otras configuraciones.
4. Ley δ*(K) con más resolución, y si `eps` la corre (acá quedó fijo en 0.3).
5. Rehacer R1–R4 con claves reales correlacionadas en lugar de puntos aleatorios.
3. Acoplar la regla de asignación a la sorpresa `β‖e‖` (sorpresa baja → gemar cerca; alta →
   lugar propio), que es lo que une esto con CENTINELA-01 y VIGÍA-03 — **bajo el supuesto de
   encoder congelado**, que es el único régimen donde R1–R4 se sostienen.

---

# R13 — Gemación implementada en un modelo chico (2026-08-09)

Primer experimento **con un modelo**, no con geometría simulada. Modelo preentrenado en MQAR y
**congelado**; archivo no paramétrico. No se aprende nada nuevo: se mide si la gemación preserva
información recuperable que la sobrescritura destruye.

## R13.1 — La tarea cross-secuencia

Tres secuencias con el estado recurrente **reseteado** entre ellas: S1 escribe L pares, S2 revisa
r de ellos con valores nuevos, S3 consulta. Dos objetivos: **VIGENTE** (último valor) y
**ANTERIOR** (valor previo de las claves revisadas).

Validada: ningún valor aparece en la secuencia de consulta (no es copiable), vigente ≠ anterior
siempre, azar = 1/64 = 0.0156.

## R13.2 — La compuerta: ¿existe un identificador estable entre secuencias?

| representación | misma clave | distinta | AUC |
|---|---|---|---|
| `emb` crudo | 1.000 | 0.062 | 1.000 |
| `ln1` (sin conv) | 1.000 | 0.114 | 1.000 |
| `W_k` sobre `ln1` | 1.000 | 0.247 | **1.000** |
| **`W_k` sobre `conv3`** | 0.681 | 0.523 | **0.789** |

El primer intento usó `W_k sobre conv3` y **falló en silencio**: con AUC 0.789 el umbral de
similitud no se alcanzaba nunca, así que la sobrescritura no sobrescribía y la gemación no gemaba
— ambas se limitaban a apilar entradas. Los números de esa corrida eran artefactos.

**Causa:** `conv3` mezcla cada token con sus dos vecinos; la misma clave rodeada de pares (S1) y
rodeada de consultas (S3) tiene representaciones distintas.

## R13.3 — Con identificador válido: la gemación domina

| archivo | VIGENTE | ANTERIOR |
|---|---|---|
| sin archivo (control) | 0.021 | 0.026 |
| sobrescritura | **1.000** | 0.026 *(azar)* |
| **gemación** | **1.000** | **1.000** |

El control da azar, como debe: sin memoria persistente el estado reseteado no puede responder.
La sobrescritura recupera el valor vigente y **pierde la historia por completo**. La gemación
recupera **ambos**, sin costo.

## R13.4 — Pero el régimen es trivial, y hay que decirlo

`W_k sobre ln1` resultó ser **función pura del token**: cos = 1.0000 exacto entre el mismo token
en posiciones y contextos distintos. El archivo es, entonces, un diccionario token → valor, y la
tarea queda trivializada — el confound que estaba anotado desde el inicio del proyecto.

## R13.5 — La frontera: contexto vs. identificabilidad

Interpolando `xin = (1−α)·ln1 + α·conv3(ln1)`:

| α | AUC ident. | sobrescr. VIG | sobrescr. ANT | gemación VIG | gemación ANT |
|---|---|---|---|---|---|
| 0.00 | 1.000 | 1.000 | 0.026 | 1.000 | **1.000** |
| 0.25 | 1.000 | 1.000 | 0.026 | 1.000 | **0.995** |
| 0.50 | 0.999 | 0.984 | 0.042 | 0.859 | 0.693 |
| 0.75 | 0.925 | 0.799 | 0.047 | 0.635 | 0.182 |
| 1.00 | 0.793 | 0.604 | 0.026 | 0.466 | 0.135 |

Tres lecturas:

1. **La gemación supera a la sobrescritura en ANTERIOR en todos los α.** La sobrescritura nunca
   pasa del azar: pierde la historia por construcción, no por dificultad.
2. **Mientras el identificador es confiable (AUC ≥ 0.999), la gemación es gratis** — iguala a la
   sobrescritura en VIGENTE y además conserva la historia. Dominancia estricta.
3. **Cuando el identificador se degrada, la gemación empieza a pagar** en VIGENTE (0.859 vs 0.984
   en α=0.5; 0.466 vs 0.604 en α=1.0): al no distinguir bien las claves, las versiones se
   confunden entre sí y la más reciente deja de ser recuperable.

La caída es abrupta entre α=0.5 y α=0.75 (AUC 0.999 → 0.925; gemación ANT 0.693 → 0.182).

**Limitación central:** en este modelo, el régimen donde la gemación es gratis (α ≤ 0.25) es
también aquel donde la representación es casi puramente el token. No se pudo exhibir un punto con
contexto sustantivo **y** memoria funcionando. El resultado defendible es **la curva**, no el
1.000/1.000.

## R13.6 — Qué haría falta

Un sustrato donde la identidad de una entidad sea estable entre contextos sin ser el token mismo.
Los embeddings de oración de un modelo real cumplen eso (R12: la identidad venía del texto
completo). El harness de d=64 no lo permite: o la representación identifica y es trivial, o lleva
contexto y no identifica.
