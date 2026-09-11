# PLAN DE PRUEBAS · lo que hoy no sabemos, y cómo se mide · 2026-09-11, 17:00

Pedido de Maxi después del resultado de la colisión (`PREREG_TOPK_ENTRENADO.md` §11): «empezá a
hacer las pruebas para tener las respuestas a todo lo que hoy no tenemos». Sus tres preguntas
(corte fijo + archivo diario; idioma primero y conocimiento como archivo; RAG contra archivo en
tokens y velocidad) dejan estas incógnitas medibles. Predicciones escritas ANTES de correr.

## P1 · ¿Hasta dónde generaliza? (escala del archivo en inferencia)

Banco `dilucion.py`, real + viejo + PERT=1, pool fresco, POOL = 32.768, X = 3.240 · 10.000 ·
30.000, 256 muestras, sobre `tt3_s0-2` (top-2) y `rq3_s0` (denso). Entrenados con 161.
**Predicción:** el top-2 baja despacio (≥ 0,80 con 10.000, ≥ 0,60 con 30.000) porque lo que decide
es que la propia salga entre las dos mejores y el sello + pertenencia no dependen de N; el denso
sigue en el piso. Si el top-2 cae por debajo de 0,5 con 10.000, la generalización tiene un techo
cerca y hace falta P4.

## P2 · El costo en archivo corto (−0,053), ¿es sistemático?

X = 0, 2.048 muestras, los siete checkpoints (`tt3_s0-2`, `rq3_s0`, `rp3_s0-2`).
**Predicción:** con 2.048 muestras la diferencia media top-2 − denso queda entre −0,02 y −0,05, o
sea existe pero es chica; si `tt3_s1` sola explica más de la mitad, es semilla y no mecanismo.

## P3 · La curva de costo real de la lectura contra N (para la pregunta 3 de Maxi)

`piloto_lectura.py` en T4: `responder` sobre un archivo de N = 40 · 1.000 · 10.000 · 100.000 ·
300.000 entradas (vectores aleatorios, D = 128, consulta de 48 tokens, batch 64), softmax completo
y top-2, tiempo por consulta y memoria pico; y el mismo tronco leyendo el episodio como TEXTO en el
prompt (los 40 enunciados concatenados, ~370 tokens) sin archivo, como haría un RAG.
**Predicción:** la lectura del archivo es lineal en N y con 100.000 entradas sigue por debajo del
costo de procesar 370 tokens de prompt; con 300.000 lo supera. El top-2 con máscara no ahorra
tiempo (se calcula sobre las N); el ahorro real necesita índice, que no está implementado.

## P4 · Escalar el entrenamiento: 52 sesiones viejas (520 entradas) con gradiente

`tw3_s0` = `tt3_s0` con `--ses-extra 52 --micro-batch 4 --batch-eval 4`, 2.000 pasos, fotos 8.
**Predicción:** ~5 s/paso; con 3.280 en el banco ≥ 0,93 (no empeora) y con 10.000 mejora sobre
`tt3` en ≥ 0,10. Si la memoria de la T4 no alcanza con micro-batch 4, se baja a 2.

## P5 · Dos experimentos nuevos, a pre-registrar por separado antes de correrlos

- **Hechos encadenados (composición).** Hoy la respuesta es un token y un hecho. La pregunta 1 de
  Maxi necesita saber si el modelo puede combinar dos entradas del archivo: «el dueño de taller es
  ana» + «la altura de ana es 40» → «¿cuánto mide el dueño de taller?». Requiere extender el idioma
  (valores que sean entidades) y una forma de pregunta nueva; es un nivel nuevo del banco.
- **RAG en el prompt contra archivo, en igualdad.** Mismo modelo, mismos datos: un brazo sin archivo
  que recibe los enunciados como texto en la consulta (T ≈ 370) y un brazo con archivo. Se mide
  exactitud, tokens por consulta y tiempo. E-I1 ya midió la mitad (inyección tardía + top-k duro,
  la receta del RAG, 0,0167 contra 1,0000); falta el brazo de texto en el prompt.

## Orden de hoy

P1 y P2 en una T4 (minutos), P3 en otra (minutos), P4 se lanza y corre solo (~3 h). P5 mañana.

## RESULTADOS (18:40, con los datos a la vista)

**P1 · generaliza hasta 30.000.** Banco real + viejo + PERT=1, pool fresco de 32.768, 256 muestras
(`controles_20260911/banco_fresco_pert1_P1escala_*.json`):

| lectura | unidad | 3.280 | 10.040 | 30.040 |
|---|---|---|---|---|
| top-2 | `tt3_s0` | 0,9336 | 0,9336 | 0,8555 |
| top-2 | `tt3_s1` | 0,9336 | 0,8750 | 0,8711 |
| top-2 | `tt3_s2` | 0,9453 | 0,8867 | 0,8594 |
| softmax | `rq3_s0` | 0,1094 | 0,0234 | 0,0000 |

Predicción cumplida y superada (pedía ≥ 0,80 con 10.000 y ≥ 0,60 con 30.000): entrenado con 161
entradas viejas, el top-2 sigue en **0,86 con 30.000**, o sea 186 veces más entradas que las que
vio entrenando; el denso está en cero. La caída de 0,93 a 0,86 entre 3.280 y 30.000 es lenta, y lo
que decide es que la propia salga entre las dos mejores: el sello y la pertenencia no dependen de N.

**P2 · el costo en archivo corto es real, chico y no es de una sola semilla.** 2.048 muestras, X = 0:
top-2 0,916 / 0,912 / 0,981 (media 0,936); softmax 0,966 / 0,964 / 0,964 / 0,996 (media 0,973).
Diferencia **−0,037**, dentro de lo predicho (−0,02 a −0,05); dos de tres semillas en 0,91, la
tercera en 0,98. Hipótesis para el prereg siguiente: con archivo corto el top-2 deja afuera la
tercera entrada cuando las dos mejores empatan (versiones), y un K=3 en archivo corto o un currículo
de K lo recuperaría. Se mide barriendo K en inferencia sobre `tt3` (barato) antes de reentrenar.

**P3 · la lectura es lineal en N** (`piloto_lectura_20260911.json`): 1.000 entradas = la pregunta
sola (0,27 ms); 3.280 = 4,4× menos que 40 enunciados como texto; cruza al texto de 368 tokens en
~30.000. El top-2 con máscara no ahorra; el ahorro a escala necesita índice. 100.000 no cabe
replicado por lote en la T4 (limitación del piloto, no del mecanismo).

**P4** (`tw3_s0`, 52 sesiones viejas con gradiente) en vuelo.
