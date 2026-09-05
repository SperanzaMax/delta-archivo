# PREREG · LA CURVA DE DILUCION · 2026-09-05

Congelado ANTES de correr el instrumento. Se hashea y se compara al informar.

## Por que

La ingenieria inversa del 5-sep deja una cosa clara: **el banco nunca probo un archivo grande.**
Un episodio archiva a lo sumo `n_sesiones * E_MAX = 40` entradas, y de esas la mask deja unas 20
vivas. El objetivo del proyecto —«que no olvide lo que le dije»— pide lo contrario: un archivo que
CRECE conversacion tras conversacion. La revision del 4-sep ya establecio que el cuello **no es la
velocidad sino la precision, por dilucion del softmax**, pero eso se afirmo por argumento y **no
esta medido en ningun lado**.

Esto lo mide, **sin entrenar nada**: sobre un checkpoint ya entrenado se le agregan al archivo `X`
entradas provenientes de OTROS episodios reales —literalmente «lo que le dije en otras
conversaciones»— y se mira cuando se rompe.

## Montaje

- Checkpoint `ckpts/kq3_s0.pkl` (kernel 5, el que resolvio la ventana) y su control `ckpts/v3_s0.pkl`.
- Archivo del episodio + `X` distractores muestreados de un pool de entradas reales de otros
  episodios, generadas por el mismo generador con otra semilla.
- `X` ∈ {0, 40, 120, 360, 1080, 3240}. X=0 es la condicion actual del banco.
- Los distractores llevan `turnos` sorteados **en el mismo rango** que el episodio: compiten con
  igualdad de sello de orden. Es el caso realista (hechos viejos de otras sesiones) y el peor.
- `mask` True para todos, asi que compiten de verdad.
- Metricas: exactitud de la respuesta; RECUP = la entrada del hecho preguntado gana la lectura
  (rank 0), medida en la posicion de maximo foco igual que `rank_hecho.py`; masa de la ganadora;
  entropia de la lectura.

## Prediccion, escrita antes del dato

1. **D-1** RECUP cae de forma monotona con `X`.
2. **D-2** La exactitud cae de forma monotona con `X`.
3. **D-3** Existe un `X` en el rango probado donde la exactitud cruza el piso trivial **0,4065**.
4. **D-4** La entropia de la lectura crece con `X` y la masa de la ganadora baja.

## Que decide

- Si D-1 a D-3 se cumplen: la dilucion es el cuello real, `PREREG_FILTRADO_PREVIO` (SHA `3b7032b0`)
  queda **justificado por dato propio** y merece la T4.
- **Si NO caen** —si el modelo aguanta 3240 distractores—, entonces la dilucion NO es el cuello, el
  filtrado previo pierde su motivacion y la T4 tiene que ir a otro lado. Este resultado seria el
  mas util de los dos y hay que informarlo igual.
- D-4 sin D-1/D-2 significaria que la lectura se ensucia y la respuesta no: el archivo tendria mas
  margen del que su distribucion sugiere.

## Limite declarado por adelantado

Mide **acceso con el modelo congelado**, no lo que un modelo ENTRENADO con archivos grandes
aprenderia a hacer. Un RECUP que cae aca no prueba que sea imposible aprenderlo; prueba que no sale
gratis. Y los distractores son del mismo idioma de 242 tokens.
