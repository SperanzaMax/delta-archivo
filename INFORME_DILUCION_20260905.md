# INFORME · LA CURVA DE DILUCION · y el control que la explico al reves · 2026-09-05

Prereg `PREREG_DILUCION.md`, SHA **`f4d91c12`**, congelado antes de correr. Instrumento
`micro_lm/dilucion.py`. Salidas en `micro_lm/dilucion_*.json` y `micro_lm/salidas/dilucion*.log`.
Todo sobre **checkpoints entrenados**, en CPU, sin entrenar nada.

## De donde sale

De la ingenieria inversa de esta mañana (`INGENIERIA_INVERSA_20260905.md`). Al desarmar el banco
aparecio una premisa que nadie habia mirado: **un episodio archiva a lo sumo 40 entradas**
(4 sesiones × `E_MAX` 10). Todo lo que el proyecto sabe —el sello de orden, la ley de la ventana, la
abstencion, el atractor— esta medido con un archivo de a lo sumo 40 entradas. El objetivo pide lo
contrario, un archivo que crece conversacion tras conversacion, y **eso nunca se probo.**

## Resultado 1 · el archivo largo rompe el modelo, y rompe temprano

`ckpts/kq3_s0.pkl`, el del kernel 5. 512 muestras por celda, piso trivial 0,4065.

| entradas | exactitud | RECUP | masa ganadora | entropia |
|---:|---:|---:|---:|---:|
| 40 | **1,0000** | 0,8555 | 0,9855 | 0,068 |
| 80 | 0,7832 | 0,5039 | 0,8123 | 0,591 |
| 160 | **0,3008** | 0,3320 | 0,6418 | 1,070 |
| 400 | 0,0605 | 0,1387 | 0,4721 | 1,662 |
| 1120 | 0,0137 | 0,0273 | 0,2875 | 2,477 |
| 3280 | **0,0039** | 0,0117 | 0,1509 | 3,450 |

Las cuatro predicciones del prereg **cumplen**. Con **cuatro conversaciones guardadas** el modelo cae
por debajo del piso trivial; con 3280 entradas acierta 4 de cada 1000.

**Y el kernel 3 colapsa igual** (1,0000 → 0,0059), asi que **arreglar la ventana no protege de esto.**
Son dos cuellos distintos y este no estaba medido.

## Resultado 2 · el control da vuelta la CAUSA · no es dilucion, es contenido

La explicacion alternativa aparecio al mirar el montaje: los distractores son hechos del mismo
idioma, y con 30 entidades × 24 relaciones = 720 combinaciones, un pool de 3240 contiene por fuerza
entradas que hablan de **la misma entidad y la misma relacion** que la pregunta. Eso no es dilucion,
es **colision**. Se corrieron dos controles antes de informar nada.

**Exactitud**, mismo checkpoint, misma curva, cambiando solo de que estan hechos los distractores:

| entradas | **ruido** | **disjunto** | **real** |
|---:|---:|---:|---:|
| 40 | 1,0000 | 1,0000 | 1,0000 |
| 160 | **0,9922** | 0,6660 | 0,3008 |
| 400 | 0,8320 | 0,3086 | 0,0605 |
| 1120 | 0,4082 | 0,1406 | 0,0137 |
| 3280 | 0,2441 | 0,0840 | 0,0039 |

**RECUP** (la entrada correcta gana la lectura), que es lo que adjudica:

| entradas | **ruido** | **disjunto** | **real** |
|---:|---:|---:|---:|
| 40 | 0,8555 | 0,8418 | 0,8555 |
| 3280 | **0,7852** | 0,4590 | **0,0117** |

- **ruido** — gaussianos con la misma media y desvio por dimension que el archivo real. No pueden
  coincidir con nada.
- **disjunto** — hechos reales del mismo generador, sorteados sobre una mitad de las entidades
  **disjunta** de la que usa el episodio. La colision exacta es **imposible por construccion**.
- **real** — hechos de otros episodios, sin restriccion.

> **EL SOFTMAX NO SE DILUYE POR NUMERO.** Con 3280 competidores sin contenido, la entrada correcta
> **sigue ganando la lectura el 78,5 % de las veces**, contra 85,5 % con archivo de 40. Lo que rompe
> la busqueda no es cuantos son, es **que dicen**.

Y el contenido rompe en **dos capas separables**, las dos grandes:

1. **Interferencia** — entradas que ni siquiera nombran la entidad preguntada ya bajan RECUP de
   0,84 a **0,46**.
2. **Colision** — permitir que hablen de la misma entidad lo baja de 0,46 a **0,012**.

**Lo que queda de dilucion pura es real pero es otra cosa:** con ruido, RECUP casi no se mueve
(0,86 → 0,79) y en cambio la exactitud cae a 0,2441 y la masa de la ganadora de 0,9855 a 0,6715. O
sea **el ranking aguanta y el VALOR leido se ensucia**: la lectura es un promedio ponderado, y con
3280 entradas el 33 % de la masa se va a otras. Es dilucion **del valor**, no **del ranking**.

## Resultado 3 · el sello de orden NO descarta lo viejo

Los distractores del Resultado 1 llevaban turnos solapados con el episodio, o sea parecian
contemporaneos. Eso esta declarado en el prereg como caso peor, **pero es justamente el caso que el
sello de orden existe para resolver y no se le habia dado la chance.** Se repitio poniendo los
distractores en los turnos 0..K−1 y corriendo el episodio a K..63, o sea el archivo largo es
literalmente «lo dicho antes».

| entradas | turnos solapados | turnos viejos |
|---:|---:|---:|
| 160 | 0,3008 | 0,3066 |
| 3280 | 0,0039 | 0,0059 |

**Identico.** Decirle al modelo que todo el archivo largo es anterior **no cambia nada**.

Interpretacion, y es una lectura mia que hay que marcar como tal: **no es que el mecanismo falle, es
que nunca aprendio ese uso.** El sello se entreno para elegir la version mas nueva **de un mismo
hecho dentro de un episodio de 40 entradas**, que es donde dio 0,4570 → 0,9956 y tiene DOI. Filtrar
por antiguedad un archivo largo es un uso distinto que el entrenamiento **nunca le presento**. La
consecuencia practica es la misma de todos modos: hoy no lo hace.

## Y se junta con el tope de 64, que es el otro hallazgo de hoy

`ord` tiene 64 filas y la indexacion de JAX **clampeaba en silencio** (turnos 63, 64, 65 y 200 daban
todos la fila 63). Puesto de otro modo: **el caso central del objetivo —«te dije Ana, despues te dije
Beto», entre conversaciones distintas— es exactamente la colision que rompe el archivo largo, y el
mecanismo que la resolveria se queda sin sello en el turno 64.**

Hoy se puso la guarda `modelo.sello` (`mode="fill"`, NaN fuera de rango), **verificada identica bit a
bit** con indices validos, en `pre`, `lat` y `lat2`: diferencia maxima `0.000e+00`. El fallo pasa de
silencioso a ruidoso. **No levanta el tope**, que es un cambio de arquitectura y va con su prereg
(`NOTA_SELLO_EXTRAPOLABLE.md`).

## Que hacer con `PREREG_FILTRADO_PREVIO` (SHA `3b7032b0`)

**No se cae, y hay que ser justo con eso.** Lo que se cae es **una de sus tres premisas de
motivacion**, la del §1: *«lo que si se rompe es la precision. Con N grande el softmax reparte su masa
entre mas candidatos y la recuperacion cae»*. Medido hoy: **con candidatos sin contenido, no cae.**

Lo que queda, y queda mejor que antes:

- **H sigue en pie** y ahora tiene **mecanismo propio medido**: el filtro ayudaria porque saca
  competidores **con contenido**, no porque baje `N`.
- **F-2 cambia de significado.** Estaba escrito como «la ventaja crece con `N` → es dilucion»; hoy
  sabemos que crecer con `N` **no** distingue dilucion de interferencia. Hay que reescribirlo o
  agregarle el brazo de ruido como control.
- **F-3 se vuelve el criterio caro.** Si el problema es contenido, el «puntaje barato» del §3
  —producto interno **sin** la proyeccion aprendida— es sospechoso de tener falsos negativos altos
  justo donde importa, porque es el que menos discrimina contenido.

**Recomendacion:** enmendar el prereg antes de correrlo, citando este informe, y agregarle el brazo
de ruido como control interno. Es media hora de escritura y evita gastar la campaña sobre una
premisa que ya sabemos falsa.

## Limites, declarados

- Mide **acceso con el modelo congelado**. Un RECUP que cae aca **no prueba** que un modelo entrenado
  con archivos largos no pueda aprenderlo; prueba que **no sale gratis**. Esa es exactamente la
  campaña que ahora tiene sentido gastar en T4.
- Idioma cerrado de 242 tokens. En texto real la interferencia sera **mayor**, no menor, porque el
  vocabulario comparte mucho mas.
- La interpretacion del Resultado 3 («nunca aprendio ese uso») es una lectura, no una medicion.
