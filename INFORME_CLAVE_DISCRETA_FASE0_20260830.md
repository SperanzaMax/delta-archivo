# INFORME · Fase 0 de `PREREG_CLAVE_DISCRETA.md` (SHA `3c89348b`)

**2026-08-30.** CPU, checkpoints `v3_s*` (`lat2`) ya en disco, **cero GPU**. n=2000 por unidad, semilla
de datos 54321 pareada entre condiciones. Product quantization con m=8 tramos.

---

## 1. Veredicto

| | criterio | resultado |
|---|---|---|
| **Q-0** bloqueante | RECUP cuantizada ≥ 0,90 × continua, 2/3 | **CUMPLE 3/3** con k≥16 |
| **Q-1** principal | AUC ≥ 0,70 en 2/3 | **NO CUMPLE 0/3**, máximo 0,6402 |
| **Q-2** nulo | 0,45-0,55 | se comporta con k≤16; **se dispara con k=256** |
| **Q-3** especificidad | ent > rel | **no interpretable** (ver §4) |

**Por el §7 del pre-registro, la clave discreta se cierra y no se prueba una segunda forma de
cuantización.**

## 2. Q-0 · la memoria tolera el código discreto, y mejor de lo previsto

RECUP con la clave reemplazada por su versión cuantizada **dentro del forward**:

| k | bits/clave | `v3_s0` | `v3_s1` | `v3_s2` |
|---:|---:|---:|---:|---:|
| continua | 4096 | 1,0000 | 1,0000 | 1,0000 |
| 4 | 16 | 0,8703 | 0,8863 | 0,8492 |
| 16 | 32 | 0,9781 | 0,9975 | 0,9553 |
| 64 | 48 | 0,9983 | 1,0000 | 0,9983 |
| **256** | **64** | **1,0000** | **1,0000** | **1,0000** |

> **★ HALLAZGO LATERAL, no buscado y positivo: la clave de 128 floats (4096 bits) se reemplaza por
> 64 bits de símbolos sin perder NI UN PUNTO de recuperación, en las tres semillas.** Compresión de
> **64×** con RECUP 1,0000 exacto. No es lo que el experimento iba a medir y es lo único que
> sobrevive de él.

**k=4 falla Q-0** (0,849-0,886). Queda excluida del juicio de Q-1, y se declara: es justo la celda
con más resolución en el estadístico, así que hay tensión entre las dos condiciones. **La celda que
satisface las dos a la vez es k=16** (Q-0 0,955-0,998 y resolución 6-7 valores).

## 3. Q-1 · la ausencia no aparece, y el negativo es limpio porque Q-0 pasa

AUC del estadístico «¿coincide algún símbolo?» separando con-respuesta de sin-respuesta. Referencia
medida del continuo (16-ago): **0,4984 / 0,5022 = el azar exacto**.

| k | res | Q-0 | `v3_s0` | `v3_s1` | `v3_s2` | nulo Q-2 |
|---:|---:|---|---:|---:|---:|---|
| 4 | 7-9 | ✗ | 0,5101 | 0,5362 | 0,5027 | 0,480 / 0,478 / 0,512 |
| **16** | 6-7 | ✓ | **0,5138** | **0,5884** | **0,5060** | 0,531 / 0,474 / 0,500 |
| 64 | 4-5 | ✓ | 0,6227 | 0,6402 | 0,4902 | 0,406 / 0,450 / 0,551 |
| 256 | 3-4 | ✓ | 0,2998 | 0,5904 | 0,2720 | **0,726 / 0,338 / 0,707** |

**Ninguna celda alcanza 0,70.** El máximo es 0,6402 en una sola semilla, con las otras dos en 0,62 y
0,49.

**Lo que hace fuerte al negativo es que Q-0 pasa.** No es «no se pudo probar»: la memoria queda
intacta —RECUP 1,0000 con k=256— y **aun así la ausencia no tiene firma**. Si Q-0 hubiera fallado,
esto sería el defecto del exploratorio del 22-ago (preguntarle a un modelo roto si sabe cuándo no
sabe) y no se podría concluir nada.

**Y el nulo con k=256 es el mejor control del experimento: SEPARA MÁS QUE EL TRATAMIENTO** (0,7260 y
0,7066 contra 0,2998 y 0,2720). O sea que en esa configuración el estadístico mide algo estructural
—escala, tamaño del episodio— y no coincidencia semántica. Sin ese nulo, el 0,5904 de `v3_s1` se
podría haber leído como señal incipiente. Es la lección del 22-ago aplicada: *lo que da el veredicto
es cruzarlo con el nulo.*

## 4. Q-3 no es interpretable, y hay que decirlo

Los valores (ent 0,5422/0,5514/0,5583 contra rel 0,5449/0,5150/0,4901 en k=16) ordenan a favor en 2
de 3, **pero los seis están alrededor de 0,50**. Ordenar la diferencia entre dos cantidades que son
ambas el azar no dice nada sobre el mecanismo. Q-3 presuponía que Q-1 hubiera encontrado separación;
sin eso, no tiene contenido. **Cuarta vez en el programa que una predicción no se evalúa porque su
régimen no ocurrió** (P-2 de E-I4/b/c, RT-5, el exploratorio de abstención QC).

## 5. Qué le hace esto al cierre del 21-ago

El 21-ago se cerró la detección desde una señal interna con cuatro vías (logit · densidad ·
desacuerdo · empate de clave), **todas sobre representación continua**. La objeción viva era que el
softmax obliga a leer algo y por eso «ninguna coincidencia» no existía por construcción.

> **Esa objeción queda respondida y en contra: con símbolos el evento existe, es observable, y sigue
> sin separar.** El cierre se amplía — la ausencia no vive en la interfaz de memoria por **ninguna**
> representación probada, continua o discreta.

Y refuerza la lectura del 27: si la información de la ausencia no está en la interfaz, la detección
tiene que salir del cómputo aguas abajo, que es donde la cabeza la encuentra. **Sigue siendo
calibración.**

## 6. Errores propios de la jornada, los tres cazados antes de reportar

1. **La v1 del chequeo del escalar era un CONTROL VACÍO** — le pasaba a la consulta el número de tema
   exacto, así que recuperar no requería buscar y dio 1,000 en todo. Mismo defecto que el control
   `m=1` del 12-ago.
2. **La v2 imprimió una conclusión que sus propios números desmentían** (dijo que otra entidad estaba
   más cerca que otra versión). Corregida con el argumento de escalas, que además es más fuerte.
3. **Q-0 se midió primero con un PROXY** (coseno de reconstrucción) en vez del acierto que el prereg
   pedía. Al medirlo bien, **el veredicto de Q-0 cambia**: por coseno k=4 daba 0,62-0,68 y parecía
   marginal; por acierto da 0,85-0,89, que es mucho mejor de lo que el coseno sugería, y k=256 pasa de
   «0,99 de coseno» a **1,0000 de acierto exacto**. El proxy subestimaba.
4. Y una de instrumento: `tipo` es un array de enteros y se comparaba contra strings, así que las dos
   celdas de Q-3 daban NaN.

**Desviación declarada:** la columna de **resolución** (valores distintos del estadístico) no estaba
en el pre-registro. Se agregó al detectar que con k=256 el estadístico toma 2-4 valores y el AUC pasa
a ser ruido de empates — el instrumento vacío del monitor v1 del 20-ago. **No se usó para elegir la
celda ganadora** (ninguna gana); se usa para declarar qué filas no se leen.

## 7. Lo que NO dice

- **No dice que un código discreto no sirva para memoria.** Dice lo contrario: Q-0 muestra que sirve
  perfectamente, y con 64× de compresión. Lo que no hace es crear la firma de ausencia.
- **No prueba el régimen co-entrenado.** Esto es cuantización POST-HOC sobre claves aprendidas en
  continuo. Un código aprendido desde cero podría organizarse distinto — pero el §7 del prereg cierra
  la vía igual, y reabrirla necesita pre-registro propio y una razón nueva.
- **Escala:** 863.859 params, 40 entradas de archivo, idioma de 242 tokens.
