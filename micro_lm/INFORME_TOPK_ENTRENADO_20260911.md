# INFORME · el top-k entrenado, el archivo largo en frío y LA FIRMA DE LOS PESOS · 2026-09-11

Pre-registro `PREREG_TOPK_ENTRENADO.md` (`03b294f9`, enmiendas §6-§9 escritas antes de cada
corrida). Este informe se cerró a las 16:30 con la campaña `tt3`/`rq3` terminada y medida. Números en `corridas_20260911/`, `controles_20260911/`,
`ckpts/*_fotos.json`.

## 1. Lo que se preguntó

El 10-sep quedó que leer el archivo con top-2 recupera la exactitud con 3.280 entradas **sin tocar
un peso** (0,2441 → 0,9961) y que el caso `real` —colisión, misma entidad y relación en otras
conversaciones— no lo rescata ninguna lectura (0,0039 / 0,0352). La pregunta de hoy era si
**entrenar** con la lectura top-k y el archivo largo adentro produce un modelo que contesta con
3.240 entradas viejas de otras conversaciones en el archivo.

## 2. Tres resultados, en el orden en que aparecieron

### 2.1 · En frío, el archivo largo no arranca (6 de 6)

`tp3` (top-2 desde el paso 0) y `dp3` (softmax completo), desde cero, con 3.240 entradas viejas
desde el primer paso: `vigente` 0,004-0,016 en el paso 3.000 en las seis unidades, y 0,000-0,012
en archivo corto. `kq3_s0` desde cero con archivo corto estaba en 0,392 al paso 1.000 y 0,605 al
2.000. No es específico del top-k: con 3.280 candidatos al azar la señal de la entrada correcta es
1/3.280 para el softmax y casi nunca cae entre las dos elegidas para el top-2. Se paró a las 08:50.

### 2.2 · ★ La métrica interna del plan B estaba inflada: el modelo aprende la ANTIGÜEDAD DE ESCRITURA

Plan B: sembrar de `kq3_sX` y seguir con el archivo largo (pool de 3.240 entradas escritas por el
propio modelo, refrescado cada 1.000 pasos). A los 500 pasos las seis unidades daban `vigente`
0,905-1,000 en archivo largo. Demasiado, así que antes de anunciarlo se corrió el control sobre
`ts3_s0` (top-2) y `ds3_s0` (denso) del paso 500, banco `dilucion.py`, real + viejo, 256 muestras:

| el relleno escrito con | bit de pertenencia | X = 3.240 · top-2 | denso |
|---|---|---|---|
| los pesos del propio checkpoint (= despliegue, modelo congelado) | no | 0,0703 | 0,0234 |
| los pesos del propio checkpoint | sí, como en el entrenamiento | 0,0703 | 0,0234 |
| los pesos del paso 0 (= lo que vio el entrenamiento) | sí | **0,9883** | **0,9922** |

Con y sin pertenencia, idéntico al cuarto decimal. **La única variable que lleva de 0,02 a 0,99 es
con qué pesos se escribió el relleno.** El RECUP acompaña (0,035 → 0,957): la discriminación está
en las claves. El modelo aprendió que «escrito con pesos anteriores» significa «ajeno». En un
sistema desplegado el modelo está congelado y todo se escribe con los mismos pesos, así que esa
señal no existe: la métrica interna no medía nada real.

**Es general.** Cualquier memoria escrita por un modelo que sigue entrenando le ofrece al modelo
la fecha de cada entrada en la firma de los pesos con que fue escrita, y el gradiente la va a
tomar porque es más fácil que aprender el sello. Regla que sale de acá: **evaluar siempre con el
archivo reescrito con los pesos actuales**, y entrenar sin darle al modelo esa señal.

### 2.3 · El relleno fresco sí enseña, y lo que falta es escala

`rp3_s0` (6-sep: sello relativo + pertenencia, 2.000 pasos con 161 entradas viejas escritas con
los pesos actuales, con gradiente) en el mismo banco de pool fresco, PERT=1:

    archivo    40  →  0,9688
    archivo   400  →  0,8867      (kq3, sin ese entrenamiento: 0,0586)
    archivo 3.280  →  0,1016      (kq3: 0,0059)

Cuando lo viejo se escribe con los pesos actuales, el modelo aprende a descartarlo de verdad, en un
banco que no comparte nada con el entrenamiento. No generaliza de 161 a 3.240; sí de 161 a 400.

## 3. ★★★ EL RESULTADO · entrenado con top-2, la colisión deja de romperlo (16:05)

La comparación limpia (prereg §10): el brazo denso ya existía y estaba publicado, `rp3_s0-2`
(6-sep, sembrado de `kq3_sX`, sello relativo + pertenencia, 161 entradas viejas escritas con los
pesos de cada paso, con gradiente, 2.000 pasos). Se corrió el gemelo con `--topk 2` y nada más
(`tt3_s0-2`) más una réplica del denso con fotos (`rq3_s0`). Banco `dilucion.py`, real + viejo,
pool escrito con los pesos del propio checkpoint, bit de pertenencia, 256 muestras:

| lectura | unidad | 40 | 400 | **3.280** |
|---|---|---|---|---|
| top-2 | `tt3_s0` | 0,9297 | 0,9688 | **0,9297** |
| top-2 | `tt3_s1` | 0,8750 | 0,9531 | **0,9375** |
| top-2 | `tt3_s2` | 0,9766 | 0,9570 | **0,9375** |
| softmax | `rq3_s0` (réplica) | 0,9766 | 0,9375 | 0,0977 |
| softmax | `rp3_s0` | 0,9688 | 0,8867 | 0,1016 |
| softmax | `rp3_s1` | — | 0,9297 | 0,1055 |
| softmax | `rp3_s2` | 0,9961 | 0,6172 | 0,0234 |

- **H2'' cumple por ocho veces lo pedido:** con 3.280, media top-2 **0,935** contra media denso
  **0,082** (+0,853; el prereg pedía ≥ 0,10). Un modelo entrenado con 161 entradas viejas
  generaliza a 3.280 con colisiones; el mismo entrenamiento con lectura densa no.
- **H1'' cumple:** con 400, 0,95-0,97 en las tres.
- **H3'' NO cumple:** con 40, 0,927 contra 0,980 (−0,053; toleraba 0,03). El costo en archivo
  corto existe y es chico; sale sobre todo de `tt3_s1`.
- **H4'' cumple:** `rq3_s0` reproduce a `rp3_s0` (0,709 / 0,636 / 0,968 exactos al paso 250;
  0,9766 / 0,9375 / 0,0977 contra 0,9688 / 0,8867 / 0,1016 en el banco).
- **Control sin el bit de pertenencia:** top-2 0,18-0,21 con 3.280, denso 0,004. El bit hace
  falta, y es información que un sistema real tiene (qué es de la conversación en curso). Aun sin
  él, el top-2 le gana al denso por 0,20.

**Por qué ahora sí y el 10-sep no.** En inferencia el top-2 no rescataba `real` (0,035) porque el
**ranking** estaba roto: el modelo no sabía descartar lo viejo. Entrenar con las entradas viejas
adentro arregla el ranking (sello relativo + pertenencia); el top-2 arregla la **dilución del
valor**. Dos problemas, dos piezas, y con las dos la memoria puede crecer conversación tras
conversación sobre las mismas cosas.

## 3b. Lo que costó llegar: la divergencia sin gradiente

`tf3`/`df3` (400 entradas viejas frescas con `--ses-extra-sin-grad`) divergió: `cruzada_corto` 0,97
→ 0,01 entre los pasos 500 y 1.000 en seis unidades. Diagnóstico (§10 del prereg, `xa`-`xd`): con
gradiente por las viejas 1,000 (cabeza y token); sin gradiente oscila y cae. Lo viejo tiene que
pasar por el tronco con gradiente; el atajo queda medido y desaconsejado.

## 4. La película

`entrenar.py --fotos` guarda cada 25 pasos las 26 submatrices fijas de 12×12, la lectura del
archivo sobre una pregunta fija capturada desde adentro de `responder`, taps, beta, pérdida y
exactitud del lote. `armar_pelicula.py` cuantiza a int8 (1,2 MB por unidad, 320 cuadros). Visor en
el artefacto `5c43008b…`, sección 8. La película publicada (v9) es la de `tt3_s0` y `rq3_s0`, la corrida válida: 250 cuadros cada una,
sobre el diagrama de las quince capas, con corte por percentil por matriz. Empaquetada en
`peliculas/pelicula_tt3rq3_20260911.json`. La de `ts3_s0`/`ds3_s0` (con la firma) queda en
`peliculas/pelicula_ts3ds3_20260911.json`.

## 5. Pendiente

- Escalar el entrenamiento (más entradas viejas con gradiente; costo ~2,4 s/paso con 161) y medir
  hasta dónde generaliza. Entender o pagar el costo en archivo corto (−0,05).
- La tercera semilla del barrido de inferencia. Por qué K=1 < K=2.
- El escalón 3 (vocabulario abierto) con TinyStories en castellano: prereg propio.
- Operación, para no repetir: `rotar_abst3.sh` es secuencial (un rotador por unidad); matar el
  tramo en la PC no mata el `entrenar.py` en la VM (OOM al reusar); sin keep-alive las sesiones se
  desasignan; `--fotos` continúa un `ck_fotos.json` ajeno; las VMs se cayeron dos veces a la vez
  hoy, y la reanudación no es bit a bit entre GPU.
