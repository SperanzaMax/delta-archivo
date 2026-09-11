# INFORME · el top-k entrenado, el archivo largo en frío y LA FIRMA DE LOS PESOS · 2026-09-11

Pre-registro `PREREG_TOPK_ENTRENADO.md` (`03b294f9`, enmiendas §6-§9 escritas antes de cada
corrida). Este informe se escribe a las 10:40 con la campaña `tf3`/`df3` **en vuelo** (§5 queda
abierta) y todo lo demás cerrado. Números en `corridas_20260911/`, `controles_20260911/`,
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

## 3. Lo que corre ahora (§9 del prereg)

`tf3_sX` (top-2) contra `df3_sX` (softmax), sembrados de `kq3_sX`, **`--ses-extra 36
--ses-extra-sin-grad`**: 400 entradas viejas escritas con los pesos actuales en cada paso, sin
gradiente por ellas (verificado: gradiente 0,0 y valores idénticos). 8.000 pasos, 2,4 s/paso en T4.
Las hipótesis se leen sólo sobre el banco de pool fresco: tf3 ≥ 0,90 con 400; con 3.280, tf3 le
gana al denso por ≥ 0,10; generalización ≥ 0,50.

## 4. La película

`entrenar.py --fotos` guarda cada 25 pasos las 26 submatrices fijas de 12×12, la lectura del
archivo sobre una pregunta fija capturada desde adentro de `responder`, taps, beta, pérdida y
exactitud del lote. `armar_pelicula.py` cuantiza a int8 (1,2 MB por unidad, 320 cuadros). Visor en
el artefacto `5c43008b…`, sección 8. Lo que ya se ve: el top-2 arranca con toda la masa en lo viejo
y en 100 pasos tiene 0,60 en la entrada correcta; el denso contesta bien desde el paso 175 con
0,3-0,6 de la masa todavía en lo viejo. Es la disociación entre recuperar y contestar, en directo.

## 5. Pendiente

- Cerrar `tf3`/`df3` y leer H1'-H4' (§9) en el banco de pool fresco, X = 360 y 3.240.
- Republicar la película con las 320 fotos de `tf3_s0`/`df3_s0`.
- Errores del día, para no repetirlos: los seis `sesion_fija` murieron de OOM porque los procesos
  de la corrida en frío seguían vivos en las VMs reusadas; sin keep-alive las sesiones se
  desasignan solas; `entrenar.py --fotos` continúa un `ck_fotos.json` ajeno que encuentre en la VM.
