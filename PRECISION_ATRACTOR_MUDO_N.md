# PRECISIÓN al `PREREG_ATRACTOR_MUDO.md` (SHA `2be4a610`) · el n de la medición de RECUP

**2026-08-29, congelada 10:55:25.** Se escribe **antes de medir un solo checkpoint nuevo de la Fase 1**.

**Corrección del propio encabezado, y hay que dejarla escrita.** La primera versión decía «todavía no
bajó ningún parcial». **Era falsa cuando se congeló.** Mientras se escribía este archivo el archivador
copió dos trazas nuevas —`b3_s6_26000` a las 10:51:59 y `b3_s3_23000` a las 10:52:32— o sea que a las
10:55:25 ya estaban en disco.

Lo que sigue siendo cierto, y es lo que importa para que esta precisión valga, es que **no se midió
ninguna**: `traza_recup.py` no se corrió sobre ellas, y el único RECUP nuevo calculado hasta este
momento es el de `b3_s3_22000`, que es el checkpoint de anoche y ya estaba informado el 29 (0,3665).
El SHA de la versión errónea (`73f0f297`) queda en `SHA_ATRACTOR_MUDO.txt` junto con el de ésta, para
que la corrección sea auditable y no una edición silenciosa.

---

## 1. Qué pasó

El pre-registro fija **F-1 · ≥ +0,0100 de RECUP entre 22000 y 26000**, y **no fija el `n` de la
medición**. Antes de correr el instrumento sobre datos nuevos se midió cuánto ruido tiene RECUP, que
es lo que el §4 del informe de A5 (27-ago) y el §3 del de CALIBRA (28-ago) dejaron anotado **dos veces**
como lección: *un umbral de mejora absoluta necesita verificar antes cuánto margen hay*. Acá el margen
no es el techo sino el ruido, pero el defecto es el mismo.

**Medido** sobre `ckpts_traza/b3_s3_22000.pkl` con n=2000 y **seis semillas de datos** (54321, 11, 222,
3333, 44444, 555555):

| | |
|---|---:|
| RECUP por semilla | 0,3665 · 0,3642 · 0,3965 · 0,3829 · 0,3758 · 0,3937 |
| media | 0,3799 |
| **desvío** | **0,0135** |
| rango | 0,0323 |
| **desvío de una diferencia NO pareada** | **0,0191** |

> **El ruido de una diferencia no pareada es 0,0191, casi el doble del efecto que F-1 pide detectar.**
> Con n=2000 y semillas distintas, F-1 **no sería decidible**. Haría falta n ≈ 29.000.

## 2. Por qué el diseño igual sirve, y qué se corrige

**El diseño ya era pareado y eso no cambia.** `traza_recup.py` mide **todos** los checkpoints con la
**misma** semilla de datos (54321), o sea con **las mismas preguntas**. Lo único que cambia entre dos
mediciones es el modelo, así que la varianza de «qué preguntas tocaron» —que es la que se acaba de
medir en 0,0135— **se cancela en la diferencia**. El número de arriba es el ruido del diseño
equivocado, no el de éste.

Lo que **no** se cancela es que dos modelos distintos aciertan preguntas distintas. Sobre el mismo
lote, la diferencia pareada tiene desvío ≈ √(p_flip / n_hay), donde `p_flip` es la fracción de
preguntas donde los dos checkpoints difieren. Con n=2000 (≈1190 con respuesta) y un `p_flip` de 5 %,
eso da **0,0065**, todavía del orden del efecto buscado.

**La corrección, y es lo único que esta precisión decide:**

> **El juicio de F-1 se hace con `N=8000`** (≈4750 preguntas con respuesta), manteniendo la semilla
> **54321** en todos los checkpoints. Con el mismo `p_flip` de 5 % el desvío de la diferencia pareada
> baja a **0,0032**, o sea **un tercio del efecto que F-1 pide**.

**Y `p_flip` se reporta medido**, no supuesto, en cuanto haya dos checkpoints consecutivos en disco.
Si `p_flip` resultara mucho mayor que 5 %, el desvío sube con su raíz y se recalcula antes de dar el
veredicto.

## 3. Lo que esto NO es

- **No es una desviación**, porque el pre-registro no fijaba `n`. Es completar una especificación que
  faltaba, y se hace **antes** de ver el primer dato nuevo, que es la única forma en que hacerlo vale.
- **No mueve el umbral de F-1.** Sigue en +0,0100, con la monotonía en ≥3 de 4 intervalos y con F-3
  disparándose por debajo de +0,0030. Lo que cambia es **la precisión con que se mide**, no lo que se
  le pide al resultado. Bajar el umbral acá sería exactamente el rescate que el proyecto no hace.
- **No toca la Fase 2.** Cuando se llegue, G-1 pide un efecto de +0,10, que con este ruido es
  cómodamente decidible incluso sin parear.

## 4. Nota sobre los hitos

`archivar_traza.sh` copia el checkpoint cuando el rotador baja un parcial, y eso pasa cada 4 ticks de
polling de 2 minutos, o sea cada ~8 minutos ≈ **~660 pasos**, no cada 1000 exactos. Los hitos reales
van a ser los que salgan. F-1 se juzga sobre **el delta total entre el primero y el último** y la
monotonía sobre **los intervalos que haya**, no sobre cinco puntos exactos en 22000/23000/24000/25000/26000.
Se declara acá para no elegir después qué puntos mirar.
