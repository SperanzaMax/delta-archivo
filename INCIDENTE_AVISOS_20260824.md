# INCIDENTE · la inundación de avisos del 24-ago, y la causa raíz de los freezes del 19

La mañana del 24-ago Maxi recibió **cientos de mensajes de Telegram** en el teléfono, en cascada y
sin parar. Todo se detuvo y se ordenó. Este documento deja qué pasó, por qué, y qué quedó arreglado.

## 1. No era una causa, eran tres a la vez

Y las tres se activaron juntas por el mismo motivo: **el pool de Colab abrió de golpe** después de
cuatro días seco, con **ocho rotadores** corriendo al mismo tiempo. Cada cosa que estaba dimensionada
para una campaña tranquila se multiplicó por ocho.

| | qué mandaba | por qué se disparó |
|---|---|---|
| `rotar_abst2.sh` | un aviso por cada cuenta que otorgaba T4, y otro por cada tramo cerrado | 8 rotadores × 13 cuentas × vueltas |
| `~/.local/bin/import` | un aviso por cada interceptación | **116 en una mañana** |
| `watchdog_tramo2.sh` | un aviso por cada tramo matado | mataba tramos **sanos** cada ~12 min |

## 2. La causa raíz, que llevaba cinco días sin aparecer

El shim `~/.local/bin/import` existe desde el 19-ago porque algo del pipeline ejecutaba el `import`
de **ImageMagick** —que hace un pointer grab de X y congela el escritorio hasta que alguien hace
clic—. Aquel día costó **3h47 y 69 min** de máquina inutilizable. El shim lo interceptaba y dejaba el
árbol de padres registrado, pero el culpable nunca se había leído.

Hoy el registro lo dio, y era esto, en `tramo_abst.sh` línea 110 (con copias idénticas en
`tramo_colab.sh` y `tramo_frontera.sh`):

```bash
cat > "$TMP/lanzar.py" <<PY          # <- SIN comillas: el heredoc se expande
...
# (2026-08-15). En TPU, el proceso que hace `import jax` se queda con el chip TOMADO, y este
                                     #        ^^^^^^^^^^^^ backticks
```

**Dentro de un heredoc sin comillas no existen los comentarios**: bash expande todo el cuerpo, y unos
backticks son sustitución de comandos. Así que esa línea, que a la vista es un comentario de Python
explicando un bug de TPU, **ejecutaba `import jax`** en cada tramo. No corría Python: corría
ImageMagick.

El delimitador no se puede citar (`<<'PY'`) porque el cuerpo necesita expandir `$NIVEL`, `$SEM`,
`$PASOS` y demás. El arreglo es quitar los backticks, y quedó hecho en los tres scripts.

> Nota de proceso: al escribir el comentario que explicaba el arreglo **se reintrodujo el mismo bug**
> —el texto nuevo tenía backticks y estaba dentro del mismo heredoc—. Lo tapó una verificación con
> `awk` mal escapada que dio un falso "ninguno". Se rehízo el chequeo en Python y ahí apareció. Vale
> registrarlo: la verificación que da verde por estar rota es peor que no verificar.

## 3. El watchdog mataba tramos sanos

`watchdog_tramo2.sh` tenía `LIMITE=720` (12 min) sobre el mtime del log del rotador. Pero el tramo
corre por `colab exec`, que **no devuelve salida hasta terminar**: entre la línea `== tramo ...` y el
resultado pasan 40-70 minutos en los que el log no crece aunque la GPU esté entrenando a pleno.

Medido hoy: `v3_s1` arrancó en la cuenta H a las 08:18 y a las 08:29 ya estaba pidiendo otra cuenta.
Los mensajes decían *"el tramo estuvo 4063s sin escribir"*, que era cierto y no era un cuelgue.

Esto no era sólo ruido: **destruía trabajo de GPU** justo el día que el pool abrió.

`LIMITE` pasa a **5400 s** (90 min), por encima de un tramo completo de 8000 pasos y muy por debajo
del episodio de 3h47 que motivó el watchdog, así que sigue cubriendo el caso para el que se escribió.

## 4. Qué quedó arreglado

- **`tramo_abst.sh`, `tramo_colab.sh`, `tramo_frontera.sh`** — sin backticks dentro del heredoc.
  Verificado funcionalmente: se expande el heredoc y el contador de interceptaciones **no se mueve**
  (115 → 115), el `lanzar.py` generado es Python válido y las variables se expanden bien.
- **`watchdog_tramo2.sh`** — `LIMITE` 720 → 5400, con la medición que lo justifica en el comentario.
- **`rotar_abst2.sh`** — se sacan los avisos por asignación de GPU y por tramo cerrado. Quedan sólo
  dos, y los dos son eventos únicos: unidad completa, y vueltas agotadas. Lo demás sigue en el log.
- **`~/.local/bin/import`** — el registro se sigue escribiendo siempre (es el instrumento), pero el
  Telegram sale como mucho **una vez cada 30 minutos**, con la cuenta total en el propio mensaje.
  Verificado: el registro creció de 115 a 116 sin enviar nada.

**Presupuesto de avisos que queda**, en régimen normal: el latido del vigía cada 30 min, más un aviso
por unidad que llega a su meta. Del orden de 2 a 4 por hora en vez de cientos.

## 5. Lo que costó, dicho sin adornos

Se detuvo todo: 8 rotadores, el vigía, los watchdogs, los tramos y las 8 VMs de Colab que estaban
tomadas. **Se perdió lo que cada tramo llevaba entrenado desde su último checkpoint bajado**, porque
en este esquema el checkpoint viaja a la PC recién al cerrar el tramo.

Lo que quedó guardado en disco, que es desde donde se reanuda:

| unidad | paso |
|---|---|
| `ed3_s0` · `ed3_s1` · `ed3_s2` | 15500 · 14750 · 7750 |
| `ef3_s1` · `ef3_s2` | 7750 · 15750 |
| `v3_s0` · `v3_s1` · `v3_s2` (`lat2`) | 4250 · 4250 · 3250 |

Ninguna unidad quedó corrupta y ningún lock quedó tomado. **Y el intento tuvo un resultado propio:
`lat2` alcanzó a entrenar hasta el paso 4250 en las tres semillas, y ya se veía aprendiendo**
(`vigente` 0,87-0,92 al paso ~4000).

Una decisión que conviene dejar escrita: **antes de relanzar los ocho rotadores a la vez hay que
pensar el volumen de avisos**, porque el modo de falla no aparece con una campaña y aparece con ocho.
