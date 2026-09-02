# D-1 · el cruce cambia `cf3_s0` por `cf3_s3` · 2026-09-02, 09:15

`PREREG_CRUCE_FORMAS.md` (SHA `410acd25`) nombra las unidades `cf3_s0/s1/s2`. **`cf3_s0` se descarta y
se reemplaza por `cf3_s3`**, entrenada de cero con la configuración correcta. Las otras dos siguen.

## Qué pasó, con la causa exacta

La campaña se lanzó dos veces. En el **primer** lanzamiento, `FORMAS_Q` **no estaba exportada** en
`rotar_abst3.sh` ni en `tramo_abst.sh`, así que el tramo usaba su valor por omisión y `cf3_s0` corrió
sus primeros 6500 pasos con `--formas-q directa`, o sea **como una copia del control**. Lo detecté a
los 20 segundos de lanzar, corté el rotador y arreglé los dos scripts, pero el tramo remoto ya estaba
lanzado en Colab y guardó su checkpoint.

**La guarda de identidad agregada hoy hizo exactamente su trabajo** y frenó la unidad en seco al
intentar el segundo tramo, con el mensaje `ABORTA: el checkpoint se entreno con formas_q='directa' y
se pidio 'directa,invertida'`. Sin esa guarda, `cf3_s0` habría corrido 6500 pasos con una tarea y
19500 con otra, **en silencio**, y el resultado se habría leído como un dato.

## Por qué se cambia de semilla en vez de reiniciar la misma

El checkpoint contaminado sobrevivía en `/content/ck.pkl` de la sesión de Colab y el tramo lo volvía a
bajar aunque el entrenamiento abortara. Cambiar a una semilla nueva elimina la ambigüedad de un lado y
del otro, sin depender de limpiar una sesión remota.

El checkpoint viejo **no se borró**, quedó en `ckpts/contaminados/cf3_s0_formas_directa_6500.pkl`.

## Qué NO cambia

Ni los criterios, ni el número de semillas, ni la configuración. `cf3_s3` es idéntica a las otras dos
salvo la semilla. **Los criterios X-1, X-2 y X-3 siguen escritos como estaban** y se aplican a las
tres unidades que corran con `formas_q='directa,invertida'`, que es lo verificable en la config de
cada checkpoint.

## Lectura preliminar, declarada como tal

Con `cf3_s1` a mitad de camino (13000 de 26000) el cruce **todavía no aparece**. `nose_rel` supera a
`nose_ent` en las **dos** formas, cuando la predicción pedía que el orden se invirtiera. El veredicto
formal es a 26000 y hoy la campaña es **NO EVALUABLE**. Esto queda escrito antes del cierre para que
no se pueda decir después que se sabía desde el principio.
