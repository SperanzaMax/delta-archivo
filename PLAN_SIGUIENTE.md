# Para retomar · escrito el 2-sep con dos campañas corriendo

## Lo que se cerró hoy

| | |
|---|---|
| **revisión de literatura** | **el hueco está libre.** Precursor a citar de frente: CAT (2407.05591). `REVISION_LITERATURA_VENTANA_20260902.md` |
| **la ley de la ventana** | **60 celdas de 60**, cero EXACTO fuera del alcance, el escalón se mueve con el kernel |
| **el cruce, mecanicista** | la misma pregunta reordenada da vuelta la ceguera. Compuerta X-0 ABIERTA |
| **validación externa** | **Mamba-130M real: 80 de 80.** El estado ve todo, la query ve tres tokens |
| **ablación de taps** | A-3 3/3, A-2 3/3 en `vigente`; **A-1 mal escrito** y el control refutó la lectura tentadora |
| **cuarto preprint** | 5 páginas, compila limpio, citas verificadas de primera mano |
| **el aviso a Maxi** | hecho, por Telegram, con el criterio de las cuatro condiciones cumplido |

## Corriendo

- **`k73_s0/s1/s2`** · kernel 7 contra kernel 5 · `PREREG_LEY_VENTANA.md` §C · juez `juzgar_k7.py`
- **`cf3_s0/s1/s2`** · el CRUCE, kernel 3 con las dos formas · `PREREG_CRUCE_FORMAS.md` · `juzgar_cruce.py`
- **`avisar_0902.sh`** vigila las dos y manda el **juicio ya hecho** por Telegram, no un «terminó»

## 1. Lo primero cuando cierren

**Leer `nose_ent` y `nose_rel` POR FORMA, nunca el `nose` global.** Todo el cruce vive en la
desagregación: la predicción es que el orden se **invierte** entre `directa` e `invertida`, y un
promedio lo borra.

Y para el kernel 7, el control es el **kernel 5**, no el 3: la pregunta ya no es si ver la relación
ayuda, es si **más ventana ensucia**. La hipótesis en contra está escrita antes del dato.

## 2. Después, lo que le falta al preprint

- Los dos resultados de arriba.
- **El paso conductual en un modelo real**, que hoy no se pudo: que una pregunta con la parte
  discriminante lejos del final se responda peor. Necesita GPU y es lo que convertiría la medición de
  arquitectura en una de comportamiento.

## 3. La regla que salió de mirar el proyecto entero

**Un hallazgo de ARQUITECTURA vale más que uno de ENTRENAMIENTO, y hay que buscarlo primero.** Entre
el 26-ago y el 1-sep hubo doce intentos de arreglar la abstención por la vía del entrenamiento, todos
negativos o parciales; el 1-sep se hizo el primer diagnóstico mecanicista y el problema se resolvió en
un día. Los de arquitectura se verifican sin entrenar, dan ceros exactos y transfieren.

**Y el corolario incómodo: siete criterios de este proyecto no se pudieron leer como estaban
escritos**, y los siete tienen la misma forma — el criterio se escribió sobre la métrica del resultado
**anterior**, o sobre un número **supuesto**, en vez de sobre lo que mide la intervención **nueva**.
Antes de congelar un prereg conviene preguntarse: *si la intervención funciona perfecto, ¿esta métrica
se mueve?*

## 4. Lo que NO hay que hacer

- **No anunciar el cruce antes de mirar `vigente` por forma.** Si una plantilla no se aprende, el
  cruce es un artefacto.
- **No leer `nose_rel` como medida de daño en una ablación.** Premia abstenerse, así que al cegar al
  modelo sube en vez de bajar. Fue el error de A-1.
- **No afirmar la causa del tap cero de Mamba.** Está medido, no explicado, y así va en el informe.

## 5. Operativo

- Venv **`/home/maxi/.venv-ligamento/bin/python`** para el micro-LM; **`/home/maxi/.venv_datasets_pandas/bin/python`**
  es el que tiene torch y transformers, para las pruebas sobre modelos reales.
- **`FORMAS_Q` viaja por los dos scripts del pipeline** (`rotar_abst3.sh` y `tramo_abst.sh`). Sin eso
  la campaña del cruce corre como una copia del control; se cazó antes de lanzar.
- Pool con 14 cuentas. Hoy k7 salió por H y el cruce por I (TPU v5e1).
