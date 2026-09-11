# Pre-registro (BORRADOR, congelar antes de correr) · HECHOS ENCADENADOS · 2026-09-11

Sale de la pregunta 1 de Maxi (`PLAN_PRUEBAS_20260911_TARDE.md` P5): un modelo de frontera con el
archivo, congelado el 1 de enero y actualizado sólo por el archivo, ¿puede **razonar** con lo que
está en el archivo, o sólo recuperarlo? El banco de hoy responde un token de un solo hecho. Esto
mide el paso mínimo de razonamiento: **dos hechos encadenados**.

## La tarea

Hoy las relaciones personales (`director`, `duenio`, `guardia`) tienen como valor un NOMBRE, y las
otras (`precio`, `altura`, `clave`) un número, ambas sobre ENTIDADES (lugares). Se extiende el idioma
para que un nombre también pueda tener `altura` y `clave` («la altura de yamil es 48»). Entonces
existe la pregunta compuesta:

    «el director de barrio es yamil»  +  «la altura de yamil es 48»
    → «¿cuánto mide el director de barrio?»  → 48

Con las mismas piezas de siempre: versiones (`no, es …`), `NOSE` cuando falta cualquiera de los dos
hechos, archivo corto y largo. Forma nueva de pregunta `compuesta`; las simples siguen (es un
nivel nuevo, no reemplaza al banco).

## Las hipótesis

- **H1 · recuperación en dos pasos.** Entrenado desde cero con preguntas simples y compuestas,
  llega a ≥ 0,85 en las compuestas con archivo de 40 en 3 semillas. El modelo lee el archivo en cada
  token de la consulta (bloque 0), así que puede leer «director de barrio» en un token y «altura de
  yamil» en otro, si la consulta del segundo token contiene lo leído en el primero.
- **H2 · la ley de la ventana, otra vez, y es la predicción que importa.** Con la conv corta de
  kernel 5 la consulta de un token ve cuatro tokens atrás; en «¿cuánto mide el director de barrio?»
  la segunda lectura necesita «altura» (o «mide») y el nombre leído, que caen fuera del alcance en
  la forma `lejana`. **Predicción:** kernel 5 falla en la forma lejana (≤ 0,5) y la atención
  completa sin proyecciones (`--donde attn`, escalón 1) llega al techo. Es la ley de cobertura del
  8-sep aplicada a composición, y si se cumple es la evidencia más directa de que la consulta
  necesita acceso global para razonar sobre el archivo.
- **H3 · en archivo largo** (161 viejas frescas, con gradiente, top-2): las compuestas no caen más
  de 0,10 respecto de las simples.

## Diseño (a congelar)

Idioma v4 con `altura`/`clave` sobre nombres; `datos.lote` con `p_compuesta`; `--formas-q
directa,lejana`; brazos `kernel 5` contra `attn`, 3 semillas, 26.000 pasos, desde cero. Métrica
por tipo de pregunta y por forma. Detector de colapso a los 2.500 pasos como siempre.
