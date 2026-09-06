# ENMIENDA · `PREREG_SELLO_RELATIVO.md` (SHA `ae66e464`) · 2026-09-06

Se escribe después del smoke del primer tramo y **antes** de lanzar a T4. Ningún criterio cambia.

## E-1. Fase 2 ya es corrible: el generador acepta archivos que no caben en la tabla

R-5 decía que probar `rel` con más de 64 turnos «requiere extender el generador y hoy no está
hecho». Ya está hecho: `datos.lote(..., turno_base=N)`. Verificado bit a bit — con el default, el
lote es **idéntico a HEAD** en 4 configuraciones (`ses_extra` 0 y 26 × niveles 3 y 4).

Con `turno_base=160` el archivo abarca turnos 0..166 y el **63,4 %** de las entradas cae fuera de las
64 filas. Medido sobre `lg3_s0`:

| | NaN en los logits |
|---|---:|
| `--sello abs` | **23.232 de 23.232 (100 %)** |
| `--sello rel` | **0** |

El episodio queda a distancia 1..7 de la consulta —bien dentro de la ventana— y saturan 578 de 976
entradas, que son las verdaderamente viejas. **Eso es la diferencia entre un techo del archivo y una
ventana de recencia, y ahora se puede medir.**

## E-2. Tres bugs que habrían quemado una cuenta, encontrados por correr el primer tramo de verdad

1. **`sembrar.py` no agregaba `pert` a los params.** Los checkpoints anteriores al 6-sep no lo
   tienen, y `modelo.marca_pert` devuelve 0,0 cuando falta —correcto para no romper lo viejo, y
   desastroso acá—: la campaña habría corrido con `--pert` puesto y **el bit no habría entrado
   nunca, en silencio**. El negativo habría sido del instrumento. Ahora se agrega en cero, que es el
   mismo valor con que nace en `init_params`, así que sembrar no cambia ningún número.
2. **La guarda de identidad se disparaba contra la propia siembra.** Preguntaba con
   `.get("sello", "abs")`, y como `sembrar.py` *borra* la clave para declarar la bifurcación, el
   default la resucitaba y abortaba siempre. Ahora pregunta por **presencia**: un checkpoint ya
   entrenado sí trae la clave y ahí la guarda muerde (verificado, aborta).
3. **El script medía sobre un checkpoint que no había entrenado.** Tras el `ABORTA` seguía de largo
   y escribía `_reloj.json` y `_geom.json` con los números de la siembra — exactamente la clase de
   archivo que después se lee como resultado. Ahora no mide si el log no dice `listo:`.

`sello` y `pert` entran en `BIFURCA` de `sembrar.py`, y quedan registrados en `sembrado_de`.

## E-3. Línea de partida del smoke, para que quede anotada antes de correr

`rp3_s0` sembrado y con **un** paso: archivo largo `vigente` **0,1772**, archivo corto **1,0000**.
Es coherente con la línea de base de la campaña del archivo largo (0,1820) y confirma que la siembra
no destruye lo que el modelo ya sabía.
