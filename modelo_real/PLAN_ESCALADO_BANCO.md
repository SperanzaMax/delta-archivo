# Escalar el banco de TinyLlama · qué falta, exactamente

2026-09-08, 22:40. Escrito con el diagnóstico hecho y **sin correr nada**, para arrancar mañana sin
volver a diagnosticar.

## Dónde estamos

El 6-sep quedó medido que **el archivo co-entrenado funciona sobre un transformer real**.
TinyLlama-1.1B congelado, 2,11 M entrenables sobre 1,10 B (**0,192 %**), tarea cross-secuencia, con
lectura **1,0000** y los dos controles en **0,0000** exacto.

Eso ya resuelve la crítica del vocabulario cerrado. **El banco de 242 tokens dejó de ser el límite el
6 de septiembre.** Lo que impide llamarlo resultado es otra cosa, y está en una nota propia.

> el diagnóstico corre sobre 4 muestras y para publicar eso es poco

## El diagnóstico, medido sobre el script

`archivo_en_real.py`, líneas 39, 40, 91 y 154.

| pieza | hoy | por qué no alcanza |
|---|---|---|
| entidades | **8** | `Ana` a `Hugo` |
| valores | **8** | ocho ciudades |
| relación | **una sola**, «vive en» | no hay consulta compuesta, así que la ley de la ventana no se puede probar acá |
| batch | **4** | de ahí las cuatro muestras |
| **versiones** | **NO EXISTEN** | no hay corrección, o sea la tarea canónica del proyecto no está |
| **abstención** | **NO EXISTE** | no hay preguntas sin respuesta, o sea `nose` no se puede medir |

**Las dos filas en negrita son las que importan.** No es que el banco sea chico, es que **le faltan
las dos cosas que el micro-LM sí tiene y que son el objetivo del programa entero**.

## Lo que hay que agregar, en orden de valor

**1. Versiones.** Es la tarea canónica. Decir «Ana vive en Córdoba» en un forward, «no, Ana vive en
Salta» en otro, y preguntar en un tercero. Es lo que el sello de orden resuelve en el micro-LM, de
0,4570 a 0,9956, y **nunca se probó sobre un modelo real**. Sin esto, lo del 6-sep demuestra
recuperación pero no memoria versionada.

**2. Abstención.** Preguntar por una entidad que nunca se dijo. Sin casos sin respuesta no hay
`nose`, no hay `invento`, y la mitad de la vara del proyecto no se puede medir. Acá además **el
vocabulario es abierto**, así que el invento tiene casos reales por primera vez.

**3. Escala.** Subir entidades y valores de 8 a algo del orden de 60 y 100, que es lo que tiene el
micro-LM (`idioma.py`, 58 nombres y 100 números), y el batch de 4 a 32 o 64. Con 8 entidades la
tarea se resuelve por eliminación y el número no significa nada.

**4. Una segunda relación.** Con una sola no hay consulta compuesta entidad x relación, que es
justamente donde vive la ley de la ventana. Agregar «trabaja en» abre eso.

## Lo que NO hay que tocar

El montaje que ya funciona. Modelo congelado, 0,192 % entrenable, capa de escritura 11, **lectura en
la capa 2** (inyección temprana, que es lo que E-I1 y el hallazgo de que el contexto es precondición
del cómputo exigen), y los dos controles, lectura apagada y archivo barajado, que dieron cero exacto.

Ese montaje es el resultado. Lo que se escala es la tarea que corre encima.

## Costo estimado

600 pasos costaron **30 minutos en CPU sin GPU**. Con batch 32 y el banco con versiones y abstención
la cosa crece, pero sigue siendo del orden de una tarde de T4. **No hace falta pedir nada nuevo.**

## Riesgo declarado antes de correr

Con versiones y abstención sobre un modelo preentrenado puede aparecer algo que el micro-LM no tiene.
TinyLlama **ya sabe cosas del mundo**, así que ante «Ana vive en» puede responder desde su
preentrenamiento y no desde el archivo. El control que lo separa ya existe y es el de archivo
barajado, pero con vocabulario abierto conviene además **medir la respuesta con el archivo vacío**,
que hoy no está.
