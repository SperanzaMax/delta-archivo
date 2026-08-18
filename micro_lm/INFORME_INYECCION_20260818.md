# La abstención SÍ consulta el archivo — y el modelo indexa por entidad, no por (relación, entidad)

**2026-08-18** · preregistrado en `../PREREG_INYECCION.md` (SHA `c45be719…`, congelado 11:17 UTC) con
**enmienda E-1** (SHA `5d7a36e7…`, 11:21 UTC, declarada antes de correr la condición nueva).
Script `sonda_inyeccion.py`. **CPU sobre checkpoints ya entrenados: cero GPU.**

## Qué había que descartar

El 17-ago la campaña `x` dio 4 de 4 pasan / 5 de 5 fallan la compuerta de abstención. La salvedad
escrita ese día era que faltaba el gemelo de permutación de etiquetas.

**Ese control se descartó por análisis, sin correrlo** (§6 del prereg): con la marginal de `NOSE`
conservada, la etiqueta miente de forma simétrica y minoritaria dentro de cada condición, así que el
argmax recupera la señal verdadera por marginalización y **el gemelo pasaría la compuerta sin que eso
signifique nada**. Además la hipótesis que iba a matar —«dispara `NOSE` por frecuencia»— ya estaba
descartada por los datos: `x2_s0` tiene `nose` 0,8635 con `falsa_abst` **0,0000 exacto**.

La alternativa que sí quedaba viva era otra: que el modelo detectara la ausencia por una **firma
marginal de la consulta**, sin consultar el archivo. El generador la hacía plausible — en una consulta
`NOSE` la relación se sortea uniforme y la entidad entre las no dichas (`idioma.py:205-222`), mientras
que en una pregunta con respuesta ambas salen del generador de hechos.

## El diseño: contraste pareado por disponibilidad del hecho

Misma consulta, mismo episodio, palabra por palabra. Lo único que cambia es si el hecho preguntado
está en el archivo:

- **A** — el episodio tal cual (respuesta correcta = `NOSE`).
- **B** — el episodio **más** un enunciado que dice el hecho preguntado.
- **C** — el hecho que esa entidad ya tenía, **reemplazado** por el preguntado (enmienda E-1).

## Resultados

Tasa de `NOSE` (n = 2000 por checkpoint, truncamiento 0,0000 en las tres condiciones):

| ckpt | tipo | A (ausente) | B (agregado) | C (reemplazado) | acierto en C |
|---|---|---:|---:|---:|---:|
| x1_s0 | `nose_ent` | 0,9592 | **0,0735** | 0,0735 | 0,9265 |
| x1_s0 | `nose_rel` | 0,8804 | 0,7186 | **0,0059** | **0,9941** |
| x2_s0 | `nose_ent` | 0,9611 | **0,0897** | 0,0897 | 0,9103 |
| x2_s0 | `nose_rel` | 0,8646 | 0,8295 | **0,0100** | **0,9900** |
| x2_s2 | `nose_ent` | 0,9641 | **0,0887** | 0,0887 | 0,9113 |
| x2_s2 | `nose_rel` | 0,9729 | 0,8495 | **0,0221** | **0,9779** |

**Contraste exploratorio, sin predicción asociada** (checkpoint que **falla** la compuerta):

| ckpt | tipo | A | B | C | acierto en C |
|---|---|---:|---:|---:|---:|
| x4_s0 | `nose_ent` | 0,6955 | 0,2537 | 0,2537 | 0,5701 |
| x4_s0 | `nose_rel` | 0,7709 | 0,2744 | **0,1648** | 0,7196 |

El que falla la compuerta **también consulta el archivo** (P-4 cumple ahí igual), pero con mucha menos
resolución: conserva 0,16-0,25 de `NOSE` residual con el hecho delante y recupera el valor sólo el
57-72 % de las veces. Ese techo no es de la abstención sino de la recuperación — coincide con el
`vigente` de su modelo base (n4_s0 = 0,7578). Y la interferencia de identidad prácticamente no aparece
en nivel 4 (0,0028): es un fenómeno del régimen fácil, donde la entidad es la única clave que hace
falta.

**P-3 (control de sanidad que podía fallar) CUMPLE en los cuatro** (x4_s0: 0,7330 vs 0,7106)**:** la tasa de `NOSE` en A reproduce el
`nose` reportado el 17-ago dentro de ±0,10 (x1_s0 0,919 vs 0,8844 · x2_s0 0,9130 vs 0,8635 · x2_s2
0,9685 vs 0,9762 · x4_s0 0,7330 vs 0,7106). El instrumento mide lo mismo que la campaña.

**P-4 cumple y P-5 falla en los tres.** La conclusión principal:

> Con el hecho disponible, la abstención se derrumba de ~0,90 a **0,006-0,022** y el modelo devuelve
> el valor recién puesto el **98-99 %** de las veces. La abstención consulta el archivo. La hipótesis
> de la firma marginal queda descartada por evidencia directa, no por descarte.

**El agregado (`todo`) NO es interpretable** y así estaba declarado en E-1 antes de mirarlo: promedia
dos regímenes de signo opuesto. P-1 «falla» ahí sólo porque B está confundida en `nose_rel`.

## El hallazgo que no se buscaba: la clave del archivo es la entidad

La condición B saca a `nose_rel` de distribución, porque `idioma.py:161` sortea las entidades con
`replace=False` y **en todo el entrenamiento cada entidad apareció con exactamente una relación**.
Inyectar el hecho le deja dos, y ahí el modelo se rompe: sigue diciendo `NOSE` el 72-85 % de las veces.

El diagnóstico exploratorio dice por qué, y es una segunda vía independiente: **de las respuestas que
no son `NOSE` en esa condición, el 22-51 % es el valor que la entidad tenía bajo su OTRA relación**
(x1_s0 0,5087 · x2_s0 0,2294 · x2_s2 0,2200), contra **0,6-1,2 %** en la condición C. No es ruido: es
**interferencia de identidad**. El modelo recupera por entidad y no separa qué se dijo *de* ella.

Esto no invalida nada de lo medido —C es la condición en distribución y es la que responde la pregunta
del prereg— pero **es una limitación de la clave del archivo con consecuencia directa sobre el
objetivo**: una persona real dice muchas cosas distintas sobre la misma entidad, y ése es justamente
el régimen en el que este modelo confunde. Encaja con lo del 16-ago: «el archivo es un banco de
evidencia parcialmente ordenado, no un índice que devuelve un registro».

## Estado del resultado del 17-ago

**Se sostiene, y ahora con el control que le faltaba.** El «4 de 4 pasan» no es reconocimiento de
distribución: es consulta a memoria. Queda en pie lo que el informe de ayer ya declaraba abierto —la
frontera del margen sin muestrear, y que dentro del grupo que falla el margen no ordena.

## Método

Novena vez que un número se lee mal antes de mirarle el control: el 0,719 de `nose_rel` en B parecía
un negativo rotundo y era enteramente un artefacto de sacar la muestra de distribución. Lo que lo
evitó fue verificar el generador (`replace=False`) **antes** de escribir el veredicto, y no después.
