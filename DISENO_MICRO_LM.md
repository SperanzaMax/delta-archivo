# MICRO-LM: el modelo más chico que puede demostrar «no olvidé lo que me dijiste»

**2026-08-13.** Propuesta de diseño, a pedido de Maxi: *«qué modelo necesitamos que realmente
podamos crear desde cero y si entra en Colab; no hace falta que sepa todo lo que sabe un LLM, con que
sepa 100 números podemos hacer pruebas de memoria; hay que pensar cómo sería la pregunta para que la
entienda y cómo la respuesta para que la entendamos nosotros; que ocupe unos pocos megas»*.

La vara está en la memoria del objetivo: **un modelo entrenado desde cero, con el mecanismo en el ADN
—no adosado—, que no olvide lo que lee ni lo que se habló.** Lo que sigue es el modelo más chico que
puede intentar eso en serio.

---

## 1. La decisión de fondo: qué tiene que saber, y qué NO

No hace falta conocimiento del mundo, ni gramática libre, ni generación fluida. Hace falta **un
idioma cerrado pero legible por un humano**, porque el criterio de éxito es que nosotros podamos leer
la pregunta y la respuesta y entender si acertó.

Ese es el corte exacto: **el lenguaje es la interfaz de la prueba, no el objeto de estudio.**

## 2. El vocabulario (~320 tokens)

| clase | cuántos | ejemplos |
|---|---|---|
| números | 100 | `0` … `99` (los «100 números» de Maxi: valores de hechos) |
| nombres propios | 60 | `ana` `beto` `carla` `dario` … |
| entidades | 30 | `norte` `sur` `taller` `tienda` `equipo` … |
| relaciones | 12 | `dirige` `cuesta` `mide` `vive_en` `atiende` … |
| funcionales | ~40 | `el` `la` `de` `es` `no` `cual` `quien` `ahora` `antes` `y` `?` `,` |
| control | 6 | `BOS` `SEP` `EOS` `USUARIO` `MODELO` `NOSE` |

`NOSE` no es decorativo: es la abstención **registrable**, lo que le faltó al banco ECO cuando el
parser confundía «no contestó» con «contestó otra cosa». Sin un token de no-sé, el error silencioso
no se puede medir, y el error silencioso es el hallazgo más fuerte de toda la línea.

## 3. La pregunta y la respuesta

Diálogo en texto plano, minúscula (por el bug del encoder), una respuesta de **un solo token**:

```
sesión 1    USUARIO  el director de norte es ana
sesión 2    USUARIO  no , el director de norte es beto
sesión 7    USUARIO  quien dirige norte ?
            MODELO   beto
```

**Por qué la respuesta es un token:** hace la métrica exacta y determinista, sin juez LLM ni parser
que interprete. Es la lección directa del 12-ago, cuando 10 de 11 «abstenciones» resultaron ser el
modelo contestando bien otra cosa y el parser anotándolo como rechazo. Acá la respuesta es `beto`,
`ana` o `NOSE`, y no hay tercera lectura.

## 4. Arquitectura y tamaño

Delta rule + **archivo co-entrenado adentro** (lo del brazo interno: E-I2 mostró que el modelo
aprende a consultarlo, E-I3 que un sello de orden resuelve el conflicto de versiones).

| config | d | capas | parámetros | fp32 | fp16 |
|---|---|---|---|---|---|
| **MICRO-A** | 128 | 4 | ~0,8 M | 3,2 MB | 1,6 MB |
| **MICRO-B** (propuesta) | 192 | 6 | ~2,8 M | 11 MB | 5,6 MB |

MICRO-B entra holgado en «unos pocos megas» y deja margen para que el idioma no sea el cuello de
botella. Comparación de escala: es **1/500 de un LLM chico** (1,5 B) y ~1/50.000 de los grandes.

**¿Entra en Colab?** Sobrado. 2,8 M de parámetros con secuencias de 64-128 tokens: una T4 gratuita
hace miles de pasos por minuto; un entrenamiento de 100 k pasos es cuestión de una o dos horas, muy
por debajo del límite de sesión. Incluso entra en esta PC, unas 10× más lento — sirve para depurar
acá y entrenar en Colab.

## 5. Los cuatro niveles, en orden de dificultad

Cada uno es un experimento con su pre-registro. **El salto real está en N2**: hasta ahí es MQAR con
ropa de español.

- **N1 — plantilla fija.** Una sola forma de decir cada hecho. Es el piso y el control de sanidad:
  si esto no anda, no seguir.
- **N2 — paráfrasis.** Cuatro formas del mismo hecho: `el director de norte es ana` · `ana dirige
  norte` · `quien dirige norte es ana` · `norte tiene como director a ana`. **Acá deja de ser
  recuperación por clave literal**: el modelo tiene que mapear formas distintas al mismo contenido, y
  el archivo tiene que indexar por contenido y no por superficie.
- **N3 — corrección elíptica.** `no , es beto`, sin nombrar la entidad. Es el caso donde el índice
  plano mide **0,0000 en 10/10 semillas, en silencio** — el resultado más fuerte del preprint recién
  publicado. Que un modelo entrenado desde cero con el archivo adentro lo resuelva sería el argumento
  que hoy no tenemos.
- **N4 — multi-sesión.** K sesiones separadas con el estado reseteado entre ellas y **el archivo
  persistiendo**, preguntas al final. Es literalmente «te lo dije en la sesión 1 y te lo pregunto en
  la 7».

## 6. Métricas

Exactitud sobre **versión vigente**, sobre **versión anterior**, tasa de **abstención**, y **SER**
(error silencioso: contestar con seguridad una versión invalidada). Desagregado por tipo de error:
*versión* (confunde v2 con v3) vs *identidad* (trae el hecho de otra entidad) — la distinción que
FAMA no separa.

## 7. Lo que puede salir mal, dicho antes

- **Presupuesto.** Es la lección que el brazo interno repitió cinco veces en dos días, la última hoy:
  a 12000 pasos dos de cinco semillas de E-I3c no habían convergido, y la que se probó con el doble
  pasó de 0,55 a 0,95. **Planificar 3-5× lo que parezca necesario, y no leer un negativo sin barrido
  de presupuesto.**
- **Que el idioma cerrado sea MQAR disfrazado.** Por eso N2 no es opcional: sin paráfrasis, cambiar
  tokens por palabras en español no agrega nada y sólo hace las figuras más lindas.
- **Que la prueba se gane sola.** Todo control tiene que poder fallar. El `m=1 → 1,000` del banco ECO
  era un control vacío: con una sola candidata, acertar no requiere leer.
- **Fuga por el contenido.** Lo de hoy en E-I3b: si dos versiones caen en la misma secuencia, el
  estado de la segunda arrastra rastro de la primera y el orden se regala. En N4 las sesiones van
  separadas de verdad.

## 7.bis El atajo de la recencia: hay que empujar al modelo, no sólo dejarlo entrenar

**Medido en E-I3d el mismo día de escribir esto, y cambia el plan.** Con turnos móviles, la condición
`sello` da ANTERIOR 0,3142 con sd 0,5330 — y por semilla: **0,0052 · 0,9297 · 0,0078**. Es bimodal,
no ruidoso. Dos semillas convergen al **atajo de la recencia** («gana el turno más alto»): resuelven
la versión vigente en 0,96 y fallan la anterior *por debajo del azar*, o sea contestan
sistemáticamente la vigente cuando se les pide la anterior. Una sola aprende a **comparar** turnos.

La solución buena existe y es alcanzable; el gradiente casi siempre prefiere la barata, porque
resuelve perfecto la pregunta frecuente.

**Consecuencia para MICRO-LM:** no alcanza con poner el archivo en la arquitectura y entrenar. Hay
que sesgar la búsqueda hacia la solución que compara:
- **Curriculum**: preguntar por versiones no vigentes desde el principio del entrenamiento, no como
  evaluación al final. Si la pregunta difícil aparece tarde, el atajo ya está fijado.
- **Balance de preguntas**: si el 90 % de las consultas son «cuál rige», la recencia es óptima en el
  entrenamiento. Subir la proporción de preguntas por versiones anteriores.
- **Pérdida auxiliar sobre el orden** (a evaluar): pedirle explícitamente cuál de dos entradas es más
  vieja, para que el sello se use como coordenada comparable y no como preferencia.
- **Reportar por semilla, nunca sólo la media.** Con distribuciones bimodales el promedio no
  describe a ninguna corrida real.

## 8. Por qué esto vale la pena aunque el modelo sea de juguete

Porque cambia la clase de afirmación. Hoy se puede decir *«el mecanismo agarra»*: un archivo
co-entrenado dentro de una red de 64 dimensiones aprende a consultarse y un sello de orden le resuelve
el conflicto de versiones. Con MICRO-LM se podría decir *«este modelo, entrenado desde cero, no
olvidó lo que le dije, y supo cuál corrección regía»* — que es la frase que Maxi puso como criterio y
que ningún resultado actual sostiene.
