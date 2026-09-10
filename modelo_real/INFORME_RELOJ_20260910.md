# La ley del reloj · lo que medía `t_techo`, y no era cuándo aprende

2026-09-10, con la campaña de la abstención corriendo y sin ningún resultado a la vista.

Ayer quedó anotado que **la ley del reloj no se puede medir con este diseño**, porque la dispersión
entre semillas del mismo punto (`t_techo` de 1.075 a 14.550) es mayor que las diferencias entre
puntos. La conclusión se sostiene. **El diagnóstico no.**

## Lo que pasó, en una fila

`ent8`, tres semillas del **mismo punto**, 20.000 pasos:

| semilla | primer cruce del techo | acierto entre 1.500 y 14.500 | `t_techo` reportado |
|---|---:|---:|---:|
| `s0` | 1.425 | 0,9986 (n = 2.084) | 1.425 |
| `s1` | **1.250** | 0,9957 (n = 2.084) | **14.550** |
| `s2` | 1.075 | 1,0000 (n = 2.084) | 1.075 |

**`s1` es la segunda más rápida en llegar al techo y la métrica la reporta como la más lenta por un
factor de doce.** Entre el paso 1.500 y el 14.500 estuvo en 0,9957 sobre 2.084 muestras. Los
13.300 pasos de diferencia los produjeron **nueve errores**.

La causa es la cláusula de permanencia. `t_techo` es el primer hito que alcanza el techo **y no
vuelve a bajar del 80 % de la meseta**, con «no vuelve a bajar» evaluado **hasta el final del
entrenamiento**. Con esa definición la métrica no marca cuándo el modelo llegó, marca **cuándo
ocurrió el último hipo**. La enmienda de ayer ya había movido la cláusula del hito suelto a la media
móvil, y eso arregló el caso de las 4 muestras sueltas; no arregla que la ventana de permanencia sea
infinita.

## La métrica arreglada

    t_primero   primer paso donde la media movil de 8 hitos cruza 0,90
                Y el acierto agregado de los 400 pasos siguientes tambien cruza 0,90

Sin cláusula hasta el final. La segunda condición es la que impide llamar techo a un pico de suerte,
y se verifica sobre cientos de muestras en vez de sobre cuatro. Comprobación de que marca el lugar
correcto: el acierto agregado **desde `t_primero` hasta el final** es ≥ 0,9859 en las siete unidades.

| punto | `t_primero` por semilla | media | sd | rango |
|---|---|---:|---:|---:|
| 8 entidades | 1.425 · 1.250 · 1.075 | 1.250 | 175 | **350** |
| 15 entidades | 1.400 · 3.000 · 3.200 | 2.533 | 987 | **1.800** |
| 30 entidades | 1.800 | (una semilla) | | |

## Qué cambia y qué no

**No cambia el veredicto.** La ley del reloj sigue sin ser medible con tres semillas por punto: la
diferencia entre 8 y 15 entidades es de 1.283 pasos y el rango dentro de `ent15` es de 1.800.

**Cambia el diagnóstico, y con él lo que habría que hacer.** El obstáculo no es que el ruido sea
enorme, porque en `ent8` la dispersión intra-punto cae de 13.475 pasos a **350**. El obstáculo es que
`ent15` tiene **dos regímenes**, una semilla en 1.400 y dos en 3.000-3.200, que es una bimodalidad y
no una dispersión continua. Eso se ataca con más semillas y estudiando la bimodalidad, no
declarando el diseño inservible.

Queda anotado además que `ent30` da 1.800, **menos** que `ent15`, con una sola semilla. Si la ley
predice que más entidades tardan más, ese punto va en contra. Con una semilla no dice nada, y por eso
mismo no puede usarse en ninguna dirección.

## La familia de error, que es la de ayer

Es el mismo error que los seis de ayer y que el mío de esta mañana en el criterio de
desestabilización: **un estadístico que depende de un extremo** (el último bache, el máximo de una
media móvil) sobre una métrica de cuatro muestras. El promedio de miles de muestras es estable; el
mínimo o el máximo de esas mismas muestras no lo es, y ninguna prolijidad de pre-registro protege
de eso.
