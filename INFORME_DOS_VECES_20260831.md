# Preguntar dos veces: cuando las respuestas difieren, el 90 % están MAL

**2026-08-31, 22:40.** Pedido de Maxi: *«hacé la misma pregunta dos veces y controlá en qué cantidad
acierta, qué cantidad de las que acierta son la misma respuesta, y las que no son iguales si son
realmente las que están mal»*. `dos_veces.py`, `n3_s0`, `n=512`, σ=0,4.

---

## 1. La respuesta, que es afirmativa

| | |
|---|---:|
| acierta (sobre las que **tienen** respuesta) | 0,7770 |
| las dos respuestas **coinciden** | 0,9043 |
| **de las que ACIERTA, coinciden** | **0,9789** |
| de las que **erra**, coinciden | 0,8529 |
| de las que **no tenían** respuesta, coinciden | 0,8357 |
| **cuando NO coinciden, están mal** | **0,8980** |
| tasa base de «mal» en todo el conjunto | 0,5371 |
| **ganancia sobre la tasa base** | **+0,3608** |
| cuando **sí** coinciden, están mal | 0,4989 |
| cobertura: qué fracción marca el desacuerdo | 0,0957 |

**Cuando el modelo sabe, es estable: el 97,9 % de sus aciertos son la misma respuesta las dos veces.
Y cuando las dos respuestas difieren, el 89,8 % de esas preguntas están mal**, contra un 53,7 % de
tasa base.

## 2. Por qué esto NO contradice el AUC 0,6054 de hace dos horas

Son dos preguntas distintas y el pedido de Maxi apuntó a la que importa.

- **El AUC** pregunta si la inestabilidad ordena **bien a todas** las preguntas. Da 0,6054: mediocre.
- **La precisión** pregunta si el grupo que el desacuerdo **señala** está podrido. Está: **90 %**.

Un detector puede tener AUC mediocre y precisión alta en el extremo, y es exactamente lo que pasa
acá: **el desacuerdo marca poco (9,6 %) y casi siempre acierta.** El AUC lo penaliza por no ordenar
el 90 % restante, que es un uso que nadie le iba a dar.

> **Es un detector de ALTA PRECISIÓN y BAJA COBERTURA.** No sirve para decidir en todas las
> preguntas; sirve para decir «en estas 1 de cada 10, no confíes». Y ése era el uso buscado.

## 3. El límite honesto, y hay que decirlo

**«Están mal» incluye a las preguntas sin respuesta**, donde cualquier valor es incorrecto por
construcción y que son el 40 % del conjunto. De ahí que la tasa base sea 0,5371 y no algo chico. La
ganancia de +0,36 es real y grande, pero **el detector no separa «me equivoqué de valor» de «no
había nada que recuperar»** — que es justamente la distinción de la que se habló a las 16:01.

**Y sigue siendo la versión post-hoc con ruido**, la más débil de las tres. Las dos fuertes (dos
queries aprendidas, y entidad contra relación) siguen sin probar.

## 4. Lo que NO dice

- Un modelo, un sigma, `n=512`. Falta réplica y falta el modelo degradado, que en la medición
  anterior resultó **50× más estable** y ahí el detector se apagaría.
- No se comparó la precisión del desacuerdo contra la precisión de la confianza de salida en el mismo
  percentil de cobertura. **Ese es el control que falta**, y es el análogo de D-4 para precisión.
