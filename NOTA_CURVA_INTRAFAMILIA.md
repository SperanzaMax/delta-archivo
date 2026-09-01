# Curva techo-vs-RECUP DENTRO de familia · lectura congelada ANTES del dato

**2026-09-01.** La curva de 10 unidades (§4 del `PLAN_20260901.md`) ya corrió y dio
**r = +0,8643 · pendiente +0,1308**, con la lectura sugerida de que «la pendiente es floja» y por lo
tanto habría que **debilitar** la frase *«toda mejora de la abstención pasa por recuperar mejor»*.

**Esa lectura no se puede dar todavía, por dos defectos del conjunto medido:**

1. **El confound de T-4 no se resolvió, se replicó.** Las unidades de RECUP bajo son de 3000 pasos con
   recompensa y las de RECUP alto son de 12000 sin ella. Poner más puntos confundidos no desconfunde.
2. **La `r` está inflada por el diseño.** Los diez puntos forman dos nubes separadas en el eje x
   (0,36-0,38 y 0,78-0,79) más un tercero aislado. Con datos agrupados la correlación mide sobre todo
   la distancia entre nubes; **la dispersión DENTRO de la nube baja (0,5945-0,6575 = 0,063) es del
   mismo tamaño que la diferencia entre las medias de las dos nubes (≈0,060)**.

**Y un error propio, cazado antes de construir encima:** creí ver que `p3_s0` y `b3_s3`/`b3_s6` eran
la misma configuración con distinta semilla —lo que habría sido el contraste limpio— y **no lo son**:
`b3` entrena con `blanco=error` y `mezcla=fija`, `p3` con ninguna de las dos. Las claves que había
mirado no incluían las que los separan.

## Lo que se corre y por qué es la medición correcta

Las 17 unidades de `nivel 3 · p_nose 0,4 · abst=cabeza · 26000 pasos · lr 1e-3` se reparten en
**cinco familias homogéneas**, y dentro de cada una **lo único que cambia es la semilla**:

| familia | `donde` | otras | semillas |
|---|---|---|---|
| `p3` | pre | — | s0 s1 s2 |
| `q3` | post | — | s0 s1 s2 |
| `v3` | lat2 | mezcla fija | s0 s1 s2 |
| `w3` | lat | mezcla fija | s0 s1 s2 |
| `b3` | pre | **`blanco=error`** + mezcla fija | s0 s1 s2 s3 s6 |

Dentro de familia, presupuesto, pérdida, arquitectura y datos son idénticos, así que **la variación de
RECUP entre semillas es la biestabilidad ya documentada** (el atractor mudo del 29-ago, el control
biestable del 31-ago) y no un confound. Es la única forma que tenemos de mover RECUP dejando todo lo
demás quieto, y no hace falta entrenar nada nuevo.

**`b3` se analiza APARTE y no entra en la pendiente principal:** entrenar con el blanco `error` es
entrenar explícitamente algo emparentado con lo que la sonda mide, así que su techo puede estar
inflado por construcción. Declararlo después de ver el número sería elegir la muestra.

## La lectura, fijada ahora

Sea `m` la pendiente mediana de las regresiones intra-familia (p3, q3, v3, w3), cada una con sus 3
semillas, y `m_glob = 0,1308` la pendiente del conjunto confundido.

- **Si `m` ≥ 0,40** → la pendiente global era un artefacto de mezclar presupuestos y pérdidas, y la
  frase *«toda mejora pasa por recuperar mejor»* **se REFUERZA**, ahora sin el confound de T-4.
- **Si `m` ≤ 0,15** (del orden de `m_glob`) → la relación es genuinamente floja y la frase **se
  DEBILITA**: recuperar mejor compra poco, y la estrategia de la línea tiene que buscar la señal en
  otro lado.
- **Si `0,15 < m < 0,40`** → **NO EVALUABLE por diseño**: la medición no distingue las dos lecturas y
  hay que mover RECUP a propósito en vez de aprovechar la semilla.

**Riesgo declarado, y protege a los tres criterios de arriba:** si dentro de una familia el rango de
RECUP entre semillas es **menor que 0,10**, esa familia **no informa la pendiente** (se estaría
dividiendo por un número chico y la pendiente es ruido); se reporta y se excluye. Si quedan **menos
de 2 familias** informativas, el resultado entero es **NO EVALUABLE** y no se lee ninguna de las tres
opciones. Esto es la regla O-6 del 31-ago aplicada por segunda vez.

**Todo lo demás es exploratorio**, incluida cualquier lectura sobre la forma de la curva (lineal
contra umbral), que con tres puntos por familia no se puede decidir.

Costo: una corrida de `sonda_techo_curva.py` sobre 17 checkpoints en Colab, ~7 min de T4.
