# La ventana no era una anécdota del generador: es una ley, y da 0,000000 exacto

**2026-09-02.** Evalúa `PREREG_LEY_VENTANA.md` (SHA `eb5e1d50`), congelado antes de correr.

## 1. El resultado principal, en una línea

**La sensibilidad de la búsqueda a un token de la consulta es CERO EXACTO en cuanto ese token pasa el
alcance de la convolución que forma la query, y el corte se mueve cuando se mueve el kernel:
60 celdas de 60.**

No es «muy chico»: es `0.000000`, en dos arquitecturas, tres semillas cada una, dos componentes de la
consulta y cinco distancias.

## 2. La tabla, y lo que hay que mirar es dónde está el escalón

La posición de lectura se corre `r` lugares sobre el relleno que ya sigue al «?». Correrla `r` lugares
es exactamente agregar `r` rellenos y **no cambia un solo token de la pregunta**, así que
`d(entidad) = 1+r` y `d(relación) = 3+r`.

| unidad | kernel | alcance | d=1 | d=2 | d=3 | d=4 | d=5 | d=6 | d=7 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `v3_s0` | 3 | 2 | 0,7413 | 0,1545 | **0,000000** | **0,000000** | **0,000000** | **0,000000** | **0,000000** |
| `v3_s1` | 3 | 2 | 0,7061 | 0,2543 | **0,000000** | **0,000000** | **0,000000** | **0,000000** | **0,000000** |
| `v3_s2` | 3 | 2 | 0,7436 | 0,1662 | **0,000000** | **0,000000** | **0,000000** | **0,000000** | **0,000000** |
| `kq3_s0` | 5 | 4 | 0,4587 | 0,0997 | 0,2719 · 0,0848 | 0,0930 · 0,0672 | **0,000000** | **0,000000** | **0,000000** |
| `kq3_s1` | 5 | 4 | 0,3594 | 0,1476 | 0,3480 · 0,0824 | 0,3265 · 0,0763 | **0,000000** | **0,000000** | **0,000000** |
| `kq3_s2` | 5 | 4 | 0,4267 | 0,1371 | 0,1755 · 0,0547 | 0,1150 · 0,0844 | **0,000000** | **0,000000** | **0,000000** |

Donde hay dos números, son la relación y la entidad medidas a esa misma distancia por caminos
distintos, y **coinciden en el veredicto**: las dos > 0 con kernel 5, las dos en cero con kernel 3.

**El mecanismo explica por qué el cero puede ser exacto y no aproximado.** La lectura de `lat2`
ocurre en el bloque 0 **antes** del mixer (`modelo.py:222`), así que `h` todavía es la embedding del
token, sin ninguna mezcla recurrente. La ventana de `convq` es literalmente todo lo que la query
puede ver, y un token afuera entra multiplicado por un peso que no existe.

## 3. Un artefacto encontrado y corregido, porque cambiaba el veredicto

La primera pasada dio **59 de 60**: la celda `kq3_s2`, entidad a distancia 5, daba **0,0106** contra
un umbral de 0,01. No era señal. La consulta «cual era antes … ?» es **dos tokens más larga**, así que
con `r` grande la posición de lectura se pasaba del tensor y quedaba **recortada** — y recortar
**acerca** el token a la ventana. Esas muestras no medían la distancia que decía la columna. Se
descartan (el 12,5 % en `r`=4, todas «anterior») en vez de recortarse, y la celda pasó a
**0,000000 exacto**. El log de la pasada con el artefacto queda en
`ley_ventana_0902_con_recorte.log` en vez de borrarse.

## 4. La ablación de taps: A-3 cumple, A-2 cumple en la métrica correcta, y **A-1 estaba mal escrito**

Se apagó un tap por vez de `convq` en los `kq3` ya entrenados, sin entrenar nada.

| | criterio | resultado | |
|---|---|---|---|
| **A-3** CONTROL + | `tap1` (entidad) baja `vigente` ≥ 0,20 | 0,9065 · 0,6504 · 0,9978 | **CUMPLE 3/3** |
| **A-2** ESPECIFICIDAD | la caída del `tap3` supera la del `tap2` y la del `tap4` | **3/3 en `vigente`**, 1/3 en `nose_rel` | ver abajo |
| **A-1** PRINCIPAL | `tap3` baja `nose_rel` ≥ 0,20 | 0,0188 · 0,0093 · 0,0390 | **NO CUMPLE 0/3** |

**A-1 no cumple porque el criterio estaba mal planteado, y eso hay que decirlo antes que cualquier
otra cosa.** `nose_rel` premia **abstenerse**. Al apagar el tap de la relación, el modelo pierde la
relación en **todas** las consultas y se abstiene **más**, así que esa métrica no puede bajar: sube o
se queda. El daño aparece donde el prereg no lo fue a buscar, en `vigente` (0,997 → 0,515 · 0,998 →
0,416 · 1,000 → 0,470) y en `falsa_abst`. El prereg supuso que apagar el tap devolvería el
comportamiento del kernel 3, y no: el kernel 3 es un modelo **entrenado** sin ese tap, que aprendió a
confiar en lo que ve; el kernel 5 con el tap apagado es un modelo que aprendió a usar la relación y
de golpe no la tiene.

**Es el quinto veredicto automático de este proyecto que no se puede leer como está escrito**, y los
cinco tienen la misma forma: el criterio se escribió sobre la métrica del resultado anterior en vez
de sobre la que mide la intervención nueva.

**A-2 quedó ambiguo en el prereg**, que no dijo en qué métrica se medía, y encadenarla a A-1 la deja
ciega por la misma razón. Se informan las dos: **3 de 3 en `vigente`** (relación 0,483 · 0,582 ·
0,530 contra «de» 0,372 · 0,013 · 0,195 y artículo 0,218 · 0,093 · 0,157) y 1 de 3 en `nose_rel`.

**Y hay un confound de norma que se midió en vez de suponerse.** Los taps aprendidos valen, en el
bloque 0: 0,29-0,39 el de la posición, 0,053-0,075 el de la entidad, 0,073-0,081 «de», **0,032-0,045
la relación** y 0,049-0,053 el artículo. El tap de la relación es **el más chico de los cinco** y aun
así hace el segundo daño más grande: por unidad de peso duele **2,1×** más que «de» y **2,6×** más que
el artículo. La especificidad sobrevive a la normalización, y la normalización **no** estaba en el
prereg: es análisis posterior y se declara como tal.

**Dato lateral que el control gratis regala:** `convq` existe en los cuatro bloques y sólo el bloque 0
recibe gradiente. Los taps 1..K−1 de los bloques 1-3 valen **0,000000 exacto** — arrancan en cero y el
weight decay los deja ahí. Entonces **todo** lo que hay en el bloque 0 es gradiente puro, y el
secundario que el prereg del kernel 5 dejó abierto queda contestado: **el modelo sí usó la ventana que
se le dio.**

## 4 bis. Un control que se agregó DESPUÉS de ver el dato, y da vuelta la lectura tentadora

Al ver la tabla, la lectura que salta es atractiva: *«cuando le sacás a la query el acceso a la
relación, el modelo se abstiene en vez de inventar»*. Y es cierto en números — entre el **81 %** y el
**89 %** de lo que el kernel 5 deja de acertar termina en `NOSE` y no en una respuesta equivocada.

Sonaba a resultado y hay que decir que **no lo es**, porque el control lo refuta. Se corrió la misma
ablación sobre el modelo de **kernel 3**, que nunca tuvo el tap de la relación:

| | fracción de la caída de `vigente` que va a abstención |
|---|---|
| kernel 5, tap de la relación | 0,81 · 0,89 · 0,85 |
| **kernel 3, tap de la entidad** | **1,000 · 1,000 · 1,001** |

El kernel 3 con la entidad ablada se calla el **100 %** de las veces: `vigente` cae a **0,0000** y
`falsa_abst` sube a **1,0000** en las tres semillas. Convertir una query degradada en silencio **no es
propiedad del kernel 5**, es propiedad de tener una cabeza de abstención entrenada, y el kernel 3 lo
hace incluso más.

El control se agregó **después** de ver el resultado y por eso se declara así, no como parte del
prereg. Es la regla de buscar la explicación alternativa antes de dar el veredicto, y esta vez la
explicación alternativa ganó.

## 5. Kernel 7

Corriendo. Tres semillas desde cero, 26000 pasos.

## 6. Y esto rompe un trade-off que el proyecto había declarado imposible

`INFORME_QUERY_CONJUNTA_20260822.md` cerró con una frase fuerte, y hasta hoy no se había vuelto a
mirar:

> *«Una query conjunta necesita contexto ya computado, y la lectura útil necesita entrar antes de que
> el cómputo ocurra. **En esta arquitectura las dos cosas son incompatibles por construcción.**»*

Era una conclusión honesta sobre las dos opciones que se habían probado. `pre` inyecta la lectura
antes del mixer y la query queda siendo un token suelto; `post` le da a la query todo el contexto y
paga la inyección tardía con el acierto cayendo de 0,97 a 0,39. Entre esas dos, la incompatibilidad
es real.

**`convq` es la tercera vía, y no estaba vista.** Una convolución local sobre la query da contexto
**sin mover el punto de inyección**: la lectura sigue entrando antes del mixer —donde el 22-ago se
midió que tiene que entrar— y la query igual ve la pregunta entera. Lo único que había que acertar
era **cuánto** contexto, y ahí estaba el defecto: con kernel 3 la ventana llegaba a dos tokens y la
relación estaba a tres.

Dicho de otro modo, la incompatibilidad del 22-ago valía para el eje que se estaba mirando —**dónde**
entra la lectura— y no para el que decidía —**cuánto ve la query en ese punto**. Es el mismo
movimiento que hace la short conv en Mamba y en Gated DeltaNet, con la diferencia de que ahí el
kernel 4 es un default heredado y nadie midió qué se pierde cuando no alcanza.

## 7. Lo que esto NO dice

- **Mide ACCESO, no uso.** Que un token entre en la ventana no obliga al modelo a usarlo: el propio
  kernel 5 tiene el artículo adentro y su tap es de los que menos duelen.
- **El relleno es OOD.** El modelo nunca vio consultas con relleno al final, y por eso §2 se lee como
  medición mecanicista y no como desempeño.
- **La distancia sigue siendo fija en el idioma.** Que el fallo lo decida **dónde está escrito** un
  componente y no **qué componente es** lo decide `PREREG_CRUCE_FORMAS.md`, que es el experimento del
  cruce y todavía no corrió.
