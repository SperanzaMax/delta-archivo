# ENMIENDA · `PREREG_FILTRADO_PREVIO` (SHA `3b7032b0`) · 2026-09-05

**Se escribe ANTES de correr la campaña y DESPUES de un dato que toca su motivacion.** El prereg
original queda intacto; esto se le adjunta, siguiendo el precedente de `ENMIENDA_RECOMPENSA_F.md` y
`ENMIENDA_DISTANCIA_REAL.md`.

## Que la motiva

`INFORME_DILUCION_20260905.md` (prereg `f4d91c12`), medido sobre checkpoints entrenados. RECUP con
3280 entradas en el archivo, cambiando solo de que estan hechos los competidores:

| competidores | RECUP |
|---|---:|
| ruido, sin contenido | **0,7852** |
| reales, entidades disjuntas de la preguntada | 0,4590 |
| reales, sin restriccion | **0,0117** |

## Lo que se cae, dicho con precision

**Una premisa de motivacion, la del §1, tercer parrafo:**

> «Segundo, lo que sí se rompe es la precisión. Con `N` grande el softmax reparte su masa entre más
> candidatos y la recuperación cae.»

**Medido: con candidatos sin contenido, NO cae.** El softmax aguanta 3280 competidores y la entrada
correcta sigue ganando el 78,5 % de las veces. La recuperacion cae **por el contenido de los
competidores**, no por su numero.

Queda un residuo de dilucion pura que **no es el que se suponia**: con ruido el ranking aguanta y lo
que se ensucia es el **valor leido** (masa de la ganadora 0,9855 → 0,6715), porque la lectura es un
promedio ponderado. Eso degrada la respuesta sin degradar la busqueda, y es un mecanismo distinto.

## Lo que NO se cae, y hay que decirlo igual de fuerte

**La hipotesis H sobrevive entera, y ahora tiene mecanismo propio medido.** Filtrar antes de buscar
sigue teniendo sentido: si el filtro saca competidores **con contenido**, ataca exactamente lo que
esta roto. Lo que cambia es **por que** se espera que funcione, y eso cambia como se lee el
resultado.

## Cambios a los criterios

**F-1 · sin cambios.** Sigue siendo el criterio principal.

**F-2 · cambia de significado y de redaccion.** Estaba escrito como:

> «la ventaja **crece** con `N` en al menos 3 de 4 tramos → es dilución y no un efecto de tamaño fijo»

Hoy sabemos que **crecer con `N` no distingue dilucion de interferencia**: las dos crecen con `N`.
Se reemplaza la lectura, no el umbral:

> **F-2 (enmendado).** La ventaja crece con `N` en al menos 3 de 4 tramos. **Esto ya no adjudica el
> mecanismo**, solo dice que el efecto escala. La adjudicacion pasa a **F-5**.

**F-3 · sube de categoria.** Si lo que rompe es el contenido, el «puntaje barato» del §3 —producto
interno **sin** la proyeccion aprendida— es justo el que menos discrimina contenido, y por lo tanto
el mas expuesto a dejar afuera la entrada correcta. **F-3 deja de ser una guarda y pasa a ser un
criterio que puede tumbar la campaña por si solo**, con el umbral del 5 % que ya tenia.

**F-5 · criterio nuevo, el que adjudica el mecanismo.** El barrido se corre **tambien con
competidores de ruido**, a igual `N`. Si la ventaja del filtrado es **sustancialmente menor** con
ruido que con distractores reales, la ganancia viene de sacar contenido y no de bajar `N`. Si es
igual en las dos, el mecanismo es el numero y **el resultado contradice al informe del 5-sep**, cosa
que hay que informar tal cual.

## Cambio al §5, abandono

Decia:

> «Si F-1 falla y F-3 muestra falsos negativos bajos, entonces **la dilución no era el mecanismo del
> techo**»

Esa conclusion **ya esta establecida por otra via** y no depende de esta campaña. Se reemplaza por:

> Si F-1 falla y F-3 muestra falsos negativos bajos, entonces **restringir el conjunto no alcanza
> para arreglar la interferencia**, y el problema esta en la CLAVE —que no separa lo suficiente— y
> no en cuantos compiten. Eso manda la linea a la clave y no a un segundo filtro.

## Lo que esta enmienda NO toca

- El §6 (relacion con TELAR-03) y el §7 (lo que la campaña no puede decidir) quedan como estan.
- El diseño de la §3 queda como esta, salvo el brazo de ruido que agrega **F-5**.
- **No se cambia ningun umbral.** Cambian las lecturas y se agrega un control.
