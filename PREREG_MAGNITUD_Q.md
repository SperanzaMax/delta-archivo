# PREREG · ¿la constante `q` es un problema de MAGNITUD? · 2026-09-04

**Congelado antes de implementar y antes de correr.** El SHA de este archivo va en
`SHA_MAGNITUD_Q.txt` y en el commit que lo agrega.

## 1. De dónde sale, y por qué esta intervención y no otra

`INFORME_RECOMPENSA_L_20260830.md` dejó el diagnóstico más preciso que tiene la línea de la
abstención. Sobre cuatro unidades de la interfaz `token`, la tasa de abstención `q` se clava en
**0,4918 · 0,4933 · 0,4960 · 0,4968** sin importar la semilla, el origen ni el valor de `L`, con
**`falsa_abst` ≈ 0,48** en las cuatro. O sea que el modelo se calla en el 48 % de las preguntas que
**sí** tienen respuesta.

> **`q` es una CONSTANTE, no una función de la pregunta.** Mudo, locuaz y medio son la misma patología
> con distinto valor, y cambiar la pérdida mueve el valor de la constante y nada más.

El mismo informe midió la causa candidata y la dejó escrita **antes** de esta campaña. Con `CE=1,0` la
recompensa entera es el **7,3 %** de la pérdida, y el logit de `NOSE` recibe **3,5× menos gradiente**
que un token de valor cualquiera. De ahí el próximo paso derivado, textual del informe, *«bajar
`--rec-ce` con el valor sacado del ratio de gradientes medido (≈3,5), con pre-registro propio»*.

Esta campaña es ese paso. No se eligió mirando resultados nuevos.

## 2. Hipótesis

**H.** El bloqueo es de **magnitud** y no de forma. Si la recompensa deja de estar aplastada por la
entropía cruzada de valor, la decisión de abstenerse deja de ser constante y pasa a depender de la
pregunta.

**H₀ (lo que se cree hoy).** `q` seguirá siendo constante y sólo cambiará de valor, como pasó con `L`.

## 3. Diseño

Seis unidades. `--rec-ce` ∈ {**1,0** control · **0,50** · **0,29**} × origen ∈ {`b3_s3` · `b3_s6`},
las dos unidades declaradas atractor absorbente el 29-ago y usadas en la campaña de `L`, para que la
comparación sea contra la misma base.

El **0,29** es `1 / 3,5` redondeado, o sea el valor que iguala el ratio de gradientes medido. **No es
un barrido exploratorio**, es el número que sale del diagnóstico.

Todo lo demás fijo e igual a la campaña de `L`. Interfaz `token`, `M`=0,5, `F`=0,2, `L`=0, 3000 pasos,
horizonte 12000. `L`=0 porque el informe ya estableció que era un subsidio al silencio.

## 4. Criterios, escritos antes del dato

| | criterio | qué decide |
|---|---|---|
| **Q-1** principal | `falsa_abst` baja al menos **0,10** respecto del control con la misma semilla | la abstención empieza a depender de la pregunta |
| **Q-2** | exactitud global **> 0,4065** en al menos 1 de 4 unidades tratadas | supera el piso trivial, que es la vara del proyecto |
| **Q-3** | `q` deja de ser constante, o sea rango de `q` entre las cuatro tratadas **> 0,10** | la constante se rompió |
| **Q-4 riesgo** | RECUP no cae más de **0,05** contra el control de la misma semilla | si cae, el modelo dejó de recuperar y el resultado no es interpretable |

**Regla de lectura.** Q-1 y Q-3 son la hipótesis. Q-2 es la vara dura. **Si Q-4 se dispara, la unidad
se declara no evaluable y no se usa para adjudicar**, igual que se hizo con la fase `cabeza` en la
campaña de `L`.

## 5. Abandono

Si las cuatro unidades tratadas fallan Q-1 **y** Q-3, la hipótesis de magnitud queda **refutada para
esta vía**, y el proyecto pasa a la única candidata escrita que puede romper la constante, el
**castigo superlineal en la confianza** (`DICTAMEN_GEMINI_20260830.md`), que hace que el costo dependa
de la pregunta concreta en vez del reloj.

**No se corre una campaña de rescate sobre `rec-ce`.** Bajar más el CE sin un número que lo justifique
sería exactamente el barrido exploratorio que este pre-registro evita.

## 6. Lo que esta campaña NO puede decidir

- **No separa magnitud de forma en general.** Sólo prueba una vía de cambiar la magnitud, la que el
  ratio de gradientes medido señala. Si falla, quedan otras.
- **No es prospectiva sobre la interfaz `cabeza`**, que en la campaña de `L` terminó muda y quedó
  declarada no evaluable por presupuesto.
- **Bajar `CE` puede degradar la recuperación**, y por eso existe Q-4. Un resultado con RECUP caída no
  es un resultado sobre la abstención.
