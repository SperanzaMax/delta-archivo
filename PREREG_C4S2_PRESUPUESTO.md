# PREREG · `c4_s2` con más presupuesto — ¿la cabeza se rompe al entrenarla de más?

Congelado el 2026-08-20 antes de lanzar el tramo. Sale del §2 de `PLAN_20260820.md`.

---

## 1 · El dato que lo motiva

`INFORME_CELDA_DIFICIL_20260819.md` cerró con una observación que no parecía ruido: en `c4_s2`
—nivel 4, la tarea más difícil, condición `cabeza`— los últimos tres puntos de `falsa_abst` suben
**monótonos** mientras `nose` sube en paralelo:

| paso | 13000 | 13250 | 13500 | 13750 | 14000 |
|---|---:|---:|---:|---:|---:|
| `falsa_abst` | 0,0817 | 0,1135 | 0,0960 | 0,1420 | 0,1498 |
| `nose` | 0,5087 | 0,6262 | 0,5546 | 0,6932 | 0,6683 |

Leído derecho: **está aprendiendo a abstenerse de más, y el exceso se lo come de preguntas que sí
tenían respuesta.** Si es tendencia y no ruido, **cambia la recomendación operativa de toda la
campaña**: la cabeza de abstención necesitaría parada temprana en tarea difícil, justo lo contrario
de lo que se hizo en todas las unidades (entrenar hasta el presupuesto).

**Por qué con lo que hay no se puede decidir:** cada punto de la historia se mide con `evaluar(n=8,
B=64)` = **512 muestras**, y sólo ~60 % son preguntas con respuesta, así que el error estándar de
`falsa_abst` ronda **0,019**. La diferencia 0,0960 → 0,1498 son 0,054, menos de tres errores
estándar, sobre **tres puntos**. No alcanza.

## 2 · Diseño

Se reanuda `c4_s2` desde su checkpoint de 14000 hasta **20000 pasos**, `--cada 250` → **24 puntos
nuevos**. La unidad se reanuda con su config exacta (`nivel 4 · semilla 2 · lr 1e-3 · d 128 ·
capas 4 · p_nose 0,4 · p_vieja 0,35 · abst cabeza · idioma 2`) y **`horizonte 20000`, que ya estaba
fijado en la corrida original**: llegar a 20000 es lo que el diseño preveía y **no cambia la curva de
decaimiento de la lr**. Esto importa —si el horizonte se moviera, cualquier degradación sería
confundible con un cambio de tasa.

El checkpoint de 14000 quedó preservado como `ckpts/c4_s2.pkl.p14000` **antes** de lanzar, porque el
tramo sobrescribe `c4_s2.pkl` y sin la copia no habría con qué comparar.

La potencia sale de la **longitud de la serie**, no de la precisión de cada punto: 24 puntos con
ruido ±0,019 detectan una pendiente que 3 puntos no pueden separar del ruido.

## 3 · Predicciones

- **T-1 (principal).** Spearman entre `falsa_abst` y el paso, sobre los 24 puntos nuevos:
  **ρ ≥ +0,41 con p < 0,05** (bilateral, n=24) ⇒ **es tendencia**. Con empates, **rangos promediados**
  (la regla que dejó el `INFORME_FRONTERA` el 19-ago).
- **T-2 (confirmación en los extremos, con muestra grande).** `falsa_abst` medida con **2048
  muestras** —rng 77000+semilla, el de prueba, para no reusar el generador con el que se evaluó
  durante el entrenamiento— sobre `c4_s2.pkl.p14000` y sobre el de 20000:
  **diferencia ≥ +0,03** ⇒ confirma.
- **T-3 (el intercambio).** `nose` sube en paralelo (ρ ≥ +0,41). **Si `falsa_abst` sube y `nose` no,
  el resultado es peor que el previsto**: pierde precisión sin ganar cobertura, y hay que decirlo así.
- **T-4 (control de sanidad, y puede fallar).** `vigente` a 20000 no cae más de **0,10** respecto de
  su valor a 14000 (0,6836). Si cae más, lo que se degrada es el modelo entero y **la conclusión no
  puede atribuirse a la cabeza**.

## 4 · Desenlaces, comprometidos por adelantado

- **T-1 y T-2 cumplen** → la cabeza se degrada al entrenarla de más en tarea difícil. Recomendación
  operativa: parada temprana gobernada por `falsa_abst`, y hay que revisar si las otras unidades de
  nivel 4 muestran lo mismo.
- **T-1 no cumple** → los tres puntos del 19-ago eran ruido de muestra chica. **Se retira la
  advertencia** del `INFORME_CELDA_DIFICIL` y se anota como la sexta vez que en este proyecto una
  serie corta parecía decir algo.
- **T-1 cumple y T-2 no** → tendencia dentro del rango de evaluación chica que no sobrevive a la
  medición buena: se reporta como no concluyente, sin elegir la mitad que más gusta.

## 5 · Alcance

**Una unidad, una semilla.** No dice nada sobre `c4_s0`/`c4_s1` ni sobre las otras condiciones; si
T-1 y T-2 cumplen, lo que corresponde es mirarlas, no generalizar desde acá.
