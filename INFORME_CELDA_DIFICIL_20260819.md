# La celda difícil — D-3 y D-4 fallan, y las dos fallas dicen algo distinto

**Prereg:** `PREREG_CELDA_DIFICIL.md`, congelado (SHA `53b780d5…`) antes de correr un paso.
**Unidades:** 2 (`t4_s2` y `c4_s2`), una semilla. **Presupuesto:** 2000 pasos, de 12000 a 14000.
`n4_s2` es la unidad que `PREREG_FRONTERA.md` §3 declaró «gratis y se agrega» y que nunca se corrió.

---

## §1 · Resultado

| unidad | condición | `vigente` | `nose` | `falsa_abst` | compuerta |
|---|---|---:|---:|---:|---|
| t4_s2 | `token` | 0,6998 | 0,6737 | **0,1419** | **falla** |
| c4_s2 | `cabeza` | 0,6836 | 0,6683 | **0,1498** | **falla** |

**Las dos fallan, y `cabeza` falla PEOR que `token`.** Es la primera unidad de toda la serie donde
eso pasa: `c4_s0` (0,0898) y `c4_s1` (0,0746) pasaban con holgura.

## §2 · D-3 falla, que era la principal

D-3 decía: *«`cabeza` le gana a `token` en `falsa_abst` también en esta unidad»*. En el número que
juzga la compuerta, **no le gana**: 0,1498 contra 0,1419.

Según el §6 del prereg, eso significa: **el aporte de `cabeza` no es general, y hay que acotarlo por
dificultad antes de recomendarlo.** Se reporta así.

**Pero la trayectoria dice otra cosa, y el prereg §7 obliga a mirarla:**

| paso | `token` | `cabeza` | |
|---:|---:|---:|---|
| 12250 | 0,3256 | 0,0734 | *`nose` 0,0960 — todavía no se abstiene, el valor bajo es trivial* |
| 12500 | 0,3467 | 0,1814 | cabeza |
| 12750 | 0,2987 | 0,2155 | cabeza |
| 13000 | 0,1342 | 0,0817 | cabeza |
| 13250 | 0,1246 | 0,1135 | cabeza |
| 13500 | 0,1461 | 0,0960 | cabeza |
| 13750 | 0,1431 | 0,1420 | cabeza |
| **14000** | **0,1419** | **0,1498** | **token ← el único punto donde gana, y es el que cuenta** |

**En los 7 puntos válidos `cabeza` gana 6.** Media 0,1908 contra 0,1400, o sea abstiene de más
**1,36 veces menos**. Test de signos una cola: **p = 0,0625** — no alcanza.

→ **El veredicto vuelve a depender de dónde cayó el corte**, igual que en la celda cruzada de la
mañana pero al revés: allá el último tick fue el único que hizo *pasar* a `cabeza`, acá es el único
que la hace *perder*. **La lección del 18-ago se repite por tercera vez: lo que está pegado al borde
del criterio no es estable.**

Lo honesto es reportar las dos cosas: **D-3 falla en su forma literal**, y **el contraste pareado
sigue favoreciendo a `cabeza` sin llegar a significancia**.

**Y hay algo que no es ruido: `c4_s2` se está degradando.** Sus últimos tres puntos van 0,0960 →
0,1420 → 0,1498, subiendo de forma monótona mientras `nose` sube de 0,5546 a 0,6683. La cabeza está
aprendiendo a abstenerse **más**, y el exceso se le va a preguntas que sí tenían respuesta. Eso es una
tendencia dentro del tramo, no un punto desafortunado, y merece mirarse con más presupuesto.

## §3 · D-4 falla, y esta falla es limpia

D-4 decía que dentro del nivel 4 `falsa_abst` bajo `token` sería monótona decreciente con el margen:

| unidad | margen | `falsa_abst` |
|---|---:|---:|
| t4_s0 | +0,1672 | 0,1342 |
| t4_s1 | +0,1787 | 0,1713 |
| **t4_s2** | **+0,2163** | **0,1419** |

**No es monótona, y `t4_s2` —la de más margen— no queda por debajo de las otras dos.** Spearman
ρ = +0,50, con el signo **contrario** al predicho.

El prereg lo dejó dicho antes de correr: *«Si `t4_s2` no queda por debajo de las dos, dentro de una
misma dificultad el margen no ordena»*. **No queda.**

→ **Dentro de una misma dificultad de tarea, el margen no ordena la abstención falsa.** Lo que
ordenaba en las 13 unidades del informe de la frontera tenía que estar acompañado por la dificultad,
porque cuando se la mantiene fija el orden desaparece.

## §4 · D-1 y D-2 no separan, como estaba declarado

`falsa_abst(t4_s2)` = 0,1419 cae **dentro del rango de las dos**: en el [0,1385 ; 0,2385] de D-1, y
entre `t4_s0` y `t4_s1` y por debajo de `mt2_s1` (0,1885) como pedía D-2. **No distingue nada.**

El §5 del prereg dijo antes de correr que con 2 unidades este contraste no se podía resolver, y no se
resolvió. Se reporta para que no se lea como si hubiera sido decisivo en ninguna dirección.

## §5 · Qué queda, y qué se mueve

**Lo que este experimento saca de circulación:**
- **El margen no es una variable explicativa dentro de una misma dificultad** (D-4). Eso debilita la
  lectura del `INFORME_FRONTERA_20260819.md`, aunque **no toca** la celda cruzada de la mañana: aquella
  comparaba **entre** dificultades a margen fijo, que es otra cosa y sigue en pie.
- **El aporte de `cabeza` deja de ser universal** (D-3). Sigue siendo el mejor de los tres ejes en 5 de
  6 unidades medidas y gana el pareado en 6 de 7 puntos acá adentro, pero **hay al menos una unidad
  donde no alcanza para pasar la compuerta**, y es de tarea difícil.

**Lo que hay que hacer y este informe no hace:**
- `c4_s2` con **más presupuesto**, porque su degradación en los últimos tres puntos parece tendencia y
  no ruido. Con 2000 pasos no se distingue una de otra.
- Una **segunda semilla** de esta celda. Todo lo de acá es n=1.

## §6 · Nota de operación

El tramo de `c4_s2` se colgó 3h47 en la cuenta K: subió el checkpoint a las 16:57 y no volvió a
escribir. La sesión de Colab se había caído —`colab status` decía «not found»— pero `tramo_abst.sh`
seguía esperando a un kernel muerto, y como el rotador espera al tramo, **la campaña entera quedó
trabada sin que nada avisara**. Esa fase no tenía timeout.

Quedó `watchdog_tramo.sh`: si hay un tramo vivo y el log del rotador no crece en 12 minutos, lo mata
para que el rotador siga con la cuenta siguiente, y avisa. **No se perdió progreso** — el tramo nunca
había arrancado, así que `c4_s2` corrió entero de una vez en la cuenta L.
