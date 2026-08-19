# La celda cruzada — el margen sobrevive al contraste, pero el veredicto binario es frágil

**Prereg:** `PREREG_CELDA_CRUZADA.md`, congelado (SHA `6b75639a…`) antes de correr un paso y antes de
mirar ningún número de `f2_s1` con `p_nose` > 0.
**Unidades:** 2 (una sola semilla). **Presupuesto:** 2000 pasos, de 18000 a 20000.

---

## §1 · Qué se fue a buscar

El §4 de `PREREG_FRONTERA.md` declaró antes de correr que el eje confundía **margen sobre el atajo**
con **grado de entrenamiento**, y el informe del mediodía confirmó que el margen predice igual por las
dos vías, **sin una sola inversión en 13 unidades** — pero como correlación, no como contraste: todos
los puntos de margen bajo eran de tarea difícil y todos los de margen alto, de tarea fácil.

`f2_s1` llena la celda que faltaba: **nivel 2 (tarea fácil), entrenada a fondo (18000 pasos), margen
+0,2124**, o sea debajo del corte de `token` (+0,2358 a +0,2826) y encima del de `cabeza` (+0,1489 a
+0,1672). **No fue fabricada para esto**: salió del fracaso de la base de la frontera, que nunca cruzó
0,85.

## §2 · Resultado

| unidad | condición | `vigente` | `nose` | `falsa_abst` | compuerta |
|---|---|---:|---:|---:|---|
| mt2_s1 | `token` | 0,5991 | 0,6755 | **0,1885** | **falla** |
| mc2_s1 | `cabeza` | 0,6897 | 0,6096 | **0,0902** | **pasa** |

**C-1 se cumple en su forma literal**, y **C-2 queda descartada**: si gobernara la dificultad de la
tarea, las dos condiciones tendrían que haber pasado, porque todas las unidades de nivel 2 medidas
hasta hoy pasan con las dos. `token` falló.

→ **El margen sobrevive al contraste cruzado.** Una unidad de tarea fácil, entrenada a fondo, con
margen bajo, se comporta como el **grupo de margen bajo** y no como el **grupo de nivel 2**. El margen
no era un proxy de la dificultad de la tarea.

## §3 · La salvedad, y es grande

**El «pasa» de `cabeza` depende del último tick.** Trayectoria de `falsa_abst` en el tramo:

| paso | `token` | `cabeza` | `nose` de cabeza |
|---:|---:|---:|---:|
| 18250 | 0,3143 | 0,0725 | 0,1271 ← *todavía no se abstiene, el valor bajo es trivial* |
| 18500 | 0,3273 | 0,2414 | 0,6362 |
| 18750 | 0,2183 | 0,1563 | 0,6559 |
| 19000 | 0,3990 | 0,1467 | 0,6247 |
| 19250 | 0,2226 | 0,1177 | 0,6117 |
| 19500 | 0,3430 | 0,1814 | 0,6690 |
| 19750 | 0,1614 | 0,1377 | 0,6961 |
| **20000** | **0,1885** | **0,0902** | 0,6096 |

**En los seis puntos válidos anteriores al último, `cabeza` está POR ENCIMA de 0,10.** Si la
evaluación final hubiera caído en el paso 19750, `cabeza` habría fallado la compuerta y este informe
diría C-3. **El veredicto binario de esta unidad es fruto de dónde cayó el corte**, y así se reporta.

Es la misma lección que dejó `sonda_umbral.py` el 18-ago, ahora del otro lado: **lo que está pegado al
borde del criterio no es estable**.

## §4 · Lo que sí es robusto en estos datos

No el veredicto binario, sino **el contraste pareado**. En los 8 puntos del tramo —mismo checkpoint
base, mismo presupuesto, mismos pasos de evaluación— `cabeza` tiene **menor `falsa_abst` que `token`
en 8 de 8**, y en 7 de 7 si se excluye el punto donde todavía no se abstenía (`nose` 0,1271, donde un
`falsa_abst` bajo no significa nada).

- **Test de signos sobre los 7 puntos válidos: p = 0,0078.**
- **Media del tramo: `token` 0,2718 contra `cabeza` 0,1430**, o sea la cabeza abstiene de más
  **1,90 veces menos**, sostenido a lo largo de todo el tramo y no en un punto.

Y `vigente` vuelve a salir **más alto** con la cabeza (0,6897 contra 0,5991), como en las tres
unidades donde P-4 falló hacia arriba el 18-ago.

## §5 · Alcance, dicho sin adornos

- **Son 2 unidades y UNA sola semilla.** No hay media, no hay intervalo, no se puede hablar de
  convergencia. El prereg lo dijo antes de correr: esto **no puede confirmar**, sólo **falsar**.
- Lo que efectivamente hizo fue **falsar C-2**, que era su trabajo. El margen queda en pie como
  variable, con evidencia ahora cruzada y no sólo correlacional.
- **Lo que no queda establecido** es que `cabeza` pase la compuerta de forma estable en este margen.
  Con el corte en 19750 habría fallado. Lo estable es que **está sistemáticamente por debajo de
  `token`**, no que cruce el umbral.
- La semilla usada es, además, **la única que no convergió** en toda la campaña de la frontera. Puede
  que su inestabilidad punto a punto sea propia de ella.
