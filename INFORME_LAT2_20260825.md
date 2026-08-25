# INFORME · `lat2` — la conv propia para la query

Evalúa `PREREG_LAT2.md` (SHA `28d6f15a`), congelado el 24-ago antes de lanzar. Unidades `v3_s0/s1/s2`
a 26000 pasos, completas desde la noche del 24 y sin analizar hasta hoy. Control reusado `p3_s0/s1/s2`
(`pre` + `cabeza`, 26000), segundo brazo `w3_s0/s1/s2` (`lat`).

Instrumentos, los declarados en el §5: `ser.py` (n=2048, semilla 54321) y `diag_relacion.py` (2048
muestras, `p_nose=0`), los dos leyendo `donde` y la regla de decisión **del checkpoint**.

## 1. Resultado, pareado por semilla

|  | s0 `pre` → `lat2` | s1 `pre` → `lat2` | s2 `pre` → `lat2` |
|---|---|---|---|
| acierto | 0,9705 → **0,9984** | 0,7769 → **0,9992** | 0,8351 → **0,9984** |
| `err_identidad` | 0,0122 → **0,0000** | 0,1289 → **0,0000** | 0,0762 → **0,0000** |
| `ident_rep` | 0,0564 → **0,0000** | 0,4683 → **0,0000** | 0,2529 → **0,0000** |
| `anterior` | 0,9471 → **1,0000** | 0,8317 → **1,0000** | 0,8125 → **1,0000** |
| `nose_ent` | 0,9016 → **0,9451** | 0,4989 → **0,9908** | 0,6888 → **0,9794** |
| `nose_rel` | 0,9235 → **0,5842** | 0,5893 → 0,5842 | 0,7755 → 0,7219 |
| `nose` | 0,9119 → 0,7744 | 0,5416 → **0,7986** | 0,7298 → **0,8577** |
| `falsa_abst` | 0,0082 → 0,0016 | 0,0041 → 0,0008 | 0,0353 → 0,0016 |

| predicción | criterio | medido | |
|---|---|---|---|
| **V-0** bloqueante | acierto ≥ 0,70 en ≥ 2/3 | 0,9984 · 0,9992 · 0,9984 | **CUMPLE 3/3** |
| **V-1** conservación, la principal | `ident_rep` ≤ 0,05 en ≥ 2/3 | **0,0000 · 0,0000 · 0,0000** | **CUMPLE 3/3** |
| **V-2** reparación de `anterior` | ≥ 0,70 en las **tres** | **1,0000 · 1,0000 · 1,0000** | **CUMPLE 3/3** |
| **V-3** reparación de `nose_rel` | no cae > 0,05 vs gemela, en ≥ 2/3 | −0,3393 · −0,0051 · −0,0536 | **NO CUMPLE 1/3** |
| **V-4** no-intercambio | `falsa_abst` ≤ 0,10 en las tres | 0,0016 · 0,0008 · 0,0016 | cumple |
| | `nose` no cae > 0,05 | −0,1375 · **+0,2569** · **+0,1279** | cumple 2/3 |

## 2. Lo que el §6 obliga a leer, y estaba escrito antes de correr

> **V-1 y V-2 pasan y V-3 falla** → la caída de `nose_rel` es **intrínseca a la query conjunta** y no
> un efecto del acoplamiento.

Es exactamente la celda que salió. La lectura no se elige hoy: quedó comprometida el 24 a la mañana,
y lo que hace es **convertir en intervención** lo que el §4 del informe del camino lateral sólo podía
sugerir por interpretación. Con la relación ausente, media query sigue coincidiendo con una entrada
real y el modelo se ancla ahí. Desacoplar la conv no lo arregla porque no era el acoplamiento.

**Y el intercambio se ve nítido en las dos mitades de `nose`**, que es lo que justifica haber sacado
`nose_rel` a predicción propia: `nose_ent` sube en las tres y se va casi a 1 (hasta +0,4919 en s1),
`nose_rel` baja. La query conjunta arregla identificar la ENTIDAD y paga en detectar que falta la
RELACIÓN. Son el mismo mecanismo mirado por sus dos lados.

## 3. `anterior` reparado, y con eso el diagnóstico del 22-ago queda confirmado por su corrección

`lat` daba 1,0000 / 1,0000 / **0,3798**; `lat2` da **1,0000 en las tres**, y además por encima del
control `pre` en las tres (0,9471 / 0,8317 / 0,8125). El desplome de la semilla 2 —lo que Maxi destapó
el 22 preguntando por qué no parar a los 4000— era la conv compartida diluyendo el marcador temporal,
tal como decía `DIAGNOSTICO_CONV_COMPARTIDA_20260822.md`. La corrección lo revierte entero.

Esto es más fuerte que un negativo evitado: el diagnóstico predijo **dónde** iba a doler y la
intervención derivada de él lo reparó sin tocar nada más.

## 4. El riesgo del §7 no se materializó, y el control gratis lo prueba

El riesgo declarado era que `lat2` se quedara en `pre` —init `[1,0,0]` deliberadamente conservador— y
que V-1 fallara por «no exploró» en vez de por la razón interesante. Se declaró por adelantado la
evidencia que separa las dos cosas: el `convq` del bloque 0 **contra los de los bloques 1-3**, que
tienen gradiente cero garantizado y son por lo tanto la trayectoria del weight decay puro.

| | tap `p0` | tap `p−1` | tap `p−2` |
|---|---|---|---|
| bloques 1-3 (decay puro, las 3 semillas) | +0,866957 | 0,000000 | 0,000000 |
| bloque 0 · s0 | +0,350873 | **+0,027589** | **+0,008449** |
| bloque 0 · s1 | +0,362838 | **+0,047819** | **+0,007554** |
| bloque 0 · s2 | +0,344694 | **+0,038005** | **+0,028448** |

Los bloques 1-3 caen a 0,8670 y **no se mueven de cero** en los taps de contexto, idénticos entre sí y
entre semillas. El bloque 0 abre los dos taps del vecino y atenúa el propio a menos de la mitad de lo
que el decay explica. **Toda esa diferencia es gradiente.** El modelo fue a buscar el contexto: no se
quedó en `pre`, y V-1 se lee como resultado y no como no-exploración.

## 5. Estado

`lat2` conserva todo lo que `lat` ganó (`err_identidad` y `ident_rep` en 0,0000, ahora en las tres
semillas y sin el residuo 0,0069 de `lat`), repara `anterior`, y sube el acierto por encima de las dos
condiciones previas con la bimodalidad entre semillas otra vez ausente. Lo que no repara —y el prereg
dice que ya no se atribuya al acoplamiento— es `nose_rel`.

Según el §6, la celda «V-1, V-2 y V-3 pasan» era la que autorizaba adoptar `lat2` como base y correrla
sobre `--idioma 3`. **Esa celda no salió**, así que esa decisión no se toma acá: queda para Maxi, con
la evidencia de que las tres predicciones de memoria cumplen 3/3 y la que falla es la de abstención.
No se prueba una cuarta forma de query.
