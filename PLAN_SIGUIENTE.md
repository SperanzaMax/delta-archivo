# Plan del 2-sep · escrito al cierre del 1

**El plan del 1-sep quedó COMPLETO** (§0/§2 el control del desacuerdo · §3 `medir_en_colab.sh`
probado · §4 la curva bien hecha), así que esto es cola nueva, no arrastre.

---

## 0. Lo que el 1-sep dejó firme

| | |
|---|---|
| el desacuerdo **no supera** a la confianza a igual cobertura | P-1 no cumple, +0,0333 IC95 cruza el cero |
| pero **la complementa** | Jaccard 0,23 · con el modelo confiado marca 75 preguntas a precisión 0,8533, y el **72 % son ausencias** |
| **recuperar mejor compra detección** | pendiente WITHIN **+0,5350**; la floja (+0,1308) era el confound |
| **y no la agota** | `v3` con RECUP 1,0000 exacto ×3 se queda en techo 0,93-0,96 |
| **evidencia causal** | arrancar de RECUP 0,78 → 3 de 3 útiles; de cero → 2 útiles y 4 mudas |

## 1. Lo primero: el residuo que la recuperación no explica

Es la pregunta que abre el §4 del informe de hoy. `v3` recupera **perfecto** y su ausencia sigue
dando 0,93-0,96 de techo, mientras `b3` con **`blanco=error` entrenado** llega a 1,0000 exacto.
La hipótesis derivada: **ese residuo se cierra entrenando la detección, no mejorando la búsqueda.**

Falsable y barato con lo que hay en disco: comparar el techo de unidades con `blanco=error` contra
`blanco=ausencia` **a RECUP igualado**, que es el pareo que nunca se hizo. Necesita pre-registro
propio —el criterio tiene que decir cuánto de brecha cuenta— y **cuidado con `b3`: mezcla dos
poblaciones y no sirve de control** (ver el §2 del informe de hoy).

## 2. El desacuerdo, en su versión que sí puede ganar

Las dos versiones fuertes siguen sin probar y hoy quedó una razón más para ir: **el detector aporta
donde la confianza es ciega**, así que la pregunta ya no es si sirve sino cuánto rinde entrenado.

1. **Dos proyecciones de query `qr1`/`qr2`** con el desacuerdo **en la pérdida**. Precedente medido:
   el blanco `error` da 0,65 post-hoc y **1,0000 entrenado**.
2. **Buscar por entidad contra buscar por relación.** Si apuntan a entradas distintas, eso *es* la
   colisión de clave, que ya está identificada como el error dominante.

Antes de cualquiera de las dos, **replicar el enriquecimiento en ausencia con pre-registro propio**:
hoy fue post-hoc, un modelo, un σ. Y falta el **degradado**, donde la inestabilidad es 50× menor y el
detector debería apagarse — es la predicción que puede matarlo barato.

## 3. Separar los tres confounds del +0,1438

`p3_s1` (RECUP 0,7945) tiene techo 0,8441 y `n3_s0` (RECUP 0,7850) tiene 0,7003. A recuperación casi
idéntica, **+0,1438 de diferencia** con `p_nose`, presupuesto e interfaz viajando juntos. Una corrida
que mueva **una sola** de las tres lo separa, y de paso dice si el techo de 0,7003 era una propiedad
del enfoque o el valor que le tocaba a esa unidad.

## 4. Lo que NO hay que hacer

- **No usar `b3` como familia homogénea.** `s0/s1/s2` arrancaron de RECUP ≈ 0,78 y `s3…s8` de cero.
- **No leer la mediana de dos familias discordantes** como pendiente. El juez ya tiene la guarda,
  pero la trampa es de lectura, no de código.
- **No re-derivar el peso del slot** ni reabrir la vía de la búsqueda: cerró el 31 con W-8-c.
- **No correr mediciones de más de ~10 min de CPU en la PC**: `medir_en_colab.sh` ya está probado y
  hace las 17 unidades en 8 minutos.

## 5. Operativo

- Venv del proyecto: **`/home/maxi/.venv-ligamento/bin/python`** (no hay jax en el sistema).
- Colab: **la cuenta A dio 503 todo el 1-sep, la C anduvo las tres veces.** Rotar igual.
- `medir_en_colab.sh <CUENTA> <script.py> <ckpts...>` pasa los checkpoints **como argumentos** al
  script, así que ya no hace falta tocar listas hardcodeadas.
