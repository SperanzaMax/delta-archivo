# PREREG · ¿aprende a buscar en un archivo largo? · 2026-09-05

**Congelado antes de correr.** El SHA de este archivo va en `SHA_ARCHIVO_LARGO.txt` y en el commit
que lo agrega. El instrumento (`--ses-extra`, commit `5651d15`) ya está escrito y verificado; lo que
este pre-registro congela es el diseño, los criterios y la cláusula de abandono.

## 1. De dónde sale

`INGENIERIA_INVERSA_20260905.md` e `INFORME_DILUCION_20260905.md` (prereg `f4d91c12`). La premisa que
nadie había mirado: **el banco nunca probó un archivo grande.** Un episodio archiva a lo sumo
`4 sesiones × E_MAX 10 = 40` entradas, y todo lo que la línea sabe —el sello de orden, la ley de la
ventana, la abstención, el atractor— está medido con ese techo. El objetivo declarado pide lo
contrario (ver `objetivo-memoria-persistente-llm`).

Medido ayer sobre checkpoints **entrenados con archivo corto**, exactitud contra tamaño del archivo:
40 → **1,0000** · 80 → 0,7832 · 160 → **0,3008** · 400 → 0,0605 · 3280 → **0,0039**. Con cuatro
conversaciones guardadas cae bajo el piso trivial 0,4065.

Y el control dio vuelta la causa: **el softmax no se diluye por número** (con 3280 competidores de
ruido RECUP sigue en 0,7852). Rompe el **contenido**, en dos capas: interferencia (0,4590) y colisión
(0,0117). La dilución pura existe pero es **del valor leído**, no del ranking.

**Todo eso es transferencia, no capacidad:** se entrenó con 40 y se evaluó con 400. Esta campaña es la
primera que entrena con el archivo largo, que es lo que `INGENIERIA_INVERSA` puso primero en la fila
para T4.

## 2. Hipótesis

**H-1 (principal).** El colapso medido es de **transferencia**. Un modelo entrenado con el archivo
largo aprende a buscar en él y supera el piso trivial en el régimen donde hoy da 0,30.

**H-2 (la que importa para el objetivo).** Entrenado así, el modelo aprende a **usar el sello de orden
para descartar lo viejo**. Las entradas de otras conversaciones llevan turnos por debajo de los del
episodio, o sea que el archivo largo está marcado como anterior. Hoy esa marca **no se usa**
(0,0059 contra 0,0039 sin marcar, medido ayer), pero **nunca se lo entrenó para ese uso**. Si H-2
cumple, R11 —*descartar lo que ya no viene al caso*— deja de ser una carencia y pasa a ser una
capacidad entrenable.

**H₀.** El archivo largo es un techo del mecanismo: entrenar no lo mueve, porque la clave no separa
lo suficiente y la interferencia es irreducible con esta representación.

## 3. Diseño

**Siembra, no entrenamiento desde cero.** Origen `kq3_s0`, `kq3_s1`, `kq3_s2` — las tres unidades de
kernel 5, 26.000 pasos, `lat2`, nivel 3, que son las que **ya resuelven la tarea con archivo corto**
(exactitud 0,988-0,993). La pregunta queda mejor puesta así: no es «¿puede aprender la tarea?», que ya
está contestada, sino **«un modelo que sabe la tarea, ¿aprende a hacerla con un archivo grande?»**.
`sembrar.py` declara la bifurcación en el checkpoint (`sembrado_de`) y `ses_extra` está en `BIFURCA`.

**Seis unidades, 3 semillas × 2 condiciones:**

| | `--ses-extra` | casilleros | entradas escritas | qué es |
|---|---:|---:|---:|---|
| control | **0** | 40 | ~5 | seguir entrenando con el archivo de siempre |
| tratada | **26** | 300 | ~161 | el régimen donde hoy la exactitud es 0,3008 |

El control existe para separar **«entrenar con archivo largo»** de **«entrenar 6000 pasos más»**, y
lleva el mismo presupuesto y la misma siembra. Sin él, cualquier mejora sería ambigua.

**Por qué 26 y no más.** Presupuesto medido hoy en CPU (`costo_paso.py`, d=128, capas=4, batch 16):
el costo por paso va con el número de **sesiones**, no de entradas — `ses-extra` 0 · 8 · 26 cuesta
**1,00× · 2,81× · 6,96×**. A 0,22 s/paso en T4, la tratada sale ~1,53 s/paso: **6000 pasos ≈ 2,5 h**
por unidad, contra ~0,4 h el control. La campaña entera son **~8,7 h de T4**, comparable a la de
`L`. Con `--ses-extra 36` (400 casilleros, el régimen 0,0605) serían ~3,5 h por unidad y la campaña no
entraría en el presupuesto.

**Todo lo demás fijo:** 6000 pasos, horizonte 6000, `--kernel-q 5`, `--donde lat2`, nivel 3,
`p_nose` 0,2, batch 64, lr 1e-3. **Las dos condiciones de archivo se evalúan siempre**, entrene con la
que entrene (`cruzada_corto` / `cruzada_largo` en el JSON): sin eso no se puede leer ni el precio ni la
transferencia.

## 4. Criterios, escritos antes del dato

| | criterio | qué decide |
|---|---|---|
| **L-1** principal | la tratada, evaluada **en archivo largo**, supera el piso trivial **0,4065** en al menos **2 de 3** semillas | H-1: entrenar con el archivo largo cambia el régimen |
| **L-2** control | el control, evaluado en archivo largo, queda **por debajo de 0,15** | replica el colapso y descarta que la ganancia venga del generador o de 6000 pasos más |
| **L-3** precio | la tratada, evaluada **en archivo corto**, no cae más de **0,05** contra el control en corto | si cae, manejar el archivo largo se paga en el corto y hay que decirlo |
| **L-4** el sello | el **índice de masa** de `masa_turnos.py` baja a **≤ 0,40** en al menos 2 de 3 semillas, **y** la brecha contra su propia celda `--barajar` es de al menos **0,20** | H-2: descarta por antigüedad, y lo hace con el sello y no por contenido |
| **L-5** riesgo | RECUP en archivo largo no cae más de **0,10** contra el origen `kq3_sX` medido hoy | si cae, el modelo dejó de recuperar y ningún otro número es interpretable |

**Regla de lectura.** L-1 y L-4 son las dos hipótesis y se leen **por separado**: se puede aprender a
buscar sin aprender a descartar, y ése sería un resultado, no un fracaso. **L-2 es bloqueante**: si el
control también sube, la campaña no midió lo que dice medir. **L-4 sin su celda barajada no cuenta**,
porque las entradas del episodio nombran la entidad preguntada y se llevan masa por contenido aunque
el sello no intervenga.

**Línea de base de L-4, medida hoy sobre `kq3_s0` antes de escribir el umbral** (`masa_turnos.py`,
`masa_turnos_kq3s0.json`), con su control:

| sesiones extra | sello | índice de masa | masa en la correcta | RECUP |
|---:|---|---:|---:|---:|
| 8 | real | 0,7287 | 0,3171 | 1,0000 |
| 8 | barajado | 0,6909 | 0,3509 | 0,9583 |
| 26 | real | **0,8886** | 0,1291 | 0,6667 |
| 26 | barajado | **0,8885** | 0,1264 | 0,7083 |

Dos cosas quedan medidas antes de gastar un solo paso de T4. **El sello no está haciendo nada:**
barajarlo mueve el índice de 0,8886 a 0,8885 —cuarta cifra— así que la preferencia leve por las
entradas del episodio es **por contenido**, no por antigüedad. Es el Resultado 3 de ayer replicado con
otro instrumento y, ahora sí, con el control que lo adjudica. Y **el modelo encuentra pero lee mal**:
RECUP 0,6667 con la masa en la correcta en 0,1291, que es la dilución del valor de ayer vista por
dentro.

Eso fija el umbral de L-4 sobre evidencia y no sobre una corazonada: hoy la brecha real−barajado es
**0,0001**, y se pide **0,20**.

## 5. Abandono

Si **L-1 falla en las tres semillas** y **L-2 cumple**, entonces entrenar con el archivo largo no
alcanza, y el cuello no es la exposición sino la **clave**: no separa contenido suficiente para que
161 competidores no se pisen. En ese caso la línea pasa a la clave —y `PREREG_FILTRADO_PREVIO`
enmendado deja de ser una alternativa y pasa a ser el paso siguiente, porque filtrar antes de buscar
es exactamente sacar competidores con contenido.

**No se corre una campaña de rescate subiendo pasos.** Si a 6000 pasos sobre un modelo que ya sabe la
tarea no se mueve, el problema no es presupuesto.

## 6. Lo que esta campaña NO puede decidir

- **No dice nada sobre 3280 entradas.** Mide 161. La curva de ayer sigue siendo la única evidencia del
  régimen grande, y es de transferencia.
- **No separa interferencia de colisión.** Las sesiones extra se sortean sin restricción, así que
  contienen las dos. Separarlas es otro experimento, con el generador restringido a entidades
  disjuntas, que ya existe en `dilucion.py`.
- **No prueba el tope de 64 turnos.** Con `TURNO_BASE = 24` y 40 turnos de episodio los índices caben
  justo. Un archivo que necesite más de 64 turnos distintos sigue sin poder sellarse, y eso es una
  limitación de `ord` que ninguna campaña arregla: hay que agrandar la tabla.
- **No es prospectiva sobre entrenar desde cero con archivo largo**, que es más caro y otra pregunta.
