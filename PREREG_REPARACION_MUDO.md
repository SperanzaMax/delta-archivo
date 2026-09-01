# ¿El atractor mudo es REPARABLE, o es absorbente de verdad? · congelado ANTES de correr

**2026-09-01, 08:55.** Pedido de Maxi, en paralelo a `PREREG_AVISAR_A_PRESUPUESTO.md`: *«hacé una
prueba en paralelo para definir esto que me dijiste, será que el atractor mudo es reparable»*.

## 1. Por qué NO la responde el experimento que ya está corriendo

Las `rk3`/`bl3` se entrenaron **desde cero** con la pérdida nueva (`SEMBRAR=0` en el informe del
29-ago). Eso mide **PREVENCIÓN**: si arrancando con `ranking`/`balance` la unidad nunca cae en el
atractor. **No dice nada sobre sacar a una unidad que YA cayó.**

El 29-ago la línea declaró el estado **ABSORBENTE**, y absorbente significa exactamente que **no se
sale**. Esa afirmación nunca se probó en la dirección que la haría falsable: **tomar una unidad muda y
tratar de repararla.** Es lo que esta prueba hace, y es la que puede refutar la palabra.

## 2. El confound que obliga a tener control

Sembrar con `sembrar.py` **conserva los pesos pero borra `opt_state` y pone el paso en 0**, así que la
corrida nueva arranca con **Adam reiniciado y warmup de `lr` otra vez**. Una unidad que salga del
silencio podría estar saliendo **por la sacudida del reinicio**, no por la pérdida nueva. Sin control,
un positivo no se puede atribuir.

- **TRATAMIENTO `rp3_s3` `rp3_s6` `rp3_s7`** — sembradas de `b3_s3/s6/s7` (mudas a 26000, abstención
  1,0000) con **`perdida_cabeza=ranking`**.
- **CONTROL `rc3_s3` `rc3_s6` `rc3_s7`** — sembradas de las MISMAS unidades, mismo reinicio, mismo
  warmup, misma `lr`, mismo presupuesto, con **`perdida_cabeza=bce`**, que es la pérdida con la que
  quedaron mudas.

**La cabeza colapsada se conserva en los dos brazos** (sin `--sin-cabeza`): el punto de partida tiene
que ser el atractor, no un modelo sin cabeza.

Presupuesto: **6000 pasos**, `horizonte=6000` en los dos brazos. La salida del silencio, cuando pasa,
se ve en 500-1500 pasos (medido esta mañana en el smoke de `rk3_s3`: 1,0000 → 0,6387 en 1000 pasos).

## 3. Criterios

- **E-1 · PRINCIPAL.** **≥2 de 3** unidades del tratamiento bajan la abstención por debajo de **0,90**
  y la **sostienen en las 3 últimas evaluaciones** (no un mínimo puntual, por
  `NOTA_LECTURA_CURVAS_20260824.md`). Si cumple, **el atractor NO es absorbente: es reparable**, y la
  palabra «absorbente» del 29-ago hay que cambiarla por «estable».
- **E-2 · ATRIBUCIÓN, y es la que decide de qué fue.** Tratamiento **menos** control en número de
  unidades que cumplen E-1. Si el control cumple **igual o más**, la reparación es **del reinicio de
  Adam**, no de la pérdida, y así se informa. **Sin este contraste E-1 no se atribuye a nada.**
- **E-3 · ¿reparada o sólo locuaz?** En las que cumplan E-1, `nose` ≥ 0,90 **y** `falsa_abst` ≤ 0,10 al
  final. **Se declara por adelantado que 6000 pasos pueden no alcanzar para esto**: si E-1 cumple y
  E-3 no, la lectura es **«sale del silencio pero no llega a útil en este presupuesto»**, que NO es un
  negativo de la reparación sino un límite de presupuesto declarado. Es la lección del 29-ago aplicada
  antes y no después.
- **E-4 · RIESGO.** Si alguna unidad del tratamiento sube `invento` por encima de 0,30, se informa el
  trade-off mudez↔invención con el mismo criterio que el 29, ahora con su control al lado.

**Riesgo de legibilidad (protege a E-1, E-2 y E-3):** si llegan a 6000 menos de 2 unidades por brazo,
**NO EVALUABLE**; no se lee ninguno de los tres.

## 4. Lo que no puede decir

Tres semillas por brazo, una tarea, un presupuesto corto y **las mismas semillas que ya se sabe que
son mudas** — elegidas por eso, así que esto no habla de tasas base. Y si la reparación funciona, no
dice **cuándo** deja de ser posible: reparar a 26000 no implica poder reparar a 60000.
