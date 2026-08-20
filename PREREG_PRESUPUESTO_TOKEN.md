# PREREG · El mismo presupuesto para `token` y `escala`

Congelado el 2026-08-20 antes de lanzar la campaña.

## 0 · Qué pregunta cierra

`INFORME_C4_REPLICA_20260820.md` mostró que **14000 → 20000 mejora `falsa_abst` en las tres unidades
de nivel 4 de `cabeza`** (0,0845 → 0,0636 · 0,0955 → 0,0423 · 0,1927 → 0,0170), y dejó escrito lo
que **no** se podía concluir: la única unidad que fallaba la compuerta era `c4_s2` y es la única que
pasó de fallar a pasar, así que **el confound del §4 del `INFORME_FRONTERA` queda TOCADO, NO
RESUELTO**; para cuantificarlo hay que ir a las unidades que sí fallan, que están en `token` y
`escala`.

Esto es exactamente eso. Y la pregunta es incómoda a propósito: el hallazgo del 18-ago —**la cabeza
de abstención pasa la compuerta en 4 de 5 unidades donde `token` y `escala` fallan 5 de 5**— se midió
con **todas las unidades a 14000**. Si `token` y `escala` también pasan cuando se les da el mismo
presupuesto que se le dio a `cabeza`, entonces parte de esa ventaja era presupuesto y no
arquitectura.

**Lo que hace la comparación limpia, verificado antes de escribir esto:** las cinco unidades tienen
`horizonte: 20000` en su config (leído de los checkpoints), igual que las `c4`. Extender de 14000 a
20000 **no toca la curva de lr** — no es un cambio de tasa disfrazado de más pasos.

## 1 · Unidades y procedimiento

`t4_s0`, `t4_s1`, `t4_s2` (`--abst token`) y `s4_s0`, `s4_s1` (`--abst escala`), de **14000 a 20000
pasos**, reanudando desde el checkpoint (Adam **no** se reinicializa: `--reinit-adam` sólo entra
cuando el tramo siembra desde la campaña base, y acá el `.pkl` ya existe). `s4_s2` no existe y **no**
se entrena desde cero: sería otra pregunta.

Copias congeladas `.p14000` hechas **antes** de lanzar, y toda medición del punto de partida sale de
ellas: es la regla D-1 del día —una unidad que entra en un análisis no puede estar entrenándose al
mismo tiempo—.

Medición con el **mismo instrumento y el mismo rng de prueba (77000 + semilla) y 2048 muestras** que
usó la réplica de `c4`, para que los números sean comparables punto por punto.

## 2 · El punto de partida, y lo que falta de él

`token` nivel 4 a 14000, ya medido (`INFORME_CELDA_DIFICIL_20260819.md`): **0,1342 · 0,1713 ·
0,1419** — las tres fallan la compuerta (`falsa_abst ≤ 0,10`).

**`escala` nivel 4 a 14000 no está medido con 2048 muestras.** Se mide desde las copias `.p14000`
**antes** de leer ningún resultado extendido, y el número queda registrado antes de comparar. Si por
lo que sea `s4_s0`/`s4_s1` ya pasaran la compuerta a 14000, quedan **fuera** de P-2 y se dice — es la
lección de R-3 de la réplica de hoy, que pidió «pasar habiéndola fallado» sin mirar que dos unidades
ya pasaban.

## 3 · Predicciones

- **P-1 (réplica del efecto).** `falsa_abst` **baja** de 14000 a 20000 en **≥ 4 de las 5** unidades, y
  el Spearman(paso, `falsa_abst`) sobre los puntos nuevos es negativo en **≥ 4 de las 5**. Es la
  réplica de R-1/R-2 en la otra condición.
- **P-2 (la principal, y decide el confound).** De las unidades que **fallaban** la compuerta a
  14000, ¿cuántas la pasan a 20000?
  - **≥ 3 de 5 pasan** → la ventaja de `cabeza` del 18-ago estaba **en parte comprada con
    presupuesto**: el hallazgo queda ACOTADO y hay que reescribirlo con esas palabras.
  - **≤ 1 de 5 pasa** → el confound queda **cerrado a favor de `cabeza`**: con el mismo presupuesto
    que arregló a `c4_s2`, las otras condiciones siguen fallando.
  - **exactamente 2 de 5** → intermedio; se reporta el número y **no** se emite veredicto binario.
- **P-3 (pareado a igual presupuesto).** A 20000 pasos, `cabeza` tiene menor `falsa_abst` que `token`
  en **las 3 semillas de nivel 4**. Es el contraste que el 18-ago se hizo a 14000 y que hasta ahora
  nunca se hizo con las dos condiciones igualadas.
- **P-4 (control de sanidad, y puede fallar).** `vigente` no cae más de **0,10** en ninguna unidad
  entre 14000 y 20000. Si se cayera, la mejora de `falsa_abst` sería un intercambio y no una mejora,
  y se archiva sin interpretar.

## 4 · Lo que no puede decir

- Cinco unidades, **una sola semilla por celda de condición×nivel** en `escala`. No es una estimación
  del efecto medio de la condición.
- Sólo nivel 4. No dice nada del nivel 3, donde `cabeza` también ganaba.
- **No prueba que 20000 sea suficiente**: si a 20000 sigue fallando, puede seguir mejorando después.
  Lo que la campaña compara es **presupuesto igualado**, no convergencia.
- El instrumento es el de la campaña (`falsa_abst`, `nose`, `vigente` con 2048 muestras y el rng de
  prueba). Todo lo que ese instrumento no ve, esto tampoco.

## 5 · Infra

Rotador `rotar_abst.sh` con `PREFIJO=t ABST=token` y `PREFIJO=s ABST=escala`, tramos de 3000 pasos,
watchdog de 12 min contra el tramo colgado (el que trabó la campaña 3h47 el 19-ago), y el arreglo del
log de esta tarde. El veredicto lo escribe una persona leyendo la tabla, no el `print`.
