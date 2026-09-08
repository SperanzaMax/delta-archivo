# Pre-registro · ESCALÓN 2 · la atención de lectura con proyecciones propias

Fecha 2026-09-08, escrito con el escalón 1 ya cerrado (`INFORME_FRONTERA_E1_20260908.md`,
`aa424ab9`) y **antes de correr una sola unidad `attnp`**. Diseño en `PLAN_ESCALON2_ATTNP.md`
(`caeed35f`).

## 1. Qué dejó abierto el escalón 1

Dos cosas, y las dos apuntan al mismo lado.

**La primera, medida.** `at3` alcanzó el techo del kernel 5 con un empate exacto de 0,9977 contra
0,9977, y su razón de selectividad quedó en 0,98, 0,99 y 0,97, o sea la de un modelo que nunca vio
acceso global. **Entrenó 26.000 pasos con acceso global y no aprendió a usarlo para discriminar.** La
explicación medida es que sin `wq`/`wk` propias no hay un solo peso entre `x` y el softmax, así que
la similitud sale de la geometría de las embeddings y la brecha de 3,13 hacia la propia posición
sale de `|x|²/sqrt(D)`, no de un parámetro.

**La segunda, observada y no pre-registrada.** `at3` **arranca más lento** que los dos brazos, va
por debajo hasta el paso 4.000 y recién los cruza en el 6.000. Consistente con lo anterior, porque
mover la geometría de las embeddings es un camino más largo que aprender una proyección.

## 2. La propiedad que este escalón estrena, y es la razón del diseño

`wqr`, `wkr` y `wvr` arrancan **en la identidad**. Verificado hoy antes de escribir esto,
`attnp` en el paso 0 es **idéntico a `attn` bit a bit**, diferencia máxima 0,000e+00 en la query de
lectura y en la salida del tronco.

Eso le da lo que a `attn` le falta. **`attnp` contiene a `attn` como caso particular**, así que no
puede ser estructuralmente peor y todo lo que aparezca lo fue a buscar el gradiente. El §6 del
informe del escalón 1 tuvo que declarar esa carencia como límite; acá deja de existir.

## 3. Hipótesis

**H1 · la que este escalón existe para contestar.** Con proyecciones propias, la razón de
selectividad `TV(d=3)/TV(d=5)` medida con N = 600 sube por encima de **1,5** en al menos dos de las
tres semillas. Se refuta si queda en el rango 0,94 a 1,05 de los controles.

Es la hipótesis que el escalón 1 no pudo contestar, porque ahí la selectividad no podía moverse por
falta de parámetros. **Acá puede.** Si igual no se mueve, el resultado es fuerte en la otra
dirección, y quiere decir que la tarea no premia discriminar y que la cobertura es todo lo que hay.

**H2 · el transitorio.** `attnp` cruza a los dos brazos **antes** del paso 6.000, que es donde los
cruzó `at3`. Se refuta si el cruce ocurre en el mismo paso o después. Es la contraparte falsable de
la observación post-hoc del escalón 1.

**H3 · el techo.** `attnp` termina en `nose_rel` **no menor** que `at3`, o sea 0,9977 o más. No es
una predicción de mejora, y esto es deliberado. El escalón 1 mostró que el techo lo fija la
cobertura, y `attnp` cubre lo mismo que `attn`. **Si `attnp` superara el techo, sería evidencia
contra la ley enunciada como cobertura** y habría que revisarla. Se declara ahora para que no se
pueda leer como éxito lo que sería un problema.

## 4. Lo que este escalón NO es

**No es el Micro LM de Frontera.** Arregla la diferencia 1, la query de lectura. **No toca la 2 ni
la 3**, que son las que deciden. El tronco sigue siendo `delta_mixer` recurrente y el vocabulario
sigue siendo 242 tokens cerrados con respuesta de un solo token.

Nada que conserve el tronco recurrente o el vocabulario cerrado se llama de Frontera.

## 5. Diseño, congelado

Tres semillas 0/1/2, **desde cero** (`SEMBRAR=0`), 26.000 pasos, horizonte 26.000, d 128, capas 4,
lr 1e-3, idioma 2, `p_vieja` 0,35, `p_nose` 0,4, `abst` cabeza, `sello` abs, `ses_extra` 0, `pert` 0.
Idéntico a `at3` salvo `--donde attnp`, que es la única variable.

    PREFIJO=ap SELLO=abs PERT=0 SES_EXTRA=0 KERNEL_Q=5 DONDE=attnp \
      P_NOSE=0.4 ABST=cabeza SEMBRAR=0 HORIZONTE=26000 \
      ./rotar_abst3.sh 3:0 26000 2000 500 <cuentas>

Métrica primaria de H1, `selectividad_attn.py` con N = 600 sobre los pesos del paso 26.000.
Secundarias, `nose_rel`, `vigente`, `anterior`, el paso de cruce.

Se reporta la corrida entera y no el mejor checkpoint.

## 6. El control que sale gratis, y acá importa más que en `convq`

Las proyecciones de los bloques 1 a 3 **no reciben gradiente**, porque la lectura entra sólo en el
bloque 0. Son la trayectoria del weight decay puro sobre una identidad.

El atractor del decay sobre una identidad es la **matriz nula**, no la identidad. Así que al cerrar,
cualquier diferencia entre el bloque 0 y los otros tres es gradiente y no decay, sin simular nada ni
suponer una tasa. Sin este control, una proyección atenuada se leería como aprendizaje.

## 7. Verificaciones hechas antes de congelar esto

1. **`attnp` idéntico a `attn` en el paso 0**, diferencia 0,000e+00. Pasa.
2. **Contabilidad contada del árbol**, y con una corrección que vale anotar. Conté primero
   **1.062.387** y el entrenador reportó **1.063.411**. La diferencia son 1.024 exactos y sale de
   que `convq` mide `KQ × D` por bloque, así que yo había contado con el default del módulo
   (`KQ = 3`) mientras la campaña corre con `KQ = 5`. **Cuatro bloques por dos taps por 128.**

   Los números buenos, en la condición real de la campaña, son **1.063.411 con `attnp` contra
   866.803 sin él**, o sea 196.608 nuevos (**22,7 %**) y 49.152 efectivos (**5,7 %**). Los
   porcentajes no cambian. Pasa.

   Es la misma lección del día por tercera vez. No alcanza con contar del árbol en vez de citar de
   memoria, hay que contar **en las condiciones en que se va a correr**, porque un global del
   módulo cambia el árbol.
3. **La compuerta de padding abre**, peor tasa 0,0000 contra límite 0,01. Pasa.
4. **Regresión sobre checkpoints viejos.** `control_attn.py` sobre `kq3_s0` devuelve exactamente los
   números del 4-sep, 0,010305 a 0,009805. Los 157 checkpoints anteriores siguen midiéndose igual.
   Pasa.
