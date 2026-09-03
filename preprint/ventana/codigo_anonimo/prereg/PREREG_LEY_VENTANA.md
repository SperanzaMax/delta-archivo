# La ventana como LEY, no como anécdota · congelado ANTES de correr

**2026-09-02.** Sale de `INFORME_KERNEL_Q5_20260901.md`. Ahí el kernel 5 llevó `nose_rel` de
0,59-0,73 a 0,99-1,00 en tres semillas sin solape, y el propio informe declara dos límites que este
prereg ataca:

1. **No distingue** si la ganancia viene de **ver la relación** o simplemente de **tener más
   contexto**. El prereg anterior dejó escrito que para eso hace falta «un control con kernel 5 donde
   el tap de la relación esté forzado a cero».
2. **La distancia 3 es una propiedad del generador**, así que el efecto es determinista acá y no se
   sabe si vale como relación general entre **ventana** y **estructura de la consulta**.

Tres mediciones. Las dos primeras son locales y baratas, la tercera va a Colab.

---

## A · ABLACIÓN DE TAPS · ¿es ver la relación, o es más contexto?

Sobre `kq3_s0/s1/s2` **ya entrenados** (kernel 5), se pone a cero **un tap por vez** de `convq` del
bloque 0 y se re-evalúa. No se entrena nada.

En «cual es `<art>` `<sust>` de `<ent>` ?» las distancias desde la posición de lectura son
**`ent`=1, `de`=2, `sust`=3, `art`=4**, y el `<sust>` **es** la relación.

| ablación | qué le saca a la query | predicción |
|---|---|---|
| **tap 3** | el sustantivo, o sea **la relación** | `nose_rel` cae al nivel del control kernel 3 |
| tap 4 | el artículo | poco o nada: `el`/`la` no identifica la relación |
| tap 2 | «de», token constante | **nada** |
| tap 1 | la **entidad** | destruye `vigente`: control POSITIVO |
| tap 0 | la posición de lectura misma | destruye todo |

**A-1 · PRINCIPAL.** `nose_rel` con el tap 3 en cero baja **≥ 0,20** respecto del modelo completo, en
**≥ 2 de 3** semillas.
**A-2 · ESPECIFICIDAD, y es la que hace la prueba.** La caída del tap 3 es **mayor que la del tap 2 y
la del tap 4** en ≥2 de 3. Sin esto, A-1 podría ser daño inespecífico por romper una activación.
**A-3 · CONTROL POSITIVO.** El tap 1 baja `vigente` ≥ 0,20. Si no lo baja, la ablación no está
haciendo efecto y **A-1 y A-2 no se leen**.

**Qué la falsa:** que todos los taps den la misma caída (inespecífico), o que el tap 3 no la dé.

## B · SENSIBILIDAD contra DISTANCIA · la ley

La sonda de sensibilidad no necesita que el modelo responda bien: mide si la búsqueda **ve** un
token, ablándolo y midiendo cuánto se mueve la distribución de lectura. Se mide la sensibilidad al
token de relación **colocado a distancia d** de la posición de lectura, con **d = 1..6**, insertando
relleno inerte en la consulta, sobre `v3` (kernel 3, alcance 2) y `kq3` (kernel 5, alcance 4).

**B-1 · PRINCIPAL, y es una predicción de escalón, no de tendencia.** La sensibilidad es
**> 0 para d ≤ alcance** y **≈ 0 para d > alcance**, con el escalón **en d=2→3 para kernel 3** y en
**d=4→5 para kernel 5**. Concretamente: `v3` tiene sensibilidad ≈0 en d=3,4,5,6 y `kq3` la tiene > 0
en d=3 y d=4 y ≈0 en d=5,6.
**B-2 · UMBRAL.** «≈0» se define **antes de mirar**: media < 0,01 en distancia TV, que es dos veces
el mayor cero medido hasta hoy (0,0000 exacto) y un orden por debajo del 0,05 que el prereg del
kernel 5 pidió como «deja de ser cero».

**Qué la falsa:** que la sensibilidad decaiga **suave** con d en vez de cortar en el alcance. Eso
diría que el efecto es de atenuación y no de ventana, y el hallazgo pasaría a ser cuantitativo.
**Confound declarado:** el relleno es OOD para el modelo, que nunca vio consultas con relleno. Por
eso B se lee como medición **mecanicista** (¿el token entra al cómputo de la query?) y **no** como
medición de desempeño. El desempeño con distancia variable es C.

## C · KERNEL 7 · ¿más ventana es mejor, o ensucia?

`k73_s0/s1/s2`, idénticos a `kq3` salvo `--kernel-q 7`. Desde cero, 26000 pasos.

Hay una **hipótesis en contra escrita antes del dato**: una ventana más ancha mete tokens
irrelevantes en la query y podría ensuciar la búsqueda. Hoy hay un indicio compatible, y es que
`nose_ent` bajó de 0,9414-1,0000 a 0,9356-0,9726 al pasar de kernel 3 a 5.

**C-1.** `nose_rel` ≥ 0,95 en ≥2 de 3 → la ventana más ancha **no rompe** lo que el kernel 5 ganó.
**C-2.** `nose_ent` comparado contra `kq3`. Si baja otro escalón, **el costo es progresivo en el
ancho** y el kernel 5 queda como óptimo declarado.
**C-3 · NO DAÑO.** `vigente` ≥ 0,95 en ≥2 de 3.
**Legibilidad:** si llegan menos de 2 unidades a 26000, **NO EVALUABLE**.

---

## Lo que ninguna de las tres puede decir

Un solo idioma, un solo nivel, una sola arquitectura y una tarea sintética. Y B mide **acceso**, no
**uso**: que un token entre en la ventana no obliga al modelo a usarlo — de hecho el propio kernel 5
tiene el tap 4 (el artículo) adentro y se predice que no aporta.
