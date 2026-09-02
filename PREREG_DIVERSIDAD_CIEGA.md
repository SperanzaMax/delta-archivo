# ¿Arregla la DIVERSIDAD o arregla VER la relación? · congelado ANTES de correr

**2026-09-02, mediodía.** Sale de un resultado que no estaba previsto y que apareció hoy en
`cf3_s1`, la primera semilla del cruce en cerrar.

## El hecho que lo motiva

Entrenando con **dos formas de pregunta mezcladas** (`directa` e `invertida`) y **sin tocar el
kernel**, `nose_rel` en la forma `directa` pasó de **0,5850 · 0,6090 · 0,7349** (control `v3`, misma
arquitectura, una sola forma) a **1,0000**, con `vigente` 1,0000 y `falsa_abst` 0,0000.

Y la sonda de sensibilidad sobre esa misma unidad dice que **la búsqueda sigue siendo ciega**. En la
forma `directa` la sensibilidad a la relación es **0,000000 exacto**, 6 celdas de 6.

O sea, el modelo se abstiene perfecto mientras su búsqueda no puede ver lo que se le pregunta. La
abstención viaja por otro lado.

## Las dos explicaciones que hay que separar

- **(V) Ver la relación alguna vez.** En la forma `invertida` la relación cae a distancia 1, adentro
  de la ventana. Puede que el modelo necesite verla **en algunas consultas** para construir la
  representación que después usa aguas abajo en todas.
- **(D) Diversidad a secas.** Puede que lo que importe sea simplemente que la pregunta tenga más de
  una forma, sin importar dónde caiga la relación, porque eso impide memorizar una plantilla y obliga
  a representar la consulta.

## El diseño que las separa

Unidades **`cl3_s0/s1/s2`**, idénticas a `cf3` salvo por las formas: **`directa` y `lejana`**.

| forma | texto | d(relación) | d(entidad) |
|---|---|---:|---:|
| `directa` | cual es `<art>` `<sust>` de `<ent>` ? | **3** | 1 |
| `lejana` | cual es `<art>` `<sust>` que tiene `<ent>` ? | **4** | 1 |

Con kernel 3 el alcance es 2, así que **la relación queda afuera en las DOS formas**, y la entidad
adentro en las dos. **Hay diversidad y no hay visión de la relación, nunca.** Es exactamente el
contraste que falta.

Las distancias están verificadas token a token en `chequeo_formas_q.py` y la sensibilidad a la
relación medida en las dos formas es **0,000000**, así que la condición del diseño no es supuesta.

## Criterios, escritos antes del dato

- **C-1 · PRINCIPAL, y adjudica entre las dos explicaciones.** `nose_rel` en la forma `directa`,
  en ≥2 de 3 semillas:
  - **≥ 0,90** → gana **(D)**, la diversidad alcanza por sí sola, y el papel de la ventana en la
    abstención queda reducido a un efecto de la distribución de entrenamiento.
  - **≤ 0,80** (o sea en el rango del control `v3`) → gana **(V)**, hay que ver la relación al menos
    a veces, y la ventana vuelve al centro.
  - entre 0,80 y 0,90 → **no adjudica**, y se informa así en vez de forzar una lectura.
- **C-2 · NO DAÑO.** `vigente` ≥ 0,90 en ≥2 de 3 y en las dos formas. Si `lejana` no se aprende, el
  contraste no se lee.
- **Legibilidad:** menos de 2 unidades a 26000 → **NO EVALUABLE**.

## Lo que no puede decir

No mide **cuánta** exposición hace falta. Si gana (V), queda abierto si alcanza con el 10 % de las
consultas o hace falta la mitad, y eso es un barrido que no se corre acá. Y sigue siendo un idioma
sintético con dos plantillas, no texto natural.
