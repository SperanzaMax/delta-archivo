# ¿La diversidad de formas compra abstención en un modelo REAL? · congelado ANTES de correr

**2026-09-02.** Es el escalón que le falta al proyecto entero. Todo lo demás está medido en un
micro-LM de 3,5 MB y nadie lo va a creer transferible hasta que se vea en un modelo que no sea
nuestro.

## De dónde sale

Hoy se midió, en el micro-LM, que entrenar con **dos formas de pregunta** y **sin tocar el kernel**
lleva `nose_rel` en la forma directa de **0,5850 · 0,6090 · 0,7349** (control con una sola forma) a
**1,0000**. Y la sonda mostró que la búsqueda **sigue siendo ciega**, con sensibilidad `0,000000`
exacto a la relación. O sea, la abstención se aprendió por otra vía.

Si eso vale en general, es lo más accionable del proyecto, porque **no le pide a nadie que cambie su
arquitectura**.

## Montaje, con todo lo que se midió antes en vez de suponerse

**Modelo** `state-spaces/mamba-370m-hf`, 371.516.416 parámetros, fine-tune completo.

**El alcance real está MEDIDO, no leído del config.** El kernel nominal es 4, pero el tap más viejo
vale **cero exacto en las 48 capas** y la intervención confirma que la conv se mueve en tres
posiciones consecutivas y no cuatro. **El alcance real es 2.**

**Las distancias están medidas en tokens del BPE**, no en palabras, y son **fijas** porque todas las
piezas del vocabulario son de un solo token, verificado sobre las 363 combinaciones.

| forma | texto | d(relación) | d(entidad) | la relación |
|---|---|---:|---:|---|
| `directa` | What is the {r} of {e}? | **3** | 1 | **AFUERA** |
| `invertida` | For {e}, what is the {r}? | **1** | 6 | ADENTRO |
| `lejana` | What is the {r} that {e} has? | **4** | 2 | AFUERA |

Es exactamente la geometría del micro-LM.

**Tarea.** Un contexto de cuatro hechos `The {rel} of {ent} is {val}.` y una pregunta. La respuesta es
**un solo token**, que hace la métrica exacta sin juez ni parser. Tres tipos, los mismos de siempre:
`vigente`, `nose_ent` (la entidad no aparece) y `nose_rel` (la entidad **sí** aparece y la relación
que se pregunta nunca se dijo, que es la que parece una alucinación real).

**Condiciones**, tres semillas cada una, y **todas se evalúan en la forma `directa`**, que es donde la
relación es invisible para la query.

- **`una`** entrena sólo con `directa`.
- **`dos`** entrena con `directa` + `invertida`.
- **`ciega`** entrena con `directa` + `lejana`, o sea **con diversidad y sin ver nunca la relación**.
  Es el control que adjudica, el mismo que está corriendo hoy en el micro-LM.

## Criterios, escritos antes del dato

Y escritos preguntándome primero **si la intervención funcionara perfecto, ¿esta métrica se mueve?**,
que es la pregunta que hoy me faltó hacer dos veces.

- **R-0 · BLOQUEANTE.** `vigente` ≥ **0,90** en las tres condiciones. Si el modelo no aprendió a
  contestar lo que sí está, **nada de lo demás se lee**.
- **R-1 · PRINCIPAL.** `nose_rel` en la forma directa: **`dos` − `una` ≥ 0,15** en **≥2 de 3**
  semillas.
- **R-2 · NO DAÑO.** `falsa_abst` ≤ **0,10** en `dos`, en ≥2 de 3. Sin esto, R-1 se podría cumplir
  simplemente porque el modelo se volvió más callado.
- **R-3 · ADJUDICA.** Comparando `ciega` contra `una` y `dos`:
  - `ciega` ≈ `dos` → gana **la diversidad sola**;
  - `ciega` ≈ `una` → gana **ver la relación al menos a veces**.
- **BASELINE.** Se mide en el paso 0 y se informa. Un modelo preentrenado ya sabe decir «unknown» y
  esa parte **no nos la podemos atribuir**.

## Lo que este experimento NO puede decir

- **Es recuerdo en contexto, no memoria persistente entre secuencias.** Es una simplificación
  deliberada, para medir una cosa por vez. La transferencia del versionado queda afuera.
- **Es un solo modelo y una sola familia.** Mamba, no Gated DeltaNet, aunque compartan el kernel 4.
- **El vocabulario son apellidos ingleses reales.** Se eligieron por ser neutros y de un token, pero
  el modelo tiene alguna estadística sobre ellos. Lo que no puede tener es el **hecho** inventado.
- **No mide cuánta exposición hace falta.** Si gana `dos`, queda abierto si alcanza con el 10 % de las
  consultas o hace falta la mitad.
