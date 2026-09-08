# Escalón 2 · `attnp`, la atención de la lectura con proyecciones propias

Escrito el 2026-09-08 con la campaña `at3` en vuelo. **No se aplica hasta que la campaña cierre**:
`tramo_abst.sh` sube `modelo.py` a cada VM en cada tramo, y agregar parámetros al árbol rompe la
reanudación de las tres unidades en curso.

## Por qué, con el número medido y no por argumento

`attn_concentracion.py` midió hoy que `attn_causal` con `q = k = v = x` **sí atiende**: 0,42-0,47 en
la propia posición, 9,6-11,3 posiciones efectivas de 24, perfil plano de d1 a d23 y dependiente del
contenido. La sospecha de que fuera una identidad disfrazada quedó refutada.

Lo que **no** puede hacer, y es lo que justifica el escalón:

1. **No puede aprender a qué atender.** La similitud sale de la geometría de las embeddings, que se
   optimiza para otra cosa. No hay un solo peso entre `x` y el softmax.
2. **La brecha de 3,13 hacia la propia posición no es aprendida.** Sale de `|x|² / sqrt(D)` con la
   norma que le deja el LayerNorm (6,09). El gradiente puede mover la ganancia de `ln1` —o sea la
   temperatura global— pero no la forma de la distribución.
3. Y la razón de selectividad de cuatro controles con N = 600 es **0,94-1,05**: acceso global
   presente y sin discriminar (enmienda 2, `077cf358`).

Ningún transformer real lee su memoria con similitud cruda. Si el objetivo es igual condición que
los grandes, hay que pagar las proyecciones.

## El diseño, con la propiedad que a `attn` le falta

**`wqr` / `wkr` / `wvr` (D×D por bloque), inicializadas en la IDENTIDAD.** No en glorot.

Con identidad, `attnp` en el paso 0 es **exactamente** `attn`, bit a bit. Eso le da la propiedad que
`lat2` tiene sobre `pre` con `convq` en [1,0,...,0] y que **hoy a `attn` le falta sobre `pre`**: la
condición nueva **contiene a la vieja como caso particular**, así que no puede ser estructuralmente
peor y todo lo que aparezca lo fue a buscar el gradiente. Sin eso, un resultado peor no se puede
leer — que es el defecto exacto que hundió a `post` el 22-ago y el que el §5 del prereg del escalón
1 tuvo que declarar como límite.

Glorot rompería esa propiedad y además arrancaría la atención en casi uniforme: las mismas
proyecciones aleatorias medidas hoy dan brecha de logits 0,01 y 23,1 posiciones efectivas de 24, o
sea el modelo empezaría promediando todo el pasado.

### Contabilidad, que hay que verificar al aplicar y no citar de memoria

Con D = 128 y NB = 4: **3 × D² × NB = 196.608 en el árbol**, de los cuales **49.152 efectivos**
(sólo el bloque 0 recibe la lectura). Se instancian en los cuatro bloques por el mismo criterio que
`convq` y `abst`: que el árbol no cambie de forma entre condiciones.

Contado del árbol hoy y no citado: **865.779**, no los 865.395 que dice el comentario de `convq` en
`modelo.py` —ese número es anterior al bit de pertenencia y al slot nulo, que suman 384—. Con las
proyecciones queda en **1.062.387**, o sea **+22,7 % en el árbol y +5,7 % efectivo**. La regla de
este mismo párrafo se aplicó a sí misma en la primera versión de este documento, que citó el número
viejo.

**Y sale el mismo control gratis que dio `convq`:** las proyecciones de los bloques 1-3 tienen
gradiente cero garantizado, así que son la trayectoria del weight decay puro. Al cerrar, cualquier
diferencia entre el bloque 0 y los otros tres es gradiente y no decay, sin simular nada. Acá importa
más que en `convq`, porque el atractor del decay sobre una identidad es la matriz nula y una
proyección atenuada se leería como aprendizaje.

## El parche

`modelo.py`, en `init_params`, dentro del `dict` de cada bloque (junto a `convq`):

```python
    # Proyecciones propias para la atencion de la LECTURA (`--donde attnp`, escalon 2). En
    # IDENTIDAD, no en glorot: asi `attnp` en el paso 0 es EXACTAMENTE `attn` y la contiene como
    # caso particular, que es la propiedad que a `attn` le falta sobre `pre`. Existen en los cuatro
    # bloques para que el arbol no cambie de forma; sólo la lectura del bloque 0 las usa, y las
    # otras tres son la trayectoria del weight decay con gradiente cero.
    "wqr": jnp.eye(D), "wkr": jnp.eye(D), "wvr": jnp.eye(D),
```

`modelo.py`, junto a `attn_causal`:

```python
def attn_causal_proy(blk, x):
    """Como `attn_causal` pero con `wqr`/`wkr`/`wvr` propias. Ver PLAN_ESCALON2_ATTNP.md."""
    T, D = x.shape
    q, k, v = x @ blk["wqr"], x @ blk["wkr"], x @ blk["wvr"]
    sim = (q @ k.T) / jnp.sqrt(D)
    sim = jnp.where(jnp.tril(jnp.ones((T, T), bool)), sim, -1e9)
    return jax.nn.softmax(sim, -1) @ v
```

`modelo.py`, en `tronco`, una rama más junto a la de `attn`:

```python
        elif lectura is not None and i == bloque and donde == "attnp":
            h = h + lectura(jax.vmap(attn_causal_proy, in_axes=(None, 0))(blk, ln(blk["ln1"], h)))
```

`entrenar.py`, línea 530: agregar `"attnp"` a los `choices` de `--donde`.

## Verificaciones antes de lanzar nada

1. **`attnp` ≡ `attn` en el paso 0**, bit a bit, sobre un checkpoint recién inicializado. Si no da
   cero exacto, la identidad está mal puesta y la propiedad de contención no existe.
2. Contabilidad de parámetros contada del árbol, no de este documento.
3. `chequeo_padding.py` sigue abriendo la compuerta.
4. Smoke de 6 pasos en CPU antes de tocar una T4, como hoy con `attn`.

## Lo que sigue faltando para el nombre

Esto arregla la diferencia 1 (la query de lectura) y **no toca la 2 ni la 3**, que son las que
deciden: el tronco sigue siendo `delta_mixer` recurrente, y el vocabulario sigue cerrado. Medido
hoy: de los 242 tokens, **158 son respuestas legales** contra un archivo de a lo sumo 40 entradas,
así que margen para inventar hay; lo que falta es la FORMA, porque la respuesta es **un solo token**
y la alucinación que importa en los grandes es composicional.

**Nada que conserve el tronco recurrente o el vocabulario cerrado se llama «de Frontera».**
