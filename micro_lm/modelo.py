"""MICRO-LM · la arquitectura: regla delta + archivo persistente co-entrenado + sello de orden.

Autocontenido a proposito. El harness de Ligamento (`telar-ligamento/src/modelos.py`) esta congelado
por pre-registro y tiene D y NB fijos; esto necesita otro tamaño y otro vocabulario, asi que se
reimplementa la misma arquitectura en vez de tocar aquello.

Lo que hereda del brazo interno, ya medido:
  · el archivo se lee por softmax e se INYECTA TEMPRANO, en el bloque 0 (E-I1: no habilita, pero
    acelera 4x y sube el techo; E-I2: temprano 0,7275 vs tardio 0,3827).
  · la clave archivada lleva un SELLO DE ORDEN co-entrenado (E-I3: 0,4570 -> 0,9956; E-I3d: con el
    sello el modelo compara turnos, sin el se queda en el azar).
  · el archivo se baraja, para que la posicion en el tensor no codifique el rol.

POLITICA DE ESCRITURA, declarada: se archiva **un vector por enunciado**, tomado en la posicion de su
ultimo token. «Un hecho dicho = una entrada». Es la politica mas simple que existe y deja para
despues la pregunta de QUE conviene guardar (la eviction sorpresa-gated de VIGIA-03).
"""
from functools import partial

import jax
import jax.numpy as jnp


def glorot(key, shape):
    lim = jnp.sqrt(6.0 / sum(shape))
    return jax.random.uniform(key, shape, minval=-lim, maxval=lim)


def init_params(seed, V, D=192, NB=6, N_TURNOS=64):
    ks = jax.random.split(jax.random.PRNGKey(seed), 4 + NB * 8)
    p = {"emb": glorot(ks[0], (V, D)) * 0.5, "blocks": [],
         "ln_f": {"g": jnp.ones(D), "b": jnp.zeros(D)},
         "head": {"w": glorot(ks[1], (D, V)), "b": jnp.zeros(V)},
         "arch": {"kw": glorot(ks[2], (D, D)), "vw": glorot(ks[3], (D, D)),
                  "qr": glorot(ks[4], (D, D)), "wo": glorot(ks[5], (D, D)),
                  "ord": glorot(ks[6], (N_TURNOS, D))},
         # Cabeza de abstencion SEPARADA (2026-08-18): un escalar por posicion, con proyeccion
         # propia. Existe siempre en los params —asi el arbol no cambia de forma entre condiciones y
         # los checkpoints son intercambiables— pero sólo la usa `--abst cabeza`; en las otras dos
         # condiciones no entra en la perdida y queda en su valor inicial.
         # Arranca en cero (no glorot): con una sola unidad de salida no hay simetria que romper, y
         # asi el logit inicial es 0 -> sigma = 0,5, sin sesgo a favor ni en contra de abstenerse.
         "abst": {"w": jnp.zeros((D, 1)), "b": jnp.zeros(1)}}
    for i in range(NB):
        b = 7 + i * 8
        p["blocks"].append({
            "ln1": {"g": jnp.ones(D), "b": jnp.zeros(D)},
            "ln2": {"g": jnp.ones(D), "b": jnp.zeros(D)},
            "conv": glorot(ks[b], (3, D)),
            # Conv PROPIA para la query de la lectura (2026-08-24, condicion `lat2`). Existe siempre
            # en los params —mismo criterio que la cabeza `abst`: el arbol no cambia de forma entre
            # condiciones y los checkpoints siguen siendo intercambiables— pero sólo la usa
            # `--donde lat2`; en las demas no entra en la perdida y queda en su valor inicial.
            #
            # Arranca en [1, 0, 0], o sea `conv3(convq, z) == z` exactamente. Eso le da a `lat2` una
            # propiedad que ninguna condicion anterior tuvo: **contiene a `pre` como caso
            # particular**, asi que no puede ser estructuralmente peor, y cualquier contexto que
            # aparezca es contexto que el modelo fue A BUSCAR y no que le vino impuesto por el
            # diseño.
            #
            # Contabilidad exacta, porque el compromiso del 22-ago decia «384 params» y el numero
            # completo es otro: `convq` se instancia en los CUATRO bloques para que el arbol no
            # cambie de forma (+1.536 params, 863.859 -> 865.395), pero la lectura entra en el
            # bloque 0 y sólo esa se usa. **384 params efectivos (0,044 %), 1.536 en el arbol
            # (0,178 %).**
            #
            # Los otros tres NO quedan clavados en [1,0,0], y esto lo escribi mal la primera vez: el
            # optimizador es `adamw(weight_decay=0.01)`, que decae TODO parametro tenga gradiente o
            # no. Medido a 60 pasos: bloque 0 se movio 0,014011 (gradiente) y los bloques 1-3
            # exactamente 0,000235 cada uno, sólo en el tap 0, que es decay puro.
            #
            # Sale gratis un control que no hay que construir: **los `convq` de los bloques 1-3 son
            # la trayectoria del weight decay con gradiente CERO garantizado.** Al cerrar la campania,
            # cualquier diferencia entre el convq del bloque 0 y los de 1-3 es gradiente y no decay,
            # sin tener que simular nada ni suponer una tasa. Es lo que hace falsable el riesgo
            # declarado en el §7 del prereg —«lat2 puede quedarse en pre»—, porque el atractor sin
            # gradiente no es [1,0,0] sino [0,0,0], y sin este control un convq atenuado se leeria
            # como aprendizaje cuando podria ser decay.
            "convq": jnp.stack([jnp.ones(D), jnp.zeros(D), jnp.zeros(D)]),
            "wq": glorot(ks[b + 1], (D, D)), "wk": glorot(ks[b + 2], (D, D)),
            "wv": glorot(ks[b + 3], (D, D)), "beta": jnp.zeros(D) + 0.5,
            "m1": {"w": glorot(ks[b + 4], (D, 4 * D)), "b": jnp.zeros(4 * D)},
            "m2": {"w": glorot(ks[b + 5], (4 * D, D)), "b": jnp.zeros(D)},
        })
    return p


def ln(p, x):
    m = x.mean(-1, keepdims=True)
    v = x.var(-1, keepdims=True)
    return (x - m) / jnp.sqrt(v + 1e-5) * p["g"] + p["b"]


def conv3(w, x):
    """Depthwise causal, kernel 3."""
    x0 = x
    x1 = jnp.pad(x, ((0, 0), (1, 0), (0, 0)))[:, :-1, :]
    x2 = jnp.pad(x, ((0, 0), (2, 0), (0, 0)))[:, :-2, :]
    return x0 * w[0] + x1 * w[1] + x2 * w[2]


def delta_mixer(blk, x):
    """Regla delta: S <- S + beta * (v - S q) k^T, leida con la misma q."""
    q, k, v = x @ blk["wq"], x @ blk["wk"], x @ blk["wv"]
    q = q / (jnp.linalg.norm(q, axis=-1, keepdims=True) + 1e-6)
    k = k / (jnp.linalg.norm(k, axis=-1, keepdims=True) + 1e-6)
    beta = jax.nn.sigmoid(blk["beta"])

    def paso(S, ent):
        qi, ki, vi, bi = ent
        pred = S @ qi
        S = S + jnp.outer(bi * (vi - S @ ki), ki)
        return S, pred

    D = x.shape[-1]
    S0 = jnp.zeros((D, D))
    _, out = jax.lax.scan(paso, S0, (q, k, v, jnp.broadcast_to(beta, k.shape)))
    return out


def tronco(params, x, lectura=None, bloque=0, donde="pre"):
    """Pasa la secuencia por los bloques; si hay `lectura`, la inyecta en `bloque`.

    `donde` decide en que punto del bloque entra la lectura, y es el eje del experimento del 22-ago:

      · "pre"  — ANTES de la conv y del mixer, sobre `ln1(h)`. Es lo que se venia haciendo. Ahi
        `h = emb[x]` todavia, asi que la query es `ln(emb[token]) @ qr`: **funcion pura del token de
        su posicion**, sin una sola operacion de contexto delante. De ahi sale que el modelo no pueda
        formar una query conjunta entidad x relacion y consulte el archivo token por token
        (`SMOKE_EMPATE_20260821.md`), que es el mecanismo al que el round-trip le atribuyo
        `err_identidad` (colision de clave: la relacion sola matchea a todas las que la comparten).

      · "post" — DESPUES de la conv y del mixer del MISMO bloque, sobre `ln2(h)`. La inyeccion sigue
        siendo temprana (quedan 3,5 bloques de computo aguas abajo, y E-I1/E-I2 penalizaban la
        inyeccion en capas PROFUNDAS, no esta), pero la query ya vio el pasado causal de la secuencia
        y puede depender de la entidad y de la relacion a la vez.

      · "lat"  — la inyeccion queda donde `pre` la tiene y la query se forma sobre
        `conv3(blk["conv"], ln1(h))`, o sea el token y los dos anteriores, con la conv COMPARTIDA con
        el mixer. Cerrada el 24-ago: disuelve la colision de clave entera (`err_identidad` 0,0000 en
        las tres semillas) pero cobra el marcador de orden, por el acoplamiento de esa conv.

      · "lat2" — igual que `lat` pero con `blk["convq"]` PROPIA, inicializada en [1,0,0]. Contiene a
        `pre` como caso particular.

    La simetria entre las dos condiciones originales es exacta: cada una reusa el LayerNorm que ya
    precede a la operacion siguiente (`ln1` para el mixer, `ln2` para el MLP), asi que ninguna
    estrena parametros que la otra no tenga. El arbol de params no cambia de forma y los checkpoints
    siguen siendo intercambiables — `convq` mantiene esa propiedad porque existe en TODAS las
    condiciones y sólo `lat2` la lee.
    """
    h = params["emb"][x]
    for i, blk in enumerate(params["blocks"]):
        if lectura is not None and i == bloque and donde == "pre":
            h = h + lectura(ln(blk["ln1"], h))
        elif lectura is not None and i == bloque and donde == "lat":
            # CAMINO LATERAL (2026-08-22, tarde). La inyeccion queda EXACTAMENTE donde `pre` la
            # tiene —antes de la conv y del mixer, sumada a `h`— y lo unico que cambia es de que se
            # forma la query: `conv3(ln1(h))` en vez de `ln1(h)`.
            #
            # Sale del informe de la mañana. `post` movio dos cosas a la vez —la forma de la query y
            # el punto donde la lectura entra al computo— y la segunda resulto devastadora (acierto
            # 0,97 -> 0,39, plano desde el paso 4000), asi que la hipotesis quedo sin probar. Aca los
            # factores se separan: la lectura entra a tiempo, y la query igual puede depender de las
            # dos posiciones anteriores, que es lo que hace falta para combinar entidad y relacion
            # —en la forma canonica del idioma caen a distancia 2 (`el director de museo es X`)—.
            #
            # La conv es la MISMA del bloque, asi que no estrena parametros: las tres condiciones
            # tienen los mismos 863.859. Y sigue siendo contexto LOCAL: la conv de kernel 3 no ve
            # mas alla de dos tokens atras, con lo cual no reintroduce la dependencia global que en
            # `post` venia del mixer.
            h = h + lectura(conv3(blk["conv"], ln(blk["ln1"], h)))
        elif lectura is not None and i == bloque and donde == "lat2":
            # CAMINO LATERAL CON CONV PROPIA (2026-08-24). Corrige el defecto de diseño de `lat`
            # diagnosticado en `DIAGNOSTICO_CONV_COMPARTIDA_20260822.md`.
            #
            # `lat` usa la MISMA `blk["conv"]` para la query de la lectura y para el mixer. Lo escribi
            # en el prereg como virtud —«no estrena parametros»— y la simetria es real, pero acopla
            # dos cosas con balances OPUESTOS: el mixer quiere el mix que le sirve a la regla delta,
            # y la query quiere mucho contexto para formar entidad x relacion (distancia 2) y POCO
            # para no diluir el marcador temporal. En `cual era antes el precio de banco ?` el token
            # `antes` cae fuera de la ventana de la conv, y encima en `lat` la query en su posicion
            # pasa a ser `conv3(cual, era, antes)` en vez de `antes` puro. La conv da la query
            # conjunta y COBRA el marcador de orden — medido: `anterior` en 0,3798 en w3_s2, contra
            # 0,8125 en su gemela `pre` (INFORME_CAMINO_LATERAL_20260824.md, §5).
            #
            # Con `convq` propia e inicializada en [1,0,0], `lat2` arranca siendo EXACTAMENTE `pre` y
            # el modelo decide por gradiente cuanto contexto quiere. 3 x D = 384 params, 0,044 %.
            h = h + lectura(conv3(blk["convq"], ln(blk["ln1"], h)))
        h = h + jax.vmap(delta_mixer, in_axes=(None, 0))(blk, conv3(blk["conv"], ln(blk["ln1"], h)))
        if lectura is not None and i == bloque and donde == "post":
            h = h + lectura(ln(blk["ln2"], h))
        h2 = ln(blk["ln2"], h)
        h = h + jax.nn.gelu(h2 @ blk["m1"]["w"] + blk["m1"]["b"]) @ blk["m2"]["w"] + blk["m2"]["b"]
    return h


def escribir(params, sesiones, cortes):
    """Procesa las sesiones y archiva un vector por enunciado.

    sesiones: (B, S, T) tokens — S sesiones independientes, estado reseteado entre ellas.
    cortes:   (B, S, E) indice del ultimo token de cada enunciado.
    Devuelve (B, S*E, D): el archivo, en orden de escritura.

    Las S sesiones se apilan como batch y pasan por UN solo forward. Son independientes por
    construccion —el estado se resetea entre ellas—, asi que hacerlas de a una era gastar S veces el
    scan secuencial de la regla delta, que en GPU es lo caro. Medido: con 4 sesiones, de a una son
    ~0,72 s por paso; apiladas, la parte secuencial baja de 4·T a T.
    """
    B, S, T = sesiones.shape
    E, D = cortes.shape[-1], params["emb"].shape[-1]
    h = tronco(params, sesiones.reshape(B * S, T))               # (B*S, T, D) — un solo scan
    idx = jnp.clip(cortes, 0, T - 1).reshape(B * S, E)
    ent = jnp.take_along_axis(h, jnp.broadcast_to(idx[:, :, None], (B * S, E, D)), axis=1)
    return ent.reshape(B, S * E, D)


def responder(params, archivo, turnos, consulta, mask_arch, bloque=0, donde="pre"):
    """Lee el archivo mientras procesa la consulta. Devuelve logits (B, T, V)."""
    a = params["arch"]
    ak = archivo @ a["kw"] + a["ord"][turnos]
    av = archivo @ a["vw"]
    penal = jnp.where(mask_arch, 0.0, -1e9)[:, None, :]          # entradas vacias no compiten

    def lectura(h):
        q = h @ a["qr"]
        sim = jnp.einsum("btd,bnd->btn", q, ak) / jnp.sqrt(h.shape[-1]) + penal
        return jnp.einsum("btn,bnd->btd", jax.nn.softmax(sim, -1), av) @ a["wo"]

    h = tronco(params, consulta, lectura, bloque, donde)
    return ln(params["ln_f"], h) @ params["head"]["w"] + params["head"]["b"]


def responder_con_abst(params, archivo, turnos, consulta, mask_arch, bloque=0, donde="pre"):
    """Igual que `responder`, mas el logit de la cabeza de abstencion. Devuelve (logits, a).

    `a` es (B, T): un escalar por posicion, con proyeccion propia desde el MISMO estado final que
    alimenta el softmax de vocabulario. La idea que prueba —ver `PREREG_CABEZA_ABSTENCION.md`— es que
    «¿esta?» y «¿que valor?» son dos decisiones de naturaleza distinta (binaria y balanceada una,
    1-entre-100 la otra) y hoy compiten por la misma masa de probabilidad, con el vector de `NOSE`
    tres veces mas corto que el de un valor (norma 0,367 contra 1,011, medido el 17-ago).
    """
    a_p = params["arch"]
    ak = archivo @ a_p["kw"] + a_p["ord"][turnos]
    av = archivo @ a_p["vw"]
    penal = jnp.where(mask_arch, 0.0, -1e9)[:, None, :]

    def lectura(h):
        q = h @ a_p["qr"]
        sim = jnp.einsum("btd,bnd->btn", q, ak) / jnp.sqrt(h.shape[-1]) + penal
        return jnp.einsum("btn,bnd->btd", jax.nn.softmax(sim, -1), av) @ a_p["wo"]

    h = tronco(params, consulta, lectura, bloque, donde)
    hn = ln(params["ln_f"], h)
    logits = hn @ params["head"]["w"] + params["head"]["b"]
    a = (hn @ params["abst"]["w"] + params["abst"]["b"])[..., 0]
    return logits, a


def contar(params):
    return sum(x.size for x in jax.tree_util.tree_leaves(params))
