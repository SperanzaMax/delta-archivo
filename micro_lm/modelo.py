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
                  "ord": glorot(ks[6], (N_TURNOS, D))}}
    for i in range(NB):
        b = 7 + i * 8
        p["blocks"].append({
            "ln1": {"g": jnp.ones(D), "b": jnp.zeros(D)},
            "ln2": {"g": jnp.ones(D), "b": jnp.zeros(D)},
            "conv": glorot(ks[b], (3, D)),
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


def tronco(params, x, lectura=None, bloque=0):
    """Pasa la secuencia por los bloques; si hay `lectura`, la inyecta en `bloque`."""
    h = params["emb"][x]
    for i, blk in enumerate(params["blocks"]):
        if lectura is not None and i == bloque:
            h = h + lectura(ln(blk["ln1"], h))
        h = h + jax.vmap(delta_mixer, in_axes=(None, 0))(blk, conv3(blk["conv"], ln(blk["ln1"], h)))
        h2 = ln(blk["ln2"], h)
        h = h + jax.nn.gelu(h2 @ blk["m1"]["w"] + blk["m1"]["b"]) @ blk["m2"]["w"] + blk["m2"]["b"]
    return h


def escribir(params, sesiones, cortes):
    """Procesa cada sesion por separado y archiva un vector por enunciado.

    sesiones: (B, S, T) tokens — S sesiones independientes, estado reseteado entre ellas.
    cortes:   (B, S, E) indice del ultimo token de cada enunciado (-1 = no hay enunciado).
    Devuelve (B, S*E, D): el archivo, en orden de escritura.
    """
    B, S, T = sesiones.shape
    E = cortes.shape[-1]
    entradas = []
    for s in range(S):
        h = tronco(params, sesiones[:, s, :])                    # (B, T, D)
        idx = jnp.clip(cortes[:, s, :], 0, T - 1)
        entradas.append(jnp.take_along_axis(h, idx[:, :, None].repeat(h.shape[-1], 2), axis=1))
    return jnp.concatenate(entradas, axis=1)                     # (B, S*E, D)


def responder(params, archivo, turnos, consulta, mask_arch, bloque=0):
    """Lee el archivo mientras procesa la consulta. Devuelve logits (B, T, V)."""
    a = params["arch"]
    ak = archivo @ a["kw"] + a["ord"][turnos]
    av = archivo @ a["vw"]
    penal = jnp.where(mask_arch, 0.0, -1e9)[:, None, :]          # entradas vacias no compiten

    def lectura(h):
        q = h @ a["qr"]
        sim = jnp.einsum("btd,bnd->btn", q, ak) / jnp.sqrt(h.shape[-1]) + penal
        return jnp.einsum("btn,bnd->btd", jax.nn.softmax(sim, -1), av) @ a["wo"]

    h = tronco(params, consulta, lectura, bloque)
    return ln(params["ln_f"], h) @ params["head"]["w"] + params["head"]["b"]


def contar(params):
    return sum(x.size for x in jax.tree_util.tree_leaves(params))
