"""Delta + archivo recuperable, en su mínima expresión.

Extiende el harness congelado de Ligamento (src/modelos.py) SIN tocarlo: importa todo lo que
puede y solo reimplementa el mixer para añadir un cache exacto de presupuesto `w`.

Diseño (deliberadamente mínimo, una sola decisión de diseño por pieza):
  - Escritura : top-w por m_t = beta_t * ||e_t||  (el residuo comprometido al estado).
                Es el criterio de HOLA (2607.02303) y la señal de CENTINELA-01.
  - Lectura   : softmax de q_t sobre el buffer, con RMSNorm desacoplado en q y k.
                HOLA reporta que sin ese normalizador el softmax promedia en vez de recuperar.
  - Mezcla    : y = y_estado + lambda_h * y_cache, con lambda_h escalar por cabeza, init 0
                (arranca siendo delta puro y tiene que APRENDER a usar el archivo).

El buffer solo contiene posiciones < t por construcción del scan, así que la causalidad se
respeta sin máscara adicional. Las ranuras vacías se enmascaran con score < 0.

kind soportados: "dcache" (4 cabezas delta + cache). El presupuesto w se pasa aparte.
"""
import sys, os
sys.path.insert(0, os.path.expanduser("~/Documentos/Nuevo Transformer/telar-ligamento/src"))

import numpy as np
import jax, jax.numpy as jnp
from datos import VOCAB
import modelos as M
from modelos import D, H, DH, NB, FFN_HID, ln, conv3, split_heads, l2n, rmsn, glorot

NEG = -1e9


def init_params(seed, w, vocab=VOCAB):
    """Params del harness base + dos tensores nuevos por bloque, ambos sin consumo de PRNG:
    `lam` (mezcla por cabeza, init 0) y `gam` (RMSNorm-gamma del cache, init 1).
    Al no consumir claves PRNG, los pesos compartidos quedan bit a bit iguales a los de
    `modelos.init_params(seed, 'delta')` — el contraste con C2 no arrastra confound de init."""
    p = M.init_params(seed, "delta", vocab=vocab)
    for blk in p["blocks"]:
        blk["lam"] = jnp.zeros(H)
        blk["gam"] = jnp.ones(DH)
    p["_w"] = w
    return p


def _delta_cache_heads(blk, x, q, k, v, B, w):
    """Idéntico a modelos._delta_heads salvo por el archivo. Devuelve (B,H,T,DH)."""
    kn = l2n(jax.nn.silu(k)); qn = l2n(jax.nn.silu(q))
    beta = jax.nn.sigmoid(x @ blk["g_beta"]["w"] + blk["g_beta"]["b"])       # (B,T,H)
    tm = lambda a: a.transpose(2, 0, 1, 3) if a.ndim == 4 else a.transpose(1, 0, 2)

    gam = blk["gam"]
    rms_g = lambda z: rmsn(z) * gam            # RMSNorm-gamma desacoplado (lectura del cache)

    S0 = jnp.zeros((B, H, DH, DH))
    K0 = jnp.zeros((B, H, w, DH))
    V0 = jnp.zeros((B, H, w, DH))
    M0 = jnp.full((B, H, w), -1.0)             # score < 0 == ranura vacía

    def step(carry, inp):
        S, Kb, Vb, Mb = carry
        qt, kt, vt, bt = inp                                   # (B,H,DH) y (B,H)

        y_state = jnp.einsum("bhij,bhj->bhi", S, qt)           # leer antes de escribir

        # --- lectura del archivo (solo posiciones ya escritas) ---
        logits = jnp.einsum("bhd,bhwd->bhw", rms_g(qt), rms_g(Kb)) / np.sqrt(DH)
        logits = jnp.where(Mb >= 0.0, logits, NEG)
        empty = jnp.all(Mb < 0.0, axis=-1, keepdims=True)      # buffer vacío -> aporte nulo
        att = jnp.where(empty, 0.0, jax.nn.softmax(logits, -1))
        y_cache = jnp.einsum("bhw,bhwd->bhd", att, Vb)

        # --- escritura al estado (delta rule, sin cambios) ---
        pred = jnp.einsum("bhij,bhj->bhi", S, kt)
        err = vt - pred
        S2 = S + bt[..., None, None] * jnp.einsum("bhi,bhj->bhij", err, kt)

        # --- escritura al archivo: top-w por m = beta * ||err|| ---
        m = bt * jnp.linalg.norm(err, axis=-1)                 # (B,H)
        j = jnp.argmin(Mb, axis=-1)                            # ranura mas debil
        m_min = jnp.take_along_axis(Mb, j[..., None], -1)[..., 0]
        win = (m > m_min)[..., None]                           # (B,H,1)
        oh = jax.nn.one_hot(j, w)                              # (B,H,w)
        sel = (oh * win).astype(Kb.dtype)[..., None]           # (B,H,w,1)
        Kb2 = Kb * (1 - sel) + sel * kt[:, :, None, :]
        Vb2 = Vb * (1 - sel) + sel * vt[:, :, None, :]
        Mb2 = jnp.where(oh * win[..., 0][..., None] > 0, m[..., None], Mb)

        y = y_state + blk["lam"][None, :, None] * y_cache
        return (S2, Kb2, Vb2, Mb2), y

    _, ys = jax.lax.scan(step, (S0, K0, V0, M0), (tm(qn), tm(kn), tm(v), tm(beta)))
    return ys.transpose(1, 2, 0, 3)


def mixer(blk, x, w):
    B, T, _ = x.shape
    q = split_heads(x @ blk["q"]); k = split_heads(x @ blk["k"]); v = split_heads(x @ blk["v"])
    y = _delta_cache_heads(blk, x, q, k, v, B, w)
    y = rmsn(y)                                                # INVARIANTE §5
    y = y.transpose(0, 2, 1, 3).reshape(B, T, D)
    return y @ blk["o"]


def forward(params, x, w):
    hx = params["emb"][x]
    for blk in params["blocks"]:
        hx = hx + mixer(blk, conv3(blk["conv"], ln(blk["ln1"], hx)), w)
        h2 = ln(blk["ln2"], hx)
        hx = hx + jax.nn.gelu(h2 @ blk["m1"]["w"] + blk["m1"]["b"]) @ blk["m2"]["w"] + blk["m2"]["b"]
    return ln(params["ln_f"], hx) @ params["head"]["w"] + params["head"]["b"]
