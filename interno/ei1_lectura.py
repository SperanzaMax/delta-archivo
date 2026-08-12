"""E-I1 — LECTURA APRENDIDA SOBRE ESCRITURA ORACULO (PROTOCOLO_INTERNO.md §4).

El archivo se llena por fuera con lo correcto; el modelo solo aprende a CONSULTARLO y a usar lo
recuperado. Aisla el obstaculo (a) del protocolo -- el gradiente no fluye por la seleccion top-k --
sin mezclarlo con la pregunta de que archivar.

El archivo tiene N entradas de las cuales solo L son las del ejemplo: el resto son DISTRACTORES
(pares de otras claves). Sin eso la recuperacion seria trivial -- con un archivo del tamano de la
consulta, traer todo equivale a resolver -- y no se estaria midiendo ninguna capacidad de consulta.

DOS FACTORES, 2x2:

  inyeccion   temprano (bloque 0) | tardio (bloque 3)
              Prediccion P-1, derivada de E2-b: TEMPRANO gana por un margen grande. Va CONTRA lo
              que hace todo pipeline RAG, que concatena lo recuperado al final.

  seleccion   densa (softmax sobre las N entradas, gradiente completo)
              topk  (solo las k mejores; el gradiente fluye por los pesos, NO por la seleccion)
              Prediccion P-2: densa >= topk. Si son equivalentes, el obstaculo que segun la
              literatura frena a todo el campo no es real a esta escala, y eso es el hallazgo.

PREDICCIONES, comprometidas antes del dato:
  P-1  acc(temprano) - acc(tardio) >= +0,15, promediando sobre seleccion.
  P-2  acc(densa) - acc(topk) >= 0 (direccional). Se reporta la magnitud.
  P-3  (bloqueante) la mejor condicion supera 0,30, contra el piso 0,0215 de E-I0. Si ninguna lo
       hace, el modelo NO aprende a formar consultas utiles y ese es el resultado del brazo.
"""
import json
import os
import sys
import time
from functools import partial

import numpy as np

sys.path.insert(0, "/home/maxi/Documentos/Nuevo Transformer/telar-ligamento/src")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jax
import jax.numpy as jnp
import optax
from datos import IGNORE, V_E001
from modelos import D, NB, conv3, glorot, ln, mixer

L = int(os.environ.get("L_CARGA", "6"))
R = L // 2
B = int(os.environ.get("BATCH", "32"))
N_ARCHIVO = int(os.environ.get("N_ARCHIVO", "64"))     # entradas totales; L son las del ejemplo
K_TOP = int(os.environ.get("K_TOP", "4"))
PASOS = int(os.environ.get("PASOS", "600"))
LR = 3e-3
KIND = "delta"
AZAR = 1.0 / V_E001.NV


def gen_lote(rng, B, L, r, voc=V_E001):
    """Devuelve (s3, y3, arch_k, arch_v): la consulta y el archivo ORACULO con distractores."""
    keys = np.argsort(rng.random((B, voc.NK)), axis=1)[:, :L]
    v1 = rng.integers(0, voc.NV, size=(B, L))
    v2 = (v1[:, :r] + 1 + rng.integers(0, voc.NV - 1, size=(B, r))) % voc.NV
    final = v1.copy()
    final[:, :r] = v2

    perm = np.argsort(rng.random((B, L)), axis=1)
    q = np.take_along_axis(keys, perm, axis=1)
    tgt = np.take_along_axis(final, perm, axis=1)

    s3 = np.full((B, L + 2), voc.PAD, dtype=np.int32)
    y3 = np.full((B, L + 2), IGNORE, dtype=np.int32)
    s3[:, 0] = voc.BOS
    s3[:, 1] = voc.SEP
    s3[:, 2:] = voc.K0 + q
    y3[:, 2:] = voc.V0 + tgt

    # --- archivo: las L entradas reales + (N-L) distractoras de claves NO consultadas ---
    n_dist = N_ARCHIVO - L
    ak = np.zeros((B, N_ARCHIVO), dtype=np.int32)
    av = np.zeros((B, N_ARCHIVO), dtype=np.int32)
    for b in range(B):
        libres = np.setdiff1d(np.arange(voc.NK), keys[b], assume_unique=False)
        dk = rng.choice(libres, size=n_dist, replace=False)
        dv = rng.integers(0, voc.NV, size=n_dist)
        ek = np.concatenate([keys[b], dk])
        ev = np.concatenate([final[b], dv])
        orden = rng.permutation(N_ARCHIVO)          # la posicion no delata cual es cual
        ak[b] = voc.K0 + ek[orden]
        av[b] = voc.V0 + ev[orden]
    return s3, y3, ak, av


def init_lector(seed):
    ks = jax.random.split(jax.random.PRNGKey(1000 + seed), 4)
    return {"wq": glorot(ks[0], (D, D)), "wo": glorot(ks[1], (D, D)),
            "wk": glorot(ks[2], (D, D)), "wv": glorot(ks[3], (D, D))}


def leer(lector, h, ek, ev, seleccion):
    """h (B,T,D) · ek/ev (B,N,D) embeddings del archivo -> (B,T,D) lo recuperado."""
    q = h @ lector["wq"]
    k = ek @ lector["wk"]
    v = ev @ lector["wv"]
    sim = jnp.einsum("btd,bnd->btn", q, k) / jnp.sqrt(D)
    if seleccion == "densa":
        w = jax.nn.softmax(sim, axis=-1)
        out = jnp.einsum("btn,bnd->btd", w, v)
    else:                                    # top-k: el gradiente NO pasa por la seleccion
        val, idx = jax.lax.top_k(sim, K_TOP)                       # (B,T,K)
        w = jax.nn.softmax(val, axis=-1)
        vk = jnp.take_along_axis(v[:, None, :, :],                  # (B,1,N,D)
                                 idx[..., None], axis=2)            # (B,T,K,D)
        out = jnp.einsum("btk,btkd->btd", w, vk)
    return out @ lector["wo"]


def forward(params, x, ak, av, cfg):
    hx = params["emb"][x]
    ek, ev = params["emb"][ak], params["emb"][av]
    for i, blk in enumerate(params["blocks"]):
        if i == cfg["bloque"]:
            hx = hx + leer(params["lector"], ln(blk["ln1"], hx), ek, ev, cfg["seleccion"])
        hx = hx + mixer(blk, conv3(blk["conv"], ln(blk["ln1"], hx)), KIND)
        h2 = ln(blk["ln2"], hx)
        hx = hx + jax.nn.gelu(h2 @ blk["m1"]["w"] + blk["m1"]["b"]) @ blk["m2"]["w"] + blk["m2"]["b"]
    return ln(params["ln_f"], hx) @ params["head"]["w"] + params["head"]["b"]


def loss_fn(params, x, y, ak, av, cfg):
    logits = forward(params, x, ak, av, cfg)
    mask = y >= 0
    yl = jnp.where(mask, y, 0)
    ce = optax.softmax_cross_entropy_with_integer_labels(logits, yl)
    return (ce * mask).sum() / mask.sum(), ((logits.argmax(-1) == yl) * mask).sum() / mask.sum()


def entrenar(bloque, seleccion, semilla, pasos=PASOS):
    from modelos import init_params
    cfg = {"bloque": bloque, "seleccion": seleccion}
    params = init_params(semilla, KIND)
    params["lector"] = init_lector(semilla)
    rng = np.random.default_rng(3000 + semilla)
    sched = optax.warmup_constant_schedule(0.0, LR, 100)
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(sched, weight_decay=0.01))
    state = opt.init(params)

    @partial(jax.jit, static_argnames="cfg_h")
    def paso(params, state, x, y, ak, av, cfg_h):
        c = {"bloque": cfg_h[0], "seleccion": cfg_h[1]}
        (l, a), g = jax.value_and_grad(loss_fn, has_aux=True)(params, x, y, ak, av, c)
        up, state = opt.update(g, state, params)
        return optax.apply_updates(params, up), state, l, a

    cfg_h = (bloque, seleccion)
    t0 = time.time()
    for s in range(1, pasos + 1):
        x, y, ak, av = gen_lote(rng, B, L, R)
        params, state, l, a = paso(params, state, jnp.array(x), jnp.array(y),
                                   jnp.array(ak), jnp.array(av), cfg_h)
        if s % 300 == 0:
            print(f"    [b{bloque}/{seleccion}/s{semilla}] paso {s:4d} loss {float(l):.4f} "
                  f"acc {float(a):.4f} ({time.time()-t0:.0f}s)", flush=True)

    ev_rng = np.random.default_rng(88000 + semilla)
    accs = []
    for _ in range(8):
        x, y, ak, av = gen_lote(ev_rng, B, L, R)
        _, a = loss_fn(params, jnp.array(x), jnp.array(y), jnp.array(ak), jnp.array(av), cfg)
        accs.append(float(a))
    return float(np.mean(accs))


def main():
    semillas = (0, 1, 2)
    print(f"E-I1 · L={L} · archivo N={N_ARCHIVO} ({L} reales + {N_ARCHIVO-L} distractores) · "
          f"k={K_TOP} · {PASOS} pasos · piso E-I0 = 0,0215\n", flush=True)
    salida = {}
    for bloque, etiqueta in ((0, "temprano"), (NB - 1, "tardio")):
        for seleccion in ("densa", "topk"):
            ms = [entrenar(bloque, seleccion, s) for s in semillas]
            salida[f"{etiqueta}_{seleccion}"] = {"media": float(np.mean(ms)),
                                                 "sd": float(np.std(ms, ddof=1)),
                                                 "por_semilla": ms}
            print(f"  {etiqueta:8s} {seleccion:5s} → acc {np.mean(ms):.4f} "
                  f"(sd {np.std(ms, ddof=1):.4f}) · {[round(m,3) for m in ms]}\n", flush=True)
            json.dump(salida, open("resultados_ei1.json", "w"), indent=1)

    temp = np.mean([salida["temprano_densa"]["media"], salida["temprano_topk"]["media"]])
    tard = np.mean([salida["tardio_densa"]["media"], salida["tardio_topk"]["media"]])
    densa = np.mean([salida["temprano_densa"]["media"], salida["tardio_densa"]["media"]])
    topk = np.mean([salida["temprano_topk"]["media"], salida["tardio_topk"]["media"]])
    mejor = max(v["media"] for v in salida.values())

    print("=" * 72)
    print(f"  P-1 posicion: temprano {temp:.4f} − tardio {tard:.4f} = {temp-tard:+.4f}  "
          f"{'CUMPLE' if temp - tard >= 0.15 else 'NO CUMPLE'} (exigido >= +0,15)")
    print(f"  P-2 seleccion: densa {densa:.4f} − topk {topk:.4f} = {densa-topk:+.4f}  "
          f"{'CUMPLE' if densa - topk >= 0 else 'NO CUMPLE'} (direccional)")
    print(f"  P-3 bloqueante: mejor condicion {mejor:.4f}  "
          f"{'CUMPLE' if mejor > 0.30 else 'NO CUMPLE'} (exigido > 0,30 vs piso 0,0215)")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
