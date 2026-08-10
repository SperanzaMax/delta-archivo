"""Desempate por CONTENIDO: usar el valor guardado, no solo la clave.

Hasta R8 todo el desempate uso unicamente las claves de la capa 0. Pero la memoria guarda pares
(clave, valor), y ademas el modelo tiene 4 bloques. Cada proyeccion de cada bloque es otra
medicion del mismo item, con su propia deriva:

    4 bloques x 3 proyecciones (k, v, q) x 4 cabezas = 48 mediciones disponibles.

R8 mostro que cruzar mediciones parcialmente independientes es lo que fija la posicion. La
pregunta ahora es cuanto queda por exprimir, y si las capas altas aportan o solo repiten a las
bajas (sus derivas deberian estar correlacionadas, porque dependen de las de abajo).

Se preentrena, se afina, y se comparan subconjuntos crecientes de mediciones.
"""
import os, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "2")
sys.path.insert(0, os.path.expanduser("~/Documentos/Nuevo Transformer/telar-ligamento/src"))

import numpy as np, jax, jax.numpy as jnp, optax
import modelos as M
from modelos import split_heads, l2n, ln, conv3, H, DH, NB
from datos import gen_mqar
from entrenar import loss_fn

CACHE = "claves_multi.npz"
KS = (1, 5, 10, 25, 100)


def repres(params, x):
    """Devuelve dict 'L{l}_{p}' -> (H, n, DH) para las 3 proyecciones de los 4 bloques."""
    out = {}
    hx = params["emb"][x]
    for l, blk in enumerate(params["blocks"]):
        xin = conv3(blk["conv"], ln(blk["ln1"], hx))
        for p in ("k", "v", "q"):
            z = split_heads(xin @ blk[p])
            if p != "v":
                z = jax.nn.silu(z)
            z = l2n(z)
            out[f"L{l}_{p}"] = np.asarray(z.transpose(1, 0, 2, 3).reshape(H, -1, DH))
        y = M.mixer(blk, xin, "delta")
        hx = hx + y
        h2 = ln(blk["ln2"], hx)
        hx = hx + jax.nn.gelu(h2 @ blk["m1"]["w"] + blk["m1"]["b"]) @ blk["m2"]["w"] + blk["m2"]["b"]
    return out


def lote(seed, batch, carga):
    x, y = gen_mqar(np.random.default_rng(seed), batch, carga)
    return jnp.asarray(x), jnp.asarray(y)


def generar():
    gv = jax.jit(jax.value_and_grad(loss_fn, has_aux=True), static_argnums=3)
    opt = optax.adam(3e-3)
    sonda = lote(999, 64, 8)[0]
    p = M.init_params(0, "delta"); st = opt.init(p)
    t0 = time.time()
    for s in range(1501):
        (l, a), g = gv(p, *lote(s, 16, 8), "delta")
        upd, st = opt.update(g, st, p); p = optax.apply_updates(p, upd)
    print(f"preentrenado acc {float(a):.3f}  [{time.time()-t0:.0f} s]")
    R0 = repres(p, sonda)
    st = opt.init(p)
    for s in range(401):
        (l, a), g = gv(p, *lote(50_000 + s, 16, 16), "delta")
        upd, st = opt.update(g, st, p); p = optax.apply_updates(p, upd)
    print(f"afinado      acc {float(a):.3f}")
    Rt = repres(p, sonda)
    np.savez(CACHE, **{f"a_{k}": v for k, v in R0.items()},
             **{f"b_{k}": v for k, v in Rt.items()})
    return R0, Rt


def normalizar(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)


def afin(A, B, anc):
    A1 = np.hstack([A[anc], np.ones((len(anc), 1))])
    W = np.linalg.lstsq(A1, B[anc], rcond=None)[0]
    return normalizar(np.hstack([A, np.ones((len(A), 1))]) @ W)


def main(n_anclas=256, seed=0):
    if os.path.exists(CACHE):
        z = np.load(CACHE)
        R0 = {k[2:]: z[k] for k in z.files if k.startswith("a_")}
        Rt = {k[2:]: z[k] for k in z.files if k.startswith("b_")}
    else:
        R0, Rt = generar()

    n = next(iter(R0.values())).shape[1]
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n); anc, hold = idx[:n_anclas], idx[n_anclas:]

    sim = {}
    for nom in R0:
        acc = []
        for h in range(H):
            A, B = R0[nom][h], Rt[nom][h]
            acc.append(B[hold] @ afin(A, B, anc).T)
        sim[nom] = np.stack(acc).sum(0)

    def evaluar(nombres):
        S = sum(sim[nm] for nm in nombres)
        orden = np.argsort(-S, 1)
        pos = np.argmax(orden == hold[:, None], 1)
        return {k: float(np.mean(pos < k)) for k in KS}, float(np.median(pos))

    grupos = {
        "L0 claves (R8)":        ["L0_k"],
        "L0 clave+valor":        ["L0_k", "L0_v"],
        "L0 clave+valor+query":  ["L0_k", "L0_v", "L0_q"],
        "claves 4 capas":        [f"L{l}_k" for l in range(NB)],
        "valores 4 capas":       [f"L{l}_v" for l in range(NB)],
        "TODO (48 mediciones)":  [f"L{l}_{p}" for l in range(NB) for p in ("k", "v", "q")],
    }
    print(f"\nDESEMPATE POR CONTENIDO — indice {n}, held-out {len(hold)}, "
          f"afin con {n_anclas} anclas")
    print(f"{'conjunto':>22} {'mediciones':>11}" + "".join(f"{'@'+str(k):>9}" for k in KS)
          + f"{'rango med':>11}")
    for nom, gs in grupos.items():
        r, med = evaluar(gs)
        print(f"{nom:>22} {len(gs)*H:>11}" + "".join(f"{r[k]:9.3f}" for k in KS)
              + f"{med:11.1f}")

    print("\nAporte individual de cada proyeccion (sola, 4 cabezas):")
    print(f"{'proyeccion':>12} {'@1':>8} {'@25':>8}")
    for nom in sorted(sim):
        r, _ = evaluar([nom])
        print(f"{nom:>12} {r[1]:8.3f} {r[25]:8.3f}")


if __name__ == "__main__":
    main()
