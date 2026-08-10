"""¿Se puede corregir la deriva al estilo GPS? Tres niveles de correccion, de rigido a flexible.

El GPS funciona porque la deformacion que sufre es PREDECIBLE y RIGIDA: se conoce su forma y basta
con pocos satelites de referencia. Aca se testea si la deriva del encoder admite el mismo trato,
estimando la correccion con n anclas re-codificadas y evaluandola en items HELD-OUT:

  ORTOGONAL  R ortogonal (Procrustes). Asume que la deriva es una rotacion: preserva distancias.
  +ESCALA    sR, rotacion mas un factor de escala global.
  LINEAL     A cualquiera (minimos cuadrados). Permite estirar y cizallar el espacio.
  AFIN       A x + b. Agrega traslacion.

Si ninguna mejora el coseno crudo, la deriva no es una deformacion de forma conocida y la
analogia del GPS no aplica: no hay correccion barata, hay que re-codificar.
"""
import os, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "2")
sys.path.insert(0, os.path.expanduser("~/Documentos/Nuevo Transformer/telar-ligamento/src"))

import numpy as np, jax, jax.numpy as jnp, optax
import modelos as M
from modelos import split_heads, l2n, ln, conv3, H, DH
from datos import gen_mqar
from entrenar import loss_fn

CACHE = "claves_deriva.npz"


def claves(params, x):
    blk = params["blocks"][0]
    hx = params["emb"][x]
    xin = conv3(blk["conv"], ln(blk["ln1"], hx))
    k = l2n(jax.nn.silu(split_heads(xin @ blk["k"])))
    return np.asarray(k.transpose(1, 0, 2, 3).reshape(H, -1, DH))


def lote(seed, batch, carga):
    x, y = gen_mqar(np.random.default_rng(seed), batch, carga)
    return jnp.asarray(x), jnp.asarray(y)


def generar():
    gv = jax.jit(jax.value_and_grad(loss_fn, has_aux=True), static_argnums=3)
    opt = optax.adam(3e-3)
    sonda = lote(999, 64, 8)[0]                    # sonda GRANDE: mas puntos para anclar
    p = M.init_params(0, "delta"); st = opt.init(p)
    t0 = time.time()
    for s in range(1501):
        x, y = lote(s, 16, 8)
        (l, a), g = gv(p, x, y, "delta")
        upd, st = opt.update(g, st, p)
        p = optax.apply_updates(p, upd)
    print(f"preentrenado: loss {float(l):.3f} acc {float(a):.3f}  [{time.time()-t0:.0f} s]")
    K0 = claves(p, sonda)
    st = opt.init(p)
    for s in range(401):
        x, y = lote(50_000 + s, 16, 16)
        (l, a), g = gv(p, x, y, "delta")
        upd, st = opt.update(g, st, p)
        p = optax.apply_updates(p, upd)
    print(f"afinado:      loss {float(l):.3f} acc {float(a):.3f}")
    Kt = claves(p, sonda)
    np.savez(CACHE, K0=K0, Kt=Kt)
    return K0, Kt


def normalizar(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)


def correcciones(A, B):
    """Estimadores ajustados sobre (A,B). Devuelve dict nombre -> f(X)."""
    U, S, Vt = np.linalg.svd(A.T @ B)
    R = U @ Vt
    s = float(np.sum(S) / (np.sum(A * A) + 1e-9))
    L = np.linalg.lstsq(A, B, rcond=None)[0]
    A1 = np.hstack([A, np.ones((len(A), 1))])
    Af = np.linalg.lstsq(A1, B, rcond=None)[0]
    return {
        "ninguna":   lambda X: X,
        "ortogonal": lambda X: X @ R,
        "+escala":   lambda X: s * (X @ R),
        "lineal":    lambda X: X @ L,
        "afin":      lambda X: np.hstack([X, np.ones((len(X), 1))]) @ Af,
    }


def evaluar(K0, Kt, n_anclas, seed=0):
    rng = np.random.default_rng(seed)
    acc = {}
    for h in range(K0.shape[0]):
        A, B = K0[h], Kt[h]
        idx = rng.permutation(len(A))
        anc, hold = idx[:n_anclas], idx[n_anclas:]
        if len(hold) < 50:
            return None
        fs = correcciones(A[anc], B[anc])
        for nom, f in fs.items():
            c = float(np.mean(np.sum(normalizar(f(A[hold])) * B[hold], 1)))
            acc.setdefault(nom, []).append(c)
    return {k: float(np.mean(v)) for k, v in acc.items()}


if __name__ == "__main__":
    if os.path.exists(CACHE):
        z = np.load(CACHE); K0, Kt = z["K0"], z["Kt"]
    else:
        K0, Kt = generar()
    print(f"\nsonda: {K0.shape[1]} vectores por cabeza, {H} cabezas, dim {DH}")
    G0 = K0[0] @ K0[0].T; Gt = Kt[0] @ Kt[0].T
    print(f"error de Gram (¿la deriva preserva distancias?): "
          f"{np.linalg.norm(G0-Gt)/np.linalg.norm(G0):.4f}\n")
    print("coseno con los items HELD-OUT segun correccion y numero de anclas")
    nombres = ["ninguna", "ortogonal", "+escala", "lineal", "afin"]
    print(f"{'anclas':>7}" + "".join(f"{n:>11}" for n in nombres))
    for n in (8, 16, 32, 64, 128, 256, 512, 1024):
        r = evaluar(K0, Kt, n)
        if r is None:
            break
        print(f"{n:7d}" + "".join(f"{r[k]:11.3f}" for k in nombres))
