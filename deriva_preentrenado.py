"""Deriva en un modelo PREENTRENADO, y si esa deriva se puede corregir al estilo GPS.

Dos preguntas:

 (1) La medicion anterior entreno desde inicializacion — el peor caso. ¿Cuanto deriva un modelo
     que YA aprendio y despues se afina? Se preentrena en carga 8 hasta converger y se afina en
     carga 16 (distribucion nueva), midiendo el desplazamiento del espacio de claves.

 (2) La idea de Maxi: el GPS se ubica corrigiendo por el tiempo transcurrido, porque la
     deformacion es PREDECIBLE. Si la deriva del encoder tambien lo fuera, no habria que
     reindexar: alcanzaria con guardar el timestamp de escritura y aplicar la correccion.
     Se testea en tres niveles, cada uno mas exigente:

       ISOMETRIA  ¿la deriva preserva las distancias entre items? Se compara la matriz de Gram
                  del batch de sondeo. Si se preserva, la deriva es (casi) una rotacion y existe
                  una correccion exacta.
       ANCLAS     Procrustes ortogonal: se re-codifican n items de anclaje, se estima la rotacion
                  que los alinea, y se aplica al RESTO del indice (held-out, nunca visto por el
                  estimador). Es literalmente el esquema GPS: pocos satelites, correccion global.
       CUANTAS    ¿cuantas anclas hacen falta? Si son pocas, la correccion es practicamente gratis.
"""
import os, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "2")
sys.path.insert(0, os.path.expanduser("~/Documentos/Nuevo Transformer/telar-ligamento/src"))

import numpy as np, jax, jax.numpy as jnp, optax
import modelos as M
from modelos import split_heads, l2n, ln, conv3, H, DH
from datos import gen_mqar
from entrenar import loss_fn

LR = 3e-3


def claves(params, x):
    blk = params["blocks"][0]
    hx = params["emb"][x]
    xin = conv3(blk["conv"], ln(blk["ln1"], hx))
    k = l2n(jax.nn.silu(split_heads(xin @ blk["k"])))        # (B,H,T,DH)
    return np.asarray(k.transpose(1, 0, 2, 3).reshape(H, -1, DH))   # (H, n, DH)


def entrenar(params, pasos, carga, batch, seed0, gv, opt, st, sonda=None, cada=25, K0=None,
             etiqueta=""):
    hist = []
    for s in range(pasos + 1):
        if sonda is not None and s % cada == 0:
            Kt = claves(params, sonda)
            fila = (s,) + ((diagnostico(K0, Kt),) if K0 is not None else ())
            (l, a), _ = gv(params, *lote(seed0 + s, batch, carga), "delta")
            hist.append((s, float(l), float(a)) + (fila[1:] if K0 is not None else ()))
        x, y = lote(seed0 + s, batch, carga)
        (l, a), g = gv(params, x, y, "delta")
        upd, st = opt.update(g, st, params)
        params = optax.apply_updates(params, upd)
    return params, st, hist


def lote(seed, batch, carga):
    x, y = gen_mqar(np.random.default_rng(seed), batch, carga)
    return jnp.asarray(x), jnp.asarray(y)


def procrustes(A, B):
    """Rotacion ortogonal R que mejor lleva A a B (por filas)."""
    U, _, Vt = np.linalg.svd(A.T @ B)
    return U @ Vt


def diagnostico(K0, Kt, n_anclas=64, seed=0):
    """Devuelve (cos crudo, cos corregido held-out, error de Gram relativo)."""
    rng = np.random.default_rng(seed)
    cos_crudo, cos_corr, gram = [], [], []
    for h in range(K0.shape[0]):
        A, B = K0[h], Kt[h]
        n = len(A)
        idx = rng.permutation(n)
        anc, hold = idx[:n_anclas], idx[n_anclas:n_anclas + 2000]
        cos_crudo.append(np.mean(np.sum(A[hold] * B[hold], 1)))
        R = procrustes(A[anc], B[anc])
        Ac = A[hold] @ R
        Ac /= np.linalg.norm(Ac, axis=1, keepdims=True) + 1e-9
        cos_corr.append(np.mean(np.sum(Ac * B[hold], 1)))
        sub = idx[:400]
        G0 = A[sub] @ A[sub].T; Gt = B[sub] @ B[sub].T
        gram.append(np.linalg.norm(G0 - Gt) / np.linalg.norm(G0))
    return float(np.mean(cos_crudo)), float(np.mean(cos_corr)), float(np.mean(gram))


def main():
    gv = jax.jit(jax.value_and_grad(loss_fn, has_aux=True), static_argnums=3)
    opt = optax.adam(LR)
    sonda = lote(999, 8, 8)[0]

    print("FASE 1 — preentrenamiento (carga 8) hasta converger")
    p = M.init_params(0, "delta"); st = opt.init(p)
    t0 = time.time()
    for s in range(1501):
        x, y = lote(s, 16, 8)
        (l, a), g = gv(p, x, y, "delta")
        upd, st = opt.update(g, st, p)
        p = optax.apply_updates(p, upd)
        if s % 300 == 0:
            print(f"   paso {s:5d}  loss {float(l):6.3f}  acc {float(a):5.3f}")
    print(f"   [{time.time()-t0:.0f} s]\n")

    K0 = claves(p, sonda)
    st = opt.init(p)                                   # optimizador nuevo para el afinado

    print("FASE 2 — afinado sobre distribucion nueva (carga 16) desde el modelo YA ENTRENADO")
    print(f"{'paso':>6} {'loss':>7} {'acc':>6} {'cos crudo':>11} "
          f"{'cos corregido':>14} {'err Gram':>10}")
    for s in range(401):
        if s % 50 == 0:
            c, cc, g_ = diagnostico(K0, claves(p, sonda))
            (l, a), _ = gv(p, *lote(50_000 + s, 16, 16), "delta")
            print(f"{s:6d} {float(l):7.3f} {float(a):6.3f} {c:11.3f} {cc:14.3f} {g_:10.4f}")
        x, y = lote(50_000 + s, 16, 16)
        (l, a), g = gv(p, x, y, "delta")
        upd, st = opt.update(g, st, p)
        p = optax.apply_updates(p, upd)

    print("\nFASE 3 — ¿cuantas anclas hacen falta? (al final del afinado)")
    Kf = claves(p, sonda)
    print(f"{'anclas':>7} {'cos corregido':>14}")
    for n in (4, 8, 16, 32, 64, 128, 256):
        _, cc, _ = diagnostico(K0, Kf, n_anclas=n)
        print(f"{n:7d} {cc:14.3f}")


if __name__ == "__main__":
    main()
