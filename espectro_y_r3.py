"""Espectro de las claves reales, y R3 rehecho con sinteticos calibrados.

R7.2 mostro que la simulacion con puntos aleatorios era optimista. Antes de rehacer nada conviene
medir CUANTO se aleja el espacio real de la ortogonalidad, porque ese numero explica la brecha.

Parte 1 — espectro: autovalores de la covarianza, dimension efectiva (participation ratio) y
coseno medio entre pares. Se compara contra puntos uniformes en la esfera de la misma dimension.

Parte 2 — sinteticos calibrados: muestrear con la covarianza empirica reproduce la estructura de
correlacion, y permite escalar a 100.000 items (imposible con claves reales: el vocabulario del
harness es de 197 tokens, asi que la diversidad genuina se agota en unos pocos miles).

Parte 3 — R3 rehecho: ¿sigue siendo invariante en N el estres de crecimiento?
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np
from gemacion_e1 import tangente, ic95, argmax_chunked

CACHE = "claves_deriva.npz"


def esfera(rng, n, d):
    x = rng.normal(size=(n, d)).astype(np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def espectro(X, nombre, rng):
    n, d = X.shape
    C = np.cov(X.T)
    lam = np.sort(np.linalg.eigvalsh(C))[::-1]
    lam = np.clip(lam, 0, None)
    pr = (lam.sum() ** 2) / (np.sum(lam ** 2) + 1e-12)      # dimension efectiva
    i = rng.integers(0, n, 20000); j = rng.integers(0, n, 20000)
    m = i != j
    cos = np.sum(X[i[m]] * X[j[m]], 1)
    print(f"{nombre:>22}  d={d:3d}  dim_efectiva={pr:6.2f}  "
          f"|cos| medio={np.mean(np.abs(cos)):.4f}  sd(cos)={np.std(cos):.4f}  "
          f"lam1/lam_tot={lam[0]/lam.sum():.3f}")
    return lam, pr


def calibrar(X, rng, N):
    """Muestrea N puntos con la covarianza empirica de X, normalizados."""
    mu = X.mean(0)
    C = np.cov(X.T) + 1e-6 * np.eye(X.shape[1])
    L = np.linalg.cholesky(C)
    Z = rng.normal(size=(N, X.shape[1])).astype(np.float32) @ L.T.astype(np.float32) + mu
    return Z / np.linalg.norm(Z, axis=1, keepdims=True)


def r3(rng, base, K=4, eps=0.3, alpha=0.4, delta=3.0, ruido=0.05, Q=300):
    """Gemacion con eje por recuerdo sobre las direcciones `base` dadas."""
    N, d = base.shape
    that = esfera(rng, N, d)
    cur = base.copy(); versiones = [base.copy()]
    for _ in range(K):
        t = tangente(that, cur)
        u = alpha * t + (1 - alpha) * esfera(rng, N, d)
        u = tangente(u, cur)
        cur = cur + eps * u
        cur /= np.linalg.norm(cur, axis=1, keepdims=True)
        versiones.append(cur.copy())
    A = np.concatenate(versiones, 0).astype(np.float32)
    mem_id = np.tile(np.arange(N), K + 1); ver = np.repeat(np.arange(K + 1), N)

    qi = rng.choice(N, min(Q, N), replace=False)
    q0 = base[qi] + ruido * esfera(rng, len(qi), d)
    q0 /= np.linalg.norm(q0, axis=1, keepdims=True)
    h1 = argmax_chunked(q0, A)
    t = tangente(that[mem_id[h1]], q0)
    q = q0 + delta * t; q /= np.linalg.norm(q, axis=1, keepdims=True)
    top1 = argmax_chunked(q, A)
    ok = mem_id[top1] == qi
    return float(np.mean(ok & (ver[top1] == K))), float(np.mean(ok))


if __name__ == "__main__":
    z = np.load(CACHE); K0 = z["K0"]
    rng = np.random.default_rng(0)
    Hh, n, d = K0.shape
    print(f"claves reales: {Hh} cabezas x {n} vectores, dim {d} por cabeza\n")

    print("PARTE 1 — espectro")
    reales = K0.reshape(-1, d)
    espectro(reales, "reales (4 cabezas)", rng)
    espectro(K0[0], "reales (cabeza 0)", rng)
    espectro(esfera(rng, n, d), f"uniforme en S^{d-1}", rng)
    espectro(esfera(rng, n, 64), "uniforme en S^63 (lo que usaba R1-R4)", rng)

    print("\nPARTE 2 — calibracion (¿el sintetico reproduce la estructura real?)")
    sint = calibrar(K0[0], rng, n)
    espectro(K0[0], "real cabeza 0", rng)
    espectro(sint, "sintetico calibrado", rng)

    print("\nPARTE 3 — R3 rehecho: estres de crecimiento")
    print(f"{'geometria':>24} {'N':>8} {'M1 vigente':>16} {'M2 cluster':>16}")
    S = 5
    for etiqueta, gen in (
            ("uniforme d=64 (R3 orig)", lambda r, N: esfera(r, N, 64)),
            ("uniforme d=16 (dim real)", lambda r, N: esfera(r, N, 16)),
            ("calibrado a claves reales", lambda r, N: calibrar(K0[0], r, N))):
        for N in (1_000, 10_000, 100_000):
            res = [r3(np.random.default_rng(9000 + s), gen(np.random.default_rng(500 + s), N))
                   for s in range(S)]
            m1, h1 = ic95([x[0] for x in res]); m2, h2 = ic95([x[1] for x in res])
            print(f"{etiqueta:>24} {N:8d} {m1:>9.3f}±{h1:.3f} {m2:>9.3f}±{h2:.3f}")
