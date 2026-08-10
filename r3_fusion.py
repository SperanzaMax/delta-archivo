"""¿La fusion de mediciones (R9) restaura la escalabilidad que R3 perdio?

R3 rehecho mostro que la invariancia en N era un artefacto de d=64: en la dimension real (16 por
cabeza) la memoria colapsa al crecer N. Pero R9 mostro que el mecanismo real no usa UNA medicion
sino varias (4 cabezas x k/v/q). Cada una vive en su propio espacio de d=16 con su propia base.
La pregunta: ¿M mediciones de d=16 recuperan lo que una sola pierde?
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np
from gemacion_e1 import tangente, ic95
from espectro_y_r3 import esfera, calibrar

z = np.load("claves_deriva.npz"); K0 = z["K0"]


def gemar(rng, base, K, eps, alpha, that):
    cur = base.copy(); vs = [base.copy()]
    for _ in range(K):
        t = tangente(that, cur)
        u = alpha * t + (1 - alpha) * esfera(rng, len(base), base.shape[1])
        u = tangente(u, cur)
        cur = cur + eps * u
        cur /= np.linalg.norm(cur, axis=1, keepdims=True)
        vs.append(cur.copy())
    return np.concatenate(vs, 0).astype(np.float32)


def corrida(seed, N, M, K=4, eps=0.3, alpha=0.4, delta=3.0, ruido=0.05, Q=200, calib=True):
    rng = np.random.default_rng(seed)
    d = 16
    bases = [calibrar(K0[m % 4], rng, N) if calib else esfera(rng, N, d) for m in range(M)]
    thats = [esfera(rng, N, d) for _ in range(M)]
    idxs = [gemar(rng, bases[m], K, eps, alpha, thats[m]) for m in range(M)]
    mem_id = np.tile(np.arange(N), K + 1); ver = np.repeat(np.arange(K + 1), N)
    qi = rng.choice(N, min(Q, N), replace=False)

    S = np.zeros((len(qi), N * (K + 1)), np.float32)
    for m in range(M):
        q0 = bases[m][qi] + ruido * esfera(rng, len(qi), d)
        q0 /= np.linalg.norm(q0, axis=1, keepdims=True)
        h1 = np.argmax(q0 @ idxs[m].T, 1)
        t = tangente(thats[m][mem_id[h1]], q0)
        q = q0 + delta * t; q /= np.linalg.norm(q, axis=1, keepdims=True)
        S += q @ idxs[m].T
    top1 = np.argmax(S, 1)
    ok = mem_id[top1] == qi
    return float(np.mean(ok & (ver[top1] == K))), float(np.mean(ok))


print("R3 CON FUSION — mediciones independientes de d=16, calibradas a claves reales")
print(f"{'mediciones':>11}" + "".join(f"{'N='+str(n):>17}" for n in (1_000, 10_000, 100_000)))
for M in (1, 2, 4, 8, 16):
    fila = f"{M:11d}"
    for N in (1_000, 10_000, 100_000):
        r = [corrida(11_000 + s, N, M) for s in range(3)]
        m1, h1 = ic95([x[0] for x in r])
        fila += f"   {m1:.3f}±{h1:.3f}"
    print(fila)
