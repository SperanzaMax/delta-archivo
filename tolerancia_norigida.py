"""(A) Tolerancia con deriva NO RIGIDA, y (B) la hipotesis del barrio, con deriva REAL.

(A) La curva de R5.1 uso la deriva como rotacion: rigida, preserva la estructura relativa dentro
    de cada cohorte. La deriva real no lo es (Gram 0.60). Aca se comparan las dos a IGUAL coseno,
    para saber si el cruce "cos 0.88 => seguro" se sostiene o era optimista.

      rigida    : x -> normalize(R x),  R ortogonal
      no rigida : x -> normalize(R (I + lam S) x + rho * n_i),  S simetrica (estira/comprime)
                  y n_i ruido IDIOSINCRATICO por item — la parte que ninguna correccion global
                  puede capturar.

(B) Idea de Maxi: si no sabes la direccion exacta de la casa de tu amigo pero sabes por donde
    queda, la busqueda se reduce muchisimo. Traducido: la correccion no necesita devolver el
    vector exacto (cos ~ 1), le alcanza con dejar al item correcto DENTRO de un vecindario chico.
    La metrica correcta no es el coseno sino el RANGO del item y el recall@k.
    Se mide sobre las claves reales guardadas por deriva_correccion.py.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np
from gemacion_e1 import esfera, tangente, ic95, argmax_chunked
from gemacion_deriva2 import rot_pequena

CACHE = "claves_deriva.npz"


# ----------------------------------------------------------------- (A)
def deriva_op(rng, d, theta, lam, rho):
    """Devuelve f(X, edad) que aplica la deriva acumulada de `edad` epocas."""
    R = rot_pequena(rng, d, theta)
    G = rng.normal(size=(d, d)) / np.sqrt(d)
    S = ((G + G.T) / 2).astype(np.float32)
    Mstep = R @ (np.eye(d, dtype=np.float32) + lam * S)

    def f(X, edad, idio):
        Y = X.copy()
        for _ in range(edad):
            Y = Y @ Mstep.T
        Y = Y + rho * edad * idio
        return Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-9)
    return f


def corrida(seed, N=1000, K=4, d=64, eps=0.3, alpha=0.4, delta=3.0,
            theta=0.0, lam=0.0, rho=0.0, gap=8, ruido=0.05, Q=300):
    rng = np.random.default_rng(seed)
    base = esfera(rng, N, d); that = esfera(rng, N, d)
    cur = base.copy(); versiones = [base.copy()]
    for _ in range(K):
        t = tangente(that, cur)
        u = alpha * t + (1 - alpha) * esfera(rng, N, d)
        u = tangente(u, cur)
        cur = cur + eps * u
        cur /= np.linalg.norm(cur, axis=1, keepdims=True)
        versiones.append(cur.copy())

    f = deriva_op(rng, d, theta, lam, rho)
    idio = esfera(rng, N, d)
    partes = [f(V, (K - j) + gap, idio) for j, V in enumerate(versiones)]
    A = np.concatenate(partes, 0).astype(np.float32)
    mem_id = np.tile(np.arange(N), K + 1); ver = np.repeat(np.arange(K + 1), N)

    qi = rng.choice(N, min(Q, N), replace=False)
    q0 = base[qi] + ruido * esfera(rng, len(qi), d)
    q0 /= np.linalg.norm(q0, axis=1, keepdims=True)
    h1 = argmax_chunked(q0, A)
    t = tangente(that[mem_id[h1]], q0)
    q = q0 + delta * t; q /= np.linalg.norm(q, axis=1, keepdims=True)
    top1 = argmax_chunked(q, A)
    ok = mem_id[top1] == qi
    cos = float(np.mean(np.sum(base * f(base, K + gap, idio), 1)))
    return float(np.mean(ok & (ver[top1] == K))), float(np.mean(ok)), cos


def parte_a(S=5):
    print("=" * 74)
    print("(A) TOLERANCIA: deriva rigida vs no rigida, a igual coseno")
    print(f"{'tipo':>10} {'cos':>7} {'M1 vigente':>17} {'M2 cluster':>17}")
    for etiqueta, kw in (("rigida", dict(theta=0.05)), ("rigida", dict(theta=0.08)),
                         ("rigida", dict(theta=0.10)), ("rigida", dict(theta=0.13)),
                         ("no rigida", dict(theta=0.02, lam=0.05, rho=0.02)),
                         ("no rigida", dict(theta=0.02, lam=0.08, rho=0.04)),
                         ("no rigida", dict(theta=0.02, lam=0.10, rho=0.06)),
                         ("no rigida", dict(theta=0.02, lam=0.12, rho=0.10))):
        r = [corrida(8000 + s, **kw) for s in range(S)]
        m1, h1 = ic95([x[0] for x in r]); m2, h2 = ic95([x[1] for x in r])
        print(f"{etiqueta:>10} {np.mean([x[2] for x in r]):7.3f} "
              f"{m1:>10.3f}±{h1:.3f} {m2:>10.3f}±{h2:.3f}")


# ----------------------------------------------------------------- (B)
def normalizar(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)


def parte_b(n_anclas=256):
    z = np.load(CACHE); K0, Kt = z["K0"], z["Kt"]     # K0 = donde se escribio; Kt = encoder de hoy
    print("\n" + "=" * 74)
    print("(B) LA HIPOTESIS DEL BARRIO — deriva REAL de un modelo preentrenado")
    print(f"    indice = claves viejas ({K0.shape[1]} por cabeza); consulta = clave de hoy")
    print(f"    correccion afin estimada con {n_anclas} anclas, evaluada en held-out\n")
    rng = np.random.default_rng(0)
    ks = (1, 5, 10, 25, 50, 100)
    acc = {"cruda": {k: [] for k in ks}, "corregida": {k: [] for k in ks}}
    rangos = {"cruda": [], "corregida": []}
    for h in range(K0.shape[0]):
        A, B = K0[h], Kt[h]
        n = len(A); idx = rng.permutation(n)
        anc, hold = idx[:n_anclas], idx[n_anclas:]
        A1 = np.hstack([A[anc], np.ones((n_anclas, 1))])
        W = np.linalg.lstsq(A1, B[anc], rcond=None)[0]
        Acorr = normalizar(np.hstack([A, np.ones((n, 1))]) @ W)
        for nom, IDX in (("cruda", A), ("corregida", Acorr)):
            sim = B[hold] @ IDX.T                       # consulta de hoy vs indice viejo
            orden = np.argsort(-sim, 1)
            pos = np.argmax(orden == hold[:, None], 1)  # rango del item correcto (0 = top-1)
            rangos[nom].append(np.median(pos))
            for k in ks:
                acc[nom][k].append(float(np.mean(pos < k)))
    print(f"{'':>11}" + "".join(f"{'@'+str(k):>9}" for k in ks) + f"{'rango med':>11}")
    for nom in ("cruda", "corregida"):
        fila = f"{nom:>11}" + "".join(f"{np.mean(acc[nom][k]):9.3f}" for k in ks)
        print(fila + f"{np.mean(rangos[nom]):11.1f}")
    print(f"\n    tamano del indice: {K0.shape[1]} entradas por cabeza")


if __name__ == "__main__":
    parte_a()
    if os.path.exists(CACHE):
        parte_b()
    else:
        print("\n(B) falta claves_deriva.npz — correr antes deriva_correccion.py")
