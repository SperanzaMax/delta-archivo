"""Gemacion E1 — ventana (alpha, delta) con IC95, estres de crecimiento, y eje por recuerdo.

Tres formas de definir la direccion temporal t_hat en la que se deposita cada revision:

  GLOBAL : un unico t_hat para todo el indice.
  CAMPO  : t_hat(x) = funcion determinista de la POSICION. Cada region del espacio tiene su
           propia direccion de "mas nuevo". Clave: como es funcion de x, la consulta puede
           computarla sin saber todavia que recuerdo va a recuperar. Los recuerdos trazan
           lineas de flujo del campo.
  LIBRE  : un t_hat aleatorio e independiente por recuerdo. NO es computable desde la consulta,
           asi que se evalua en dos saltos: hop1 sin sesgo recupera el cluster, se lee el eje
           guardado junto a la entrada, hop2 consulta sesgada.

Metricas: M1 = la version VIGENTE es el top-1.  M2 = el top-1 es del recuerdo correcto.
Todo tangencial a la esfera y con el mismo tratamiento en las tres condiciones, para que la
unica diferencia sea el origen de t_hat.
"""
import numpy as np, time, sys

T95 = {5: 2.571, 10: 2.262, 20: 2.093}          # t de Student, dos colas, n-1 gl


def esfera(rng, n, d):
    x = rng.normal(size=(n, d)).astype(np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def tangente(t, x):
    """Componente de t ortogonal a x, normalizada. Mantiene el paso sobre la esfera."""
    t = t - (t * x).sum(-1, keepdims=True) * x
    return t / (np.linalg.norm(t, axis=-1, keepdims=True) + 1e-8)


def ejes(modo, X, that_g, R, that_libre=None):
    """Direccion temporal en cada punto de X segun el modo."""
    if modo == "global":
        return tangente(np.broadcast_to(that_g, X.shape).copy(), X)
    if modo == "campo":
        return tangente(X @ R.T, X)
    return tangente(that_libre, X)


def construir(rng, N, K, d, eps, alpha, modo, R, dtype=np.float32):
    base = esfera(rng, N, d)
    that_g = esfera(rng, 1, d)[0]
    that_libre = esfera(rng, N, d) if modo == "libre" else None
    cur = base.copy()
    versiones = [base]
    for _ in range(K):
        t = ejes(modo, cur, that_g, R, that_libre)
        u = alpha * t + (1 - alpha) * esfera(rng, N, d)
        u = tangente(u, cur)
        cur = cur + eps * u
        cur = cur / np.linalg.norm(cur, axis=1, keepdims=True)
        versiones.append(cur.copy())
    A = np.concatenate(versiones, 0).astype(dtype)
    mem_id = np.tile(np.arange(N), K + 1)
    ver = np.repeat(np.arange(K + 1), N)
    return A, mem_id, ver, base, that_g, that_libre


def argmax_chunked(Q, A, chunk=200_000):
    """argmax_j Q@A.T sin materializar la matriz completa."""
    best_i = np.zeros(len(Q), np.int64); best_v = np.full(len(Q), -np.inf, np.float32)
    for s in range(0, len(A), chunk):
        sim = Q @ A[s:s + chunk].T
        j = np.argmax(sim, 1); v = sim[np.arange(len(Q)), j]
        m = v > best_v
        best_i[m] = (s + j)[m]; best_v[m] = v[m]
    return best_i


def corrida(seed, N, K, d, eps, alpha, delta, modo, ruido=0.05, Q=None, R=None):
    rng = np.random.default_rng(seed)
    if R is None:
        R = np.linalg.qr(rng.normal(size=(d, d)))[0].astype(np.float32)
    A, mem_id, ver, base, that_g, that_libre = construir(rng, N, K, d, eps, alpha, modo, R)

    qi = np.arange(N) if (Q is None or Q >= N) else rng.choice(N, Q, replace=False)
    q0 = base[qi] + ruido * esfera(rng, len(qi), d)
    q0 /= np.linalg.norm(q0, axis=1, keepdims=True)

    if modo == "libre":                      # hop1 sin sesgo -> leer el eje guardado -> hop2
        h1 = argmax_chunked(q0, A)
        t = tangente(that_libre[mem_id[h1]], q0)
    else:
        t = ejes(modo, q0, that_g, R)

    q = q0 + delta * t
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    top1 = argmax_chunked(q, A)
    ok_mem = mem_id[top1] == qi
    return float(np.mean(ok_mem & (ver[top1] == K))), float(np.mean(ok_mem))


def ic95(vals):
    v = np.array(vals); n = len(v)
    if n < 2:
        return float(v.mean()), 0.0
    t = T95.get(n, 2.093)
    return float(v.mean()), float(t * v.std(ddof=1) / np.sqrt(n))


# ---------------------------------------------------------------- Parte 1
def parte1(semillas=10, d=64, eps=0.3, K=4, N=200):
    print("=" * 78)
    print(f"PARTE 1 — ventana (alpha, delta).  N={N} K={K} d={d} eps={eps}, "
          f"{semillas} semillas, IC95")
    for modo in ("global", "campo", "libre"):
        print(f"\n### eje {modo.upper()}")
        print(f"{'a\\d':>5}" + "".join(f"{dd:>14.1f}" for dd in (0.6, 0.8, 1.0, 1.2)))
        for alpha in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
            fila = f"{alpha:5.1f}"
            for delta in (0.6, 0.8, 1.0, 1.2):
                r = [corrida(1000 + s, N, K, d, eps, alpha, delta, modo)[0]
                     for s in range(semillas)]
                m, h = ic95(r)
                fila += f"  {m:.3f}±{h:.3f}"
            print(fila)


# ---------------------------------------------------------------- Parte 2
def parte2(alpha, delta, semillas=5, d=64, eps=0.3):
    print("\n" + "=" * 78)
    print(f"PARTE 2 — estres de crecimiento.  alpha={alpha} delta={delta} d={d} eps={eps}, "
          f"{semillas} semillas, IC95")
    print(f"{'modo':>7} {'N':>8} {'K':>4} {'entradas':>10} "
          f"{'M1 vigente':>16} {'M2 cluster':>16} {'seg':>6}")
    for modo in ("global", "campo", "libre"):
        for N, K in ((1_000, 4), (1_000, 16), (1_000, 64),
                     (10_000, 4), (10_000, 16), (100_000, 4)):
            t0 = time.time()
            r = [corrida(2000 + s, N, K, d, eps, alpha, delta, modo, Q=300)
                 for s in range(semillas)]
            m1, h1 = ic95([x[0] for x in r]); m2, h2 = ic95([x[1] for x in r])
            print(f"{modo:>7} {N:8d} {K:4d} {N*(K+1):10d} "
                  f"{m1:>9.3f}±{h1:.3f} {m2:>9.3f}±{h2:.3f} {time.time()-t0:6.1f}")
            sys.stdout.flush()


if __name__ == "__main__":
    parte1()
    parte2(alpha=0.5, delta=1.0)
