"""Prueba de Basu: ¿sobrevive la gemacion cuando el espacio de coordenadas SE MUEVE?

Basu (2603.22858) midio que cualquier sistema de coordenadas aprendido junto al modelo es
inestable. Query Drift Compensation (2506.00037) agrega que la deriva es DESIGUAL entre consulta
y entrada almacenada. Produccion reporta recall 0.92 -> 0.74 por embeddings obsoletos.

La gemacion tiene una exposicion especifica que un indice plano no tiene: sus versiones se
escriben en MOMENTOS DISTINTOS, asi que cada una arrastra una deriva distinta. La informacion
esta codificada en distancias del orden de eps=0.3; si la deriva mueve las entradas mas que eso,
el cluster se desarma.

Modelo de deriva: la entrada escrita hace `edad` pasos aparece hoy desplazada en proporcion a esa
edad (drift_rate por paso). La consulta se calcula con el encoder de HOY (deriva 0) -> deriva
desigual, tal como se reporta.

Baseline de contraste: SOBRESCRITURA — una sola entrada por recuerdo, reescrita siempre con el
encoder actual. Pierde la historia pero es inmune a la deriva. Es el trade-off real.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")          # cuidar la maquina
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import numpy as np
from gemacion_e1 import esfera, tangente, ic95, argmax_chunked


def corrida_deriva(seed, N=1000, K=4, d=64, eps=0.3, alpha=0.4, delta=3.0,
                   drift=0.0, ruido=0.05, Q=300, sobrescritura=False):
    """drift = desplazamiento por paso de antiguedad. La version j tiene edad (K-j)."""
    rng = np.random.default_rng(seed)
    base = esfera(rng, N, d)
    that = esfera(rng, N, d)                                # eje por recuerdo (el ganador)

    cur = base.copy(); versiones = [base.copy()]
    for _ in range(K):
        t = tangente(that, cur)
        u = alpha * t + (1 - alpha) * esfera(rng, N, d)
        u = tangente(u, cur)
        cur = cur + eps * u
        cur /= np.linalg.norm(cur, axis=1, keepdims=True)
        versiones.append(cur.copy())

    if sobrescritura:                                       # solo la ultima, siempre fresca
        A = versiones[-1].astype(np.float32)
        mem_id = np.arange(N); ver = np.full(N, K)
    else:
        derivadas = []
        for j, V in enumerate(versiones):
            edad = K - j
            W = V + drift * edad * esfera(rng, N, d)        # deriva proporcional a la edad
            derivadas.append(W / np.linalg.norm(W, axis=1, keepdims=True))
        A = np.concatenate(derivadas, 0).astype(np.float32)
        mem_id = np.tile(np.arange(N), K + 1)
        ver = np.repeat(np.arange(K + 1), N)

    qi = rng.choice(N, min(Q, N), replace=False)
    q0 = base[qi] + ruido * esfera(rng, len(qi), d)         # consulta con encoder de HOY
    q0 /= np.linalg.norm(q0, axis=1, keepdims=True)

    if sobrescritura:
        top1 = argmax_chunked(q0, A)
    else:
        h1 = argmax_chunked(q0, A)
        t = tangente(that[mem_id[h1]], q0)
        q = q0 + delta * t
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        top1 = argmax_chunked(q, A)

    ok = mem_id[top1] == qi
    return float(np.mean(ok & (ver[top1] == K))), float(np.mean(ok))


if __name__ == "__main__":
    S = 5
    print("PRUEBA DE BASU — deriva del sistema de coordenadas")
    print("N=1000 d=64 eps=0.30 alpha=0.4 delta=3.0 K=4, eje por recuerdo, "
          f"{S} semillas, IC95\n")
    print(f"{'drift/paso':>11} {'drift·K/eps':>12} {'M1 vigente':>17} {'M2 cluster':>17}")
    for drift in (0.0, 0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5):
        r = [corrida_deriva(4000 + s, drift=drift) for s in range(S)]
        m1, h1 = ic95([x[0] for x in r]); m2, h2 = ic95([x[1] for x in r])
        print(f"{drift:11.2f} {drift*4/0.30:12.2f} {m1:>10.3f}±{h1:.3f} {m2:>10.3f}±{h2:.3f}")

    print("\nBaseline SOBRESCRITURA (inmune a la deriva, sin historia):")
    r = [corrida_deriva(4000 + s, sobrescritura=True) for s in range(S)]
    m1, h1 = ic95([x[0] for x in r]); m2, h2 = ic95([x[1] for x in r])
    print(f"{'—':>11} {'—':>12} {m1:>10.3f}±{h1:.3f} {m2:>10.3f}±{h2:.3f}")

    print("\n¿Sirve agrandar eps para tolerar mas deriva? (drift=0.10 fijo)")
    print(f"{'eps':>6} {'M1 vigente':>17} {'M2 cluster':>17}")
    for eps in (0.15, 0.30, 0.60, 1.00):
        r = [corrida_deriva(5000 + s, eps=eps, drift=0.10) for s in range(S)]
        m1, h1 = ic95([x[0] for x in r]); m2, h2 = ic95([x[1] for x in r])
        print(f"{eps:6.2f} {m1:>10.3f}±{h1:.3f} {m2:>10.3f}±{h2:.3f}")
