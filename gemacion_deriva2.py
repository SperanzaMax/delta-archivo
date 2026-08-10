"""Prueba de Basu, version correcta.

Que estaba mal en el primer intento (gemacion_deriva.py): modele la deriva como ruido
INDEPENDIENTE por entrada. Eso no es deriva, es dispersion — y encima ayudaba, porque desparramaba
las versiones viejas y dejaba sola a la vigente. Resultado 1.000 en todas las celdas: sintoma de
prueba mal especificada, no de robustez.

La deriva real es una TRANSFORMACION DEL ESPACIO: si el encoder cambia, todas las representaciones
escritas en la misma epoca se mueven JUNTAS (una rotacion las preserva entre si), y lo que se
desalinea es una cohorte respecto de otra. Como el coseno es invariante a rotaciones globales, el
dano no viene de que el espacio rote: viene de que las entradas quedaron CONGELADAS en el marco de
su epoca mientras la consulta se calcula en el marco de hoy. Eso es exactamente la "deriva
desigual" que reporta Query Drift Compensation (2506.00037).

Segundo faltante: el tiempo entre la ultima escritura y la consulta (`gap`). Con gap=0 se consulta
justo despues de escribir y no hay nada que medir.

Modelo: R_paso es una rotacion pequena; la entrada de edad e aparece hoy como R_paso^e aplicada a
su posicion. La consulta vive en el marco actual (edad 0). `desalineamiento` = cos(R^e x, x).
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import numpy as np
from gemacion_e1 import esfera, tangente, ic95, argmax_chunked


def rot_pequena(rng, d, theta):
    """Rotacion ortogonal proxima a la identidad, angulo caracteristico ~theta."""
    G = rng.normal(size=(d, d))
    A = (G - G.T) / np.sqrt(2 * d)                   # antisimetrica normalizada
    Q, R = np.linalg.qr(np.eye(d) + theta * A)
    return (Q * np.sign(np.diag(R))).astype(np.float32)


def corrida(seed, N=1000, K=4, d=64, eps=0.3, alpha=0.4, delta=3.0,
            theta=0.0, gap=0, ruido=0.05, Q=300, modo="gemacion"):
    rng = np.random.default_rng(seed)
    base = esfera(rng, N, d)
    that = esfera(rng, N, d)

    cur = base.copy(); versiones = [base.copy()]
    for _ in range(K):
        t = tangente(that, cur)
        u = alpha * t + (1 - alpha) * esfera(rng, N, d)
        u = tangente(u, cur)
        cur = cur + eps * u
        cur /= np.linalg.norm(cur, axis=1, keepdims=True)
        versiones.append(cur.copy())

    Rp = rot_pequena(rng, d, theta)
    Rpow = {0: np.eye(d, dtype=np.float32)}
    for e in range(1, K + gap + 1):
        Rpow[e] = Rpow[e - 1] @ Rp

    def congelar(V, edad):                            # la entrada quedo en el marco de su epoca
        W = V @ Rpow[edad].T
        return W / np.linalg.norm(W, axis=1, keepdims=True)

    if modo == "sobrescritura":                       # una sola entrada, reescrita al final
        A = congelar(versiones[-1], gap).astype(np.float32)
        mem_id = np.arange(N); ver = np.full(N, K)
    else:
        partes = [congelar(V, (K - j) + gap) for j, V in enumerate(versiones)]
        A = np.concatenate(partes, 0).astype(np.float32)
        mem_id = np.tile(np.arange(N), K + 1)
        ver = np.repeat(np.arange(K + 1), N)

    qi = rng.choice(N, min(Q, N), replace=False)
    q0 = base[qi] + ruido * esfera(rng, len(qi), d)   # consulta en el marco de HOY
    q0 /= np.linalg.norm(q0, axis=1, keepdims=True)

    if modo == "sobrescritura":
        top1 = argmax_chunked(q0, A)
    else:
        h1 = argmax_chunked(q0, A)
        t = tangente(that[mem_id[h1]], q0)
        q = q0 + delta * t
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        top1 = argmax_chunked(q, A)

    ok = mem_id[top1] == qi
    desal = float(np.mean(np.sum(base * (base @ Rpow[K + gap].T), 1)))   # cos vieja-vs-hoy
    return float(np.mean(ok & (ver[top1] == K))), float(np.mean(ok)), desal


if __name__ == "__main__":
    S = 5
    print("PRUEBA DE BASU (corregida) — la deriva como rotacion del espacio")
    print(f"N=1000 d=64 eps=0.30 K=4 alpha=0.4 delta=3.0, eje por recuerdo, {S} semillas, IC95\n")
    print("A) barrido de deriva, gap=8 pasos desde la ultima escritura")
    print(f"{'theta':>7} {'cos(hoy,vieja)':>15} {'M1 vigente':>17} {'M2 cluster':>17}")
    for theta in (0.0, 0.02, 0.05, 0.1, 0.2, 0.4, 0.8):
        r = [corrida(6000 + s, theta=theta, gap=8) for s in range(S)]
        m1, h1 = ic95([x[0] for x in r]); m2, h2 = ic95([x[1] for x in r])
        print(f"{theta:7.2f} {np.mean([x[2] for x in r]):15.3f} "
              f"{m1:>10.3f}±{h1:.3f} {m2:>10.3f}±{h2:.3f}")

    print("\nB) mismo barrido, baseline SOBRESCRITURA (sin historia)")
    print(f"{'theta':>7} {'cos(hoy,vieja)':>15} {'M1 vigente':>17} {'M2 cluster':>17}")
    for theta in (0.0, 0.05, 0.2, 0.4, 0.8):
        r = [corrida(6000 + s, theta=theta, gap=8, modo="sobrescritura") for s in range(S)]
        m1, h1 = ic95([x[0] for x in r]); m2, h2 = ic95([x[1] for x in r])
        print(f"{theta:7.2f} {np.mean([x[2] for x in r]):15.3f} "
              f"{m1:>10.3f}±{h1:.3f} {m2:>10.3f}±{h2:.3f}")

    print("\nC) antiguedad: cuanto aguanta sin reescribir (theta=0.1)")
    print(f"{'gap':>5} {'cos(hoy,vieja)':>15} {'M1 gemacion':>17} {'M1 sobrescr.':>17}")
    for gap in (0, 2, 4, 8, 16, 32, 64):
        rg = [corrida(7000 + s, theta=0.1, gap=gap) for s in range(S)]
        rs = [corrida(7000 + s, theta=0.1, gap=gap, modo="sobrescritura") for s in range(S)]
        m1, h1 = ic95([x[0] for x in rg]); m2, h2 = ic95([x[0] for x in rs])
        print(f"{gap:5d} {np.mean([x[2] for x in rg]):15.3f} "
              f"{m1:>10.3f}±{h1:.3f} {m2:>10.3f}±{h2:.3f}")
