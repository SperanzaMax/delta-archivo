"""Modelo minimo absoluto de la memoria por gemacion: geometria pura, sin red, sin entrenar.

Idea a testear (Maxi, 2026-08-08): al revisar un recuerdo no se sobrescribe — se deposita una
version nueva EN UN LUGAR CERCANO, porque la cercania codifica la correlacion con la version
anterior. La pregunta previa a cualquier arquitectura es geometrica:

    ¿existe un radio de gemacion eps que mantenga las versiones lo bastante juntas para que la
    consulta las encuentre como grupo, y lo bastante separadas para no confundirse entre si?

Montaje: N recuerdos base en la esfera S^{d-1}. Cada uno recibe K revisiones sucesivas, cada una
depositada a distancia eps de la anterior en direccion aleatoria. Se consulta con la clave
ORIGINAL mas ruido (el caso de uso real: preguntas por el concepto y queres la info al dia).

Tres metricas, que separan tres cosas distintas:
  M1 recall@1 de la version MAS RECIENTE     -> ¿la geometria sola alcanza para desempatar?
  M2 recall de CLUSTER (top-1 es del recuerdo correcto, cualquier version) -> ¿agrupa bien?
  M3 pureza del vecindario top-K            -> si es alta, se puede desempatar por metadato
                                                (recencia) y M1 deja de importar.
"""
import numpy as np

RNG = np.random.default_rng(20260808)


def esfera(rng, n, d):
    x = rng.normal(size=(n, d))
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def construir(rng, N, K, d, eps):
    """Devuelve direcciones (N*(K+1), d), id de recuerdo, indice de version."""
    base = esfera(rng, N, d)
    dirs = [base]
    for _ in range(K):
        u = esfera(rng, N, d)                       # direccion de la revision
        nxt = dirs[-1] + eps * u
        dirs.append(nxt / np.linalg.norm(nxt, axis=1, keepdims=True))
    A = np.concatenate(dirs, 0)                     # orden: v0 de todos, v1 de todos, ...
    mem_id = np.tile(np.arange(N), K + 1)
    ver = np.repeat(np.arange(K + 1), N)
    return A, mem_id, ver


def evaluar(rng, N=200, K=4, d=64, eps=0.3, ruido=0.05, topk=5, reps=20):
    m1 = m2 = m3 = 0.0
    for _ in range(reps):
        A, mem_id, ver = construir(rng, N, K, d, eps)
        base = A[:N]                                            # v0 de cada recuerdo
        q = base + ruido * esfera(rng, N, d)
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        sim = q @ A.T                                           # (N, N*(K+1))
        top1 = np.argmax(sim, 1)
        m1 += np.mean((mem_id[top1] == np.arange(N)) & (ver[top1] == K))
        m2 += np.mean(mem_id[top1] == np.arange(N))
        idx = np.argpartition(-sim, topk, axis=1)[:, :topk]
        m3 += np.mean(mem_id[idx] == np.arange(N)[:, None])
    return m1 / reps, m2 / reps, m3 / reps


if __name__ == "__main__":
    print("Gemacion: N=200 recuerdos, K=4 revisiones c/u, ruido de consulta 0.05, top-5\n")
    for d in (16, 64, 256):
        print(f"--- d = {d} ---")
        print(f"{'eps':>6} {'M1 rec@1 reciente':>18} {'M2 cluster ok':>14} {'M3 pureza top5':>16}")
        for eps in (0.0, 0.05, 0.1, 0.2, 0.4, 0.8, 1.5):
            a, b, c = evaluar(RNG, d=d, eps=eps)
            print(f"{eps:6.2f} {a:18.3f} {b:14.3f} {c:16.3f}")
        print()

    print("Capacidad del vecindario: ¿cuantas versiones distinguibles caben a distancia eps?")
    print(f"{'d':>6} {'eps':>6} {'K=2':>8} {'K=8':>8} {'K=32':>8}   (M3 pureza top-5)")
    for d in (16, 64, 256):
        for eps in (0.1, 0.3):
            fila = [evaluar(RNG, K=K, d=d, eps=eps)[2] for K in (2, 8, 32)]
            print(f"{d:6d} {eps:6.2f} {fila[0]:8.3f} {fila[1]:8.3f} {fila[2]:8.3f}")


# ---------------------------------------------------------------------------
# Variante 2: EJE TEMPORAL. La direccion de la revision no es aleatoria: tiene
# una componente comun t_hat compartida por todas las revisiones del indice.
# Si eso funciona, la recencia se vuelve LEGIBLE EN LA GEOMETRIA y no hace
# falta metadato: consultar sesgado hacia +t da la version vigente, hacia -t
# da la historia.
# ---------------------------------------------------------------------------
def evaluar_eje(rng, N=200, K=4, d=64, eps=0.3, alpha=0.7, delta=0.0,
                ruido=0.05, reps=20):
    """alpha = cuanto de la revision va en la direccion comun t_hat.
    delta  = cuanto se sesga la CONSULTA hacia +t_hat."""
    m1 = m2 = 0.0
    for _ in range(reps):
        that = esfera(rng, 1, d)[0]
        base = esfera(rng, N, d)
        dirs = [base]
        for _ in range(K):
            u = alpha * that + (1 - alpha) * esfera(rng, N, d)
            u /= np.linalg.norm(u, axis=1, keepdims=True)
            nxt = dirs[-1] + eps * u
            dirs.append(nxt / np.linalg.norm(nxt, axis=1, keepdims=True))
        A = np.concatenate(dirs, 0)
        mem_id = np.tile(np.arange(N), K + 1); ver = np.repeat(np.arange(K + 1), N)
        q = base + ruido * esfera(rng, N, d) + delta * that
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        top1 = np.argmax(q @ A.T, 1)
        m1 += np.mean((mem_id[top1] == np.arange(N)) & (ver[top1] == K))
        m2 += np.mean(mem_id[top1] == np.arange(N))
    return m1 / reps, m2 / reps


if __name__ == "__main__":
    print("\n" + "=" * 66)
    print("EJE TEMPORAL (d=64, eps=0.3, K=4): ¿se vuelve legible la recencia?")
    print(f"{'alpha':>6} {'delta':>6} {'M1 rec@1 reciente':>18} {'M2 cluster ok':>14}")
    for alpha in (0.0, 0.5, 0.9):
        for delta in (0.0, 0.3, 0.6, 1.0):
            a, b = evaluar_eje(RNG, alpha=alpha, delta=delta)
            print(f"{alpha:6.1f} {delta:6.1f} {a:18.3f} {b:14.3f}")
