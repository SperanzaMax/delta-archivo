"""Desempate dentro del barrio: como elegir el correcto entre los top-k candidatos.

R7 dejo el mecanismo a mitad de camino: la geometria entrega un barrio chico (el item correcto en
la posicion 12 de 1664) pero no sabe cual de los vecinos es. Falta el desempate.

Recurso disponible que todavia no se uso: el modelo tiene H=4 cabezas, cada una con su PROPIO
espacio de claves y su propia deriva. Son cuatro mediciones parcialmente independientes del mismo
item — la misma logica por la que el GPS cruza varios satelites en vez de confiar en uno.

Criterios comparados:
  1cabeza      similitud cruda en una sola cabeza (la linea de base de R7.2)
  1cabeza+af   idem, con la correccion afin
  suma         suma de similitudes de las 4 cabezas
  RRF          Reciprocal Rank Fusion: sum_h 1/(60 + rank_h). Robusto a escalas distintas.
  suma+af      suma de similitudes ya corregidas
  RRF+af       RRF sobre los rankings corregidos
  mutuo        entre el top-k, se queda con el candidato que TAMBIEN rankea alto a la consulta

Y por ultimo la idea del burbujeo: si la lista de candidatos ya viene casi ordenada, reordenarla
sale casi gratis. Se mide cuantas INVERSIONES hay realmente entre el orden crudo y el orden final,
que es lo que determina el costo de un burbujeo adaptativo.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np

CACHE = "claves_deriva.npz"
KS = (1, 5, 10, 25, 50, 100)


def normalizar(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)


def afin(A, B, anc):
    A1 = np.hstack([A[anc], np.ones((len(anc), 1))])
    W = np.linalg.lstsq(A1, B[anc], rcond=None)[0]
    return normalizar(np.hstack([A, np.ones((len(A), 1))]) @ W)


def rangos_de(sim, hold):
    orden = np.argsort(-sim, 1)
    return np.argmax(orden == hold[:, None], 1)


def recalls(pos):
    return {k: float(np.mean(pos < k)) for k in KS}


def main(n_anclas=256, seed=0):
    z = np.load(CACHE); K0, Kt = z["K0"], z["Kt"]
    Hh, n, d = K0.shape
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n); anc, hold = idx[:n_anclas], idx[n_anclas:]

    sims, sims_af = [], []
    for h in range(Hh):
        A, B = K0[h], Kt[h]
        sims.append(B[hold] @ A.T)
        sims_af.append(B[hold] @ afin(A, B, anc).T)
    S = np.stack(sims)          # (H, |hold|, n)
    Saf = np.stack(sims_af)

    def rrf(SS, k0=60.0):
        out = np.zeros_like(SS[0])
        for h in range(SS.shape[0]):
            r = np.argsort(np.argsort(-SS[h], 1), 1)      # rango de cada candidato
            out += 1.0 / (k0 + r)
        return out

    variantes = {
        "1cabeza":    S[0],
        "1cabeza+af": Saf[0],
        "suma":       S.sum(0),
        "RRF":        rrf(S),
        "suma+af":    Saf.sum(0),
        "RRF+af":     rrf(Saf),
    }

    print("DESEMPATE — deriva real, indice de", n, "entradas,", Hh, "cabezas")
    print(f"correccion afin con {n_anclas} anclas; evaluado en {len(hold)} items held-out\n")
    print(f"{'criterio':>11}" + "".join(f"{'@'+str(k):>9}" for k in KS) + f"{'rango med':>11}")
    pos_ref = None
    for nom, sim in variantes.items():
        pos = rangos_de(sim, hold)
        if nom == "1cabeza":
            pos_ref = pos
        r = recalls(pos)
        print(f"{nom:>11}" + "".join(f"{r[k]:9.3f}" for k in KS)
              + f"{np.median(pos):11.1f}")

    # ---- rank mutuo dentro del barrio ----
    print("\nRANK MUTUO dentro del top-k (¿el candidato tambien me elige a mi?)")
    base = Saf.sum(0)
    A_all = [afin(K0[h], Kt[h], anc) for h in range(Hh)]
    inv = np.stack([A_all[h][:, :] @ Kt[h][hold].T for h in range(Hh)]).sum(0)   # (n, |hold|)
    print(f"{'top-k':>7} {'cobertura':>11} {'acierto tras mutuo':>20} {'recall@1 final':>16}")
    for k in (5, 10, 25, 50):
        orden = np.argsort(-base, 1)[:, :k]
        cob = np.mean([hold[i] in orden[i] for i in range(len(hold))])
        aciertos = 0; dentro = 0
        for i in range(len(hold)):
            cand = orden[i]
            if hold[i] not in cand:
                continue
            dentro += 1
            r_mutuo = inv[cand, i]                       # cuanto me elige cada candidato
            comb = base[i, cand] + r_mutuo
            if cand[int(np.argmax(comb))] == hold[i]:
                aciertos += 1
        print(f"{k:7d} {cob:11.3f} {aciertos/max(dentro,1):20.3f} "
              f"{aciertos/len(hold):16.3f}")

    # ---- ¿cuan desordenada esta la lista? (la pregunta del burbujeo) ----
    print("\nBURBUJEO — ¿que tan cerca esta el orden crudo del orden final?")
    print(f"{'top-k':>7} {'inversiones medias':>20} {'max posible':>12} {'% desorden':>11}")
    crudo = S.sum(0)
    fino = Saf.sum(0)
    for k in (10, 25, 50):
        orden = np.argsort(-crudo, 1)[:, :k]
        invs = []
        for i in range(0, len(hold), 7):                 # submuestreo, es O(k^2)
            cand = orden[i]
            v = fino[i, cand]
            invs.append(int(np.sum(v[:, None] < v[None, :] * np.tri(k, k, -1).T)))
        m = float(np.mean(invs)); mx = k * (k - 1) / 2
        print(f"{k:7d} {m:20.1f} {mx:12.0f} {100*m/mx:11.1f}")


if __name__ == "__main__":
    main()
