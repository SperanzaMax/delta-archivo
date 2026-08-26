#!/usr/bin/env python3
"""DOS DETECTORES, NO UNO — `PREREG_DOS_DETECTORES.md` (SHA 91494aa0...).

    python sonda_dos_detectores.py ckpts/v3_s0.pkl --n 6000 --salida out.json
    python sonda_dos_detectores.py ckpts/v3_s0.pkl --censo        # solo D-0, barato

La hipotesis: el detector unico falla porque resuelve DOS problemas con un solo numero. Hay dos
fallos distintos y `ser.py` los separa desde el 15-ago —`invento` (la respuesta no estaba y contesto
igual) y `err_identidad` (estaba y trajo la de otro)— con mecanismos distintos. Se prueba si
especializar un detector por fallo le gana al mejor detector unico.

Y su corolario espacial, que es la idea de la bandera de Maxi: la señal vive en la posicion de
MAXIMO FOCO de lectura y se diluye antes de llegar a `pos_q`, que es donde hoy decide la cabeza
(verificado en `entrenar.py:113`). En `pos_q` la entropia de lectura es 1,71-1,77 contra un techo de
ln(6)~1,79, o sea casi uniforme.

UN PROCESO POR UNIDAD, a proposito: `jax.jit` cachea el trace y `donde` queda horneado en el
compilado. Comparar `lat2` contra `pre` en el mismo proceso ya dio una vez numeros identicos que casi
se reportan como «el bug no tenia efecto».
"""
import argparse, json, os, pickle, sys

import numpy as np
import jax, jax.numpy as jnp

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
import idioma as I, datos as DAT, modelo as M
from ser import clasificar

NOSE = I.STOI["NOSE"]
SEM_AJUSTE, SEM_PRUEBA = 90000, 77000          # PREREG §3, fijadas antes de mirar nada
L2 = 1.0                                        # PREREG §3, sin busqueda de hiperparametros
REPS_NULO = 20                                  # PREREG §4 D-3


# ------------------------------------------------------------------ la pasada, con los internos
def internos(params, archivo, turnos, consulta, mask_arch, donde):
    """Replica `responder_con_abst` y ademas devuelve el estado y la distribucion de lectura.

    No se usa `E.predecir_cabeza` a proposito: depende de las globales `_ABST`/`_DONDE` de
    `entrenar.py`, que es justo el desfase que rompio cuatro scripts entre el 18 y el 25-ago. Aca la
    prediccion se reconstruye de `logits` y `ab`, que es lo mismo que hace `predecir_cabeza`.
    """
    a = params["arch"]
    ak = archivo @ a["kw"] + a["ord"][turnos]
    av = archivo @ a["vw"]
    penal = jnp.where(mask_arch, 0.0, -1e9)[:, None, :]
    cap = {}

    def lectura(h):
        q = h @ a["qr"]
        sim = jnp.einsum("btd,bnd->btn", q, ak) / jnp.sqrt(h.shape[-1]) + penal
        p = jax.nn.softmax(sim, -1)
        cap["p"], cap["sim"] = p, sim
        return jnp.einsum("btn,bnd->btd", p, av) @ a["wo"]

    h = M.tronco(params, consulta, lectura, 0, donde)
    hn = M.ln(params["ln_f"], h)
    logits = hn @ params["head"]["w"] + params["head"]["b"]
    ab = (hn @ params["abst"]["w"] + params["abst"]["b"])[..., 0]
    return hn, logits, ab, cap["p"], cap["sim"]


def escalares_lectura(p, sim):
    """4 escalares por posicion: entropia, masa top-1, margen top1-top2, logsumexp. PREREG §3."""
    ent = -(p * np.log(p + 1e-12)).sum(-1)
    orden = np.sort(p, axis=-1)
    top1, top2 = orden[..., -1], orden[..., -2]
    lse = np.log(np.exp(sim - sim.max(-1, keepdims=True)).sum(-1) + 1e-12) + sim.max(-1)
    return np.stack([ent, top1, top1 - top2, lse], -1)


def extraer(ck, n, B, rng_semilla, donde, nivel):
    """Una pasada completa. Devuelve un dict de arrays por muestra."""
    with open(ck, "rb") as f:
        params = jax.tree_util.tree_map(jnp.asarray, pickle.load(f)["params"])
    fn = jax.jit(lambda *a: internos(*a, donde=donde))
    rng = np.random.default_rng(rng_semilla)
    acc = {k: [] for k in ("est_q", "est_foco", "lect_q", "lect_foco", "salida", "cab",
                           "cat", "tgt_nose", "pos", "foco")}
    vistos = 0
    while vistos < n:
        b = min(B, n - vistos)
        ses, cor, tur, mask, cons, pos, tgt, tipo, meta = DAT.lote(
            rng, b, nivel=nivel, n_hechos=4, n_sesiones=4, p_nose=0.4, con_meta=True)
        jses, jcor, jtur = jnp.array(ses), jnp.array(cor), jnp.array(tur)
        jmask, jcons = jnp.array(mask), jnp.array(cons)
        arch = M.escribir(params, jses, jcor)
        hn, lg, ab, p, sim = fn(params, arch, jtur, jcons, jmask)
        hn, lg, ab = np.asarray(hn), np.asarray(lg), np.asarray(ab)
        p, sim = np.asarray(p), np.asarray(sim)
        esc = escalares_lectura(p, sim)
        ent = esc[..., 0]
        for i in range(b):
            pq = int(pos[i])
            f = int(ent[i, :pq + 1].argmin())            # posicion de MAXIMO foco
            lgi = lg[i, pq].copy(); lgi[NOSE] = -np.inf
            pred = NOSE if ab[i, pq] > 0.0 else int(lgi.argmax())
            acc["cat"].append(clasificar(I.ITOS[pred], I.ITOS[int(tgt[i])], meta[i]))
            acc["tgt_nose"].append(int(tgt[i]) == NOSE)
            acc["est_q"].append(hn[i, pq]); acc["est_foco"].append(hn[i, f])
            acc["lect_q"].append(esc[i, pq]); acc["lect_foco"].append(esc[i, f])
            sm = np.exp(lgi - lgi.max()); sm = sm / sm.sum()
            o = np.sort(sm)
            acc["salida"].append([o[-1], o[-1] - o[-2], -(sm * np.log(sm + 1e-12)).sum()])
            acc["cab"].append(float(ab[i, pq]))
            acc["pos"].append(pq); acc["foco"].append(f)
        vistos += b
    out = {k: np.array(v) for k, v in acc.items() if k != "cat"}
    out["cat"] = np.array(acc["cat"])
    return out


# ------------------------------------------------------------------------------ sonda lineal
def auc(y, s):
    """AUC por rangos, con empates promediados."""
    y = np.asarray(y).astype(bool); s = np.asarray(s, float)
    n1, n0 = y.sum(), (~y).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    o = np.argsort(s, kind="mergesort")
    r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1)
    # empates -> rango promedio
    su = np.sort(s); i = 0
    while i < len(su):
        j = i
        while j + 1 < len(su) and su[j + 1] == su[i]:
            j += 1
        if j > i:
            m = (i + j + 2) / 2.0
            r[np.isin(s, su[i])] = m
        i = j + 1
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def logistica(X, y, l2=L2, iters=300, lr=0.5):
    """Newton amortiguado sobre features ya estandarizadas. Sin busqueda de hiperparametros."""
    X = np.c_[X, np.ones(len(X))]
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        z = np.clip(X @ w, -30, 30)
        p = 1 / (1 + np.exp(-z))
        g = X.T @ (p - y) / len(y) + l2 * np.r_[w[:-1], 0.0] / len(y)
        s = p * (1 - p) + 1e-6
        H = (X * s[:, None]).T @ X / len(y) + np.eye(X.shape[1]) * (l2 / len(y) + 1e-6)
        try:
            w = w - lr * np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            w = w - lr * g
    return w


def sonda(Xa, ya, Xp, l2=L2):
    """Ajusta en A, puntua en P. Estandariza con la media y el desvio DE A (PREREG §3)."""
    mu, sd = Xa.mean(0), Xa.std(0) + 1e-8
    w = logistica((Xa - mu) / sd, ya.astype(float), l2)
    return (np.c_[(Xp - mu) / sd, np.ones(len(Xp))] @ w)


def bloque(d, nombres):
    return np.c_[tuple(d[n].reshape(len(d[n]), -1) for n in nombres)]


# ---------------------------------------------------------------------------------- el censo
def censo(d):
    cats, n = d["cat"], len(d["cat"])
    c = {k: int((cats == k).sum()) for k in np.unique(cats)}
    con = ~d["tgt_nose"]; sin = d["tgt_nose"]
    r = {
        "n": n,
        "por_categoria": c,
        "n_con_respuesta": int(con.sum()), "n_sin_respuesta": int(sin.sum()),
        # las tres metricas publicadas, para D-0
        "acierto": float((cats[con] == "acierto").sum() / max(1, con.sum())),
        "nose": float((cats[sin] == "acierto_nose").sum() / max(1, sin.sum())),
        "falsa_abst": float((cats[con] == "abstencion").sum() / max(1, con.sum())),
        # lo que decide si D-2 es medible
        "err_identidad": float((cats == "err_identidad").sum() / max(1, con.sum())),
        "invento": float((cats == "invento").sum() / max(1, sin.sum())),
        "foco_igual_posq": float((d["foco"] == d["pos"]).mean()),
        "foco_medio": float(d["foco"].mean()), "pos_media": float(d["pos"].mean()),
    }
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pesos")
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--B", type=int, default=64)
    ap.add_argument("--censo", action="store_true", help="solo D-0, una sola muestra")
    ap.add_argument("--salida", default=None)
    a = ap.parse_args()

    bulto = pickle.load(open(a.pesos, "rb"))
    cfg = bulto["config"]
    donde, nivel = cfg.get("donde", "pre"), cfg["nivel"]
    sem = cfg.get("semilla", 0)
    unidad = os.path.basename(a.pesos).replace(".pkl", "")
    print(f"# {unidad} · nivel {nivel} · donde={donde} · abst={cfg.get('abst')} · "
          f"pasos {cfg.get('pasos')}", flush=True)

    A = extraer(a.pesos, a.n, a.B, SEM_AJUSTE + sem, donde, nivel)
    res = {"unidad": unidad, "donde": donde, "nivel": nivel, "semilla": sem,
           "n": a.n, "censo_ajuste": censo(A)}
    print(json.dumps(res["censo_ajuste"], indent=2, ensure_ascii=False), flush=True)
    if a.censo:
        if a.salida:
            json.dump(res, open(a.salida, "w"), indent=2, ensure_ascii=False)
        return

    P = extraer(a.pesos, a.n, a.B, SEM_PRUEBA + sem, donde, nivel)
    res["censo_prueba"] = censo(P)

    def err(d):    # blanco compuesto de D-1: «el modelo se equivoco»
        return np.isin(d["cat"], ["invento", "err_identidad", "err_version", "err_fuera"])

    todo = ["est_q", "est_foco", "lect_q", "lect_foco", "salida", "cab"]
    r = {}

    # ---- D-1 · unico contra compuesto
    s_unico = sonda(bloque(A, todo), err(A), bloque(P, todo))
    r["D1_unico"] = auc(err(P), s_unico)

    conA, conP = ~A["tgt_nose"], ~P["tgt_nose"]
    sA = sonda(bloque(A, todo), A["tgt_nose"], bloque(P, todo))          # detector de AUSENCIA
    malA = A["cat"][conA] == "err_identidad"
    sB_p = sonda(bloque(A, todo)[conA], malA, bloque(P, todo))           # detector de ATRIBUCION
    pa, pb = 1 / (1 + np.exp(-np.clip(sA, -30, 30))), 1 / (1 + np.exp(-np.clip(sB_p, -30, 30)))
    r["D1_compuesto"] = auc(err(P), pa + (1 - pa) * pb)
    r["D1_solo_ausencia"] = auc(err(P), sA)
    r["D1_delta"] = r["D1_compuesto"] - r["D1_unico"]
    r["D1_auc_ausencia"] = auc(P["tgt_nose"], sA)

    # ---- D-2 · foco contra pos_q, sobre el blanco de mala atribucion
    malP = P["cat"][conP] == "err_identidad"
    if malA.sum() >= 20 and malP.sum() >= 20:
        f_foco = sonda(bloque(A, ["est_foco", "lect_foco"])[conA], malA,
                       bloque(P, ["est_foco", "lect_foco"])[conP])
        f_q = sonda(bloque(A, ["est_q", "lect_q"])[conA], malA,
                    bloque(P, ["est_q", "lect_q"])[conP])
        r["D2_foco"], r["D2_posq"] = auc(malP, f_foco), auc(malP, f_q)
        r["D2_delta"] = r["D2_foco"] - r["D2_posq"]
    else:
        r["D2_foco"] = r["D2_posq"] = r["D2_delta"] = None
        r["D2_motivo"] = f"casos insuficientes: ajuste {int(malA.sum())}, prueba {int(malP.sum())}"

    # ---- D-3 · nulo, tiene que fallar
    rng = np.random.default_rng(4242)
    nulos = []
    for _ in range(REPS_NULO):
        yp = rng.permutation(err(A))
        nulos.append(auc(err(P), sonda(bloque(A, todo), yp, bloque(P, todo))))
    r["D3_nulo_medio"] = float(np.mean(nulos)); r["D3_nulo_sd"] = float(np.std(nulos))

    # ---- D-5 · contraste honesto contra el detector que ya existe
    r["D5_cabeza_sola_error"] = auc(err(P), P["cab"])
    r["D5_cabeza_sola_ausencia"] = auc(P["tgt_nose"], P["cab"])
    r["D5_salida_sola_error"] = auc(err(P), -P["salida"][:, 0])

    res["resultados"] = r
    print(json.dumps(r, indent=2, ensure_ascii=False), flush=True)
    if a.salida:
        json.dump(res, open(a.salida, "w"), indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
