#!/usr/bin/env python3
"""A3 (calibrar) y A4 (ensamble de semillas) — `ALTERNATIVAS_DETECCION_20260826.md`.

    python sonda_calibra_ensamble.py --dump ckpts/p3_s0.pkl          # un proceso por unidad
    python sonda_calibra_ensamble.py --analizar dump/p3_s0.npz dump/p3_s1.npz dump/p3_s2.npz

**EXPLORATORIO Y DECLARADO COMO TAL.** No tiene pre-registro y NO confirma nada, igual que
`sonda_umbral.py` del 18-ago. Son dos contrastes que le ponen piso y techo a cualquier detector
propio, y que el proyecto nunca midio:

  A3 · el techo medido NO es de capacidad (AUC del logit 0,777-0,998) sino de CALIBRACION. Se mide
       cuanto de la brecha cierra elegir el corte bien, con la leccion del 19-ago aplicada: el umbral
       se elige pidiendo MARGEN (0,07) y se juzga con el criterio real (0,10), porque el optimo
       pegado al borde no generaliza.

  A4 · el ensamble de semillas es el baseline fuerte de la literatura y aca nunca se corrio. Las tres
       unidades tienen que ver LAS MISMAS preguntas, asi que la semilla de generacion es fija y NO
       depende de la unidad — es la diferencia con `sonda_dos_detectores.py`, donde cada unidad tiene
       su propia muestra.

       ⚠ En este banco la variacion entre semillas es BIMODAL (E-I3c): las semillas difieren en
       CAPACIDAD y no solo en ruido, asi que el desacuerdo puede estar midiendo «esta semilla es la
       mala» en vez de «esta pregunta es dificil». Se reporta con esa advertencia.

UN PROCESO POR UNIDAD para el dump: `jax.jit` hornea `donde` en el trace.
"""
import argparse, json, os, pickle, sys

import numpy as np
import jax, jax.numpy as jnp

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
import idioma as I, datos as DAT, modelo as M
from ser import clasificar

NOSE = I.STOI["NOSE"]
SEM_A, SEM_P = 55000, 66000        # FIJAS y sin sumar la semilla: el ensamble necesita el mismo lote
MARGEN_ELEGIR, CRITERIO = 0.07, 0.10          # leccion del 19-ago
N_CORTES = 400


def pasada(ck, n, B, semilla):
    b = pickle.load(open(ck, "rb"))
    params = jax.tree_util.tree_map(jnp.asarray, b["params"])
    donde, nivel = b["config"].get("donde", "pre"), b["config"]["nivel"]

    def fn(params, arch, tur, cons, mask):
        a = params["arch"]
        ak = arch @ a["kw"] + a["ord"][tur]
        av = arch @ a["vw"]
        penal = jnp.where(mask, 0.0, -1e9)[:, None, :]

        def lectura(h):
            q = h @ a["qr"]
            sim = jnp.einsum("btd,bnd->btn", q, ak) / jnp.sqrt(h.shape[-1]) + penal
            return jnp.einsum("btn,bnd->btd", jax.nn.softmax(sim, -1), av) @ a["wo"]

        h = M.tronco(params, cons, lectura, 0, donde)
        hn = M.ln(params["ln_f"], h)
        return (hn @ params["head"]["w"] + params["head"]["b"],
                (hn @ params["abst"]["w"] + params["abst"]["b"])[..., 0])

    jfn = jax.jit(fn)
    rng = np.random.default_rng(semilla)
    P = {k: [] for k in ("pred", "cab", "conf", "cat", "tgt_nose", "arg", "tgt")}
    vistos = 0
    while vistos < n:
        bb = min(B, n - vistos)
        ses, cor, tur, mask, cons, pos, tgt, tipo, meta = DAT.lote(
            rng, bb, nivel=nivel, n_hechos=4, n_sesiones=4, p_nose=0.4, con_meta=True)
        lg, ab = jfn(params, M.escribir(params, jnp.array(ses), jnp.array(cor)),
                     jnp.array(tur), jnp.array(cons), jnp.array(mask))
        lg, ab = np.asarray(lg), np.asarray(ab)
        for i in range(bb):
            pq = int(pos[i])
            v = lg[i, pq].copy(); v[NOSE] = -np.inf
            pred = NOSE if ab[i, pq] > 0.0 else int(v.argmax())
            sm = np.exp(v - v.max()); sm = sm / sm.sum()
            P["pred"].append(pred); P["cab"].append(float(ab[i, pq]))
            P["conf"].append(float(sm.max()))
            P["cat"].append(clasificar(I.ITOS[pred], I.ITOS[int(tgt[i])], meta[i]))
            P["tgt_nose"].append(int(tgt[i]) == NOSE)
            # el argmax SIN mirar la cabeza y el target crudo: son lo unico con lo que se puede
            # armar el blanco no contaminado («si contestara, ¿estaria mal?»). Guardarlos aca evita
            # repetir el error D-D3, que fue juzgar un detector contra un blanco que el ya modifico.
            P["arg"].append(int(v.argmax())); P["tgt"].append(int(tgt[i]))
        vistos += bb
    return {k: np.array(v) for k, v in P.items()}, donde, nivel


def auc(y, s):
    """AUC de Mann-Whitney con rangos promediados en los empates.

    Los empates importan de verdad aca: `acuerdo` del ensamble toma pocos valores distintos (con
    tres unidades, solo 1/3, 2/3 y 1), asi que una implementacion que rompa empates por el orden del
    array daria un numero que depende de como venia ordenado — el mismo error que en el Spearman del
    19-ago, donde el coeficiente se movia entre corridas con los mismos datos.
    """
    y = np.asarray(y).astype(bool); s = np.asarray(s, float)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    o = np.argsort(s, kind="mergesort")
    su = s[o]
    r = np.empty(len(s), float)
    i = 0
    while i < len(su):
        j = i
        while j + 1 < len(su) and su[j + 1] == su[i]:
            j += 1
        r[o[i:j + 1]] = (i + j + 2) / 2.0        # rango promedio, 1-indexado
        i = j + 1
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def compuerta(cat, tgt_nose, decide_nose):
    """`nose` y `falsa_abst` bajo una regla de decision arbitraria de abstencion."""
    con, sin = ~tgt_nose, tgt_nose
    # con `decide_nose` True se abstiene; si no, mantiene lo que el modelo contesto
    nose = float((decide_nose[sin]).mean()) if sin.sum() else float("nan")
    fa = float((decide_nose[con]).mean()) if con.sum() else float("nan")
    return nose, fa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default=None)
    ap.add_argument("--analizar", nargs="*", default=None)
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--B", type=int, default=64)
    a = ap.parse_args()

    if a.dump:
        u = os.path.basename(a.dump).replace(".pkl", "")
        os.makedirs(f"{AQUI}/dump", exist_ok=True)
        A, donde, nivel = pasada(a.dump, a.n, a.B, SEM_A)
        P, _, _ = pasada(a.dump, a.n, a.B, SEM_P)
        np.savez(f"{AQUI}/dump/{u}.npz", donde=donde, nivel=nivel,
                 **{f"A_{k}": v for k, v in A.items()},
                 **{f"P_{k}": v for k, v in P.items()})
        print(f"{u} · donde={donde} · n={a.n} · guardado", flush=True)
        return

    res = {}
    ds = {os.path.basename(f).replace(".npz", ""): np.load(f, allow_pickle=True)
          for f in a.analizar}

    # ---------------- A3 · calibracion, por unidad
    for u, d in ds.items():
        catA, catP = d["A_cat"], d["P_cat"]
        nA, nP = d["A_tgt_nose"], d["P_tgt_nose"]
        cabA, cabP = d["A_cab"], d["P_cab"]
        r = {"AUC_cab_ausencia": auc(nP, cabP)}
        # el corte que se usa hoy
        r["sigma05"] = compuerta(catP, nP, cabP > 0.0)
        # elegido en A pidiendo MARGEN, juzgado en P con el criterio real
        cortes = np.quantile(cabA, np.linspace(0.001, 0.999, N_CORTES))
        mejor, best = None, -9
        for c in cortes:
            no, fa = compuerta(catA, nA, cabA > c)
            if fa <= MARGEN_ELEGIR and no > best:
                best, mejor = no, float(c)
        if mejor is None:
            r["calibrado"] = None; r["umbral"] = None
        else:
            r["umbral"] = mejor
            r["calibrado"] = compuerta(catP, nP, cabP > mejor)
        # oraculo: el mejor corte MIRANDO P. Es techo, no metodo.
        best, orac = -9, None
        for c in cortes:
            no, fa = compuerta(catP, nP, cabP > c)
            if fa <= CRITERIO and no > best:
                best, orac = no, (no, fa)
        r["oraculo"] = orac
        res[u] = r

    # ---------------- A4 · ensamble, sobre el lote COMPARTIDO
    us = list(ds)
    if len(us) >= 3:
        preds = np.stack([ds[u]["P_pred"] for u in us])          # (U, n)
        cat0 = ds[us[0]]["P_cat"]
        # el desacuerdo: cuantas de las U unidades coinciden con la moda
        # el acuerdo se mide sobre el ARGMAX de cada unidad, no sobre `pred`: `pred` ya incorpora
        # la decision de la cabeza, y meterla aca haria que el ensamble estuviera midiendo en parte
        # su propio detector.
        args = np.stack([ds[u]["P_arg"] for u in us])
        moda = np.array([np.bincount(args[:, i]).argmax() for i in range(args.shape[1])])
        acuerdo = (args == moda).mean(0)
        tgt = ds[us[0]]["P_tgt"]
        mal0 = ds[us[0]]["P_arg"] != tgt          # BLANCO LIMPIO: «si u0 contestara, estaria mal»
        mal_moda = moda != tgt
        res["A4_ensamble"] = {
            "unidades": us,
            "blanco": "argmax != tgt, sin mirar la cabeza (no contaminado)",
            "tasa_mal_u0": float(mal0.mean()),
            "AUC_acuerdo_vs_error_de_u0": auc(mal0, -acuerdo),
            "contraste_conf_u0": auc(mal0, -ds[us[0]]["P_conf"]),
            "contraste_cab_u0": auc(mal0, ds[us[0]]["P_cab"]),
            "acuerdo_medio": float(acuerdo.mean()),
            "acierto_argmax_u0": float((~mal0).mean()),
            "acierto_argmax_moda": float((~mal_moda).mean()),
        }
    print(json.dumps(res, indent=2, ensure_ascii=False, default=float), flush=True)
    json.dump(res, open(f"{AQUI}/dos_detectores/calibra_ensamble.json", "w"),
              indent=2, ensure_ascii=False, default=float)


if __name__ == "__main__":
    main()
