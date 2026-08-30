"""Juez de `PREREG_RECOMPENSA_L.md` (SHA 96e750b6). Adjudica L-1 a L-6 sobre los checkpoints.

Por que no se reusa `exactitud.py`: pasa por `ser_cobertura.sondear`, que ABORTA a proposito ante una
unidad `token` —«no tiene score que barrer, su abstencion es un argmax»—. Y la condicion PRINCIPAL de
este pre-registro es justamente `token`.

La regla de decision es la MISMA que usa el entrenamiento (`entrenar.py::_recompensa`, linea del
`pred`): se abstiene cuando **q > 0,5**, con q la masa de NOSE en el softmax de vocabulario para
`token` y sigmoid(a) para `cabeza`. NO es «argmax == NOSE»: con 242 tokens NOSE puede ganar el argmax
con masa 0,3, y juzgar con una regla distinta de la que se entreno seria medir otro modelo.
"""

import argparse
import collections
import os
import pickle
import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.getcwd())
import datos as DAT
import idioma as I
import entrenar as E
from ser import clasificar

PISO = 0.4065


def correr(ruta, n, B, semilla):
    with open(ruta, "rb") as f:
        bulto = pickle.load(f)
    params, cfg = bulto["params"], bulto["config"]
    E._DONDE = cfg.get("donde", "pre")
    E._ABST = cfg.get("abst", "token")
    if "abst" not in params:
        d = params["ln_f"]["g"].shape[-1]
        params = dict(params)
        params["abst"] = {"w": jnp.zeros((d, 1)), "b": jnp.zeros((1,))}
        E._ABST = "token"

    @jax.jit
    def partes(p, ses, cortes, turnos, mask, cons, pos):
        return E._partes(p, ses, cortes, turnos, mask, cons, pos)

    rng = np.random.default_rng(semilla)
    cat = collections.Counter()
    recup_ok = recup_n = 0
    vistos = 0
    while vistos < n:
        b = min(B, n - vistos)
        ses, cortes, turnos, mask, cons, pos, tgt, tipo, meta = DAT.lote(
            rng, b, nivel=cfg["nivel"], n_hechos=4, n_sesiones=4,
            p_nose=cfg.get("p_nose", 0.4), con_meta=True)
        lg, a = partes(params, jnp.array(ses), jnp.array(cortes), jnp.array(turnos),
                       jnp.array(mask), jnp.array(cons), jnp.array(pos))
        lg = np.asarray(lg, dtype=np.float64)
        tgt = np.asarray(tgt)

        lg_v = lg.copy()
        lg_v[:, E.NOSE] = -1e9
        arg = lg_v.argmax(-1)
        if E._ABST == "token":
            p_all = np.exp(lg - lg.max(-1, keepdims=True))
            p_all /= p_all.sum(-1, keepdims=True)
            q = p_all[:, E.NOSE]
        else:
            q = 1.0 / (1.0 + np.exp(-np.asarray(a, dtype=np.float64)))

        for i in range(len(tgt)):
            tok = "NOSE" if q[i] > 0.5 else I.ITOS[int(arg[i])]
            cat[clasificar(tok, I.ITOS[int(tgt[i])], meta[i])] += 1
            if int(tgt[i]) != E.NOSE:                      # RECUP: argmax, sin la decision de callarse
                recup_n += 1
                recup_ok += int(arg[i] == tgt[i])
        vistos += b

    n_tot = sum(cat.values())
    return {
        "paso": bulto.get("paso"),
        "exactitud": (cat["acierto"] + cat["acierto_nose"]) / n_tot,
        "acierto": cat["acierto"] / n_tot,
        "acierto_nose": cat["acierto_nose"] / n_tot,
        "abstencion": (cat["acierto_nose"] + cat["abstencion"]) / n_tot,
        "invento": cat["invento"] / n_tot,
        "err_ident": cat["err_identidad"] / n_tot,
        "recup": recup_ok / max(recup_n, 1),
        "origen": bulto.get("sembrado_de", {}).get("ruta", "-"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("unidades", nargs="+", help="nombres sin .pkl, p.ej. t03_s3")
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--lote", type=int, default=64)
    ap.add_argument("--semilla", type=int, default=54321)
    a = ap.parse_args()

    print("=" * 100)
    print(f"JUEZ · PREREG_RECOMPENSA_L (96e750b6) · n={a.n} · semilla {a.semilla} pareada · "
          f"piso trivial {PISO}")
    print("=" * 100)
    print(f"{'unidad':10} {'paso':>6} {'EXACT':>8} {'>piso':>6} {'abstenc':>8} {'L-3':>5} "
          f"{'acierto':>8} {'noseOK':>7} {'invento':>8} {'RECUP':>7}")
    R = {}
    for u in a.unidades:
        ruta = f"ckpts/{u}.pkl"
        if not os.path.exists(ruta):
            print(f"{u:10}   (sin checkpoint todavia)")
            continue
        r = correr(ruta, a.n, a.lote, a.semilla)
        R[u] = r
        l3 = "si" if 0.05 < r["abstencion"] < 0.95 else "NO"
        print(f"{u:10} {r['paso']:>6} {r['exactitud']:8.4f} "
              f"{'SI' if r['exactitud'] > PISO else 'no':>6} {r['abstencion']:8.4f} {l3:>5} "
              f"{r['acierto']:8.4f} {r['acierto_nose']:7.4f} {r['invento']:8.4f} {r['recup']:7.4f}")

    # --- L-2: contraste pareado L=0 contra L=0,5, mismo origen y misma semilla ------------------
    print("\nL-2 · contraste pareado (L=0 debe superar a su par L=0,5):")
    pares = 0
    gana = 0
    for fam0, fam5 in (("t0", "t5"), ("h0", "h5")):
        for sem in ("s3", "s6"):
            u0, u5 = f"{fam0}3_{sem}", f"{fam5}3_{sem}"
            if u0 in R and u5 in R:
                pares += 1
                d = R[u0]["exactitud"] - R[u5]["exactitud"]
                gana += d > 0
                print(f"   {u0} - {u5} = {d:+.4f}   {'gana L=0' if d > 0 else 'gana L=0,5'}")
    if pares:
        print(f"   -> {gana} de {pares} pares a favor de L=0   "
              f"(L-2 pide 3 de 4)")

    # --- L-4: convergencia desde los dos extremos -----------------------------------------------
    print("\nL-4 · convergencia desde los dos extremos (|abstencion(T0) - abstencion(H0)| < 0,20):")
    for sem in ("s3", "s6"):
        t, h = f"t03_{sem}", f"h03_{sem}"
        if t in R and h in R:
            d = abs(R[t]["abstencion"] - R[h]["abstencion"])
            print(f"   {sem}: T0 {R[t]['abstencion']:.4f} (arranco locuaz) vs "
                  f"H0 {R[h]['abstencion']:.4f} (arranco mudo)  ->  |d| = {d:.4f}  "
                  f"{'CUMPLE' if d < 0.20 else 'no cumple'}")


if __name__ == "__main__":
    main()
