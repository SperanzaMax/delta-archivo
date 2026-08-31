"""PREREG_TECHO_EVIDENCIA.md (SHA d7e3383b) · ¿el 0,70 es techo del ESTADO o del LECTOR?

Cinco lectores sobre los mismos dos checkpoints, todos por SOLUCION CERRADA (ridge) para que no haya
convergencia que verificar —la leccion del 13-ago: un negativo sin barrido de lr no es un negativo, y
la forma de esquivarla es no tener lr—.

  L1  lineal sobre los LOGITS (242)          <- replica del 0,7003 de sonda_ausencia_lineal.py
  L2  lineal sobre el ESTADO FINAL `hn` (128)
  L3  NO LINEAL sobre `hn`: random features (proyeccion aleatoria fija + tanh, 1024) + ridge.
      Sigue siendo solucion cerrada: la no linealidad esta en las features, no en el ajuste.
  L4  lineal sobre la salida de CADA bloque
  L5  lineal sobre el resumen de la BUSQUEDA (s_max, brecha top-2, entropia, norma de lo leido)

Controles en todos, y si fallan no se lee el numero: NULO (etiquetas barajadas -> ~0,50) y TECHO
(«¿el argmax es un nombre?» -> ~1,00).

Costo: CPU, minutos. No toca checkpoints ni corridas.
"""
import json
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)

import datos as DAT      # noqa: E402
import entrenar as E     # noqa: E402
import idioma as I       # noqa: E402
import medir_ratio_ce as R   # noqa: E402
import modelo as M       # noqa: E402

N = 6144
SEMILLA = 54321
IDS_NOM = None           # se llena despues de fijar la version del idioma


def auc(s, pos):
    o = np.argsort(s)
    r = np.empty(len(s))
    r[o] = np.arange(1, len(s) + 1)
    n1, n0 = pos.sum(), (~pos).sum()
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else float("nan")


def ridge_auc(Xtr, ytr, Xte, yte, lams=(1e-2, 1e0, 1e2, 1e4)):
    G = Xtr.T @ Xtr
    b = Xtr.T @ ytr
    mejor = (-1.0, None)
    for lam in lams:
        w = np.linalg.solve(G + lam * np.eye(G.shape[0]), b)
        a = auc(Xte @ w, yte)
        if a > mejor[0]:
            mejor = (a, lam)
    return mejor


def evaluar(nombre, F, no, es_nom, tr, te, rng):
    """Corre el lector sobre las features F y devuelve (senal, nulo, techo, lambda)."""
    mu, sd = F[tr].mean(0), F[tr].std(0) + 1e-6
    X = np.hstack([(F - mu) / sd, np.ones((len(F), 1))])
    Xtr, Xte = X[tr], X[te]
    a_sen, lam = ridge_auc(Xtr, no[tr].astype(float), Xte, no[te])
    yb = rng.permutation(no[tr].astype(float))
    a_nul, _ = ridge_auc(Xtr, yb, Xte, no[te])
    a_tec, _ = ridge_auc(Xtr, es_nom[tr].astype(float), Xte, es_nom[te])
    ok = a_tec > 0.90 and abs(a_nul - 0.5) < 0.05
    print(f"  {nombre:34s} senal {a_sen:.4f}   techo {a_tec:.4f}   nulo {a_nul:.4f}"
          f"   {'OK' if ok else '** CONTROLES FALLAN, no se lee **'}")
    return {"senal": a_sen, "nulo": a_nul, "techo": a_tec, "lambda": lam, "confiable": bool(ok)}


def cosechar(params, cfg, n, B, semilla, p_nose):
    """Junta logits, estado final, salida por bloque y resumen de la busqueda. Un solo barrido."""
    nivel = cfg["nivel"]
    a_p = params["arch"]

    def partes(params, ses, cortes, turnos, mask, cons, pos):
        archivo = M.escribir(params, ses, cortes)
        ak = archivo @ a_p["kw"] + a_p["ord"][turnos]
        av = archivo @ a_p["vw"]
        penal = jnp.where(mask, 0.0, -1e9)[:, None, :]
        guard = {}

        def lectura(h):
            q = h @ a_p["qr"]
            sim = jnp.einsum("btd,bnd->btn", q, ak) / jnp.sqrt(h.shape[-1]) + penal
            p = jax.nn.softmax(sim, -1)
            leido = jnp.einsum("btn,bnd->btd", p, av) @ a_p["wo"]
            guard["sim"], guard["p"], guard["leido"] = sim, p, leido
            return leido

        # L4 · el estado DESPUES de cada bloque. Se obtiene corriendo el mismo `tronco` con el arbol
        # recortado a los primeros k bloques: es exacto —el loop simplemente recorre menos— y evita
        # copiar aca la logica de la conv, el mixer y el MLP, que es lo que se rompe en silencio
        # cuando `modelo.py` cambia. La lectura entra en el bloque 0, asi que con k >= 1 siempre esta.
        hs = []
        for k in range(1, len(params["blocks"]) + 1):
            pk = dict(params)
            pk["blocks"] = params["blocks"][:k]
            hs.append(M.tronco(pk, cons, lectura, 0, cfg.get("donde", "pre")))

        h = M.tronco(params, cons, lectura, 0, cfg.get("donde", "pre"))
        hn = M.ln(params["ln_f"], h)
        lg = hn @ params["head"]["w"] + params["head"]["b"]
        return lg, hn, guard, hs

    rng = np.random.default_rng(semilla)
    LG, HN, TGT, BUS = [], [], [], []
    CAPAS = None
    vistos = 0
    while vistos < n:
        b = min(B, n - vistos)
        ses, cortes, turnos, mask, cons, pos, tgt, tipo = DAT.lote(
            rng, b, nivel=nivel, n_hechos=4, n_sesiones=4, p_nose=p_nose)
        aj = [jnp.array(x) for x in (ses, cortes, turnos, mask, cons, pos)]
        lg, hn, g, hs = partes(params, *aj)
        pos_j = aj[5]
        if CAPAS is None:
            CAPAS = [[] for _ in hs]
        tomar = lambda x: np.asarray(jnp.take_along_axis(
            x, pos_j[:, None, None], axis=1)[:, 0, :])
        LG.append(tomar(lg))
        HN.append(tomar(hn))
        sim = tomar(g["sim"])
        p = tomar(g["p"])
        leido = tomar(g["leido"])
        srt = np.sort(sim, axis=-1)
        BUS.append(np.stack([
            srt[:, -1],                                   # s_max
            srt[:, -1] - srt[:, -2],                      # brecha top-2
            -(p * np.log(p + 1e-9)).sum(-1),              # entropia de la lectura
            np.linalg.norm(leido, axis=-1),               # cuanto se leyo
            p.max(-1),                                    # masa de la entrada ganadora
        ], axis=-1))
        for j, hk in enumerate(hs):
            CAPAS[j].append(tomar(hk))
        TGT.append(np.asarray(tgt))
        vistos += b
    return (np.concatenate(LG).astype(np.float64), np.concatenate(HN).astype(np.float64),
            np.concatenate(BUS).astype(np.float64), np.concatenate(TGT),
            [np.concatenate(c).astype(np.float64) for c in CAPAS])


def main():
    global IDS_NOM
    res = {}
    print("=" * 92)
    print("SONDA DEL TECHO · PREREG_TECHO_EVIDENCIA.md (SHA d7e3383b)")
    print("=" * 92)

    for ruta in ("ckpts/n3_s0.pkl", "ckpts/t03_s3.pkl"):
        params, cfg, paso = R.cargar(os.path.join(AQUI, ruta))
        I.fijar_version(cfg.get("idioma", 2))
        IDS_NOM = np.array([I.STOI[t] for t in I.NOMBRES])
        lg, hn, bus, tgt, capas = cosechar(params, cfg, N, 64, SEMILLA, 0.4)
        no = (tgt == E.NOSE)
        lg_v = lg.copy()
        lg_v[:, E.NOSE] = -1e9
        es_nom = np.isin(lg_v.argmax(-1), IDS_NOM)

        n = len(lg)
        idx = np.random.default_rng(0).permutation(n)
        tr, te = idx[:n // 2], idx[n // 2:]
        rng = np.random.default_rng(1)

        print(f"\n--- {ruta}  paso={paso}  ·  held-out {len(te)}  ·  sin respuesta {no.mean():.4f} ---")
        r = {}

        # --- T-0 · el control ARITMETICO, y va primero -----------------------------------------
        W = np.asarray(params["head"]["w"], dtype=np.float64)
        rango = int(np.linalg.matrix_rank(W))
        print(f"  T-0 · rango de head.w = {rango} de {min(W.shape)}"
              f"   ({'L1 y L2 son el MISMO espacio de funciones' if rango >= W.shape[0] else 'rango deficiente: L2 puede tener MAS'})")
        r["rango_head"] = rango

        r["L1_logits"] = evaluar("L1 lineal · logits (242)", lg, no, es_nom, tr, te, rng)
        r["L2_estado"] = evaluar("L2 lineal · estado final (128)", hn, no, es_nom, tr, te, rng)

        # --- L3 · no lineal, sin optimizador ----------------------------------------------------
        d = hn.shape[1]
        rf = np.random.default_rng(7)
        P = rf.normal(size=(d, 1024)) / np.sqrt(d)
        bsh = rf.uniform(0, 2 * np.pi, size=1024)
        sd0 = hn.std(0) + 1e-6
        Z = np.tanh((hn - hn.mean(0)) / sd0 @ P + bsh)
        r["L3_nolineal"] = evaluar("L3 NO LINEAL · random features", Z, no, es_nom, tr, te, rng)

        for j, hk in enumerate(capas):
            r[f"L4_bloque{j + 1}"] = evaluar(f"L4 lineal · tras el bloque {j + 1}", hk,
                                             no, es_nom, tr, te, rng)

        r["L5_busqueda"] = evaluar("L5 lineal · busqueda cruda (5)", bus, no, es_nom, tr, te, rng)
        res[ruta] = r

    # --- veredicto contra los criterios, escritos antes -----------------------------------------
    print(f"\n{'=' * 92}")
    sanas = res["ckpts/n3_s0.pkl"]
    mejor = max((v["senal"], k) for k, v in sanas.items()
                if isinstance(v, dict) and v.get("confiable"))
    print(f"MEJOR lector confiable en n3_s0: {mejor[1]} con {mejor[0]:.4f}")
    print(f"  T-1 (nada supera 0,73): {'CUMPLE' if mejor[0] <= 0.73 else 'NO CUMPLE'}")
    print(f"  T-2 (algo supera 0,75): {'SE DISPARA -> el 0,70 era del LECTOR' if mejor[0] > 0.75 else 'no'}")
    l5 = sanas["L5_busqueda"]["senal"]
    print(f"  T-3 (L5 <= 0,55): {'CUMPLE' if l5 <= 0.55 else 'NO CUMPLE'}   (L5 = {l5:.4f})")
    dif = mejor[0] - max(v["senal"] for k, v in res["ckpts/t03_s3.pkl"].items()
                         if isinstance(v, dict) and v.get("confiable"))
    print(f"  T-4 (n3_s0 supera a t03_s3 por >= 0,08): "
          f"{'CUMPLE' if dif >= 0.08 else 'NO CUMPLE'}   (diferencia {dif:+.4f})")
    res["veredicto"] = {"mejor": mejor[1], "auc": mejor[0], "T1": bool(mejor[0] <= 0.73),
                        "T2": bool(mejor[0] > 0.75), "T3": bool(l5 <= 0.55),
                        "T4": bool(dif >= 0.08), "dif_recup": dif}

    sal = os.path.join(AQUI, "sonda_techo_20260831.json")
    with open(sal, "w") as f:
        json.dump(res, f, indent=1)
    print(f"-> {os.path.basename(sal)}")


if __name__ == "__main__":
    main()
