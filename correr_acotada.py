"""Gemación con desplazamiento ACOTADO — implementa PREREG_GEMACION_ACOTADA.md (SHA ab3115e2…).

Nada acá decide nada: geometría, umbrales, punto de test y orden de variantes están congelados en el
pre-registro. Este archivo sólo los ejecuta.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from correr_hechos import tangente, ic_t, EPS, K, N_SEMILLAS, N_SUB, MARGEN

SALIDA_K = "hechos_revisiones.npz"
K_REVISIONES = (1, 2, 4, 8)
GAMMA = 0.5                       # §3 del prereg: cota 2ε, no ajustado a ningún resultado
INFORME = "INFORME_GEMACION_ACOTADA.md"
CONDS = ("g_orbita", "g_decay", "g_fija", "duplicados")


def direcciones(cond, EV, i, rng, k_rev):
    """Entradas (dirección, versión) de la entidad `i` con k_rev revisiones."""
    E0 = EV[0][i]
    out = [(E0, 0)]
    if cond == "g_fija":                       # la caminata falsada por P4 (control de mecanismo)
        eje = tangente(rng.normal(size=EV.shape[2]).astype(np.float32), E0)
        prev = E0
        for r in range(1, k_rev + 1):
            n = prev + EPS * tangente(eje, prev)
            n /= np.linalg.norm(n) + 1e-8
            out.append((n, r)); prev = n
    elif cond == "g_decay":                    # paso ε·γ^(r−1): arco total ≤ 2ε
        eje = tangente(rng.normal(size=EV.shape[2]).astype(np.float32), E0)
        prev = E0
        for r in range(1, k_rev + 1):
            n = prev + EPS * (GAMMA ** (r - 1)) * tangente(eje, prev)
            n /= np.linalg.norm(n) + 1e-8
            out.append((n, r)); prev = n
    elif cond == "g_orbita":                   # PRINCIPAL: siempre a ε del ancla ORIGINAL
        for r in range(1, k_rev + 1):
            t = tangente(rng.normal(size=EV.shape[2]).astype(np.float32), E0)
            n = E0 + EPS * t
            n /= np.linalg.norm(n) + 1e-8
            out.append((n, r))
    else:                                      # duplicados: posición real del texto
        for r in range(1, k_rev + 1):
            out.append((EV[r][i], r))
    return out


def evaluar(cond, EV, EQ, idx, semilla, k_rev):
    dirs, ver, due = [], [], []
    for i in idx:
        rng = np.random.default_rng(hash((int(semilla), int(i))) % (2 ** 32))
        for v, r in direcciones(cond, EV, i, rng, k_rev):
            dirs.append(v); ver.append(r); due.append(i)
    dirs = np.stack(dirs); ver = np.array(ver); due = np.array(due)
    S = EQ[idx] @ dirs.T
    top = np.argsort(-S, axis=1)[:, :K]
    vig = cob = 0
    coss = []
    for fila, i in enumerate(idx):
        vers = {ver[j] for j in top[fila] if due[j] == i}
        if k_rev in vers:
            vig += 1
        if k_rev in vers and (k_rev - 1) in vers:
            cob += 1
        # P-A4: coseno de la versión VIGENTE contra la consulta
        mios = [j for j in range(len(due)) if due[j] == i and ver[j] == k_rev]
        if mios:
            coss.append(float(EQ[i] @ dirs[mios[0]]))
    n = len(idx)
    return vig / n, cob / n, float(np.mean(coss))


def main():
    dk = np.load(SALIDA_K)
    EV, EQ = dk["EV"], dk["EQ"]
    res = {c: {k: {"vig": [], "cob": [], "cos": []} for k in K_REVISIONES} for c in CONDS}
    for s in range(N_SEMILLAS):
        idx = np.random.default_rng(1000 + s).choice(EV.shape[1], N_SUB, replace=False)
        for k_rev in K_REVISIONES:
            for c in CONDS:
                v, co, cs = evaluar(c, EV, EQ, idx, 2000 + s, k_rev)
                res[c][k_rev]["vig"].append(v)
                res[c][k_rev]["cob"].append(co)
                res[c][k_rev]["cos"].append(cs)
        print(f"  semilla {s} ok", flush=True)

    L = ["# Gemación con desplazamiento acotado — resultados\n",
         f"Prereg `PREREG_GEMACION_ACOTADA.md` (SHA ab3115e2…), congelado antes del dato.",
         f"ε = {EPS} (sin re-ajustar) · γ = {GAMMA} · k = {K} · {N_SEMILLAS} semillas × {N_SUB} · "
         f"margen {MARGEN} · IC95 t de Student, 9 gl\n",
         "## COBERTURA (ambas versiones en el top-k)\n",
         "| K | " + " | ".join(f"`{c}`" for c in CONDS) + " |",
         "|---|" + "---|" * len(CONDS)]
    for k_rev in K_REVISIONES:
        L.append(f"| {k_rev} | " + " | ".join(
            f"{ic_t(res[c][k_rev]['cob'])[0]:.4f}" for c in CONDS) + " |")

    L += ["\n## Coseno de la versión vigente contra la consulta (P-A4)\n",
          "| K | " + " | ".join(f"`{c}`" for c in CONDS) + " |",
          "|---|" + "---|" * len(CONDS)]
    for k_rev in K_REVISIONES:
        L.append(f"| {k_rev} | " + " | ".join(
            f"{ic_t(res[c][k_rev]['cos'])[0]:+.4f}" for c in CONDS) + " |")

    # ---- veredictos ----
    L.append("\n## Veredictos pre-registrados\n")

    # P-A3 primero: es bloqueante
    f4 = ic_t(res["g_fija"][4]["cob"])
    pa3 = bool(f4[0] < 0.10)
    L.append(f"**P-A3 (control de mecanismo, BLOQUEANTE)** `g_fija` a K=4 = {f4[0]:.4f} "
             f"[{f4[1]:.4f}, {f4[2]:.4f}] · exige < 0,10 → "
             f"**{'reproduce el colapso, el harness es comparable' if pa3 else 'NO REPRODUCE — nada de lo demás es válido'}**")
    if not pa3:
        txt = "\n".join(L) + "\n"
        open(INFORME, "w").write(txt); print(txt)
        return 1

    d1 = np.array(res["g_orbita"][8]["cob"]) - np.array(res["duplicados"][8]["cob"])
    m1, lo1, hi1 = ic_t(d1)
    pa1 = bool(m1 >= MARGEN and lo1 > 0)
    L.append(f"\n**P-A1 (PRINCIPAL)** COBERTURA a K=8, `g_orbita` − `duplicados` = **{m1:+.4f}** "
             f"IC95 [{lo1:+.4f}, {hi1:+.4f}] · exige ≥ {MARGEN} sin cruzar cero → "
             f"**{'CONFIRMA' if pa1 else 'NO CONFIRMA'}**")

    L.append("\n**P-A2 (no-regresión)**")
    pa2 = True
    for k_rev in (1, 2):
        d = np.array(res["g_orbita"][k_rev]["cob"]) - np.array(res["duplicados"][k_rev]["cob"])
        m, lo, hi = ic_t(d)
        ok = bool(m >= -MARGEN)
        pa2 &= ok
        L.append(f"  - K={k_rev}: {m:+.4f} IC95 [{lo:+.4f}, {hi:+.4f}] · piso −{MARGEN} → "
                 f"{'ok' if ok else 'REGRESIONA'}")

    # P-A4: pendiente del coseno por revisión
    L.append("\n**P-A4 (mecanicista)** pendiente del coseno por revisión "
             "(exige ≥ −0,01 en `g_orbita`)")
    pend = {}
    for c in ("g_orbita", "g_fija"):
        x = np.array(K_REVISIONES, float)
        y = np.array([ic_t(res[c][k]["cos"])[0] for k in K_REVISIONES])
        pend[c] = float(np.polyfit(x, y, 1)[0])
        L.append(f"  - `{c}`: {pend[c]:+.4f} por revisión")
    pa4 = bool(pend["g_orbita"] >= -0.01)
    L.append(f"  → P-A4 **{'CUMPLE' if pa4 else 'NO CUMPLE'}**")

    # secundaria
    d2 = np.array(res["g_decay"][8]["cob"]) - np.array(res["duplicados"][8]["cob"])
    m2, lo2, hi2 = ic_t(d2)
    L.append(f"\n**Secundaria** `g_decay` − `duplicados` a K=8 = {m2:+.4f} "
             f"IC95 [{lo2:+.4f}, {hi2:+.4f}] → "
             f"{'supera' if m2 >= MARGEN and lo2 > 0 else 'no supera'}")

    if not pa1:
        L.append("\n## Falsación global (§5 del prereg)\n")
        gana = any(ic_t(np.array(res["g_orbita"][k]["cob"])
                        - np.array(res["duplicados"][k]["cob"]))[0] >= MARGEN
                   for k in K_REVISIONES)
        if not gana:
            L.append("`g_orbita` **no supera a `duplicados` en ningún K**. Según lo comprometido por "
                     "adelantado, la gemación queda **descartada como mecanismo de indexación en este "
                     "régimen**, y no se prueba una tercera geometría.")
        else:
            L.append("`g_orbita` no confirma en K=8 pero supera el margen en algún otro K; se reporta "
                     "como tal, con la predicción principal caída.")

    txt = "\n".join(L) + "\n"
    open(INFORME, "w").write(txt); print(txt)
    json.dump({c: {str(k): v for k, v in d2_.items()} for c, d2_ in res.items()},
              open("resultados_acotada.json", "w"), indent=1, default=float)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
