"""Régimen elíptico: barrido de τ y punto de cruce — implementa PREREG_ELIPTICA.md (299edbd8…).

Nada acá decide nada: condiciones, τ, métrica principal, umbrales y cláusula de falsación están
congelados en el pre-registro. Este archivo sólo los ejecuta.

Dos decisiones de IMPLEMENTACIÓN (no de diseño), declaradas porque afectan el ruido, no el estimador:
  1. Números aleatorios comunes: la moneda de hidratación se sortea UNA vez por (semilla, entidad,
     revisión) como u ~ U(0,1), y cada τ usa `u < τ`. Las curvas en τ quedan anidadas y comparables;
     con monedas independientes por τ el mismo estimador tendría más varianza.
  2. El rango se calcula sin ordenar, pero **con desempate explícito por índice ascendente**, que es
     como rompe empates el `np.argsort` de `correr_hechos.py` y `correr_acotada.py`. Acá NO es un
     detalle: sólo existen 240 textos elípticos posibles (4 formas × 60 valores únicos, porque el
     texto no nombra la entidad), así que los 24 000 embeddings elípticos son repeticiones de 240
     vectores y los empates EXACTOS son masivos. Contar sólo `>` habría dado por recuperadas todas
     las entradas colisionadas y habría inflado justo las condiciones que el experimento discrimina.
     La degeneración no es un artefacto: es la forma aritmética del fenómeno bajo estudio.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from correr_hechos import tangente, ic_t, EPS, K, N_SEMILLAS, N_SUB, MARGEN

DATOS = "hechos_elipticas.npz"
INFORME = "INFORME_ELIPTICA.md"
K_REVISIONES = (1, 2, 4, 8)
TAUS = (0.0, 0.05, 0.10, 0.20, 0.40, 1.00)      # §3 del prereg
TAU_MAX_ACEPTABLE = 0.25                         # §4, P-E1


def construir(cond, tau, E0, EL, EH, idx, k_rev, u, rng):
    """Direcciones del índice, en bloques contiguos por entidad.

    Con el layout por bloques, la columna j tiene dueño j//(k_rev+1) y revisión j%(k_rev+1);
    eso permite localizar la entrada vigente sin recorrer nada.
    """
    n, d = len(idx), E0.shape[1]
    dirs = np.empty((n * (k_rev + 1), d), dtype=np.float32)
    for fila, i in enumerate(idx):
        base = fila * (k_rev + 1)
        dirs[base] = E0[i]                                   # el ancla, en todas las condiciones
        for r in range(1, k_rev + 1):
            if cond == "g_orbita":
                t = tangente(rng.normal(size=d).astype(np.float32), E0[i])
                v = E0[i] + EPS * t
                dirs[base + r] = v / (np.linalg.norm(v) + 1e-8)
            else:                                            # hidratada_τ
                elip = u[fila, r - 1] < tau
                dirs[base + r] = EL[r - 1][i] if elip else EH[r - 1][i]
    return dirs


def _rango(S, cols, orden):
    """Rango de la columna `cols[fila]` en su fila, desempatando por índice ascendente.

    Idéntico a `np.argsort(-S)` del harness previo. Imprescindible acá: ver nota 2 del encabezado.
    """
    s = S[np.arange(S.shape[0]), cols]
    mayores = (S > s[:, None]).sum(1)
    iguales_antes = ((S == s[:, None]) & (orden[None, :] < cols[:, None])).sum(1)
    return mayores + iguales_antes


def evaluar(cond, tau, E0, EQ, EL, EH, idx, k_rev, u, rng):
    dirs = construir(cond, tau, E0, EL, EH, idx, k_rev, u, rng)
    S = EQ[idx] @ dirs.T                                     # (n, n*(k_rev+1))
    n = len(idx)
    orden = np.arange(S.shape[1])
    col = np.arange(n) * (k_rev + 1)
    rank_vig = _rango(S, col + k_rev, orden)                 # entrada de la última revisión
    vigente = float(np.mean(rank_vig < K))
    # la "anterior" es la revisión previa; con una sola revisión, es el ancla
    col_ant = col + k_rev - 1 if k_rev >= 2 else col
    rank_ant = _rango(S, col_ant, orden)
    cobertura = float(np.mean((rank_vig < K) & (rank_ant < K)))
    return vigente, cobertura, float(np.mean(S[np.arange(n), col + k_rev]))


def main():
    if not os.path.exists(DATOS):
        print(f"falta {DATOS} — correr generar_elipticas.py primero")
        return 1
    d = np.load(DATOS)
    E0, EQ, EL, EH = d["E0"], d["EQ"], d["EL"], d["EH"]
    N = E0.shape[0]
    conds = [("g_orbita", None)] + [("hidratada", t) for t in TAUS]

    res = {}
    for s in range(N_SEMILLAS):
        idx = np.random.default_rng(1000 + s).choice(N, N_SUB, replace=False)
        u = np.random.default_rng(3000 + s).random((N_SUB, max(K_REVISIONES)))
        for k_rev in K_REVISIONES:
            for cond, tau in conds:
                rng = np.random.default_rng(2000 + s)
                v, c, cs = evaluar(cond, tau, E0, EQ, EL, EH, idx, k_rev, u, rng)
                key = (cond if tau is None else f"hidratada_{tau:.2f}", k_rev)
                res.setdefault(key, {"vig": [], "cob": [], "cos": []})
                res[key]["vig"].append(v)
                res[key]["cob"].append(c)
                res[key]["cos"].append(cs)
        print(f"  semilla {s} ok", flush=True)

    nom = ["g_orbita"] + [f"hidratada_{t:.2f}" for t in TAUS]
    L = ["# Régimen elíptico — resultados\n",
         f"Prereg `PREREG_ELIPTICA.md` (SHA 299edbd8…), congelado antes de generar los textos.",
         f"N = {N} · {N_SEMILLAS} semillas × {N_SUB} · k = {K} · ε = {EPS} (sin re-ajustar) · "
         f"margen {MARGEN} · IC95 t de Student, 9 gl · encoder `nomic-embed-text` en minúscula\n",
         "## VIGENTE — métrica principal (§3)\n",
         "| K | " + " | ".join(f"`{c}`" for c in nom) + " |",
         "|---|" + "---|" * len(nom)]
    for k_rev in K_REVISIONES:
        L.append(f"| {k_rev} | " + " | ".join(
            f"{ic_t(res[(c, k_rev)]['vig'])[0]:.4f}" for c in nom) + " |")

    L += ["\n## COBERTURA — secundaria\n",
          "| K | " + " | ".join(f"`{c}`" for c in nom) + " |",
          "|---|" + "---|" * len(nom)]
    for k_rev in K_REVISIONES:
        L.append(f"| {k_rev} | " + " | ".join(
            f"{ic_t(res[(c, k_rev)]['cob'])[0]:.4f}" for c in nom) + " |")

    L += ["\n## Coseno de la entrada vigente contra la consulta (P-E3)\n",
          "| K | " + " | ".join(f"`{c}`" for c in nom) + " |",
          "|---|" + "---|" * len(nom)]
    for k_rev in K_REVISIONES:
        L.append(f"| {k_rev} | " + " | ".join(
            f"{ic_t(res[(c, k_rev)]['cos'])[0]:.4f}" for c in nom) + " |")

    L.append("\n## Veredictos pre-registrados\n")

    # ---- P-E0: BLOQUEANTE ----
    d0 = np.array(res[("hidratada_0.00", 8)]["vig"]) - np.array(res[("g_orbita", 8)]["vig"])
    m0, lo0, hi0 = ic_t(d0)
    pe0 = bool(m0 >= 0)
    L.append(f"**P-E0 (control BLOQUEANTE de comparabilidad)** a τ=0 y K=8, "
             f"`hidratada_0` − `g_orbita` en VIGENTE = **{m0:+.4f}** IC95 [{lo0:+.4f}, {hi0:+.4f}] "
             f"· exige ≥ 0 → **{'reproduce el régimen del 10-ago' if pe0 else 'NO REPRODUCE — el harness cambió, nada más se lee'}**")
    if not pe0:
        txt = "\n".join(L) + "\n"
        open(INFORME, "w").write(txt); print(txt)
        return 1

    # ---- P-E2: control de régimen ----
    m2, lo2, hi2 = ic_t(res[("hidratada_1.00", 8)]["vig"])
    pe2 = bool(m2 < 0.10)
    L.append(f"\n**P-E2 (control de régimen)** VIGENTE de `hidratada_1` (corrección cruda) a K=8 = "
             f"{m2:.4f} IC95 [{lo2:.4f}, {hi2:.4f}] · exige < 0,10 → "
             f"**{'el régimen elíptico es duro, como el smoke indicaba' if pe2 else 'LA CRUDA SE RECUPERA IGUAL — P-E1 pierde sentido aunque confirme'}**")

    # ---- P-E1: principal ----
    L.append(f"\n**P-E1 (PRINCIPAL)** — `g_orbita` − `hidratada_τ` en VIGENTE a K=8, por τ:\n")
    L.append("| τ | dif | IC95 | ¿supera el margen? |")
    L.append("|---|---|---|---|")
    tau_cruce = None
    for t in TAUS:
        dd = np.array(res[("g_orbita", 8)]["vig"]) - np.array(res[(f"hidratada_{t:.2f}", 8)]["vig"])
        m, lo, hi = ic_t(dd)
        ok = bool(m >= MARGEN and lo > 0)
        if ok and tau_cruce is None:
            tau_cruce = t
        L.append(f"| {t:.2f} | {m:+.4f} | [{lo:+.4f}, {hi:+.4f}] | {'**sí**' if ok else 'no'} |")
    pe1 = bool(tau_cruce is not None and tau_cruce <= TAU_MAX_ACEPTABLE)
    if tau_cruce is None:
        L.append(f"\n→ **P-E1 NO CONFIRMA**: no existe τ en la grilla donde `g_orbita` supere. "
                 f"Ver la cláusula de falsación de §5.")
    else:
        L.append(f"\n→ τ* observado (primer punto de la grilla que supera) = **{tau_cruce:.2f}** · "
                 f"exigido ≤ {TAU_MAX_ACEPTABLE} · predicción puntual del prereg ≈ 0,075 → "
                 f"**P-E1 {'CONFIRMA' if pe1 else 'NO CONFIRMA (existe cruce pero muy tarde)'}**")

    # ---- P-E3: mecanicista ----
    x = np.array(TAUS)
    y = np.array([ic_t(res[(f"hidratada_{t:.2f}", 8)]["cos"])[0] for t in TAUS])
    pend_h = float(np.polyfit(x, y, 1)[0])
    yg = ic_t(res[("g_orbita", 8)]["cos"])[0]
    pend_g = float(np.polyfit(x, np.full_like(x, yg), 1)[0])
    pe3 = bool(pend_h <= -0.30 and abs(pend_g) <= 0.01)
    L += [f"\n**P-E3 (mecanicista)** pendiente del coseno de la entrada vigente por unidad de τ:",
          f"  - `hidratada_τ`: **{pend_h:+.4f}** (exige ≤ −0,30)",
          f"  - `g_orbita`: {pend_g:+.4f} — constante en {yg:.4f} por construcción (exige |·| ≤ 0,01)",
          f"  → P-E3 **{'CUMPLE' if pe3 else 'NO CUMPLE'}**"]

    # ---- falsación (§5) ----
    if tau_cruce is None:
        L += ["\n## Falsación global (§5 del prereg)\n",
              "`g_orbita` no supera a `hidratada_τ` en **ningún** τ, ni siquiera con la corrección "
              "cruda. La geometría no sirve tampoco donde el texto **no lleva la clave de "
              "recuperación**. Según lo comprometido por adelantado, la gemación queda descartada "
              "**con generalidad**, no sólo en el régimen auto-contenido, y no se prueba otra "
              "variante."]
    elif not pe1:
        L += ["\n## Lectura (§5 del prereg)\n",
              f"El cruce existe pero en τ* = {tau_cruce:.2f} > {TAU_MAX_ACEPTABLE}: el mecanismo sólo "
              "paga cuando la co-referencia es muy mala. **Negativo práctico**: en un sistema con "
              "hidratación razonable no vale la pena."]

    txt = "\n".join(L) + "\n"
    open(INFORME, "w").write(txt)
    print(txt)
    json.dump({f"{c}|{k}": v for (c, k), v in res.items()},
              open("resultados_elipticas.json", "w"), indent=1, default=float)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
