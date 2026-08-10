"""P4 — ley de escala en K: la predicción del prereg que SÍ tiene margen para discriminar.

**Versión 2.** La v1 fue inválida por error de implementación (ver D2 en DESVIACIONES_HECHOS.md):
modelaba la geometría de las revisiones en vez de medirla. Acá las revisiones son **textos reales**
que se embeben con el mismo encoder, así que la posición de cada versión es un dato, no un supuesto.

La tanda principal dejó P1 sin confirmar con las dos condiciones en el techo (duplicados 0,9988 ·
gemacion 0,9928): con K = 1 revisión, dos entradas por entidad y k = 5, la cobertura es trivialmente
1,0 para cualquier forma de guardar la historia. Un empate en el techo **no es equivalencia**, es
falta de potencia.

P4 está pre-registrado así (§4): «con K revisiones por entidad (K ∈ {2,4,8}), el sesgo δ necesario
para recuperar la versión vigente crece **superlinealmente** en K, y con K=8 la recuperación de
ANTERIOR cae por debajo de 0.5 a δ fijo».

Es donde el clúster se puebla y la geometría tiene que trabajar: con K revisiones hay K+1 versiones
del mismo recuerdo compitiendo dentro del mismo top-k.

Nada acá cambia umbrales: ε = 0,30, k = 5, 10 semillas, margen 0,02, todo de PREREG_HECHOS.md y la
enmienda E2.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from correr_hechos import tangente, ic_t, EPS, K, N_SEMILLAS, N_SUB, MARGEN

SALIDA_K = "hechos_revisiones.npz"
K_REVISIONES = (1, 2, 4, 8)          # 1 es la tanda principal, de referencia
INFORME = "INFORME_P4.md"
K_MAX = 8


def archivo_multi(cond, EV, idx, rng, k_rev):
    """Índice con k_rev revisiones por entidad. `EV` (K+1, N, d) son los embeddings REALES de cada
    versión del texto.

    - `duplicados`: la revisión r se indexa en `emb(v_r)` — su posición real, medida.
    - `gemacion`:   la revisión r se ancla al lado de la anterior — **caminata sobre la esfera con
      paso ε re-tangentado en cada paso**, igual que `exp_gemacion.py:106` (D2 corrige el rayo
      lineal de la v1).
    """
    dirs, ver, due = [], [], []
    for i in idx:
        if cond == "sobrescritura":
            dirs.append(EV[k_rev][i]); ver.append(k_rev); due.append(i)
            continue
        dirs.append(EV[0][i]); ver.append(0); due.append(i)
        eje = tangente(rng.normal(size=EV.shape[2]).astype(np.float32), EV[0][i])
        prev = EV[0][i]
        for r in range(1, k_rev + 1):
            if cond == "duplicados":
                nueva = EV[r][i]                       # posición real del texto de esa revisión
            else:
                paso = tangente(eje, prev)             # re-tangentar en cada paso (R13)
                nueva = prev + EPS * paso
                nueva = nueva / (np.linalg.norm(nueva) + 1e-8)
                prev = nueva
            dirs.append(nueva); ver.append(r); due.append(i)
    return np.stack(dirs), np.array(ver), np.array(due)


def evaluar_k(cond, EV, EQ, idx, rng, k_rev):
    dirs, ver, due = archivo_multi(cond, EV, idx, rng, k_rev)
    S = EQ[idx] @ dirs.T
    top = np.argsort(-S, axis=1)[:, :K]
    vig = ant = cob = 0
    for fila, i in enumerate(idx):
        mio = [j for j in top[fila] if due[j] == i]
        if mio:
            v = ver[mio]
            if v[int(np.argmax(v))] == k_rev:          # VIGENTE = la última revisión
                vig += 1
            if len(mio) >= 2:
                orden = np.argsort(v)
                if ver[mio[orden[-2]]] == k_rev - 1:   # ANTERIOR = la penúltima
                    ant += 1
        vers = {ver[j] for j in top[fila] if due[j] == i}
        if k_rev in vers and (k_rev - 1) in vers:
            cob += 1
    n = len(idx)
    return vig / n, ant / n, cob / n


def main():
    if not os.path.exists(SALIDA_K):
        print("faltan los embeddings de revisiones; correr generar_revisiones.py")
        return 1
    dk = np.load(SALIDA_K)
    EV, EQ = dk["EV"], dk["EQ"]          # EV (K_MAX+1, N, d)
    conds = ("duplicados", "gemacion")
    res = {c: {k: {"vig": [], "ant": [], "cob": []} for k in K_REVISIONES} for c in conds}

    for s in range(N_SEMILLAS):
        rs = np.random.default_rng(1000 + s)
        idx = rs.choice(EV.shape[1], N_SUB, replace=False)
        for k_rev in K_REVISIONES:
            for c in conds:
                v, a, co = evaluar_k(c, EV, EQ, idx, np.random.default_rng(2000 + s), k_rev)
                res[c][k_rev]["vig"].append(v)
                res[c][k_rev]["ant"].append(a)
                res[c][k_rev]["cob"].append(co)
        print(f"  semilla {s} ok", flush=True)

    L = ["# P4 — ley de escala en K revisiones\n",
         f"k = {K} (top-k de lectura) · ε = {EPS} · {N_SEMILLAS} semillas × {N_SUB} · "
         f"margen {MARGEN}. Umbrales sin cambios.\n",
         "**Por qué esta tanda:** con K = 1 la tarea está en el techo (duplicados 0,9988 · "
         "gemacion 0,9928) y P1 no puede discriminar. Al poblar el clúster con más revisiones, "
         "la geometría tiene que trabajar.\n",
         "## VIGENTE (recuperar la versión al día)\n",
         "| K | duplicados | gemacion | gemacion − duplicados |", "|---|---|---|---|"]
    for k_rev in K_REVISIONES:
        a = ic_t(res["duplicados"][k_rev]["vig"]); b = ic_t(res["gemacion"][k_rev]["vig"])
        dif = ic_t(np.array(res["gemacion"][k_rev]["vig"])
                   - np.array(res["duplicados"][k_rev]["vig"]))
        L.append(f"| {k_rev} | {a[0]:.4f} | {b[0]:.4f} | **{dif[0]:+.4f}** "
                 f"[{dif[1]:+.4f}, {dif[2]:+.4f}] |")

    L += ["\n## ANTERIOR (recuperar la versión previa)\n",
          "| K | duplicados | gemacion | gemacion − duplicados |", "|---|---|---|---|"]
    for k_rev in K_REVISIONES:
        a = ic_t(res["duplicados"][k_rev]["ant"]); b = ic_t(res["gemacion"][k_rev]["ant"])
        dif = ic_t(np.array(res["gemacion"][k_rev]["ant"])
                   - np.array(res["duplicados"][k_rev]["ant"]))
        L.append(f"| {k_rev} | {a[0]:.4f} | {b[0]:.4f} | **{dif[0]:+.4f}** "
                 f"[{dif[1]:+.4f}, {dif[2]:+.4f}] |")

    L += ["\n## COBERTURA (ambas versiones en el top-k) — la métrica de P1\n",
          "| K | duplicados | gemacion | gemacion − duplicados |", "|---|---|---|---|"]
    for k_rev in K_REVISIONES:
        a = ic_t(res["duplicados"][k_rev]["cob"]); b = ic_t(res["gemacion"][k_rev]["cob"])
        dif = ic_t(np.array(res["gemacion"][k_rev]["cob"])
                   - np.array(res["duplicados"][k_rev]["cob"]))
        marca = " ← supera el margen" if abs(dif[0]) > MARGEN else ""
        L.append(f"| {k_rev} | {a[0]:.4f} | {b[0]:.4f} | **{dif[0]:+.4f}** "
                 f"[{dif[1]:+.4f}, {dif[2]:+.4f}]{marca} |")

    # veredicto de P4: ANTERIOR de gemación por debajo de 0,5 a K=8
    a8 = ic_t(res["gemacion"][8]["ant"])
    L += [f"\n## Veredicto de P4\n",
          f"El prereg predice que **con K = 8 la recuperación de ANTERIOR cae por debajo de 0,5 a "
          f"δ fijo**. Medido: **{a8[0]:.4f}** IC95 [{a8[1]:.4f}, {a8[2]:.4f}] → "
          f"**{'CONFIRMA' if a8[2] < 0.5 else 'NO CONFIRMA'}**."]

    txt = "\n".join(L) + "\n"
    open(INFORME, "w").write(txt)
    print(txt)
    json.dump({c: {str(k): v for k, v in d2.items()} for c, d2 in res.items()},
              open("resultados_p4.json", "w"), indent=1, default=float)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
