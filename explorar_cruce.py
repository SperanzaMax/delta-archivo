"""EXPLORATORIO — dónde cae realmente el cruce, entre τ=0,40 y τ=1,00.

Se corre DESPUÉS de emitir los veredictos de PREREG_ELIPTICA.md y NO los modifica: P-E1 ya quedó
NO CONFIRMA por τ* > 0,25. Esto sólo refina el enunciado práctico ("la geometría conviene si tu
co-referencia falla más de X"), y por venir después del dato se reporta como exploratorio.
"""
import numpy as np, sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from correr_hechos import ic_t, N_SEMILLAS, N_SUB, MARGEN
import correr_elipticas as CE

d = np.load("hechos_elipticas.npz")
E0, EQ, EL, EH = d["E0"], d["EQ"], d["EL"], d["EH"]
TAUS = (0.50, 0.60, 0.70, 0.80, 0.90, 0.95)
K_REV = 8

res = {t: [] for t in TAUS}; res["g"] = []
for s in range(N_SEMILLAS):
    idx = np.random.default_rng(1000 + s).choice(E0.shape[0], N_SUB, replace=False)
    u = np.random.default_rng(3000 + s).random((N_SUB, 8))
    res["g"].append(CE.evaluar("g_orbita", None, E0, EQ, EL, EH, idx, K_REV, u,
                               np.random.default_rng(2000 + s))[0])
    for t in TAUS:
        res[t].append(CE.evaluar("hidratada", t, E0, EQ, EL, EH, idx, K_REV, u,
                                 np.random.default_rng(2000 + s))[0])
    print(f"  semilla {s} ok", flush=True)

g = np.array(res["g"])
print(f"\n  g_orbita VIGENTE @K=8 = {ic_t(g)[0]:.4f}\n")
print("  | τ | hidratada_τ | g_orbita − hidratada_τ | IC95 | supera |")
print("  |---|---|---|---|---|")
cruce = None
for t in TAUS:
    h = np.array(res[t]); m, lo, hi = ic_t(g - h)
    ok = m >= MARGEN and lo > 0
    if ok and cruce is None: cruce = t
    print(f"  | {t:.2f} | {ic_t(h)[0]:.4f} | {m:+.4f} | [{lo:+.4f}, {hi:+.4f}] | {'sí' if ok else 'no'} |")
print(f"\n  primer τ de esta grilla fina donde la geometría supera: {cruce}")
json.dump({str(k): list(map(float, v)) for k, v in res.items()},
          open("resultados_cruce.json", "w"), indent=1)
