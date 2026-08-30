"""Veredicto de `PREREG_PERDIDA_CABEZA.md` (0f57609d) + `ENMIENDA` (fe058151).

    N=8000 python juzgar_perdida.py

Mide, en el punto de operación propio de cada unidad, todo lo que los criterios piden:

  P-1  ¿salió del silencio?                        -> `abst` < 1,0000
  P-3  ¿la cabeza DISCRIMINA o sólo se movió?      -> AUC sobre su propio blanco, pide > 0,60
  P-4  ¿salió del silencio INVENTANDO?             -> `invento` y la EXACTITUD GLOBAL contra 0,4065

El control se mide en el MISMO paso 3000, no al final: a 3000 pasos incluso `b3_s0`, que termina en
RECUP 0,9996, tiene `vigente` 0,0222. Comparar una unidad de 3000 pasos contra un control de 26000
mediría el presupuesto y no la condición.
"""
import sys, os, glob, re, collections, pickle
import numpy as np
sys.path.insert(0, os.getcwd())
import idioma as I
from ser import clasificar
from ser_cobertura import sondear

N = int(os.environ.get("N", "8000"))
PISO = 0.4065


def auc(score, pos):
    pos = np.asarray(pos, dtype=bool)
    if pos.all() or not pos.any():
        return float("nan")
    r = np.empty(len(score), dtype=float)
    o = np.argsort(score, kind="stable"); s = score[o]; i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        r[o[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    n1, n0 = pos.sum(), (~pos).sum()
    return float((r[pos].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


print(f"n={N}, semilla de datos 54321, punto de operación propio (a > 0)\n")
print(f"{'unidad':9} {'paso':>5} {'EXACT':>7} {'RECUP':>7} {'AUC':>7} {'abst':>7} "
      f"{'acierto':>8} {'noseOK':>7} {'invento':>8}")
print("-" * 78)

for u in sys.argv[1:]:
    ruta = f"ckpts/{u}.pkl"
    if not os.path.exists(ruta):
        print(f"{u:9} sin checkpoint")
        continue
    paso = pickle.load(open(ruta, "rb")).get("paso")
    sc, pv, tg, mt, cfg = sondear(ruta, N, 64, None, None, 54321)

    hay = np.array([I.ITOS[int(t)] != "NOSE" for t in tg])
    ok_val = np.array([I.ITOS[int(pv[i])] == I.ITOS[int(tg[i])] for i in range(len(tg))])

    c = collections.Counter()
    for i in range(len(sc)):
        tok = "NOSE" if sc[i] > 0.0 else I.ITOS[int(pv[i])]
        c[clasificar(tok, I.ITOS[int(tg[i])], mt[i])] += 1
    n = len(sc)
    exact = (c["acierto"] + c["acierto_nose"]) / n

    print(f"{u:9} {paso:>5} {exact:7.4f} {ok_val[hay].mean():7.4f} {auc(sc, ~ok_val):7.4f} "
          f"{(sc > 0).mean():7.4f} {c['acierto']/n:8.4f} {c['acierto_nose']/n:7.4f} "
          f"{c['invento']/n:8.4f}", flush=True)

print(f"\npiso trivial de EXACT = {PISO:.4f} (abstenerse de todo)")
print("P-3 pide AUC > 0,60 · P-4 mira invento y que EXACT supere el piso")
