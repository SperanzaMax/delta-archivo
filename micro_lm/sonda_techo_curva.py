"""AMPLIACION de `PREREG_TECHO_EVIDENCIA.md`, declarada POST-HOC y por eso separada del script del
prereg. Dos cosas, y la primera es una correccion a un defecto propio.

**(A) L3' · el lector no lineal, ahora bien construido.** En `sonda_techo.py` el L3 son 1024 random
features `tanh(hn·P + b)` y NADA MAS, asi que **no contiene al lector lineal**: puede dar menos que
L2 —y dio 0,6424 contra 0,7003— sin que eso pruebe que no hay senal no lineal. Solo prueba que ESA
proyeccion aleatoria la pierde. La version honesta concatena el estado con las features:

    L3' = ridge sobre [hn , tanh(hn·P + b)]     ->     L3' >= L2 por construccion

Recien con esa garantia «L3' no supera a L2» significa **no hay senal no lineal accesible**, que es
lo que T-1 y T-2 querian preguntar.

**(B) La curva techo-contra-RECUP, y es la que decide una frase de la estrategia.** T-4 comparaba DOS
checkpoints —`n3_s0` (12000 pasos, base) contra `t03_s3` (3000 pasos, recompensa)— que difieren en
mucho mas que su RECUP: presupuesto y funcion de perdida incluidos. **Esa comparacion esta
confundida**, asi que ni su CUMPLE ni su NO CUMPLE se pueden leer como «la deteccion escala con la
recuperacion». Con varios checkpoints la relacion se ve como curva y no como un contraste de dos
puntos.

Se declara antes de mirar: **no hay criterio pre-registrado para (B)**. Es exploratorio y se informa
como tal; lo unico que se fija de antemano es que se reportan TODOS los checkpoints medidos, no un
subconjunto elegido despues.

Costo: CPU, minutos.
"""
import glob
import json
import os
import sys

import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)

import entrenar as E             # noqa: E402
import idioma as I               # noqa: E402
import medir_ratio_ce as R       # noqa: E402
import sonda_techo as ST         # noqa: E402

N = 4096


def main():
    rutas = sys.argv[1:] or sorted(
        p for p in glob.glob(os.path.join(AQUI, "ckpts", "*.pkl"))
        if os.path.basename(p)[:-4] in (
            "n3_s0", "n3_s1", "n3_s2", "b3_s3", "b3_s6",
            "t03_s3", "t03_s6", "r03_s3", "r03_s6", "p3_s0"))

    print("=" * 100)
    print("AMPLIACION · L3' (no lineal que CONTIENE al lineal) y la curva techo-vs-RECUP")
    print("=" * 100)
    filas = []
    for ruta in rutas:
        nom = os.path.basename(ruta)[:-4]
        try:
            params, cfg, paso = R.cargar(ruta)
        except Exception as e:
            print(f"  [{nom}] no se pudo cargar: {e}")
            continue
        I.fijar_version(cfg.get("idioma", 2))
        ST.IDS_NOM = np.array([I.STOI[t] for t in I.NOMBRES])
        lg, hn, bus, tgt, _ = ST.cosechar(params, cfg, N, 64, ST.SEMILLA, 0.4)
        no = (tgt == E.NOSE)
        lg_v = lg.copy()
        lg_v[:, E.NOSE] = -1e9
        arg = lg_v.argmax(-1)
        es_nom = np.isin(arg, ST.IDS_NOM)
        recup = float((arg == tgt)[~no].mean())

        n = len(lg)
        idx = np.random.default_rng(0).permutation(n)
        tr, te = idx[:n // 2], idx[n // 2:]
        rng = np.random.default_rng(1)

        lin = ST.evaluar(f"{nom} · L2 lineal", hn, no, es_nom, tr, te, rng)
        d = hn.shape[1]
        rf = np.random.default_rng(7)
        P = rf.normal(size=(d, 1024)) / np.sqrt(d)
        b = rf.uniform(0, 2 * np.pi, size=1024)
        z = (hn - hn.mean(0)) / (hn.std(0) + 1e-6)
        # LA CORRECCION: el estado va concatenado, asi que este lector CONTIENE al lineal.
        Z = np.hstack([hn, np.tanh(z @ P + b)])
        nol = ST.evaluar(f"{nom} · L3' NO LINEAL (contiene al lineal)", Z, no, es_nom, tr, te, rng)
        print(f"      RECUP {recup:.4f}   ganancia no lineal {nol['senal'] - lin['senal']:+.4f}\n")
        filas.append({"unidad": nom, "paso": paso, "recup": recup,
                      "lineal": lin["senal"], "no_lineal": nol["senal"],
                      "confiable": bool(lin["confiable"] and nol["confiable"])})

    print("=" * 100)
    print(f"{'unidad':10s} {'paso':>6s} {'RECUP':>8s} {'techo lin':>10s} {'techo no-lin':>13s}"
          f" {'ganancia':>9s}")
    for f in sorted(filas, key=lambda x: x["recup"]):
        marca = "" if f["confiable"] else "   <- controles fallan"
        print(f"{f['unidad']:10s} {f['paso']:6d} {f['recup']:8.4f} {f['lineal']:10.4f}"
              f" {f['no_lineal']:13.4f} {f['no_lineal'] - f['lineal']:+9.4f}{marca}")

    buenos = [f for f in filas if f["confiable"]]
    if len(buenos) >= 3:
        x = np.array([f["recup"] for f in buenos])
        y = np.array([max(f["lineal"], f["no_lineal"]) for f in buenos])
        r = float(np.corrcoef(x, y)[0, 1])
        pend = float(np.polyfit(x, y, 1)[0])
        print(f"\ncorrelacion techo-vs-RECUP  r = {r:+.4f}   ·   pendiente {pend:+.4f} de AUC por")
        print(f"punto de RECUP, sobre {len(buenos)} unidades confiables")
        print(f"  lectura: con pendiente {pend:+.4f}, pasar RECUP de 0,36 a 0,79 compra "
              f"{pend * (0.79 - 0.36):+.4f} de AUC")
    with open(os.path.join(AQUI, "sonda_techo_curva_20260831.json"), "w") as f:
        json.dump(filas, f, indent=1)
    print("-> sonda_techo_curva_20260831.json")


if __name__ == "__main__":
    main()
