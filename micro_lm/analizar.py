"""MICRO-LM · consolida las corridas de una campania en una tabla legible.

    python analizar.py corridas_20260814

Reporta **por semilla y no solo la media**. No es prolijidad: en E-I3d la metrica `ANTERIOR` resulto
BIMODAL (0,0052 · 0,9297 · 0,0078) — dos semillas cayeron en el atajo de la recencia y una aprendio
la operacion completa. La media de eso (0,314) no describe a ninguna de las tres corridas y habria
hecho leer como «ruido» lo que era un mecanismo partido en dos.
Por eso ademas de la media va el RANGO, y se marca con «⚠ bimodal» toda celda donde las semillas se
separen mas de 0,3.
"""
import argparse
import glob
import json
import os
import re

import numpy as np

METRICAS = ["vigente", "anterior"]


def cargar(carpeta):
    corridas = {}
    for f in sorted(glob.glob(os.path.join(carpeta, "n*_s*.json"))):
        m = re.match(r"n(\d+)_s(\d+)\.json", os.path.basename(f))
        if not m:
            continue
        d = json.load(open(f))
        if not d.get("historia"):
            continue
        corridas[(int(m[1]), int(m[2]))] = d
    return corridas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("carpeta")
    a = ap.parse_args()
    corridas = cargar(a.carpeta)
    if not corridas:
        raise SystemExit(f"no hay corridas en {a.carpeta}")

    niveles = sorted({n for n, _ in corridas})
    semillas = sorted({s for _, s in corridas})
    print(f"{len(corridas)} corridas · niveles {niveles} · semillas {semillas}\n")

    # la compuerta primero: si algo se trunco, la accuracy no significa nada
    peor = max(d["historia"][-1].get("truncados", 0.0) for d in corridas.values())
    print(f"truncamiento maximo en toda la campania: {peor:.4f}"
          f"  {'✓ limpio' if peor <= 0.01 else '⚠⚠ SE ESTA MIDIENDO EL PADDING'}\n")

    for met in METRICAS:
        print(f"── {met.upper()}")
        print(f"{'nivel':>6} " + " ".join(f"{'s'+str(s):>8}" for s in semillas)
              + f" {'media':>8} {'rango':>8}   pasos")
        for n in niveles:
            vals, pasos = [], []
            celdas = []
            for s in semillas:
                d = corridas.get((n, s))
                if d is None:
                    celdas.append(f"{'—':>8}")
                    continue
                v = d["historia"][-1][met]
                vals.append(v)
                pasos.append(d["historia"][-1]["paso"])
                celdas.append(f"{v:>8.4f}")
            if vals:
                rango = max(vals) - min(vals)
                aviso = "  ⚠ bimodal" if rango > 0.3 else ""
                print(f"{n:>6} " + " ".join(celdas)
                      + f" {np.mean(vals):>8.4f} {rango:>8.4f}   {min(pasos)}-{max(pasos)}{aviso}")
            else:
                print(f"{n:>6} " + " ".join(celdas))
        print()

    faltan = [(n, s) for n in niveles for s in semillas if (n, s) not in corridas]
    if faltan:
        print("faltan:", ", ".join(f"n{n}s{s}" for n, s in faltan))


if __name__ == "__main__":
    main()
