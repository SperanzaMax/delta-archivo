#!/usr/bin/env python3
"""P-1 y P-4 de `PREREG_PRESUPUESTO_TOKEN.md`, sobre las series de entrenamiento.

    python analizar_presupuesto.py

P-1 pide dos cosas sobre los puntos NUEVOS (14000 -> 20000): que `falsa_abst` baje en >= 4 de 5
unidades, y que Spearman(paso, `falsa_abst`) sea negativo en >= 4 de 5.
P-4 pide que `vigente` no caiga mas de 0,10 en ninguna unidad.

Las series son la eval interna del tramo, con 512 muestras: sirven para la TENDENCIA, que es lo que
P-1 pregunta. Los extremos declarados de la campaña se miden aparte con 2048
(`medir_compuerta.py`), y son otros numeros: no se mezclan en la misma columna. Es la leccion del
19/20-ago —con 512 muestras el error estandar es 0,019 y una serie corta parece decir cosas—, y por
eso aca la potencia sale de la LONGITUD de la serie y no de la precision de cada punto.
"""
import glob
import json
import os
import re
import sys

import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
UNIDADES = ["t4_s0", "t4_s1", "t4_s2", "s4_s0", "s4_s1"]
DESDE, HASTA = 14000, 20000


def spearman(x, y):
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean(); ry -= ry.mean()
    d = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / d) if d else float("nan")


def serie(unidad):
    """Junta los puntos de la unidad de TODAS las carpetas de corridas, sin duplicar pasos."""
    pts = {}
    for f in sorted(glob.glob(os.path.join(AQUI, "corridas_*", f"{unidad}.json"))):
        txt = open(f).read()
        for m in re.finditer(r"\{[^{}]*\"paso\":\s*(\d+)[^{}]*\}", txt):
            try:
                d = json.loads(m.group(0))
            except json.JSONDecodeError:
                continue
            if "falsa_abst" in d and "vigente" in d:
                pts[int(d["paso"])] = d
    return [pts[k] for k in sorted(pts) if DESDE <= k <= HASTA]


print("P-1 y P-4 · puntos nuevos de la serie (eval interna, 512 muestras)\n")
print(f"{'unidad':<8} {'n':>3} {'fa 14000':>9} {'fa final':>9} {'paso':>6} "
      f"{'rho':>8} {'baja':>5} | {'vig 14000':>10} {'vig final':>10} {'caida':>7}")
print("-" * 88)

baja, rho_neg, p4 = [], [], []
for u in UNIDADES:
    s = serie(u)
    if len(s) < 3:
        print(f"{u:<8} {len(s):>3}   sin serie suficiente todavia")
        continue
    paso = np.array([d["paso"] for d in s], float)
    fa = np.array([d["falsa_abst"] for d in s], float)
    vig = np.array([d["vigente"] for d in s], float)
    r = spearman(paso, fa)
    b = fa[-1] < fa[0]
    caida = vig[0] - vig[-1]
    baja.append(b); rho_neg.append(r < 0); p4.append(caida <= 0.10)
    print(f"{u:<8} {len(s):>3} {fa[0]:>9.4f} {fa[-1]:>9.4f} {int(paso[-1]):>6} "
          f"{r:>8.4f} {'si' if b else 'NO':>5} | {vig[0]:>10.4f} {vig[-1]:>10.4f} {caida:>7.4f}")

n = len(baja)
print(f"\nP-1a  `falsa_abst` baja        {sum(baja)} de {n}   (pide >= 4 de 5)")
print(f"P-1b  Spearman negativo        {sum(rho_neg)} de {n}   (pide >= 4 de 5)")
print(f"P-4   `vigente` no cae >0,10   {sum(p4)} de {n}   (pide 5 de 5)")
if n < len(UNIDADES):
    print(f"\n-> faltan {len(UNIDADES) - n} unidades; el veredicto no se escribe hasta tenerlas.")
