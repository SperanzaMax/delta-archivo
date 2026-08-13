"""REPLICA de E-I2 con lr=1e-3, la tasa que si aprende.

La corrida principal de E-I2 uso lr=3e-3, heredado del harness de Ligamento y calibrado para OTRA
tarea. Con ese paso todo quedaba en el azar. El barrido mostro que la ventana es estrecha:

    1e-3 -> 0,7305      3e-3 -> 0,0265      1e-2 -> 0,0130

Una sola corrida no es un resultado. Esto replica con 5 semillas y las dos posiciones de inyeccion,
y desglosa por claves revisadas / no revisadas, que es donde estaba lo interesante: 0,9987 cuando
hay una sola version archivada y 0,4622 cuando compiten la vieja y la nueva -- el mismo modo de
falla que R1/R4 midieron con un indice NO parametrico ("agrupa perfecto pero no ordena").
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ei2_consulta as E

E.LR = 1e-3
SEMILLAS = (0, 1, 2, 3, 4)

salida = {}
for bloque, etiqueta in ((0, "temprano"), (3, "tardio")):
    rs = []
    for s in SEMILLAS:
        r = E.entrenar(bloque, s, pasos=4000)
        rs.append(r)
        print(f"  {etiqueta} s{s} → acc {r[0]:.4f} (rev {r[1]:.4f} · no-rev {r[2]:.4f})", flush=True)
        json.dump(dict(salida, **{etiqueta: {"parcial": np.array(rs).tolist()}}),
                  open("resultados_ei2_replica.json", "w"), indent=1)
    a = np.array(rs)
    salida[etiqueta] = {"acc": float(a[:, 0].mean()), "sd": float(a[:, 0].std(ddof=1)),
                        "revisadas": float(a[:, 1].mean()),
                        "no_revisadas": float(a[:, 2].mean()),
                        "por_semilla": a[:, 0].tolist()}
    print(f"\n  ►► {etiqueta}: acc {a[:,0].mean():.4f} (sd {a[:,0].std(ddof=1):.4f}) · "
          f"revisadas {a[:,1].mean():.4f} · no revisadas {a[:,2].mean():.4f}\n", flush=True)
    json.dump(salida, open("resultados_ei2_replica.json", "w"), indent=1)

t = salida["temprano"]
print("=" * 70)
print(f"  P-1 bloqueante: {max(v['acc'] for v in salida.values()):.4f} "
      f"(exigido > 0,30 · piso 0,0215)")
print(f"  P-2 conflicto de versiones: no-rev − rev = "
      f"{t['no_revisadas'] - t['revisadas']:+.4f} (exigido >= +0,10)")
print(f"  P-3 posicion: temprano − tardio = {t['acc'] - salida['tardio']['acc']:+.4f}")
print("=" * 70)
