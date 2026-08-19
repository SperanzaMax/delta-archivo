#!/usr/bin/env python3
"""Arma la tabla de las 18 fases del PREREG_FRONTERA a partir de los cortes ya guardados.

El eje A del prereg corta la base POR VALOR de `vigente` (0,85 · 0,90 · 0,95), no por número de
paso, así que cada corte cae en un paso distinto: 3000/3250/3500 en la semilla 0 y 3000/3250/3750
en la 2. El presupuesto de la fase, en cambio, está fijado en §7 en **2000 pasos para todas**.

Las dos cosas juntas obligan a que `--pasos` (que es absoluto, porque el checkpoint trae su propio
`paso`) sea distinto en cada fase: paso del corte + 2000. Esta tabla lo resuelve de una vez y deja
el número escrito en disco, en vez de que cada script lo recalcule y se desincronicen.

Se escribe `fases.tsv` con una fila por fase:

    unidad  base_ckpt  paso_base  pasos_total  abst  vigente_del_corte

No toca ningún checkpoint: sólo los lee.
"""
import os
import pickle
import sys

AQUI = os.path.dirname(os.path.abspath(__file__))
CKPTS = os.path.join(AQUI, "ckpts")
PRESUPUESTO = 2000          # §7 del prereg, idéntico al de la campaña de la cabeza del 18-ago
SEMILLAS = (0, 2)           # la 1 se estancó en 0,7777 y no cruzó 0,85: entra cuando cruce
CORTES = (85, 90, 95)
CONDICIONES = {"t": "token", "c": "cabeza"}     # §3, eje B: `escala` no entra (falló 5 de 5)


def main():
    filas, faltan = [], []
    for v in CORTES:
        for s in SEMILLAS:
            base = os.path.join(CKPTS, f"f2_s{s}.pkl.v{v}")
            if not os.path.exists(base):
                faltan.append(base)
                continue
            with open(base, "rb") as f:
                ck = pickle.load(f)
            paso = ck["paso"]
            vig = ck["historia"][-1]["vigente"]
            # Compuertas de sanidad: un corte que no cruzó su umbral, o que trae la abstención ya
            # entrenada, no sirve como punto de partida y es mejor que reviente acá que en la VM.
            assert vig >= v / 100 - 1e-9, f"{base}: vigente {vig:.4f} no llega a {v/100}"
            assert ck["config"].get("p_nose", 0.0) == 0.0, f"{base}: la base traía p_nose>0"
            assert ck["config"].get("horizonte") == 20000, f"{base}: horizonte distinto de 20000"
            for letra, abst in CONDICIONES.items():
                filas.append((f"k{v}{letra}2_s{s}", os.path.basename(base), paso,
                              paso + PRESUPUESTO, abst, f"{vig:.4f}"))
    if faltan:
        print("FALTAN checkpoints de corte:", *faltan, sep="\n  ", file=sys.stderr)

    destino = os.path.join(AQUI, "fases.tsv")
    with open(destino, "w") as f:
        f.write("# unidad\tbase\tpaso_base\tpasos_total\tabst\tvigente_corte\n")
        for fila in filas:
            f.write("\t".join(str(x) for x in fila) + "\n")
    print(f"{len(filas)} fases escritas en {destino}\n")
    print(f"{'unidad':12}{'base':22}{'paso':>6}{'->':>6}{'abst':>9}{'vigente':>10}")
    for u, b, p, pt, a, vg in filas:
        print(f"{u:12}{b:22}{p:>6}{pt:>6}{a:>9}{vg:>10}")
    return 0 if not faltan else 1


if __name__ == "__main__":
    sys.exit(main())
