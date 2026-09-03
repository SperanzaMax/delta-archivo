"""Juez de `PREREG_DISTANCIA_REAL.md` + `ENMIENDA_DISTANCIA_REAL.md` · 3-sep

Cuatro veces este año un juez automatico imprimio un veredicto que sus propios numeros desmentian, y
las cuatro por lo mismo: calculo un criterio en condiciones donde ese criterio no mide nada, y lo
informo como si midiera. Asi que aca las GUARDAS van primero y pueden devolver NO EVALUABLE, que es
un resultado legitimo y no un error.

Guardas, en orden, y ninguna se saltea:

  G-0  BLOQUEANTE. vigente >= 0,90 en las cuatro condiciones. Si el modelo no aprendio a contestar
       lo que si esta, NADA de lo demas se lee.
  G-L  TECHO. Si `lejos` da nose_rel >= 0,95, G-1 y G-2 son NO EVALUABLES: no queda margen para que
       `cerca` sea mejor. Declarado en el prereg ANTES de correr.
  P-0  PRESUPUESTO. Las unidades que se comparan tienen que haber llegado al mismo paso. Comparar
       una de 800 con una de 100 no mide la condicion, mide el presupuesto.

Recien despues:

  G-1   cerca - lejos >= 0,10 en nose_rel FINAL, en >=2 de 3 semillas
  G-1v  lo mismo sobre el PROMEDIO de las evaluaciones (velocidad). Declarado en la enmienda como
        POST-HOC, porque nace de mirar la compuerta.
  G-2   lejos_dos - lejos >= 0,10  ·  G-2v el equivalente en velocidad
  G-3   adjudica: lejos_relleno cerca de lejos_dos -> gana la diversidad sola
                  lejos_relleno cerca de lejos     -> gana que la relacion entre a veces
  G-4   NO DAÑO. falsa_abst <= 0,10 en lejos_dos, en >=2 de 3
"""
import glob
import json
import os
import sys

import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
CONDS = ("cerca", "lejos", "lejos_dos", "lejos_relleno")


def cargar():
    D = {}
    for f in sorted(glob.glob(os.path.join(AQUI, "g?_*.json"))):
        d = json.load(open(f))
        c = d["config"]
        h = [m for m in d["historia"] if m["paso"] > 0]
        if not h:
            continue
        D[(c["condicion"], c["semilla"])] = dict(
            paso=h[-1]["paso"], n_evals=len(h), pedidos=c["pasos"],
            rel=[m["nose_rel"] for m in h], vig=h[-1]["vigente"],
            falsa=h[-1]["falsa_abst"], ent=h[-1]["nose_ent"],
            archivo=os.path.basename(f), sens=d.get("sensibilidad_final"))
    return D


def main():
    D = cargar()
    if not D:
        print("no hay unidades todavia"); return
    print(f"{len(D)} unidades\n")
    print(f"{'condicion':<16} {'sem':>3} {'paso':>6}/{'ped':<5} {'evals':>5} | "
          f"{'nose_rel':>9} {'AUC':>7} {'vigente':>8} {'nose_ent':>9} {'falsa':>7}")
    print("-" * 92)
    for k in sorted(D):
        v = D[k]
        print(f"{k[0]:<16} {k[1]:>3} {v['paso']:>6}/{v['pedidos']:<5} {v['n_evals']:>5} | "
              f"{v['rel'][-1]:>9.4f} {np.mean(v['rel']):>7.4f} {v['vig']:>8.4f} "
              f"{v['ent']:>9.4f} {v['falsa']:>7.4f}")

    print("\n" + "=" * 92 + "\nGUARDAS\n")
    problemas = []

    # ---- G-0
    malos = [(k, v["vig"]) for k, v in D.items() if v["vig"] < 0.90]
    if malos:
        print("  G-0 BLOQUEANTE **NO CUMPLE**: " +
              ", ".join(f"{k[0]}_s{k[1]} vigente {x:.4f}" for k, x in malos))
        print("      -> NADA de lo demas se lee. Fin.")
        return
    print(f"  G-0 BLOQUEANTE CUMPLE · vigente minimo "
          f"{min(v['vig'] for v in D.values()):.4f} sobre 0,90")

    # ---- P-0: presupuesto homogeneo por comparacion
    incompletas = [f"{k[0]}_s{k[1]} ({v['paso']}/{v['pedidos']})"
                   for k, v in D.items() if v["paso"] < v["pedidos"]]
    if incompletas:
        print(f"  P-0 PRESUPUESTO: {len(incompletas)} unidades NO llegaron al paso pedido -> "
              f"{', '.join(incompletas)}")
        problemas.append("presupuesto desparejo")

    # ---- G-L: techo
    techo = [k for k, v in D.items() if k[0].startswith("lejos") and v["rel"][-1] >= 0.95]
    if techo:
        print(f"  G-L TECHO: {', '.join(f'{k[0]}_s{k[1]}' for k in techo)} dan nose_rel >= 0,95 -> "
              f"esas comparaciones son NO EVALUABLES")

    def contraste(a, b, campo):
        """Devuelve [(semilla, delta)] solo para pares con IGUAL ultimo paso."""
        out, saltados = [], []
        for s in (0, 1, 2):
            if (a, s) not in D or (b, s) not in D:
                continue
            va, vb = D[(a, s)], D[(b, s)]
            if va["paso"] != vb["paso"]:
                saltados.append(f"s{s} ({va['paso']} vs {vb['paso']})")
                continue
            f = (lambda v: v["rel"][-1]) if campo == "final" else (lambda v: float(np.mean(v["rel"])))
            out.append((s, f(va) - f(vb)))
        return out, saltados

    print("\n" + "=" * 92 + "\nCRITERIOS\n")
    for nombre, a, b, campo, umbral in (
            ("G-1  ", "cerca", "lejos", "final", 0.10),
            ("G-1v ", "cerca", "lejos", "auc", 0.10),
            ("G-2  ", "lejos_dos", "lejos", "final", 0.10),
            ("G-2v ", "lejos_dos", "lejos", "auc", 0.10)):
        pares, saltados = contraste(a, b, campo)
        etq = f"{nombre} {a} - {b} ({campo})"
        if not pares:
            print(f"  {etq}: **NO EVALUABLE**, no hay ningun par con igual presupuesto"
                  + (f" (saltados: {', '.join(saltados)})" if saltados else ""))
            continue
        cumplen = [s for s, d in pares if d >= umbral]
        det = " · ".join(f"s{s} {d:+.4f}" for s, d in pares)
        if len(pares) < 2:
            print(f"  {etq}: **NO EVALUABLE**, {len(pares)} par con igual presupuesto (pide 2 de 3). "
                  f"{det}")
        else:
            veredicto = "CUMPLE" if len(cumplen) >= 2 else "NO CUMPLE"
            print(f"  {etq}: **{veredicto}** {len(cumplen)}/{len(pares)} sobre {umbral} · {det}")
        if saltados:
            print(f"        (saltados por presupuesto distinto: {', '.join(saltados)})")

    # ---- G-4
    fs = [(s, D[("lejos_dos", s)]["falsa"]) for s in (0, 1, 2) if ("lejos_dos", s) in D]
    if fs:
        ok = [s for s, x in fs if x <= 0.10]
        estado = "CUMPLE" if len(ok) >= 2 else ("NO EVALUABLE (menos de 2 semillas)"
                                                if len(fs) < 2 else "NO CUMPLE")
        print(f"  G-4   NO DAÑO en lejos_dos: **{estado}** · " +
              " · ".join(f"s{s} {x:.4f}" for s, x in fs))
    else:
        print("  G-4   **NO EVALUABLE**, no hay unidades de lejos_dos")

    # ---- G-3 adjudica
    tri = [s for s in (0, 1, 2)
           if all((c, s) in D for c in ("lejos", "lejos_dos", "lejos_relleno"))
           and len({D[(c, s)]["paso"] for c in ("lejos", "lejos_dos", "lejos_relleno")}) == 1]
    if not tri:
        print("  G-3   **NO EVALUABLE**, falta alguna de las tres condiciones a igual presupuesto")
    else:
        print("  G-3   ADJUDICA")
        for s in tri:
            l, d, r = (D[(c, s)]["rel"][-1] for c in ("lejos", "lejos_dos", "lejos_relleno"))
            cual = "la DIVERSIDAD SOLA" if abs(r - d) < abs(r - l) else "que la relacion ENTRE A VECES"
            print(f"        s{s}: lejos {l:.4f} · lejos_dos {d:.4f} · lejos_relleno {r:.4f}"
                  f"  -> gana {cual}")

    # Un veredicto sobre el paso 100 de 800 no es el veredicto del prereg. Si NINGUNA unidad
    # llego al presupuesto pedido, se dice PROVISORIO arriba de todo y no se deja pasar como firme.
    if D and all(v["paso"] < v["pedidos"] for v in D.values()):
        peor = max(v["paso"] for v in D.values())
        print(f"\n  ⚠⚠ VEREDICTO PROVISORIO: ninguna unidad llego al presupuesto pedido "
              f"(la mas avanzada va por el paso {peor} de {list(D.values())[0]['pedidos']}). "
              f"La compuerta del 3-sep mostro que `lejos` puede ALCANZAR a `cerca` mas adelante, "
              f"asi que un CUMPLE temprano no decide G-1: puede ser un efecto de VELOCIDAD.")
    if problemas:
        print(f"\n  ⚠ leer con: {', '.join(problemas)}")

    sens = {k: v["sens"] for k, v in D.items() if v.get("sens")}
    if sens:
        print("\n" + "=" * 92 + "\nSENSIBILIDAD DESPUES del fine-tune (conv1d @ entidad)\n")
        for k, s in sorted(sens.items()):
            print(f"  {k[0]}_s{k[1]}: alcance {s['alcance']} · capa0 {s['conv_ent'][0]:.3e} · "
                  f"capa1 {s['conv_ent'][1]:.3e} · capa12 {s['conv_ent'][12]:.3e}")


if __name__ == "__main__":
    main()
