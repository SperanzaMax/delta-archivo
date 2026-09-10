"""La desestabilizacion tardia, medida como una CAIDA y no contra un umbral inventado.

Historia corta de por que el criterio es este y no otro, porque el camino importa.

  1er intento (el que quedo escrito en ENMIENDA_ABSTENCION_20260910.md, E-4): pico = maximo de la
  media movil, DESESTABILIZADA si el minimo de la cola cae bajo 0,80 x pico. Dio 15 de 28, y esta
  MAL por la misma razon que denunciamos ayer: compara un MAXIMO contra un MINIMO sobre hitos de
  cuatro muestras, o sea dos extremos de ruido. Ademas marcaba corridas que nunca llegaron al techo.

  2do intento: promedios contra promedios con umbral 0,80. Dio 0 de 13, y tambien esta mal, en la
  otra direccion. `ent15_s0` tiene el tramo medio en 1,0000 EXACTO sobre 800 muestras y la cola con
  hitos en 0,00 y 0,50: un modelo al 100 % no produce un hito de 4 de 4 mal. La caida es real y el
  umbral proporcional no la ve porque el promedio de la cola sigue en 0,9074.

  Este: se compara la cola contra el tramo medio como dos PROPORCIONES, con el n real de muestras
  detras de cada tramo. Es una caida si es grande (>= --minimo en valor absoluto) Y es clara
  (z >= --z). No hay umbral proporcional en ningun lado.
"""
import argparse, glob, json, math, os

TRAMOS = {"medio": (0.50, 0.75), "cola": (0.90, 1.00)}


def cuenta(hist, ini, fin, campo="acierto"):
    """Aciertos y muestras de un tramo. `acierto` vive sobre las clases con respuesta."""
    h = [m for m in hist if m[campo] == m[campo]]
    tr = h[int(ini * len(h)):(int(fin * len(h)) if fin < 1 else len(h))]
    clases = {"acierto": ("una", "dos"), "vigente": ("dos",), "global": None,
              "nose": ("nose_rel", "nose_aus"), "nose_rel": ("nose_rel",),
              "nose_aus": ("nose_aus",)}[campo]
    n = sum(sum(m["n"].values()) if clases is None else sum(m["n"][c] for c in clases) for m in tr)
    k = sum(m[campo] * (sum(m["n"].values()) if clases is None
                        else sum(m["n"][c] for c in clases)) for m in tr)
    return k, n


def juzgar(hist, campo="acierto", minimo=0.05, zmin=3.0, techo=0.80):
    k1, n1 = cuenta(hist, *TRAMOS["medio"], campo)
    k2, n2 = cuenta(hist, *TRAMOS["cola"], campo)
    if n1 < 100 or n2 < 100:
        return None
    p1, p2 = k1 / n1, k2 / n2
    if p1 < techo:
        return {"p1": p1, "p2": p2, "nota": "NO llego al techo", "des": False, "z": None,
                "n1": n1, "n2": n2}
    p = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    z = (p1 - p2) / se if se > 0 else 0.0
    des = (p1 - p2) >= minimo and z >= zmin
    return {"p1": p1, "p2": p2, "z": z, "des": des, "n1": n1, "n2": n2,
            "nota": "DESESTABILIZADA" if des else "estable"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--campo", default="acierto")
    ap.add_argument("--minimo", type=float, default=0.05, help="caida absoluta minima")
    ap.add_argument("--z", type=float, default=3.0)
    ap.add_argument("archivos", nargs="*")
    a = ap.parse_args()
    fs = a.archivos or sorted(glob.glob("banco_*_s*.json"))
    print(f"{'unidad':20s} {'pasos':>6s} {'medio':>7s} {'cola':>7s} {'caida':>7s} "
          f"{'z':>6s} {'n':>6s}  veredicto")
    print("-" * 78)
    nd = nt = 0
    for f in fs:
        if not os.path.exists(f) or ".parcial." in f:
            continue
        d = json.load(open(f))
        r = juzgar(d.get("hist", []), a.campo, a.minimo, a.z)
        u = os.path.basename(f).replace("banco_", "").replace(".json", "")
        if r is None:
            print(f"{u:20s} {d['args']['pasos']:6d}  pocas muestras")
            continue
        if r["z"] is None:
            print(f"{u:20s} {d['args']['pasos']:6d} {r['p1']:7.4f} {r['p2']:7.4f} "
                  f"{'':>7s} {'':>6s} {r['n1']:6d}  {r['nota']}")
            continue
        nt += 1; nd += r["des"]
        print(f"{u:20s} {d['args']['pasos']:6d} {r['p1']:7.4f} {r['p2']:7.4f} "
              f"{r['p1']-r['p2']:+7.4f} {r['z']:6.2f} {r['n1']:6d}  {r['nota']}")
    print("-" * 78)
    print(f"{nd} de {nt} desestabilizadas · campo `{a.campo}` · caida >= {a.minimo} y z >= {a.z}")


if __name__ == "__main__":
    main()
