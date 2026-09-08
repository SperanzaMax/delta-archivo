"""`at3` contra sus dos brazos, paso a paso · 2026-09-08

`PREREG_FRONTERA_E1.md` (`050341b4`), H1: si el corte de la ley de la ventana era ALCANCE, un modelo
que aprende con acceso global tiene que llegar a donde llega el kernel 5 y no a donde llega el 3,
sin necesitar un kernel mas grande. Los dos brazos ya estan corridos a 26.000 pasos con los mismos
hiperparametros, asi que la comparacion no cuesta computo.

    v3_s*   lat2, kernel_q 3   la RELACION cae afuera de la ventana   nose_rel final 0,6430
    kq3_s*  lat2, kernel_q 5   la relacion entra                      nose_rel final 0,9977
    at3_s*  attn               toda la secuencia causal               lo pre-registrado

Metrica primaria `nose_rel` en el paso 26.000; se reporta la corrida entera y NO el mejor
checkpoint (`regla-lectura-de-curvas`). La brecha entre los brazos recien es legible desde el paso
6.000 (+0,15) y es inequivoca en el 12.000 (+0,29): antes de eso las tres familias son
indistinguibles y mirar es engañarse.

    python curvas_frontera.py [--cada 2000]
"""
import argparse
import glob
import json
from collections import defaultdict

FAMILIAS = [("v3", "kernel 3"), ("kq3", "kernel 5"), ("at3", "attn"), ("ap3", "attnp")]

# 2026-09-08 · `ap3` NO SE PUEDE PROMEDIAR. Es bimodal: dos semillas colapsan la atencion a la
# propia posicion (1,41 y 2,08 posiciones efectivas de 24) y una se abre mas que `attn` (16,17). El
# `nose_rel` acompania exacto, 0,11 y 0,00 contra 0,88 en el paso 6.000. La media de esas tres no
# describe a ninguna, que es la leccion de metodo que este mismo repo ya tenia escrita. Para `ap3`
# hay que mirar `--por-semilla`.
BIMODALES = {"ap3"}
METRICA = "nose_rel"


def curva(prefijo, metrica=METRICA):
    """{paso: [valor por semilla]} juntando todas las corridas de la familia."""
    out = defaultdict(list)
    for f in sorted(glob.glob("corridas_*/%s_s*.json" % prefijo)):
        for r in json.load(open(f))["historia"]:
            out[r["paso"]].append(r[metrica])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cada", type=int, default=2000)
    ap.add_argument("--metrica", default=METRICA)
    ap.add_argument("--por-semilla", action="store_true",
                    help="abre las familias bimodales en una columna por semilla")
    a = ap.parse_args()

    cs = {p: curva(p, a.metrica) for p, _ in FAMILIAS}
    pasos = sorted(set(cs["at3"]))
    print(f"{a.metrica} · media por familia (n semillas entre parentesis)\n")
    print(f"{'paso':>7}" + "".join(f"{n:>22}" for _, n in FAMILIAS) + "   at3 vs kernel 3 / 5")
    for p in pasos:
        if p % a.cada and p != pasos[-1]:
            continue
        cel, val = "", {}
        for pre, _ in FAMILIAS:
            v = cs[pre].get(p, [])
            val[pre] = sum(v) / len(v) if v else None
            cel += f"{('%.4f (%d)' % (val[pre], len(v))) if v else '—':>22}"
        extra = ""
        if val["at3"] is not None and val["v3"] and val["kq3"]:
            extra = f"   {val['at3'] - val['v3']:+.4f} / {val['at3'] - val['kq3']:+.4f}"
        print(f"{p:>7}{cel}{extra}")

    for pre in sorted(BIMODALES & set(cs)):
        if not cs[pre]:
            continue
        print(f"\n!! `{pre}` es BIMODAL y su media no describe a ninguna semilla. "
              f"Correr con --por-semilla.")
        if a.por_semilla:
            import glob as _g
            fs = sorted(_g.glob("corridas_*/%s_s*.json" % pre))
            hs = [{r["paso"]: r[a.metrica] for r in json.load(open(f))["historia"]} for f in fs]
            print(f"   {'paso':>7}" + "".join(f"{f.split('/')[-1][:-5]:>12}" for f in fs))
            for p in sorted(hs[0]):
                if p % a.cada and p != max(hs[0]):
                    continue
                print(f"   {p:>7}" + "".join(
                    f"{h[p]:12.4f}" if p in h else f"{'—':>12}" for h in hs))

    print("\nLectura: at3 tiene que terminar CERCA de kernel 5 y LEJOS de kernel 3.")
    print("H1 se refuta si el `nose_rel` final queda bajo 0,80 en dos de las tres semillas.")
    ult = max(cs["at3"])
    v = cs["at3"][ult]
    if ult >= 26000:
        bajo = sum(x < 0.80 for x in v)
        print(f"\nPaso {ult}, semillas: {', '.join('%.4f' % x for x in v)}")
        print("H1 REFUTADA" if bajo >= 2 else "H1 sostenida en la metrica primaria")
    else:
        print(f"\n(at3 va por el paso {ult} de 26000 — sin veredicto todavia)")


if __name__ == "__main__":
    main()
