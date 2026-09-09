"""`t_techo`, el tiempo hasta el techo. Implementa PREREG_LEY_DEL_RELOJ.md (c3b5f1af).

La regla esta fijada en el prereg y aca no se toca:

  meseta   media de `acierto` sobre el ultimo 25 % de los hitos
  si meseta < 0,80 -> la corrida NO llego al techo, `t_techo` indefinido, no se extrapola
  t_techo  primer hito donde la media movil de 8 hitos alcanza 0,90 x meseta Y la MEDIA MOVIL no
           vuelve a bajar de 0,80 x meseta

⚠️ ENMIENDA del mismo dia, ver ENMIENDA_LEY_DEL_RELOJ.md. El prereg pedia que ningun HITO SUELTO
bajara de 0,80 x meseta, y con batch 4 eso es imposible de sostener: un hito son 4 muestras, asi
que 3 de 4 = 0,7500 ya perfora un umbral de 0,7871. El criterio terminaba midiendo cuando ocurrio
el ultimo lote con mala suerte (246 de 801 hitos lo perforaban con el modelo al 98 %). La clausula
pasa a la media movil. El criterio original queda disponible con `--regla original`.
"""
import argparse, glob, json, os


def t_techo(hist, campo="acierto", regla="enmendada"):
    h = [(m["paso"], m[campo]) for m in hist if m[campo] == m[campo]]
    if len(h) < 12:
        return None, None, "muy pocos hitos"
    y = [v for _, v in h]
    corte = int(len(y) * 0.75)
    meseta = sum(y[corte:]) / len(y[corte:])
    if meseta < 0.80:
        return None, meseta, "NO llego al techo"
    alto, bajo = 0.90 * meseta, 0.80 * meseta
    mm = [sum(y[i:i+8]) / 8 for i in range(len(y) - 7)]      # media movil de 8 hitos
    for i in range(len(mm)):
        if mm[i] < alto:
            continue
        resto = (y[i+8:] if regla == "original" else mm[i+1:]) or [1.0]
        if min(resto) >= bajo:
            return h[i][0], meseta, "ok"
    return None, meseta, "nunca se sostiene"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--campo", default="acierto")
    ap.add_argument("--regla", default="enmendada", choices=["enmendada", "original"])
    ap.add_argument("archivos", nargs="*", default=None)
    a = ap.parse_args()
    fs = a.archivos or sorted(glob.glob("banco_*_s*.json"))
    print(f"{'unidad':18s} {'ent':>4s} {'val':>4s} {'arch':>5s} {'pasos':>7s} "
          f"{'meseta':>7s} {'t_techo':>8s}")
    print("-" * 62)
    for f in fs:
        if not os.path.exists(f):
            continue
        d = json.load(open(f))
        g = d["args"]
        t, mes, nota = t_techo(d["hist"], a.campo, a.regla)
        u = os.path.basename(f).replace("banco_", "").replace(".json", "")
        print(f"{u:18s} {g.get('n_ent') or 60:4d} {g.get('n_val') or 100:4d} "
              f"{g['arch']:5d} {g['pasos']:7d} "
              f"{mes if mes is None else round(mes,4):>7} "
              f"{t if t is not None else nota:>8}")


if __name__ == "__main__":
    main()
