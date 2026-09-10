"""`t_primero`, el reloj sin la clausula de permanencia infinita. Ver INFORME_RELOJ_20260910.md.

    t_primero   primer paso donde la media movil de 8 hitos cruza el umbral
                Y el acierto agregado de los `--ventana` pasos siguientes tambien lo cruza

`t_techo` (PREREG_LEY_DEL_RELOJ + su enmienda) exige ademas que la media movil no vuelva a bajar
del 80 % de la meseta HASTA EL FINAL del entrenamiento, y con eso deja de marcar cuando el modelo
llego para marcar cuando ocurrio el ultimo hipo. `ent8_s1` cruza en 1.250, se queda en 0,9957 sobre
2.084 muestras, y `t_techo` la reporta en 14.550 por nueve errores.
"""
import argparse, glob, json, os, statistics as st


def t_primero(hist, umbral=0.90, ventana=400, campo="acierto"):
    h = [m for m in hist if m[campo] == m[campo]]
    if len(h) < 12:
        return None
    y = [(m["paso"], m[campo], m["n"]["una"] + m["n"]["dos"]) for m in h]
    mm = [(y[i][0], sum(v for _, v, _ in y[i:i + 8]) / 8) for i in range(len(y) - 7)]
    for p, v in mm:
        if v < umbral:
            continue
        sel = [(a, n) for q, a, n in y if p <= q <= p + ventana]
        n = sum(n for _, n in sel)
        if n and sum(a * n for a, n in sel) / n >= umbral:
            return p
    return None


def acierto_desde(hist, paso, campo="acierto"):
    sel = [m for m in hist if m[campo] == m[campo] and m["paso"] >= paso]
    n = sum(m["n"]["una"] + m["n"]["dos"] for m in sel)
    return (sum(m[campo] * (m["n"]["una"] + m["n"]["dos"]) for m in sel) / n, n) if n else (float("nan"), 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--umbral", type=float, default=0.90)
    ap.add_argument("--ventana", type=int, default=400)
    ap.add_argument("--campo", default="acierto")
    ap.add_argument("archivos", nargs="*")
    a = ap.parse_args()
    fs = a.archivos or sorted(glob.glob("banco_*_s*.json"))
    print(f"{'unidad':18s} {'ent':>4s} {'pasos':>6s} {'t_primero':>10s} {'acierto desde':>14s}")
    print("-" * 58)
    por_punto = {}
    for f in fs:
        if not os.path.exists(f) or ".parcial." in f:
            continue
        d = json.load(open(f))
        t = t_primero(d["hist"], a.umbral, a.ventana, a.campo)
        u = os.path.basename(f).replace("banco_", "").replace(".json", "")
        ac, n = acierto_desde(d["hist"], t or 0, a.campo)
        ent = d["args"].get("n_ent") or 60
        if t:
            por_punto.setdefault((u.rsplit("_s", 1)[0], ent), []).append(t)
        print(f"{u:18s} {ent:4d} {d['args']['pasos']:6d} {str(t):>10s} "
              f"{ac:9.4f} (n={n})")
    hay = {k: v for k, v in por_punto.items() if len(v) > 1}
    if hay:
        print("-" * 58)
        print("por punto, con mas de una semilla:")
        for (b, ent), ts in sorted(hay.items()):
            print(f"  {b:12s} {ent:2d} ent · {sorted(ts)} · media {st.mean(ts):7.1f} · "
                  f"sd {st.stdev(ts):6.1f} · rango {max(ts)-min(ts)}")


if __name__ == "__main__":
    main()
