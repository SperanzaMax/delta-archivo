"""El veredicto de la campania de la abstencion, escrito ANTES de ver los datos.

Implementa PREREG_ABSTENCION.md + ENMIENDA_ABSTENCION_20260910.md (f89071b6) y su correccion de
E-4. Se escribio con las doce corridas en marcha y ninguna terminada, para que la forma de leerlas
no dependa de lo que digan.

  H1  no se va al mudo   GLOBAL > 0,4000 (piso del mudo) Y acierto >= 0,80. Las dos juntas.
  H2  nose_rel < nose_aus                (la entidad esta, con la otra relacion, y el valor tienta)
  H3  el calentamiento NO cambia el resultado   abst3s contra abstcal3s

  par A  abstcal3s contra abstcal6p   calentamiento 3.000 vs 6.000, presupuesto POST igualado
  par D  abstcal3s contra abstlargo   13.000 vs 19.000 pasos post-calentamiento

Cada numero sale de la EVALUACION FINAL de 512 muestras frescas, no de la cola de la curva. La cola
se imprime al lado para que se vea cuanto cambiaba el instrumento viejo.
"""
import argparse, glob, json, math, os, sys, statistics as st

BRAZOS = ["abst3s", "abstcal3s", "abstcal6p", "abstlargo"]
PISO_MUDO = 0.4000
CAMPOS = ["global", "acierto", "vigente", "nose", "nose_rel", "nose_aus",
          "invento_rel", "invento_aus"]


def cargar(brazo):
    us = []
    for f in sorted(glob.glob(f"banco_{brazo}_s*.json")):
        if ".parcial." in f:
            continue
        d = json.load(open(f))
        if not d.get("final") or "ERROR" in d.get("final", {}):
            print(f"  !! {os.path.basename(f)} sin evaluacion final: {d.get('final')}")
            continue
        us.append((os.path.basename(f), d))
    return us


def cola(hist, campo):
    """El mismo campo, sobre el ultimo 10 % de la curva. Para comparar instrumentos."""
    from desestabiliza import cuenta
    k, n = cuenta(hist, 0.90, 1.00, campo)
    return k / n if n else float("nan")


def resumen(us, campo, brazo_de="con archivo"):
    v = [d["final"][brazo_de][campo] for _, d in us]
    v = [x for x in v if x == x]
    if not v:
        return float("nan"), float("nan"), 0
    return st.mean(v), (st.stdev(v) if len(v) > 1 else 0.0), len(v)


def es(p, n):
    return math.sqrt(p * (1 - p) / n) if n and 0 <= p <= 1 else float("nan")


def dif(a, b, campo):
    """Diferencia entre dos brazos, pareada por semilla cuando se puede."""
    pa = {f[-7:-5]: d["final"]["con archivo"][campo] for f, d in a}
    pb = {f[-7:-5]: d["final"]["con archivo"][campo] for f, d in b}
    com = sorted(set(pa) & set(pb))
    ds = [pa[k] - pb[k] for k in com if pa[k] == pa[k] and pb[k] == pb[k]]
    if len(ds) < 2:
        return None
    m, s = st.mean(ds), st.stdev(ds)
    t = m / (s / math.sqrt(len(ds))) if s > 0 else (float("inf") if m else 0.0)
    return {"n": len(ds), "media": m, "sd": s, "t": t}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-n", type=int, default=512)
    a = ap.parse_args()
    datos = {b: cargar(b) for b in BRAZOS}

    print("=" * 96)
    print("LA ABSTENCION, tres semillas por brazo · evaluacion final de muestras frescas")
    print("=" * 96)
    for b in BRAZOS:
        us = datos[b]
        if not us:
            print(f"\n{b}: sin unidades cerradas")
            continue
        n_ev = sum(sum(d["final"]["con archivo"]["n"].values()) for _, d in us) // len(us)
        print(f"\n### {b} · {len(us)} semillas · {n_ev} muestras por semilla")
        print(f"  {'campo':12s} {'media':>8s} {'sd':>8s} {'ES':>7s}   {'cola de la curva':>18s}")
        for c in CAMPOS:
            m, s, k = resumen(us, c)
            cs = [cola(d["hist"], c) for _, d in us if c in ("global", "acierto", "vigente",
                                                             "nose", "nose_rel", "nose_aus")]
            cm = st.mean([x for x in cs if x == x]) if cs and any(x == x for x in cs) else float("nan")
            print(f"  {c:12s} {m:8.4f} {s:8.4f} {es(m, n_ev):7.4f}   {cm:18.4f}")
        print("  controles (mismas muestras):")
        for ctrl in ("BARAJADO", "VACIO (ceros)", "lectura APAGADA"):
            m, s, k = resumen(us, "global", ctrl)
            ma, _, _ = resumen(us, "acierto", ctrl)
            print(f"    {ctrl:18s} GLOBAL {m:.4f} ± {s:.4f}   acierto {ma:.4f}")

    print("\n" + "=" * 96)
    print("HIPOTESIS")
    print("=" * 96)
    for b in BRAZOS:
        us = datos[b]
        if not us:
            continue
        g, sg, _ = resumen(us, "global")
        ac, sa, _ = resumen(us, "acierto")
        c1, c2 = g > PISO_MUDO, ac >= 0.80
        print(f"H1 · {b:11s} GLOBAL {g:.4f} ± {sg:.4f} {'>' if c1 else '<='} {PISO_MUDO} · "
              f"acierto {ac:.4f} ± {sa:.4f} {'>=' if c2 else '<'} 0,80  ->  "
              f"{'SE CUMPLE' if (c1 and c2) else 'NO se cumple'}")
    print()
    for b in BRAZOS:
        us = datos[b]
        if not us:
            continue
        r, _, _ = resumen(us, "nose_rel")
        au, _, _ = resumen(us, "nose_aus")
        pares = [(d["final"]["con archivo"]["nose_rel"], d["final"]["con archivo"]["nose_aus"])
                 for _, d in us]
        ok = sum(1 for x, y in pares if x == x and y == y and x < y)
        print(f"H2 · {b:11s} nose_rel {r:.4f} contra nose_aus {au:.4f} · "
              f"{ok} de {len(pares)} semillas en la direccion predicha")
    print()
    for tag, x, y in [("H3 · calentar 3.000 contra no calentar", "abstcal3s", "abst3s"),
                      ("par A · calentar 6.000 contra 3.000, post igualado", "abstcal6p", "abstcal3s"),
                      ("par D · 19.000 post contra 13.000 post", "abstlargo", "abstcal3s")]:
        if not (datos[x] and datos[y]):
            continue
        print(f"{tag}  ({x} - {y})")
        for c in ("global", "acierto", "nose_rel"):
            d_ = dif(datos[x], datos[y], c)
            if d_:
                print(f"    {c:10s} {d_['media']:+.4f} ± {d_['sd']:.4f}  t={d_['t']:+.2f} "
                      f"(n={d_['n']} semillas pareadas)")

    print("\n" + "=" * 96)
    print("DESESTABILIZACION (criterio de la correccion de E-4)")
    print("=" * 96)
    fs = [f for b in BRAZOS for f in sorted(glob.glob(f"banco_{b}_s*.json"))
          if ".parcial." not in f]
    sys.stdout.flush()
    if fs:
        os.system("python3 desestabiliza.py " + " ".join(f'"{f}"' for f in fs))
    else:
        print("  (todavia no hay unidades cerradas)")


if __name__ == "__main__":
    main()
