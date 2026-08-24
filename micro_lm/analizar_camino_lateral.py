"""Analisis de la campania del CAMINO LATERAL (`PREREG_CAMINO_LATERAL.md`, SHA c440ec93).

Evalua W-0..W-4 sobre `lat` (w3_s*) contra el control `pre` (p3_s*), los seis en el paso 26000.

    python analizar_camino_lateral.py --paso 26000

Por que no se reuso `analizar_query_conjunta.py`: ese script tiene FAM = {pre: p, post: q} cableado,
y su P-1..P-4 son los criterios de OTRO prereg (la campania de la mañana del 22). Los criterios de
esta campania son distintos —W-1 va sobre `ident_rep` y va PAREADA POR SEMILLA, no por mediana, por
el riesgo declarado en el §7— asi que mezclarlos habria sido evaluar el prereg equivocado.

Dos reglas de procedimiento heredadas, que no son decorativas:

  · **D-1 del 20-ago** — una unidad que entra en un analisis no puede estar entrenandose al mismo
    tiempo. Los checkpoints se COPIAN a `ckpts/lat_congelados/` y todo se mide sobre la copia.
  · **el paso se verifica, no se supone** — se lee DEL CHECKPOINT y se aborta si alguna unidad no
    llego. Y ademas se verifica que `donde` sea el que la familia declara, que es la guarda contra
    el bug mas caro posible: que una familia haya corrido con la arquitectura de la otra.

El control `pre` se REUSA de `qc_26000/` (§4 del prereg: verificado bit a bit). Se revalida que los
json que estan en disco declaren `donde=pre`; con --recalcular-pre se vuelven a correr los dos
instrumentos sobre el control en vez de leerlos.
"""
import argparse
import json
import os
import pickle
import shutil
import subprocess
import sys

AQUI = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
SEMILLAS = (0, 1, 2)


def congelar(prefijo, semillas, paso, destino, donde_esperado):
    """Copia los checkpoints, verifica el paso Y la arquitectura declarada."""
    os.makedirs(destino, exist_ok=True)
    faltan = []
    for s in semillas:
        uni = f"{prefijo}3_s{s}"
        src = os.path.join(AQUI, "ckpts", f"{uni}.pkl")
        if not os.path.exists(src):
            faltan.append(f"{uni} (sin checkpoint)")
            continue
        with open(src, "rb") as f:
            d = pickle.load(f)
        if d.get("paso") != paso:
            faltan.append(f"{uni} (paso {d.get('paso')}, se pidio {paso})")
            continue
        donde = d.get("donde", d.get("config", {}).get("donde"))
        if donde != donde_esperado:
            faltan.append(f"{uni} (donde={donde}, se esperaba {donde_esperado})")
            continue
        shutil.copy2(src, os.path.join(destino, f"{uni}.pkl"))
    return faltan


def correr_ser(prefijo, semillas, dir_cong, n, salida_dir):
    out = {}
    for s in semillas:
        uni = f"{prefijo}3_s{s}"
        js = os.path.join(salida_dir, f"ser_{uni}.json")
        cmd = [PY, os.path.join(AQUI, "ser.py"), os.path.join(dir_cong, f"{uni}.pkl"),
               "--n", str(n), "--B", "64", "--semilla", "54321", "--json", js]
        r = subprocess.run(cmd, cwd=AQUI, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"  !! ser.py fallo en {uni}:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
            continue
        with open(js) as f:
            out[s] = json.load(f)
        print(f"  {uni}: acierto {out[s]['acierto']:.4f} · err_identidad "
              f"{out[s]['err_identidad']:.4f} · nose {out[s]['nose']:.4f} · "
              f"falsa_abst {out[s]['falsa_abst']:.4f}  [lectura {out[s]['donde']}]")
    return out


def correr_diag(prefijo, semillas, dir_cong, n_lotes, salida):
    unidades = ",".join(f"3_s{s}" for s in semillas)
    cmd = [PY, os.path.join(AQUI, "diag_relacion.py"), "--n", str(n_lotes), "--batch", "64",
           "--p-nose", "0.0", "--unidades", unidades, "--prefijo", prefijo,
           "--dir-ckpt", dir_cong, "--salida", salida]
    r = subprocess.run(cmd, cwd=AQUI, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  !! diag_relacion fallo:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
        return {}
    print(r.stdout.split("-" * 82)[-1].strip())
    with open(salida) as f:
        return json.load(f)["unidades"]


def leer_pre(salida_dir, semillas):
    """Reusa el control ya medido (§4 del prereg). Revalida `donde` antes de aceptarlo."""
    ser = {}
    for s in semillas:
        p = os.path.join(salida_dir, f"ser_p3_s{s}.json")
        if not os.path.exists(p):
            return None, None
        with open(p) as f:
            d = json.load(f)
        if d.get("donde") != "pre":
            sys.exit(f"ABORTA: {p} declara donde={d.get('donde')}, no 'pre'.")
        ser[s] = d
    p = os.path.join(salida_dir, "diag_pre.json")
    if not os.path.exists(p):
        return ser, None
    with open(p) as f:
        return ser, json.load(f)["unidades"]


def evaluar(res):
    ser_lat, ser_pre = res["ser"]["lat"], res["ser"]["pre"]
    dg_lat, dg_pre = res["diag"]["lat"], res["diag"]["pre"]
    comunes = sorted(set(ser_lat) & set(ser_pre))
    print("=" * 78)
    print("EVALUACION DEL PRE-REGISTRO · PREREG_CAMINO_LATERAL.md")
    print("=" * 78)
    veredicto = {}

    # --- W-0 · BLOQUEANTE. lat aprende la tarea: acierto >= 0,70 en >= 2 de 3 ------------------
    ac = {s: ser_lat[s]["acierto"] for s in sorted(ser_lat)}
    n_ok = sum(1 for v in ac.values() if v >= 0.70)
    w0 = n_ok >= 2
    print(f"\nW-0 · BLOQUEANTE · acierto de lat {[f'{ac[s]:.4f}' for s in sorted(ac)]}")
    print(f"     >= 0,70 en {n_ok}/3 semillas (hace falta >= 2)")
    print(f"     W-0: {'CUMPLE' if w0 else 'NO CUMPLE'}")
    veredicto["W-0"] = w0
    if not w0:
        print("\n  >> W-0 falla: por el §6 del prereg TODO LO DEMAS QUEDA NO EVALUABLE.")
        print("     Se archiva sin interpretar y se declara que en esta arquitectura la query no")
        print("     se puede tocar sin romper el modelo, ni dejando la inyeccion en su lugar.")
        res["veredicto"] = veredicto
        return

    # --- W-1 · PRINCIPAL. ident_rep menor en lat, PAREADO POR SEMILLA, >= 2 de 3 ---------------
    print(f"\nW-1 · PRINCIPAL · ident_rep, pareado por semilla (§7: NO se usa la media)")
    baja, det = 0, []
    for s in comunes:
        a = dg_pre.get(f"3_s{s}", {}).get("ident_rep")
        b = dg_lat.get(f"3_s{s}", {}).get("ident_rep")
        if a is None or b is None:
            det.append(f"s{s}: sin dato")
            continue
        ok = b < a
        baja += ok
        det.append(f"s{s}: pre {a:.4f} -> lat {b:.4f}  ({b - a:+.4f}) {'OK' if ok else 'no'}")
    for x in det:
        print(f"     {x}")
    w1 = baja >= 2
    print(f"     baja en {baja}/3 semillas (hace falta >= 2)")
    print(f"     W-1: {'CUMPLE' if w1 else 'NO CUMPLE'}")
    veredicto["W-1"] = w1

    # --- W-2 · MECANICISTA. brecha unica-repetida en lat <= la mitad de pre, >= 2 de 3 ---------
    print(f"\nW-2 · MECANICISTA · brecha acierto(unica) - acierto(repetida)")
    ok2, det = 0, []
    for s in comunes:
        dp, dl = dg_pre.get(f"3_s{s}"), dg_lat.get(f"3_s{s}")
        if not dp or not dl:
            det.append(f"s{s}: sin dato")
            continue
        bp = dp["ac_unica"] - dp["ac_rep"]
        bl = dl["ac_unica"] - dl["ac_rep"]
        ok = bl <= bp / 2
        ok2 += ok
        det.append(f"s{s}: pre {bp:.4f} -> lat {bl:.4f}  (hace falta <= {bp / 2:.4f}) "
                   f"{'OK' if ok else 'no'}")
    for x in det:
        print(f"     {x}")
    w2 = ok2 >= 2
    print(f"     W-2: {'CUMPLE' if w2 else 'NO CUMPLE'}  ({ok2}/3 semillas)")
    veredicto["W-2"] = w2

    # --- W-3 · ESPECIFICIDAD. ident_unica <= 0,03 en las TRES ----------------------------------
    print(f"\nW-3 · ESPECIFICIDAD · ident_unica de lat (compuerta <= 0,03 en las tres)")
    vals, ok3 = {}, True
    for s in sorted(ser_lat):
        v = dg_lat.get(f"3_s{s}", {}).get("ident_unica")
        vals[s] = v
        if v is None or v > 0.03:
            ok3 = False
        vp = dg_pre.get(f"3_s{s}", {}).get("ident_unica")
        print(f"     s{s}: lat {v if v is None else f'{v:.4f}'}   (pre "
              f"{vp if vp is None else f'{vp:.4f}'})")
    print(f"     W-3: {'CUMPLE' if ok3 else 'NO CUMPLE'}")
    veredicto["W-3"] = ok3
    if not ok3:
        print("     >> por el §5, si lat empeora tambien con relacion UNICA lo que hubo fue daño")
        print("        general y no disolucion de la colision.")

    # --- W-4 · NO-INTERCAMBIO. falsa_abst <= 0,10 en las tres y nose no cae mas de 0,05 --------
    print(f"\nW-4 · NO-INTERCAMBIO")
    fa_ok, nose_ok = True, True
    for s in sorted(ser_lat):
        fa = ser_lat[s]["falsa_abst"]
        if fa > 0.10:
            fa_ok = False
        print(f"     s{s}: falsa_abst {fa:.4f} (compuerta <= 0,10)")
    for s in comunes:
        np_, nl = ser_pre[s]["nose"], ser_lat[s]["nose"]
        caida = np_ - nl
        if caida > 0.05:
            nose_ok = False
        print(f"     s{s}: nose pre {np_:.4f} -> lat {nl:.4f}  ({-caida:+.4f}, tolerancia -0,05)")
    w4 = fa_ok and nose_ok
    print(f"     W-4: {'CUMPLE' if w4 else 'NO CUMPLE'}")
    veredicto["W-4"] = w4

    # --- OBSERVACION declarada como hueco propio el 22-ago: `anterior` NO esta en W-4 ----------
    print(f"\nOBSERVACION · `anterior` (HUECO PROPIO: W-4 no lo vigila y deberia haberlo hecho;")
    print(f"              se reporta, NO se usa como criterio — mover el arco seria peor)")
    for s in comunes:
        ap_ = ser_pre[s]["por_tipo"].get("anterior", {}).get("acierto")
        al = ser_lat[s]["por_tipo"].get("anterior", {}).get("acierto")
        if ap_ is None or al is None:
            continue
        print(f"     s{s}: pre {ap_:.4f} -> lat {al:.4f}  ({al - ap_:+.4f})")

    print("\n" + "-" * 78)
    print("RESUMEN: " + " · ".join(f"{k} {'CUMPLE' if v else 'NO'}" for k, v in veredicto.items()))
    if veredicto.get("W-1") and not veredicto.get("W-2"):
        print("Regla del §6: hay mejora SIN mecanismo. Se reporta asi, sin adjudicarsela al")
        print("round-trip.")
    if w0 and not veredicto.get("W-1"):
        print("Regla del §6: W-0 pasa y W-1 falla -> la forma de la query NO es la causa de la")
        print("colision de clave, con un experimento que esta vez SI aisla el factor. El mecanismo")
        print("del 21-ago queda como correlacion y la linea se cierra: no se prueba una tercera")
        print("forma de query.")
    res["veredicto"] = veredicto


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paso", type=int, default=26000)
    ap.add_argument("--n-ser", type=int, default=2048)
    ap.add_argument("--n-diag", type=int, default=32)      # x64 = 2048 muestras
    ap.add_argument("--recalcular-pre", action="store_true")
    ap.add_argument("--salida", default=os.path.join(AQUI, "camino_lateral_20260824.json"))
    a = ap.parse_args()

    dir_cong = os.path.join(AQUI, "ckpts", "lat_congelados")
    salida_dir = os.path.join(AQUI, f"qc_{a.paso}")
    os.makedirs(salida_dir, exist_ok=True)

    print(f"== congelando checkpoints de lat en el paso {a.paso}")
    faltan = congelar("w", SEMILLAS, a.paso, dir_cong, "lat")
    if faltan:
        for x in faltan:
            print(f"    · {x}")
        sys.exit("ABORTA: la campania lat no llego, o la arquitectura no es la declarada.")
    print("  las tres unidades lat estan en el paso pedido y declaran donde=lat\n")

    res = {"paso": a.paso, "n_ser": a.n_ser, "n_diag": a.n_diag * 64,
           "prereg": "PREREG_CAMINO_LATERAL.md", "ser": {}, "diag": {}}

    print("== SER · familia lat")
    res["ser"]["lat"] = correr_ser("w", SEMILLAS, dir_cong, a.n_ser, salida_dir)
    print("\n== relacion unica vs repetida · familia lat")
    res["diag"]["lat"] = correr_diag("w", SEMILLAS, dir_cong, a.n_diag,
                                     os.path.join(salida_dir, "diag_lat.json"))

    ser_pre, diag_pre = (None, None) if a.recalcular_pre else leer_pre(salida_dir, SEMILLAS)
    if ser_pre is None or diag_pre is None:
        print("\n== el control pre no estaba medido (o se pidio recalcular): se corre")
        dir_pre = os.path.join(AQUI, "ckpts", "qc_congelados")
        f_pre = congelar("p", SEMILLAS, a.paso, dir_pre, "pre")
        if f_pre:
            for x in f_pre:
                print(f"    · {x}")
            sys.exit("ABORTA: el control pre no esta en el paso pedido.")
        ser_pre = correr_ser("p", SEMILLAS, dir_pre, a.n_ser, salida_dir)
        diag_pre = correr_diag("p", SEMILLAS, dir_pre, a.n_diag,
                               os.path.join(salida_dir, "diag_pre.json"))
    else:
        print("\n== control pre REUSADO de qc_26000 (§4 del prereg, verificado bit a bit)")
    res["ser"]["pre"] = {int(k): v for k, v in ser_pre.items()}
    res["diag"]["pre"] = diag_pre

    # La guarda que mas caro saldria: que una familia haya corrido con la arquitectura de la otra.
    mal = [f"{fam}/s{s}: el checkpoint dice donde={d['donde']}"
           for fam, esp in (("lat", "lat"), ("pre", "pre"))
           for s, d in res["ser"][fam].items() if d["donde"] != esp]
    if mal:
        print("!! ARQUITECTURA CRUZADA — el analisis no es valido:")
        for x in mal:
            print(f"   · {x}")
        sys.exit(1)

    print()
    evaluar(res)
    with open(a.salida, "w") as f:
        json.dump(res, f, indent=1)
    print(f"\n-> {a.salida}")


if __name__ == "__main__":
    main()
