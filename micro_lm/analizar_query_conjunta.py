"""Analisis de la campania de la QUERY CONJUNTA (`PREREG_QUERY_CONJUNTA.md` + enmienda E-1).

Corre los dos instrumentos declarados sobre las seis unidades y evalua P-1..P-4 con los criterios
tal como quedaron congelados. P-5 (la razon top2/top1 de la lectura) va aparte, con la sonda del
empate.

    python analizar_query_conjunta.py --paso 26000

Dos reglas de procedimiento que este script implementa y no son decorativas:

  · **D-1 del 20-ago** — una unidad que entra en un analisis no puede estar entrenandose al mismo
    tiempo. Los checkpoints se COPIAN a `ckpts/qc_congelados/` y todo se mide sobre la copia. El
    20-ago una sonda midio `c4_s2` en el paso 15000 mientras Colab lo sobreescribia y hubo que
    descartar la corrida entera.
  · **el paso se verifica, no se supone** — la D-1 de la replica: esperar a que «el JSON diga 20000»
    no alcanza, porque el JSON se escribe antes de que baje el .pkl. Se lee el paso DEL CHECKPOINT y
    se aborta si alguna unidad no llego.
"""
import argparse
import json
import os
import pickle
import shutil
import statistics
import subprocess
import sys

AQUI = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
SEMILLAS = (0, 1, 2)
FAM = {"pre": "p", "post": "q"}          # familia -> prefijo de la corrida


def congelar(prefijo, semillas, paso, destino):
    """Copia los checkpoints y verifica que TODOS esten en el paso pedido."""
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
        print(f"  {uni}: err_identidad {out[s]['err_identidad']:.4f} · "
              f"falsa_abst {out[s]['falsa_abst']:.4f} · nose {out[s]['nose']:.4f} · "
              f"acierto {out[s]['acierto']:.4f}  [lectura {out[s]['donde']}]")
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paso", type=int, default=26000)
    ap.add_argument("--n-ser", type=int, default=2048)
    ap.add_argument("--n-diag", type=int, default=32)      # x64 = 2048 muestras
    ap.add_argument("--salida", default=os.path.join(AQUI, "query_conjunta_20260822.json"))
    ap.add_argument("--parcial", action="store_true",
                    help="mide lo que haya llegado al paso pedido en vez de abortar. Para mirar la "
                         "campania a mitad de camino; un resultado parcial NO evalua el prereg")
    a = ap.parse_args()

    dir_cong = os.path.join(AQUI, "ckpts", "qc_congelados")
    salida_dir = os.path.join(AQUI, f"qc_{a.paso}")
    os.makedirs(salida_dir, exist_ok=True)

    print(f"== congelando checkpoints en el paso {a.paso}")
    faltan = []
    for fam, pre in FAM.items():
        faltan += [f"[{fam}] {x}" for x in congelar(pre, SEMILLAS, a.paso, dir_cong)]
    if faltan:
        print("  unidades que NO estan en el paso pedido:")
        for x in faltan:
            print(f"    · {x}")
        if not a.parcial:
            sys.exit("ABORTA: la campania no llego. Con --parcial se mide lo que haya, sin evaluar "
                     "el prereg.")

    listas = {fam: [s for s in SEMILLAS
                    if os.path.exists(os.path.join(dir_cong, f"{FAM[fam]}3_s{s}.pkl"))]
              for fam in FAM}
    print(f"  medibles: pre {listas['pre']} · post {listas['post']}\n")

    res = {"paso": a.paso, "n_ser": a.n_ser, "n_diag": a.n_diag * 64, "ser": {}, "diag": {}}
    for fam, pre in FAM.items():
        print(f"== SER · familia {fam}")
        res["ser"][fam] = correr_ser(pre, listas[fam], dir_cong, a.n_ser, salida_dir)
        print(f"\n== relacion unica vs repetida · familia {fam}")
        res["diag"][fam] = correr_diag(pre, listas[fam], dir_cong, a.n_diag,
                                       os.path.join(salida_dir, f"diag_{fam}.json"))
        print()

    # --- guarda contra el bug que mas caro saldria: que una familia haya corrido con la otra
    # arquitectura. `donde` se lee del checkpoint, asi que esto compara lo que el ckpt DECLARA con
    # lo que la familia tenia que ser.
    mal = [f"{fam}/s{s}: el checkpoint dice donde={d['donde']}"
           for fam in FAM for s, d in res["ser"][fam].items() if d["donde"] != fam]
    if mal:
        print("!! ARQUITECTURA CRUZADA — el analisis no es valido:")
        for x in mal:
            print(f"   · {x}")
        sys.exit(1)

    evaluar(res)
    with open(a.salida, "w") as f:
        json.dump(res, f, indent=1)
    print(f"\n-> {a.salida}")


def med(xs):
    return statistics.median(xs) if xs else float("nan")


def evaluar(res):
    ser, diag = res["ser"], res["diag"]
    comunes = sorted(set(ser["pre"]) & set(ser["post"]))
    if not comunes:
        print("sin semillas comparables: no se evalua el prereg")
        return

    print("=" * 78)
    print("EVALUACION DEL PRE-REGISTRO")
    print("=" * 78)

    # --- P-1 · err_identidad baja en post, en >=2 de 3 semillas, mediana >= 0,03 absoluto ---------
    e_pre = {s: ser["pre"][s]["err_identidad"] for s in comunes}
    e_post = {s: ser["post"][s]["err_identidad"] for s in comunes}
    baja = [s for s in comunes if e_post[s] < e_pre[s]]
    caida = med(list(e_pre.values())) - med(list(e_post.values()))
    p1 = len(baja) >= 2 and caida >= 0.03
    print(f"\nP-1 · err_identidad   pre {[f'{e_pre[s]:.4f}' for s in comunes]}")
    print(f"                      post {[f'{e_post[s]:.4f}' for s in comunes]}")
    print(f"     baja en {len(baja)}/{len(comunes)} semillas · caida de la mediana {caida:+.4f} "
          f"(hace falta >= 0,03 y >= 2 semillas)")
    print(f"     P-1: {'CUMPLE' if p1 else 'NO CUMPLE'}")

    # --- P-2 · la brecha unica-repetida en post <= la mitad de la de pre, en >=2 de 3 -------------
    def brecha(fam, s):
        d = diag[fam].get(f"3_s{s}")
        return None if not d else d["ac_unica"] - d["ac_rep"]

    p2_ok, det2 = [], []
    for s in comunes:
        bp, bq = brecha("pre", s), brecha("post", s)
        if bp is None or bq is None:
            det2.append(f"s{s}: sin dato")
            continue
        ok = bq <= bp / 2
        p2_ok.append(ok)
        det2.append(f"s{s}: pre {bp:.4f} -> post {bq:.4f} {'OK' if ok else 'no'}")
    p2 = sum(p2_ok) >= 2
    print(f"\nP-2 · brecha acierto(relacion unica) - acierto(relacion repetida)")
    for x in det2:
        print(f"     {x}")
    print(f"     P-2: {'CUMPLE' if p2 else 'NO CUMPLE'}  ({sum(p2_ok)}/{len(comunes)} semillas)")

    # --- P-3 · no-intercambio: falsa_abst <= 0,10 en las tres y vigente no baja mas de 0,02 -------
    fa = {s: ser["post"][s]["falsa_abst"] for s in comunes}
    vig_pre = [ser["pre"][s]["por_tipo"].get("vigente", {}).get("acierto", 0.0) for s in comunes]
    vig_post = [ser["post"][s]["por_tipo"].get("vigente", {}).get("acierto", 0.0) for s in comunes]
    caida_vig = med(vig_pre) - med(vig_post)
    p3 = all(v <= 0.10 for v in fa.values()) and caida_vig <= 0.02
    print(f"\nP-3 · falsa_abst post {[f'{fa[s]:.4f}' for s in comunes]} (compuerta <= 0,10)")
    print(f"     vigente  pre {med(vig_pre):.4f} -> post {med(vig_post):.4f} "
          f"(caida {caida_vig:+.4f}, tolerancia 0,02)")
    print(f"     P-3: {'CUMPLE' if p3 else 'NO CUMPLE'}")

    # --- P-4 · especificidad: con relacion unica, err_identidad en post <= 0,03 -------------------
    iu = {s: diag["post"].get(f"3_s{s}", {}).get("ident_unica") for s in comunes}
    iu = {s: v for s, v in iu.items() if v is not None}
    p4 = bool(iu) and all(v <= 0.03 for v in iu.values())
    print(f"\nP-4 · err_identidad con relacion UNICA en post "
          f"{[f'{iu[s]:.4f}' for s in sorted(iu)]} (tope 0,03)")
    print(f"     P-4: {'CUMPLE' if p4 else 'NO CUMPLE'}")

    print("\n" + "-" * 78)
    if p1 and p2:
        print("LECTURA: la forma de la query es CAUSA de la colision de clave. El efecto existe")
        print("y el eslabon mecanico lo sostiene.")
    elif p1 and not p2:
        print("LECTURA: el efecto existe pero SIN mecanismo. Por el §5 se reporta como mejora, y")
        print("NO se le adjudica al hallazgo del round-trip.")
    else:
        print("LECTURA: por el §5, la forma de la query NO es la causa de la colision. El mecanismo")
        print("del 21-ago queda como correlacion y no se prueba una tercera posicion de inyeccion.")
    if not p3:
        print("AVISO: P-3 no cumple — lo que haya bajado se pago con abstencion o con la vigente.")
    if p1 and not p4:
        print("AVISO: P-4 no cumple — mejora pareja, compatible con mas capacidad y no con haber")
        print("disuelto la colision. Ver tambien el riesgo del §6 (profundidad de computo).")
    res["veredicto"] = {"P1": bool(p1), "P2": bool(p2), "P3": bool(p3), "P4": bool(p4)}


if __name__ == "__main__":
    main()
