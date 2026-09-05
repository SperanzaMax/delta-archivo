"""Los criterios L-1 a L-5 con sus numeros. `PREREG_ARCHIVO_LARGO` c769a4ef + enmienda 0410e957.

NO adjudica solo. Imprime cada criterio con el numero que lo decide y el umbral al lado, y marca
CUMPLE / NO con aritmetica simple. Seis veces en este proyecto un juez automatico dio un veredicto
incorrecto (ver `regla-verificar-antes-de-veredicto`), asi que la salida esta pensada para LEERSE:
todos los numeros crudos quedan a la vista y ninguno se resume en un booleano solo.

    python3 juzgar_archivo_largo.py [--meta 2000]
"""
import argparse, glob, json, os

AQUI = os.path.dirname(os.path.abspath(__file__))
PISO = 0.4065          # el piso trivial del banco


def ultimo(u):
    """La ultima EVALUACION de la unidad. El JSON de corrida guarda `config` y una `historia`; lo
    que interesa es su ultima entrada, que es la evaluacion mas reciente."""
    js = sorted(glob.glob(os.path.join(AQUI, "corridas_*", f"{u}.json")))
    if not js:
        return None
    h = json.load(open(js[-1])).get("historia") or []
    return h[-1] if h else None


def masa(u):
    js = sorted(glob.glob(os.path.join(AQUI, "corridas_*", f"{u}_masa.json")))
    if not js:
        js = sorted(glob.glob(os.path.join(AQUI, f"masa_turnos_{u}.json")))
    return json.load(open(js[-1])) if js else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", type=int, default=2000)
    a = ap.parse_args()
    TR = ["lg3_s0", "lg3_s1", "lg3_s2"]
    CO = ["lc3_s0", "lc3_s1", "lc3_s2"]

    print(f"CAMPANIA DEL ARCHIVO LARGO · meta {a.meta} pasos · piso trivial {PISO}")
    print(f"{'unidad':8s} {'paso':>6s} {'largo':>8s} {'corto':>8s} {'anterior':>9s} {'nose':>7s}")
    d = {}
    for u in TR + CO:
        j = ultimo(u)
        d[u] = j
        if not j:
            print(f"{u:8s}  sin arrancar"); continue
        cruz = j.get("cruzada_corto") or {}
        largo = j.get("vigente")
        corto = cruz.get("vigente")
        if u in CO:                       # el control entrena en corto: su cruzada es la LARGA
            largo, corto = (j.get("cruzada_largo") or {}).get("vigente"), j.get("vigente")
        f4 = lambda x: "   —" if x is None else f"{x:.4f}"
        print(f"{u:8s} {j.get('paso', 0):6d} {f4(largo):>8} {f4(corto):>8} "
              f"{f4(j.get('anterior')):>9} {f4(j.get('nose')):>7}")

    def val(u, cual):
        j = d.get(u)
        if not j:
            return None
        if u in TR:
            return j.get("vigente") if cual == "largo" else (j.get("cruzada_corto") or {}).get("vigente")
        return (j.get("cruzada_largo") or {}).get("vigente") if cual == "largo" else j.get("vigente")

    print()
    l1 = [val(u, "largo") for u in TR]
    ok1 = sum(1 for x in l1 if x is not None and x > PISO)
    print(f"L-1 principal · tratada en archivo LARGO > {PISO} en >=2 de 3")
    print(f"    {[None if x is None else round(x, 4) for x in l1]}  ->  {ok1}/3  "
          f"{'CUMPLE' if ok1 >= 2 else 'NO'}")

    l2 = [val(u, "largo") for u in CO]
    ok2 = all(x is not None and x < 0.15 for x in l2)
    print(f"L-2 BLOQUEANTE · control en archivo LARGO < 0,15")
    print(f"    {[None if x is None else round(x, 4) for x in l2]}  ->  "
          f"{'CUMPLE' if ok2 else 'NO — si el control tambien sube, la campania no midio lo que dice'}")

    print("L-3 precio · tratada en CORTO no cae mas de 0,05 contra su control en corto")
    ok3 = 0
    for t, c in zip(TR, CO):
        a_, b_ = val(t, "corto"), val(c, "corto")
        if a_ is None or b_ is None:
            print(f"    {t}: falta dato"); continue
        cae = b_ - a_
        ok3 += cae <= 0.05
        print(f"    {t} {a_:.4f} vs {c} {b_:.4f} · cae {cae:+.4f}  "
              f"{'ok' if cae <= 0.05 else 'PAGA'}")

    print("L-4 el sello · indice <= 0,40 y brecha real-barajado >= 0,20 en >=2 de 3")
    ok4 = 0
    for u in TR:
        m = masa(u)
        if not m:
            print(f"    {u}: sin medir"); continue
        real = next((c for c in m["celdas"] if c["ses_extra"] == 26 and not c.get("barajado")), None)
        bar = next((c for c in m["celdas"] if c["ses_extra"] == 26 and c.get("barajado")), None)
        if not real or not bar:
            print(f"    {u}: falta una celda"); continue
        br = bar["indice"] - real["indice"]
        cumple = real["indice"] <= 0.40 and br >= 0.20
        ok4 += cumple
        print(f"    {u} indice {real['indice']:.4f} · barajado {bar['indice']:.4f} · "
              f"brecha {br:+.4f} · masa_corr {real['masa_correcta']:.4f}  "
              f"{'CUMPLE' if cumple else 'no'}")
    print(f"    -> {ok4}/3   (linea de base de kq3_s0: indice 0,8886 · brecha 0,0001)")

    print("L-5 riesgo · RECUP en largo no cae mas de 0,10 contra el origen (kq3_sX: 0,6667)")
    for u in TR:
        m = masa(u)
        real = next((c for c in (m or {}).get("celdas", [])
                     if c["ses_extra"] == 26 and not c.get("barajado")), None)
        if not real:
            print(f"    {u}: sin medir"); continue
        print(f"    {u} RECUP {real['recup']:.4f} · cae {0.6667 - real['recup']:+.4f}  "
              f"{'ok' if 0.6667 - real['recup'] <= 0.10 else 'RIESGO'}")

    print()
    print("Leer L-1 y L-4 POR SEPARADO: se puede aprender a buscar sin aprender a descartar, y eso")
    print("seria un resultado y no un fracaso. L-2 es bloqueante. L-4 sin su celda barajada no cuenta.")


if __name__ == "__main__":
    main()
