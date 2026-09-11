"""ANALIZADOR DEL PLAN B DEL TOP-K ENTRENADO · PREREG_TOPK_ENTRENADO.md §3 y §7 · 2026-09-11

Escrito a las 09:20 con las evaluaciones de los pasos 500 y 1.000 ya vistas y las del final NO.
Lee los seis JSON de `corridas_20260911/` y contesta H1-H4 tal como estan escritas, mas la lectura
de la pelicula (masa en la entrada correcta, en lo propio y en lo viejo, ultimo cuadro).

    python3 analizar_topk.py [corridas_20260911]
"""
import json, os, sys
import numpy as np

DIA = sys.argv[1] if len(sys.argv) > 1 else "corridas_20260911"
AQUI = os.path.dirname(os.path.abspath(__file__))
TS, DS = ["ts3_s0", "ts3_s1", "ts3_s2"], ["ds3_s0", "ds3_s1", "ds3_s2"]
FIN, DETECTOR = 8000, 2500


def historia(u):
    f = os.path.join(AQUI, DIA, u + ".json")
    return json.load(open(f))["historia"] if os.path.exists(f) else []


def en(h, paso):
    c = [e for e in h if e["paso"] == paso]
    return c[-1] if c else None


def ultimo(h):
    return h[-1] if h else None


print(f"{'unidad':8} {'paso':>5} {'K':>2} {'largo vig':>9} {'ant':>6} {'nose':>6} {'falsa':>6} {'corto vig':>9} {'corto nose':>10}")
fin = {}
for u in TS + DS:
    h = historia(u); e = ultimo(h)
    if not e:
        print(f"{u:8} (sin datos)"); continue
    ac = e.get("archivo_corto", {})
    fin[u] = e
    print(f"{u:8} {e['paso']:>5} {e.get('topk', 0):>2} {e['vigente']:>9.4f} {e['anterior']:>6.3f} {e['nose']:>6.3f} "
          f"{e['falsa_abst']:>6.3f} {ac.get('vigente', float('nan')):>9.4f} {ac.get('nose', float('nan')):>10.3f}")

completas = all(u in fin and fin[u]["paso"] >= FIN for u in TS + DS)
print("\n" + ("las seis llegaron al paso 8.000" if completas else "OJO: no todas llegaron al 8.000; lo de abajo es PARCIAL"))

# H1: ts3 vigente largo >= 0,90 en >= 2 de 3
ts_v = [fin[u]["vigente"] for u in TS if u in fin]
print(f"\nH1  ts3 vigente largo al final: {ts_v}  ->  {'CUMPLE' if sum(v >= 0.90 for v in ts_v) >= 2 else 'NO cumple'} (>= 0,90 en >= 2 de 3)")
# H2: media ts - media ds >= 0,10
ds_v = [fin[u]["vigente"] for u in DS if u in fin]
if ts_v and ds_v:
    d = float(np.mean(ts_v) - np.mean(ds_v))
    print(f"H2  media ts3 {np.mean(ts_v):.4f} - media ds3 {np.mean(ds_v):.4f} = {d:+.4f}  ->  "
          f"{'CUMPLE' if d >= 0.10 else 'NO cumple: ' + ('EMPATE, lo que enseño fue el archivo largo y no la seleccion' if abs(d) < 0.10 else 'el denso gana')}")
# H3: corto ts >= corto ds - 0,05
tc = [fin[u].get("archivo_corto", {}).get("vigente", np.nan) for u in TS if u in fin]
dc = [fin[u].get("archivo_corto", {}).get("vigente", np.nan) for u in DS if u in fin]
if tc and dc:
    d3 = float(np.nanmean(tc) - np.nanmean(dc))
    print(f"H3  archivo corto: ts3 {np.nanmean(tc):.4f} contra ds3 {np.nanmean(dc):.4f} ({d3:+.4f})  ->  {'CUMPLE' if d3 >= -0.05 else 'NO cumple: el top-k cuesta en archivo corto'}")
# H4: nose >= 0,20 en el paso 2500
for u in TS:
    e = en(historia(u), DETECTOR)
    print(f"H4  {u} nose en {DETECTOR}: {e['nose'] if e else 'sin evaluacion'}  ->  "
          f"{'sano' if e and e['nose'] >= 0.20 else ('COLAPSO' if e else '?')}")

# la pelicula: ultimo cuadro
print("\nPELICULA (ultimo cuadro): masa en la correcta / en lo propio / en lo viejo · prediccion")
for u in TS + DS:
    f = os.path.join(AQUI, "ckpts", u + "_fotos.json")
    if not os.path.exists(f):
        continue
    d = json.load(open(f)); c = d["cuadros"][-1]; cor = d["muestra"]["correctos"]
    prop = sum(c["atencion"][k] for k in cor); tot = sum(c["atencion"])
    print(f"  {u:8} paso {c['paso']:>5} K{c['topk']} correcta {prop:.3f} propio {tot:.3f} viejo {c.get('masa_relleno', 0):.3f} "
          f"pred {c['pred']} {'ok' if c['pred'] == d['respuesta_correcta'] else '(era ' + d['respuesta_correcta'] + ')'} · {len(d['cuadros'])} cuadros")
