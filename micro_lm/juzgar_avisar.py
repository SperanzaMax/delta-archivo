"""Juez de PREREG_AVISAR_A_PRESUPUESTO.md (SHA cd7c78e9) · escrito ANTES de ver el dato.

La pregunta de Maxi: «que aprenda la cabeza a avisar cuando no encuentra la informacion antes de dar
una informacion equivocada». Las tres metricas que hacen falta juntas y que R-1 exige juntas:

    vigente >= 0,60      responde bien lo que SI sabe
    falsa_abst <= 0,10   no se calla cuando sabe
    nose >= 0,90         AVISA cuando no encuentra

Control historico ya medido, no se re-corre: b3_s3/s6/s7/s8, mudas, 0 de 4 utiles.
Referencia superior: b3_s0/b3_s1, exactitud global 1,0000.

Devuelve NO EVALUABLE si llegan menos de 3 unidades (riesgo declarado en el §3 del prereg).
"""
import glob
import json
import os
import sys

import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)

TRAT = ["rk3_s3", "rk3_s6", "rk3_s7", "rk3_s8", "bl3_s3", "bl3_s6"]
CTRL = ["b3_s3", "b3_s6", "b3_s7", "b3_s8"]
PISO = 0.4065          # el piso trivial: abstenerse siempre
META = 26000


def util(m):
    return m["vigente"] >= 0.60 and m["falsa_abst"] <= 0.10 and m["nose"] >= 0.90


def exactitud(m):
    """(acierto + acierto_nose)/n · la metrica de `metrica-exactitud-global`, no `nose` sola."""
    p = m.get("p_nose", 0.4)
    return (1 - p) * (m["vigente"] + m["anterior"]) / 2 + p * m["nose"]


def medir(u):
    """Evalua el checkpoint de verdad, no el json: el json puede ser de un tramo viejo."""
    import jax
    import entrenar as E, idioma as I, medir_ratio_ce as R
    p = os.path.join(AQUI, "ckpts", f"{u}.pkl")
    if not os.path.exists(p):
        return None
    params, cfg, paso = R.cargar(p)
    params = jax.tree_util.tree_map(jax.numpy.asarray, params)
    I.fijar_version(cfg.get("idioma", 2))
    fn = E.predecir_cabeza if cfg.get("abst") == "cabeza" else E.predecir
    sem = int(u.split("_s")[1])
    m = E.evaluar(params, np.random.default_rng(90000 + sem), nivel=cfg["nivel"],
                  p_vieja=cfg.get("p_vieja", 0.35), p_nose=cfg.get("p_nose", 0.4), pred_fn=fn)
    m["paso"] = paso
    m["p_nose"] = cfg.get("p_nose", 0.4)
    m["cond"] = cfg.get("perdida_cabeza", "bce")
    return m


def main():
    print("=" * 100)
    print("¿APRENDE LA CABEZA A AVISAR CON PRESUPUESTO?   criterios SHA cd7c78e9, fijados antes")
    print("=" * 100)
    filas = {}
    for u in TRAT + CTRL:
        m = medir(u)
        if m:
            filas[u] = m

    print(f"\n{'unidad':8s} {'cond':9s} {'paso':>6s} {'vigente':>8s} {'nose':>7s} {'falsa':>7s}"
          f" {'exact':>7s}  útil")
    for grupo, nom in ((TRAT, "TRATAMIENTO"), (CTRL, "CONTROL (ya medido, no se re-corre)")):
        print(f"--- {nom}")
        for u in grupo:
            m = filas.get(u)
            if not m:
                print(f"{u:8s} (sin checkpoint)")
                continue
            marca = "SÍ" if util(m) else "no"
            corto = "" if m["paso"] >= META else f"   <- sólo {m['paso']} de {META}"
            print(f"{u:8s} {m['cond']:9s} {m['paso']:6d} {m['vigente']:8.4f} {m['nose']:7.4f}"
                  f" {m['falsa_abst']:7.4f} {exactitud(m):7.4f}  {marca}{corto}")

    llegaron = [u for u in TRAT if u in filas and filas[u]["paso"] >= META]
    print(f"\nllegaron a {META}: {len(llegaron)} de {len(TRAT)}   {llegaron}")
    if len(llegaron) < 3:
        print(f"\n** NO EVALUABLE ** llegaron {len(llegaron)} y el §3 del prereg pide al menos 3.")
        print("   R-1, R-2 y R-3 NO se leen: el riesgo declarado los protege a los tres.")
        return

    leidas = [u for u in TRAT if u in filas and filas[u]["paso"] >= META]
    if len(leidas) < len(TRAT):
        print(f"   (se lee sobre {len(leidas)}, no sobre {len(TRAT)}: se declara)")

    r0 = sum(1 for u in leidas if filas[u]["falsa_abst"] < 0.90)
    utiles = [u for u in leidas if util(filas[u])]
    r3 = sum(1 for u in leidas if exactitud(filas[u]) >= PISO + 0.15)

    print("\n" + "-" * 100)
    print(f"R-0 BLOQUEANTE  no quedan en abstención total: {r0} de {len(leidas)}"
          f"   {'CUMPLE' if r0 >= min(4, len(leidas)) else 'NO CUMPLE'}")
    if r0 < min(4, len(leidas)):
        print("   ** El remedio no aguanta el presupuesto. Lo demás no se lee (R-0 es bloqueante). **")
        return
    print(f"R-1 PRINCIPAL   útiles (vig>=0,60 Y falsa<=0,10 Y nose>=0,90): {len(utiles)} de "
          f"{len(leidas)}   {'CUMPLE' if len(utiles) >= 3 else 'NO CUMPLE'}   (control: 0 de 4)")
    if utiles:
        inv = [filas[u].get("invento", float("nan")) for u in utiles]
        ok2 = all(not np.isfinite(i) or i <= 0.10 for i in inv)
        print(f"R-2 invención en las útiles: "
              + ", ".join(f"{u}={filas[u].get('invento', float('nan')):.4f}" for u in utiles)
              + f"   {'CUMPLE' if ok2 else 'SE DISPARA'}")
    else:
        print("R-2 no evaluable: ninguna unidad cumple R-1")
    print(f"R-3 exactitud >= {PISO+0.15:.4f}: {r3} de {len(leidas)}"
          f"   {'CUMPLE' if r3 >= 3 else 'NO CUMPLE'}")
    print("-" * 100)

    if len(utiles) >= 3:
        print("\n** LA CABEZA APRENDE A AVISAR SI SE LE DA EL PRESUPUESTO. **")
        print("   El negativo del 29-ago era por impaciencia, no por la condición.")
    else:
        print("\n** R-1 no cumple: con presupuesto completo tampoco alcanza. **")
        print("   Recién ahora el negativo del 29-ago queda fundado.")


if __name__ == "__main__":
    main()
