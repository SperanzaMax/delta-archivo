"""Juez del CRUCE · criterios de PREREG_CRUCE_FORMAS.md (SHA 410acd25), escritos ANTES del dato.

X-1 PRINCIPAL  dentro de `invertida` nose_ent < nose_rel, y dentro de `directa` nose_rel < nose_ent,
               en >=2 de 3. Es una INTERACCION: lo que se afirma es que el orden se INVIERTE.
X-2 MAGNITUD   la diferencia (nose_ent - nose_rel) cambia de signo con un salto >= 0,15 en >=2 de 3.
X-3 NO DAÑO    vigente >= 0,90 en >=2 de 3 y en las DOS formas.
X-0 ya se cumplio ANTES de lanzar (`chequeo_formas_q.py`, compuerta ABRE).

OJO con la leccion del 2-sep: `nose_rel` y `nose_ent` premian ABSTENERSE, asi que una unidad muda
las tiene altas por no hacer nada. Por eso se imprime tambien `falsa_abst` y la exactitud global
POR FORMA, y el juez no adjudica sin mirar `vigente`.
"""
import os
import sys

import numpy as np
import jax

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)

import entrenar as E, idioma as I, medir_ratio_ce as R, modelo as M

TRAT = ["cl3_s0", "cl3_s1", "cl3_s2"]   # D-1: s0 se descarto por formas_q contaminada
META = 26000
FORMAS = ("directa", "lejana")


def metricas(u):
    p = os.path.join(AQUI, "ckpts", f"{u}.pkl")
    if not os.path.exists(p):
        return None
    params, cfg, paso = R.cargar(p)
    M.KQ = cfg.get("kernel_q", 3)
    params = jax.tree_util.tree_map(jax.numpy.asarray, params)
    I.fijar_version(cfg.get("idioma", 2))
    fn = E.predecir_cabeza if cfg.get("abst") == "cabeza" else E.predecir
    s = int(u.split("_s")[1])
    fq = tuple(x.strip() for x in cfg.get("formas_q", "directa").split(","))
    m = E.evaluar(params, np.random.default_rng(90000 + s), n=12, nivel=cfg["nivel"],
                  p_vieja=cfg.get("p_vieja", 0.35), p_nose=cfg.get("p_nose", 0.4), pred_fn=fn,
                  formas_q=fq, por_forma=True)
    m["paso"], m["kq"], m["formas"] = paso, cfg.get("kernel_q", 3), cfg.get("formas_q")
    return m


def main():
    print("=" * 104)
    print("EL CONTROL QUE ADJUDICA · ¿arregla la DIVERSIDAD, o arregla VER la relacion?")
    print("prereg SHA 2b480ce7 · las DOS formas tienen la relacion AFUERA de la ventana")
    print("=" * 104)
    filas = {}
    for u in TRAT:
        m = metricas(u)
        if m:
            filas[u] = m
        else:
            print(f"  (sin checkpoint: {u})")
    if not filas:
        return

    print(f"\n{'unidad':8s} {'paso':>6s} {'formas':22s} {'vigente':>8s} {'nose':>7s} {'falsa':>7s}")
    for u, m in filas.items():
        corto = "" if m["paso"] >= META else f"  <- {m['paso']} de {META}"
        print(f"{u:8s} {m['paso']:6d} {str(m['formas']):22s} {m['vigente']:8.4f} {m['nose']:7.4f}"
              f" {m['falsa_abst']:7.4f}{corto}")

    print(f"\n{'unidad':8s} {'forma':11s} {'d_rel':>5s} {'d_ent':>5s} {'vigente':>8s}"
          f" {'nose_ent':>9s} {'nose_rel':>9s} {'ent-rel':>9s} {'falsa':>7s}")
    for u, m in filas.items():
        for f in FORMAS:
            d = m.get("por_forma", {}).get(f)
            if not d:
                continue
            dif = d["nose_ent"] - d["nose_rel"]
            print(f"{u:8s} {f:11s} {I.DIST_Q[f]['rel']:5d} {I.DIST_Q[f]['ent']:5d}"
                  f" {d['vigente']:8.4f} {d['nose_ent']:9.4f} {d['nose_rel']:9.4f}"
                  f" {dif:+9.4f} {d['falsa_abst']:7.4f}")

    llegaron = [u for u in filas if filas[u]["paso"] >= META]
    print(f"\n  llegaron a {META}: {len(llegaron)} de {len(TRAT)}")
    if len(llegaron) < 2:
        print("  ** NO EVALUABLE ** el prereg pide >=2.")
        return

    alto = bajo = c2 = 0
    for u in llegaron:
        pf = filas[u]["por_forma"]
        nr = pf["directa"]["nose_rel"]
        if nr >= 0.90:
            alto += 1
        if nr <= 0.80:
            bajo += 1
        if all(pf[f]["vigente"] >= 0.90 for f in FORMAS):
            c2 += 1

    print("\n" + "-" * 104)
    print("CRITERIOS, escritos antes del dato   (control v3, misma arquitectura y UNA forma: 0,59-0,73)")
    print(f"  C-1  nose_rel en `directa` >= 0,90   {alto} de {len(llegaron)}"
          f"   -> gana DIVERSIDAD")
    print(f"  C-1  nose_rel en `directa` <= 0,80   {bajo} de {len(llegaron)}"
          f"   -> gana VER LA RELACION")
    print(f"  C-2  NO DAÑO, vigente >= 0,90 en las DOS formas   {c2} de {len(llegaron)}"
          f"   {'CUMPLE' if c2 >= 2 else 'NO CUMPLE'}")
    print("-" * 104)
    if c2 < 2:
        print("  -> OJO: sin C-2 el contraste NO se lee, `lejana` puede no haberse aprendido.")
    elif alto >= 2:
        print("  -> gana (D): la DIVERSIDAD alcanza sola, sin ver nunca la relacion. La ventana")
        print("     queda como un efecto de la distribucion de entrenamiento, no una barrera.")
    elif bajo >= 2:
        print("  -> gana (V): hay que VER la relacion al menos a veces. La ventana vuelve al centro")
        print("     y `cf3` funciona porque la forma `invertida` la mete adentro en la mitad de los casos.")
    else:
        print("  -> NO ADJUDICA, queda entre 0,80 y 0,90. Se informa asi y no se fuerza la lectura.")


if __name__ == "__main__":
    main()
