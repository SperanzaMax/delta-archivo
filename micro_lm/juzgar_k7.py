"""Juez del kernel 7 · criterios de PREREG_KERNEL_Q5.md (SHA 50c4503d), escritos ANTES del dato.

K-0 BLOQUEANTE  la sensibilidad a la RELACION deja de ser 0,0000 (media > 0,05 en >=2 de 3)
K-1 PRINCIPAL   el AUC contra `nose_rel` sube de 0,4914 (azar) a >= 0,60 en >=2 de 3
K-2 UTILIDAD    nose >= 0,90 con falsa_abst <= 0,10 en >=2 de 3
K-3 NO DAÑO     RECUP >= 0,95 en >=2 de 3
Control ya medido, NO se re-corre: v3_s0/s1/s2 (mismo todo, kernel 3).
"""
import os
import sys

import numpy as np
import jax

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)

import entrenar as E, idioma as I, medir_ratio_ce as R, modelo as M
import sensibilidad_busqueda as SB

TRAT = ["k73_s0", "k73_s1", "k73_s2"]
# El control del kernel 7 es el kernel 5, no el 3: la pregunta ya no es «¿ver la relacion ayuda?»
# sino «¿MAS ventana mejora o ENSUCIA?», y la hipotesis en contra esta escrita en PREREG_LEY_VENTANA §C.
CTRL = ["kq3_s0", "kq3_s1", "kq3_s2"]
META = 26000


def metricas(u):
    p = os.path.join(AQUI, "ckpts", f"{u}.pkl")
    if not os.path.exists(p):
        return None
    params, cfg, paso = R.cargar(p)
    M.KQ = cfg.get("kernel_q", 3)                 # la forma de convq la decide el checkpoint
    params = jax.tree_util.tree_map(jax.numpy.asarray, params)
    I.fijar_version(cfg.get("idioma", 2))
    fn = E.predecir_cabeza if cfg.get("abst") == "cabeza" else E.predecir
    s = int(u.split("_s")[1])
    m = E.evaluar(params, np.random.default_rng(90000 + s), nivel=cfg["nivel"],
                  p_vieja=cfg.get("p_vieja", 0.35), p_nose=cfg.get("p_nose", 0.4), pred_fn=fn)
    m["paso"] = paso
    m["kq"] = cfg.get("kernel_q", 3)
    return m


def main():
    print("=" * 96)
    print("KERNEL 7 · ¿mas ventana mejora, o mete tokens irrelevantes y ENSUCIA?  prereg eb5e1d50 §C")
    print("=" * 96)
    filas = {}
    for u in TRAT + CTRL:
        m = metricas(u)
        if m:
            filas[u] = m
    print(f"\n{'unidad':8s} {'kq':>3s} {'paso':>6s} {'vigente':>8s} {'nose':>7s} {'falsa':>7s}")
    for g, nom in ((TRAT, "TRATAMIENTO kernel 5"), (CTRL, "CONTROL kernel 3 (ya medido)")):
        print(f"--- {nom}")
        for u in g:
            m = filas.get(u)
            if not m:
                print(f"{u:8s} (sin checkpoint)")
                continue
            corto = "" if m["paso"] >= META else f"  <- {m['paso']} de {META}"
            print(f"{u:8s} {m['kq']:3d} {m['paso']:6d} {m['vigente']:8.4f} {m['nose']:7.4f}"
                  f" {m['falsa_abst']:7.4f}{corto}")

    llegaron = [u for u in TRAT if u in filas and filas[u]["paso"] >= META]
    if len(llegaron) < 2:
        print(f"\n** NO EVALUABLE ** llegaron {len(llegaron)} de 3 y el prereg pide >=2.")
        return

    print("\n--- K-0 y K-1: la sonda sobre la busqueda (esto tarda unos minutos) ---")
    sens, aucs = {}, {}
    for u in llegaron:
        print(f"\n>>> {u}")
        SB.correr(os.path.join(AQUI, "ckpts", f"{u}.pkl"))

    print("\n" + "-" * 96)
    print("Los numeros de K-0 (sensibilidad a la RELACION) y K-1 (AUC contra nose_rel) salen de la")
    print("tabla de arriba: K-0 pide que `TV_rel` deje de ser 0,0000 con media > 0,05 en >=2 de 3,")
    print("y K-1 que el AUC restringido a HAY vs nose_REL suba de 0,4914 a >= 0,60 en >=2 de 3.")
    ok2 = sum(1 for u in llegaron
              if filas[u]["nose"] >= 0.90 and filas[u]["falsa_abst"] <= 0.10)
    print(f"\nK-2 UTILIDAD  nose>=0,90 y falsa<=0,10: {ok2} de {len(llegaron)}"
          f"   {'CUMPLE' if ok2 >= 2 else 'NO CUMPLE'}   (control v3: nose 0,78-0,81)")
    print("-" * 96)


if __name__ == "__main__":
    main()
