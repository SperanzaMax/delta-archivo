"""ABLACION DE TAPS de `convq` · ¿es VER LA RELACION o es simplemente MAS CONTEXTO? · 2-sep

Evalua `PREREG_LEY_VENTANA.md` §A (SHA eb5e1d50), congelado antes de correr.

`INFORME_KERNEL_Q5_20260901.md` dejo el hallazgo con un agujero declarado: kernel 5 gana, pero no se
sabe si gana porque la query VE la relacion o porque tiene MAS CONTEXTO. La forma barata de separarlo
es no entrenar nada y apagar UN TAP POR VEZ del modelo ya entrenado.

En «cual es <art> <sust> de <ent> ?» las distancias desde la posicion de lectura son
    tap 0 = la posicion misma · tap 1 = <ent> · tap 2 = «de» · tap 3 = <sust> (LA RELACION) · tap 4 = <art>
La prediccion es de ESPECIFICIDAD, no de daño: el tap 3 tiene que doler MAS que el 2 y el 4. Si todos
duelen igual, apagar un tap rompe activaciones y no prueba nada.
"""
import os
import sys

import numpy as np
import jax
import jax.numpy as jnp

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)

import entrenar as E, idioma as I, medir_ratio_ce as R, modelo as M

UNIDADES = ["kq3_s0", "kq3_s1", "kq3_s2"]
# CONTROL agregado el 2-sep DESPUES de ver el resultado, y se declara asi. La lectura tentadora de
# la tabla es «al degradar la query el modelo se ABSTIENE en vez de inventar», y eso solo se puede
# afirmar si el modelo de kernel 3 NO hace lo mismo. Es la explicacion alternativa obligatoria.
CONTROL = ["v3_s0", "v3_s1", "v3_s2"]
N_LOTES = 12          # 12 x 64 = 768 consultas por celda; el juez del kernel 5 uso 8


def con_tap_en_cero(params, tap):
    """Copia de los params con `convq` del bloque 0 anulada en un tap. El resto intacto."""
    if tap is None:
        return params
    p = jax.tree_util.tree_map(lambda x: x, params)          # copia superficial del arbol
    bloques = list(p["blocks"])
    b0 = dict(bloques[0])
    b0["convq"] = b0["convq"].at[tap].set(0.0)
    bloques[0] = b0
    p = dict(p); p["blocks"] = bloques
    return p


def medir(ruta, taps):
    params, cfg, paso = R.cargar(ruta)
    M.KQ = cfg.get("kernel_q", 3)
    params = jax.tree_util.tree_map(jnp.asarray, params)
    I.fijar_version(cfg.get("idioma", 2))
    fn = E.predecir_cabeza if cfg.get("abst") == "cabeza" else E.predecir
    s = int(ruta.split("_s")[1].split(".")[0])
    filas = {}
    for tap in taps:
        # MISMA semilla de evaluacion en todas las celdas: las condiciones ven las MISMAS consultas,
        # asi que la comparacion no arrastra ruido de muestreo.
        m = E.evaluar(con_tap_en_cero(params, tap), np.random.default_rng(90000 + s),
                      n=N_LOTES, nivel=cfg["nivel"], p_vieja=cfg.get("p_vieja", 0.35),
                      p_nose=cfg.get("p_nose", 0.4), pred_fn=fn)
        m["exactitud"] = (1 - m["falsa_abst"]) * 0 + np.nan   # se calcula abajo con los pesos reales
        filas[tap] = m
    return filas, cfg, paso


def main():
    _corrida(UNIDADES, [None, 0, 1, 2, 3, 4], "TRATAMIENTO kernel 5")
    print("\n\n" + "#" * 104)
    print("# CONTROL kernel 3 · ¿tambien convierte la query rota en ABSTENCION, o inventa?")
    print("#" * 104)
    _corrida(CONTROL, [None, 0, 1, 2], "CONTROL kernel 3")


def _corrida(unidades, taps, titulo):
    nombre = {None: "completo", 0: "tap0 posicion", 1: "tap1 ENTIDAD", 2: "tap2 «de»",
              3: "tap3 RELACION", 4: "tap4 articulo"}
    print("=" * 104)
    print(f"ABLACION DE TAPS de convq · {titulo} · prereg SHA eb5e1d50 §A")
    print("=" * 104)
    todo = {}
    for u in unidades:
        ruta = os.path.join(AQUI, "ckpts", f"{u}.pkl")
        if not os.path.exists(ruta):
            print(f"  (falta {u})")
            continue
        filas, cfg, paso = medir(ruta, taps)
        todo[u] = filas
        print(f"\n--- {u}  paso={paso}  kq={cfg.get('kernel_q')}")
        print(f"  {'condicion':16s} {'vigente':>8s} {'anterior':>9s} {'nose':>7s} {'nose_ent':>9s}"
              f" {'nose_rel':>9s} {'falsa':>7s}")
        for t in taps:
            m = filas[t]
            print(f"  {nombre[t]:16s} {m['vigente']:8.4f} {m['anterior']:9.4f} {m['nose']:7.4f}"
                  f" {m['nose_ent']:9.4f} {m['nose_rel']:9.4f} {m['falsa_abst']:7.4f}")

    if not todo:
        return
    print("\n" + "=" * 104)
    print("CAIDAS respecto del modelo completo (positivo = empeora)")
    print("=" * 104)
    print("\n  ** ¿adonde va lo que deja de acertar? · fraccion de la caida de `vigente` que")
    print("     termina en ABSTENCION en vez de en una respuesta EQUIVOCADA **")
    print(f"  {'unidad':8s} " + " ".join(f"{nombre[t]:>15s}" for t in taps[1:]))
    for u, filas in todo.items():
        cel = []
        for t in taps[1:]:
            dv = filas[None]["vigente"] - filas[t]["vigente"]
            df = filas[t]["falsa_abst"] - filas[None]["falsa_abst"]
            cel.append(f"{df / dv:15.3f}" if dv > 0.01 else f"{'-':>15s}")
        print(f"  {u:8s} " + " ".join(cel))
    print(f"  {'unidad':8s} " + " ".join(f"{nombre[t]:>15s}" for t in taps[1:]))
    for clave in ("nose_rel", "vigente"):
        print(f"\n  ** {clave} **")
        for u, filas in todo.items():
            base = filas[None][clave]
            print(f"  {u:8s} " + " ".join(f"{base - filas[t][clave]:15.4f}" for t in taps[1:]))

    if len(taps) < 6:
        print("-" * 104)
        return
    print("\n" + "-" * 104)
    print("CRITERIOS del prereg §A, escritos antes del dato")
    a1 = sum(1 for f in todo.values() if f[None]["nose_rel"] - f[3]["nose_rel"] >= 0.20)
    a2 = sum(1 for f in todo.values()
             if (f[None]["nose_rel"] - f[3]["nose_rel"]) >
                max(f[None]["nose_rel"] - f[2]["nose_rel"], f[None]["nose_rel"] - f[4]["nose_rel"]))
    # El prereg NO dijo en que metrica se mide A-2, y lo encadeno a A-1, que estaba en `nose_rel`.
    # Se informan las dos, y la ambiguedad se declara en el informe en vez de elegir la que conviene.
    a2v = sum(1 for f in todo.values()
              if (f[None]["vigente"] - f[3]["vigente"]) >
                 max(f[None]["vigente"] - f[2]["vigente"], f[None]["vigente"] - f[4]["vigente"]))
    a3 = sum(1 for f in todo.values() if f[None]["vigente"] - f[1]["vigente"] >= 0.20)
    n = len(todo)
    print(f"  A-1 PRINCIPAL   tap3 baja nose_rel >= 0,20        {a1} de {n}"
          f"   {'CUMPLE' if a1 >= 2 else 'NO CUMPLE'}")
    print(f"  A-2 ESPECIFICO  medido en nose_rel                {a2} de {n}"
          f"   {'CUMPLE' if a2 >= 2 else 'NO CUMPLE'}")
    print(f"  A-2 ESPECIFICO  medido en vigente                 {a2v} de {n}"
          f"   {'CUMPLE' if a2v >= 2 else 'NO CUMPLE'}"
          f"   <- el prereg no dijo la metrica; van las dos")
    print(f"  A-3 CONTROL +   tap1 baja vigente >= 0,20         {a3} de {n}"
          f"   {'CUMPLE' if a3 >= 2 else 'NO CUMPLE'}"
          f"   {'' if a3 >= 2 else '<- sin esto, A-1 y A-2 NO SE LEEN'}")
    print("-" * 104)


if __name__ == "__main__":
    main()
