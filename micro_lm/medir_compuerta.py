#!/usr/bin/env python3
"""`falsa_abst` / `nose` / `vigente` de una lista de checkpoints, con el instrumento de la campaña.

`PREREG_PRESUPUESTO_TOKEN.md` §1 pide medir con el MISMO instrumento y el mismo rng de prueba
(77000 + semilla) y 2048 muestras que uso la replica de `c4`. Asi que no se reimplementa nada: se
importa `evaluar` de `entrenar.py` —el import es seguro, todo su codigo de corrida esta bajo
`main()`— y se elige la regla de decision por CONDICION, que es donde esta el detalle que importa:

  `token` y `escala` -> `predecir`, con NOSE como un token mas del softmax;
  `cabeza`           -> `predecir_cabeza`, con la salida binaria separada.

Medir `token` con el criterio de `cabeza` (`a > 0`) daria un numero sin sentido: en esas dos
condiciones la cabeza nunca entro en la perdida y quedo en su valor inicial.
"""
import os, sys, json, pickle, argparse
import numpy as np
import jax, jax.numpy as jnp

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
import entrenar as E

SEM_PRUEBA = 77000


def pasa(nose, falsa):
    return bool(falsa <= 0.10 and nose >= 0.50)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+", help="rutas de checkpoints")
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--p-nose", type=float, default=0.4)
    ap.add_argument("--nivel", type=int, default=4)
    ap.add_argument("--salida", default=os.path.join(AQUI, "compuerta_20260820.json"))
    a = ap.parse_args()

    print(f"COMPUERTA · {a.n * a.batch} muestras · rng {SEM_PRUEBA}+semilla · p_nose {a.p_nose}")
    print(f"{'checkpoint':<26} {'cond':<7} {'paso':>6} {'falsa_abst':>11} {'nose':>7} "
          f"{'vigente':>8} {'pasa':>5}")
    print("-" * 78)

    res = {}
    for ck in a.ckpts:
        p = ck if os.path.isabs(ck) else os.path.join(AQUI, ck)
        if not os.path.exists(p):
            print(f"{os.path.basename(ck):<26} sin checkpoint"); continue
        with open(p, "rb") as f:
            d = pickle.load(f)
        cfg = d.get("config", {})
        cond = cfg.get("abst", "token")
        nivel = cfg.get("nivel", a.nivel)
        base = os.path.basename(p)
        # la semilla sale del nombre: <fam><nivel>_s<semilla>[.pkl[.pXXXXX]]
        sem = int(base.split("_s")[1].split(".")[0].split("_")[0])
        params = jax.tree_util.tree_map(jnp.asarray, d["params"])
        # 2026-08-25 · DOS arreglos, y el primero era el mas grave de los tres scripts auditados:
        #  · `_DONDE` NO se fijaba, asi que una unidad `lat`/`lat2` se media con la lectura `pre` y
        #    devolvia un numero invalido EN SILENCIO. `donde` existe desde el 22-ago, este script es
        #    anterior.
        #  · `slot` comparte el camino binario de `cabeza`, asi que el `==` lo dejaba afuera.
        E._DONDE = cfg.get("donde", "pre")
        E._ABST = cond
        fn = E.predecir_cabeza if cond in ("cabeza", "slot") else E.predecir
        rng = np.random.default_rng(SEM_PRUEBA + sem)
        m = E.evaluar(params, rng, n=a.n, B=a.batch, nivel=nivel, p_vieja=0.35,
                      p_nose=a.p_nose, pred_fn=fn)
        m["paso"], m["cond"], m["ckpt"] = d.get("paso"), cond, base
        m["pasa"] = pasa(m["nose"], m["falsa_abst"])
        res[base] = m
        print(f"{base:<26} {cond:<7} {str(m['paso']):>6} {m['falsa_abst']:>11.4f} "
              f"{m['nose']:>7.4f} {m['vigente']:>8.4f} {'SI' if m['pasa'] else 'no':>5}")

    json.dump({"instrumento": "entrenar.evaluar, regla por condicion", "rng": SEM_PRUEBA,
               "n": a.n * a.batch, "p_nose": a.p_nose, "unidades": res},
              open(a.salida, "w"), indent=1, default=float)
    print(f"\n-> {a.salida}")
    print("Compuerta: falsa_abst <= 0,10 y nose >= 0,50. El veredicto lo escribe una persona.")


if __name__ == "__main__":
    main()
