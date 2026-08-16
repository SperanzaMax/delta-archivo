"""MICRO-LM · ¿el hecho no esta archivado, o esta y le gana otro?

    python foco_lectura.py ckpts/n4_s0.pkl --n 4000

Las dos sondas dejaron el hecho propio INALCANZABLE en todas sus versiones, y desde afuera «no se
escribio» y «esta pero pierde» se ven igual. Se separan mirando la FORMA de la distribucion de
lectura —softmax(sim) sobre las entradas del archivo— en la posicion desde la que se responde:

  · si el hecho NO esta          -> nada matchea fuerte -> distribucion DISPERSA (entropia alta,
                                    masa del top-1 baja);
  · si esta y le gana el vecino  -> hay un ganador claro, solo que el equivocado -> distribucion
                                    CONCENTRADA, igual que en los aciertos.

Predicciones (comprometidas antes de correr, 2026-08-16):
  F-1  entropia(err_identidad) > entropia(acierto)          -> compatible con «no esta»
  F-2  masa top-1(err_identidad) < masa top-1(acierto)      -> lo mismo, por la otra cara
  F-3  si F-1 y F-2 NO se cumplen, el hecho SI esta archivado y el problema es de competencia de
       claves: el direccionamiento elige mal, no la escritura omite.
"""
import argparse
import pickle

import numpy as np
import jax
import jax.numpy as jnp

import datos as DAT
import idioma as I
import entrenar as E
from ser import clasificar
from score_archivo import scores_archivo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pesos")
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--B", type=int, default=64)
    ap.add_argument("--semilla", type=int, default=31415)
    a = ap.parse_args()

    with open(a.pesos, "rb") as f:
        bulto = pickle.load(f)
    params = jax.tree_util.tree_map(jnp.asarray, bulto["params"])
    nivel = bulto["config"]["nivel"]

    rng = np.random.default_rng(a.semilla)
    col = {g: {"ent": [], "top1": [], "n_val": []} for g in ("err_identidad", "acierto")}
    vistos = 0

    while vistos < a.n:
        B = min(a.B, a.n - vistos)
        ses, cortes, turnos, mask, cons, pos, tgt, tipo, meta = DAT.lote(
            rng, B, nivel=nivel, n_hechos=4, n_sesiones=4, p_nose=0.0, con_meta=True)
        jses, jcor, jtur = jnp.array(ses), jnp.array(cortes), jnp.array(turnos)
        jmask, jcons, jpos = jnp.array(mask), jnp.array(cons), jnp.array(pos)

        pred = np.asarray(E.predecir(params, jses, jcor, jtur, jmask, jcons, jpos))
        s = np.asarray(scores_archivo(params, jses, jcor, jtur, jmask, jcons, jpos))
        p = np.asarray(jax.nn.softmax(jnp.array(s), -1))

        for i in range(B):
            cat = clasificar(I.ITOS[int(pred[i])], I.ITOS[int(tgt[i])], meta[i])
            if cat not in col:
                continue
            pi = p[i]
            col[cat]["ent"].append(float(-(pi * np.log(pi + 1e-12)).sum()))
            col[cat]["top1"].append(float(pi.max()))
            col[cat]["n_val"].append(int(mask[i].sum()))
        vistos += B

    print(f"pesos: {a.pesos}")
    print(f"nivel {nivel} · paso {bulto.get('paso','?')} · n={vistos}\n")
    res = {}
    for g in ("err_identidad", "acierto"):
        e = np.array(col[g]["ent"]); t = np.array(col[g]["top1"])
        nv = np.array(col[g]["n_val"])
        res[g] = (e.mean(), t.mean())
        print(f"{g}  (n={len(e)})")
        print(f"   entropia de la lectura   {e.mean():.4f} ± {e.std():.4f}   "
              f"(maxima posible ln({nv.mean():.0f}) = {np.log(nv.mean()):.4f})")
        print(f"   masa del top-1           {t.mean():.4f} ± {t.std():.4f}\n")

    f1 = res["err_identidad"][0] > res["acierto"][0]
    f2 = res["err_identidad"][1] < res["acierto"][1]
    print("PREDICCIONES")
    print(f"  F-1  entropia err > acierto     {res['err_identidad'][0]:.4f} vs {res['acierto'][0]:.4f}   "
          f"{'CUMPLE' if f1 else 'NO CUMPLE'}")
    print(f"  F-2  top1 err < acierto         {res['err_identidad'][1]:.4f} vs {res['acierto'][1]:.4f}   "
          f"{'CUMPLE' if f2 else 'NO CUMPLE'}")
    print(f"  F-3  ninguna cumple -> el hecho ESTA y pierde por competencia   "
          f"{'ACTIVA' if not f1 and not f2 else 'no activa'}")


if __name__ == "__main__":
    main()
