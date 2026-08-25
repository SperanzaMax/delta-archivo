#!/usr/bin/env python3
"""¿Por que el slot nulo nunca se abstiene? — diagnostico POST-HOC de la campania `y3_*`.

NO es una prediccion de `PREREG_SLOT_NULO.md`. El prereg ya quedo evaluado con `ser.py` (S-1 falla,
`nose = 0,0000` en las tres semillas) y este script no puede cambiar ese veredicto. Existe para
separar dos lecturas de ese cero, que la metrica de salida no distingue:

  (a) EL SLOT NO APRENDIO NADA  -> la masa del nulo es plana y no separa «hay respuesta» de «no hay»
      -> AUC ~ 0,50. El mecanismo no capta pertenencia y S-2 falla por la misma razon.
  (b) EL SLOT APRENDIO Y NO ALCANZA -> la masa separa (AUC alto) pero nunca cruza el 0,5 que la
      regla de decision exige, porque compite contra 40 entradas reales en un softmax.

La diferencia importa y no es cosmetica. En (a) el slot es un mecanismo muerto; en (b) la señal
existe y lo que falla es el UMBRAL heredado de `cabeza`, donde el logit es libre y aca esta acotado
por la geometria del softmax. Se reporta como diagnostico, no como rescate: la regla de decision fue
la MISMA en entrenamiento (`perdida_cabeza`, BCE sobre este logit) y en evaluacion, asi que el
gradiente tuvo 26000 pasos para empujar la masa arriba de 0,5 y no lo logro. Eso es un resultado
sobre el mecanismo; este script solo dice de que forma.

    python diag_slot.py --unidades y3_s0,y3_s1,y3_s2 --n 2048
"""
import argparse
import json
import os
import pickle

import numpy as np
import jax
import jax.numpy as jnp

import datos as DAT
import entrenar as E

AQUI = os.path.dirname(os.path.abspath(__file__))
TIPOS = {0: "vigente", 1: "anterior", 2: "nose_ent", 3: "nose_rel"}


def auc(pos, neg):
    """AUC de Mann-Whitney: P(un positivo puntue mas alto que un negativo). Empates cuentan 0,5."""
    if not len(pos) or not len(neg):
        return float("nan")
    todo = np.concatenate([pos, neg])
    r = np.argsort(np.argsort(todo)) + 1.0
    # promedio de rangos en los empates
    orden = np.argsort(todo)
    vals = todo[orden]
    i = 0
    while i < len(vals):
        j = i
        while j + 1 < len(vals) and vals[j + 1] == vals[i]:
            j += 1
        if j > i:
            r[orden[i:j + 1]] = np.mean(r[orden[i:j + 1]])
        i = j + 1
    n1, n2 = len(pos), len(neg)
    return (r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--unidades", default="y3_s0,y3_s1,y3_s2")
    ap.add_argument("--dir-ckpt", default=os.path.join(AQUI, "ckpts"))
    ap.add_argument("--n", type=int, default=2048)
    ap.add_argument("--B", type=int, default=64)
    ap.add_argument("--p-nose", type=float, default=0.4)
    ap.add_argument("--semilla", type=int, default=54321)
    ap.add_argument("--salida", default=os.path.join(AQUI, "reanalisis_20260825", "diag_slot.json"))
    a = ap.parse_args()

    res = {}
    for u in a.unidades.split(","):
        with open(os.path.join(a.dir_ckpt, f"{u}.pkl"), "rb") as f:
            bulto = pickle.load(f)
        params, cfg = bulto["params"], bulto["config"]
        # La arquitectura y la regla de decision salen del checkpoint, nunca de flags.
        E._DONDE = cfg.get("donde", "pre")
        E._ABST = cfg.get("abst", "token")
        nivel = cfg["nivel"]

        rng = np.random.default_rng(a.semilla)
        masas, tipos, vistos = [], [], 0
        while vistos < a.n:
            B = min(a.B, a.n - vistos)
            ses, cortes, turnos, mask, cons, pos, tgt, tipo = DAT.lote(
                rng, B, nivel=nivel, n_hechos=4, n_sesiones=4, p_nose=a.p_nose)
            # Con `--abst slot`, `_partes` devuelve el logit binario = log(m/(1-m)) de la masa del
            # nulo (`modelo.py:303-307`), asi que la masa se recupera exacta con la sigmoide y no
            # hace falta reimplementar la lectura.
            _, lg_a = E._partes(params, jnp.array(ses), jnp.array(cortes), jnp.array(turnos),
                                jnp.array(mask), jnp.array(cons), jnp.array(pos))
            masas.extend(np.asarray(jax.nn.sigmoid(lg_a)).tolist())
            tipos.extend(np.asarray(tipo).tolist())
            vistos += B

        m = np.array(masas)
        t = np.array(tipos)
        con = t <= 1                     # la respuesta ESTA en el archivo
        sin = t >= 2                     # no esta: el slot deberia ganar
        d = {
            "abst": cfg.get("abst"), "donde": cfg.get("donde"), "paso": cfg.get("paso"),
            "masa_media_con_respuesta": float(m[con].mean()),
            "masa_media_sin_respuesta": float(m[sin].mean()),
            "masa_max_global": float(m.max()),
            "masa_p99_sin_respuesta": float(np.percentile(m[sin], 99)),
            "frac_masa_mayor_0.5": float((m > 0.5).mean()),
            "AUC_masa_sin_vs_con": float(auc(m[sin], m[con])),
            "por_tipo": {TIPOS[k]: float(m[t == k].mean()) for k in sorted(TIPOS) if (t == k).any()},
        }
        res[u] = d
        print(f"\n== {u} ({d['donde']} · {d['abst']}) ==")
        print(f"   masa del slot   con respuesta {d['masa_media_con_respuesta']:.4f}   "
              f"sin respuesta {d['masa_media_sin_respuesta']:.4f}")
        print(f"   por tipo        " + "  ".join(f"{k} {v:.4f}" for k, v in d["por_tipo"].items()))
        print(f"   maxima masa     {d['masa_max_global']:.4f}   "
              f"p99 sin respuesta {d['masa_p99_sin_respuesta']:.4f}")
        print(f"   frac > 0,5      {d['frac_masa_mayor_0.5']:.4f}   <- lo que la regla exige")
        print(f"   AUC (sin vs con) {d['AUC_masa_sin_vs_con']:.4f}   <- (a) ~0,50 muerto · (b) alto = umbral")

    os.makedirs(os.path.dirname(a.salida), exist_ok=True)
    with open(a.salida, "w") as f:
        json.dump({"post_hoc": True, "prereg": None, "unidades": res}, f, indent=1)
    print(f"\n-> {a.salida}")


if __name__ == "__main__":
    main()
