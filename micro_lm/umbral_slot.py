#!/usr/bin/env python3
"""¿El `nose = 0,0000` del slot es del mecanismo o del UMBRAL heredado de `cabeza`?

POST-HOC. No es una predicción de `PREREG_SLOT_NULO.md` y NO puede cambiar su veredicto: S-1 está
evaluado con la regla de decisión del checkpoint, que es la misma que usó el entrenamiento. Esto
existe porque el §3 del `DISENO_ATRIBUCION.md` dice, textual y por adelantado:

    «"Que gane el slot nulo" NO puede ser el criterio de abstención... El nulo tiene que competir
     por masa relativa, no por victoria.»

Y la implementación hace exactamente lo que ese párrafo prohíbe: el logit binario es
`log(m/(1-m))` (`modelo.py:306`) y la decisión es `a > 0` (`entrenar.py:123`), o sea **masa > 0,5**.
Con 41 entradas eso exige que el nulo se lleve más atención que las otras cuarenta JUNTAS.

Este script barre el umbral sobre la masa y reporta la compuerta S-1 (`nose` ≥ 0,50 y `falsa_abst`
≤ 0,10) en cada punto. Dos lecturas posibles, declaradas antes de correr:

  (a) NINGÚN umbral pasa la compuerta -> el negativo es del mecanismo. El trípode cierra limpio y
      la restricción del softmax es una nota al pie, no una objeción.
  (b) ALGÚN umbral la pasa -> lo que falló fue la regla de decisión importada de `cabeza`, y eso
      hay que decirlo antes de publicar el trípode, porque entonces `slot` y `cabeza` no compitieron
      en igualdad de condiciones.

Se reporta también el umbral óptimo por Youden (J = TPR - FPR) como referencia, y el AUC, que NO
depende del umbral y por eso es el que decide si hay señal.
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


def auc(pos, neg):
    if not len(pos) or not len(neg):
        return float("nan")
    todo = np.concatenate([pos, neg])
    orden = np.argsort(todo)
    r = np.empty(len(todo), float)
    r[orden] = np.arange(1, len(todo) + 1)
    vals = todo[orden]
    i = 0
    while i < len(vals):
        j = i
        while j + 1 < len(vals) and vals[j + 1] == vals[i]:
            j += 1
        if j > i:
            r[orden[i:j + 1]] = r[orden[i:j + 1]].mean()
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
    ap.add_argument("--salida", default=os.path.join(AQUI, "reanalisis_20260825", "umbral_slot.json"))
    a = ap.parse_args()

    res = {}
    for u in a.unidades.split(","):
        with open(os.path.join(a.dir_ckpt, f"{u}.pkl"), "rb") as f:
            bulto = pickle.load(f)
        params, cfg = bulto["params"], bulto["config"]
        E._DONDE = cfg.get("donde", "pre")
        E._ABST = cfg.get("abst", "token")

        rng = np.random.default_rng(a.semilla)
        masas, tipos, vistos = [], [], 0
        while vistos < a.n:
            B = min(a.B, a.n - vistos)
            ses, cortes, turnos, mask, cons, pos, tgt, tipo = DAT.lote(
                rng, B, nivel=cfg["nivel"], n_hechos=4, n_sesiones=4, p_nose=a.p_nose)
            _, lg_a = E._partes(params, jnp.array(ses), jnp.array(cortes), jnp.array(turnos),
                                jnp.array(mask), jnp.array(cons), jnp.array(pos))
            masas.extend(np.asarray(jax.nn.sigmoid(lg_a)).tolist())
            tipos.extend(np.asarray(tipo).tolist())
            vistos += B

        m, t = np.array(masas), np.array(tipos)
        sin, con = m[t >= 2], m[t <= 1]          # sin respuesta / con respuesta
        A = auc(sin, con)

        # Barrido sobre los umbrales que el propio dato ofrece.
        cands = np.unique(np.quantile(m, np.linspace(0.001, 0.999, 400)))
        filas, mejor_j, mejor = [], -9, None
        pasa_alguno = None
        for th in cands:
            nose = float((sin > th).mean())            # abstiene cuando no hay respuesta: acierto
            fa = float((con > th).mean())              # abstiene habiendo respuesta: costo
            j = nose - fa
            filas.append((float(th), nose, fa))
            if j > mejor_j:
                mejor_j, mejor = j, (float(th), nose, fa)
            if nose >= 0.50 and fa <= 0.10 and pasa_alguno is None:
                pasa_alguno = (float(th), nose, fa)

        res[u] = {"AUC_masa": float(A), "umbral_heredado_0.5": {
                      "nose": float((sin > 0.5).mean()), "falsa_abst": float((con > 0.5).mean())},
                  "mejor_youden": {"umbral": mejor[0], "nose": mejor[1], "falsa_abst": mejor[2],
                                   "J": float(mejor_j)},
                  "pasa_compuerta_S1": pasa_alguno}

        print(f"\n== {u} ==")
        print(f"   AUC de la masa (no depende del umbral) : {A:.4f}")
        print(f"   umbral 0,5 heredado de `cabeza`        : nose {(sin>0.5).mean():.4f} · "
              f"falsa_abst {(con>0.5).mean():.4f}")
        print(f"   mejor umbral (Youden J={mejor_j:+.4f})   : th={mejor[0]:.4f} · "
              f"nose {mejor[1]:.4f} · falsa_abst {mejor[2]:.4f}")
        if pasa_alguno:
            print(f"   ⚠ HAY umbral que pasa S-1            : th={pasa_alguno[0]:.4f} · "
                  f"nose {pasa_alguno[1]:.4f} · falsa_abst {pasa_alguno[2]:.4f}  -> lectura (b)")
        else:
            print(f"   NINGÚN umbral pasa S-1 (nose>=0,50 y falsa_abst<=0,10) -> lectura (a)")

    os.makedirs(os.path.dirname(a.salida), exist_ok=True)
    with open(a.salida, "w") as f:
        json.dump({"post_hoc": True, "no_cambia_veredicto_S1": True, "unidades": res}, f, indent=1)
    print(f"\n-> {a.salida}")


if __name__ == "__main__":
    main()
