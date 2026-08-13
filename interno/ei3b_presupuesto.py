"""E-I3b · CHEQUEO DE PRESUPUESTO antes de firmar el negativo.

La corrida principal de E-I3b deja la pregunta por la version ANTERIOR muy por debajo de P-1 (0,80)
mientras la VIGENTE queda en 0,95-0,99. La lectura natural es que el lector no usa el orden como
orden. Pero este programa ya se equivoco tres veces de la misma manera:

  E-I1  la celda "que no aprendia" daba 0,0167 a 1500 pasos y 0,418/0,528 a 6000: era impaciencia.
  E-I2  la corrida principal dio el azar en las tres predicciones por un lr heredado de otra tarea;
        la ventana resulto ser 1e-3 -> 0,7305 · 3e-3 -> 0,0265 · 1e-2 -> 0,0130.
  E-I3  (banco ECO, otro brazo) un control de sanidad vacio hizo pasar por dificultad lo que era
        incapacidad del sujeto.

De ahi la regla: UN NEGATIVO SIN BARRIDO DE PRESUPUESTO NO ES UN NEGATIVO. E-I3b hereda lr = 1e-3 y
4000 pasos de E-I3, y su tarea NO es la misma: el archivo pasa de 9 a 12 entradas y aparecen dos
tipos de consulta que el modelo tiene que separar. Antes de escribir "no usa el orden" hay que darle
el presupuesto.

QUE SE CORRE. Solo la celda que decide -- `sello`, la accuracy en ANTERIOR -- con 12000 pasos (3x) y
dos tasas: la heredada 1e-3 y una mas conservadora 3e-4, que es lo que suele hacer falta cuando la
tarea agrega un factor. Se registra la curva de ANTERIOR cada 1000 pasos, que es lo que distingue
"plano" de "sube despacio": un negativo con la curva plana en 12000 pasos es un negativo; una curva
que todavia sube al final significa que hay que seguir dandole.

CRITERIO, fijado antes de mirar:
  - si algun (lr, paso) alcanza ANTERIOR >= 0,80  -> P-1 CUMPLE y el negativo se cae.
  - si el mejor queda < 0,80 pero la curva SUBE en el ultimo tramo (ultimo tercio > tercio previo
    por >= 0,05) -> presupuesto insuficiente, no se firma nada y se vuelve a correr mas largo.
  - si el mejor queda < 0,80 con la curva plana -> NEGATIVO FIRMADO: el lector no usa el orden como
    orden, y el informe de E-I3 se corrige, no se matiza.
"""
import json
import os
import sys
import time
from functools import partial

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jax
import jax.numpy as jnp
import optax
import ei2_consulta as E
import ei3b_orden_vs_recencia as B
from modelos import init_params

PASOS = int(os.environ.get("PASOS", "12000"))
TASAS = (1e-3, 3e-4)
SEMILLAS = (0, 1)
TRAZA = 1000


def entrenar(lr, semilla, pasos=PASOS):
    params = init_params(semilla, E.KIND)
    params["extra"] = B.init_extra(semilla)
    rng = np.random.default_rng(5000 + semilla)
    sched = optax.warmup_constant_schedule(0.0, lr, 100)
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(sched, weight_decay=0.01))
    state = opt.init(params)

    @partial(jax.jit, static_argnames="m")
    def paso(params, state, lote, m):
        (l, aux), g = jax.value_and_grad(B.loss_fn, has_aux=True)(params, lote, m)
        up, state = opt.update(g, state, params)
        return optax.apply_updates(params, up), state, l, aux

    curva, t0 = [], time.time()
    for s in range(1, pasos + 1):
        params, state, l, aux = paso(params, state, B.gen_lote(rng), "sello")
        if s % TRAZA == 0:
            a, av, an, au = (float(v) for v in aux)
            curva.append({"paso": s, "acc": a, "vigente": av, "anterior": an})
            print(f"    [lr {lr:g}/s{semilla}] paso {s:5d} vig {av:.4f} · ANT {an:.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    ev = np.random.default_rng(99000 + semilla)
    res = np.mean([[float(v) for v in B.loss_fn(params, B.gen_lote(ev), "sello")[1]]
                   for _ in range(8)], axis=0)
    return res, curva


def main():
    print(f"E-I3b · chequeo de presupuesto · {PASOS} pasos · tasas {TASAS} · "
          f"semillas {SEMILLAS}\nreferencia: la corrida principal dio ANTERIOR ~0,12 con "
          f"lr 1e-3 y 4000 pasos\n", flush=True)
    salida = {}
    for lr in TASAS:
        finales, curvas = [], []
        for s in SEMILLAS:
            res, curva = entrenar(lr, s)
            finales.append(res)
            curvas.append(curva)
            print(f"  lr {lr:g} s{s} → vigente {res[1]:.4f} · ANTERIOR {res[2]:.4f}\n", flush=True)
            salida[f"{lr:g}"] = {"finales": np.array(finales).tolist(), "curvas": curvas}
            json.dump(salida, open("resultados_ei3b_presupuesto.json", "w"), indent=1)

    print("=" * 74)
    mejor, mejor_cfg = 0.0, None
    for lr in TASAS:
        a = np.array(salida[f"{lr:g}"]["finales"])
        print(f"  lr {lr:g} → vigente {a[:,1].mean():.4f} · ANTERIOR {a[:,2].mean():.4f}")
        if a[:, 2].mean() > mejor:
            mejor, mejor_cfg = a[:, 2].mean(), lr

    # ¿la curva todavia sube al final?
    sube = False
    for lr in TASAS:
        for curva in salida[f"{lr:g}"]["curvas"]:
            ant = [p["anterior"] for p in curva]
            t = len(ant) // 3
            if t and np.mean(ant[-t:]) - np.mean(ant[-2*t:-t]) >= 0.05:
                sube = True
    print("-" * 74)
    if mejor >= 0.80:
        print(f"  P-1 CUMPLE con presupuesto: ANTERIOR {mejor:.4f} con lr {mejor_cfg:g}. "
              f"El negativo se cae y E-I3b se re-corre entero con esa tasa.")
    elif sube:
        print(f"  NO SE FIRMA: mejor {mejor:.4f} pero la curva todavia sube en el ultimo tercio. "
              f"Presupuesto insuficiente, hay que correr mas largo.")
    else:
        print(f"  NEGATIVO FIRMADO: mejor {mejor:.4f} < 0,80 con la curva plana. El lector NO usa "
              f"el orden como orden; corregir el informe de E-I3.")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
