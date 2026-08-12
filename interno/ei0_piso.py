"""E-I0 — EL PISO. Etapa bloqueante del brazo interno (PROTOCOLO_INTERNO.md §4).

Dos mitades, y hacen falta LAS DOS:

  CROSS (piso)     el modelo ve SOLO S3: las claves consultadas, sin ningun valor. La informacion
                   se escribio en S1 y se reviso en S2, dos forwards independientes que no dejan
                   estado. Debe quedar en el AZAR (1/NV = 0,0156). Si no queda, hay fuga y toda
                   medicion posterior seria ilegible.

  INTRA (control)  la misma informacion concatenada en una secuencia. DEBE resolverse. Si no se
                   resuelve, no se puede leer nada del fracaso de la version cross: seria
                   incapacidad del modelo y no ausencia de memoria.

El control es el que puede FALLAR, y por eso existe. Es la leccion que costo el veredicto del
11-ago: se valido un chequeo con una celda (m=1) donde acertar no requeria leer nada, y quedo sin
poder distinguir "tarea dificil" de "sujeto incapaz".
"""
import json
import os
import sys
import time
from functools import partial

import numpy as np

sys.path.insert(0, "/home/maxi/Documentos/Nuevo Transformer/telar-ligamento/src")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jax
import jax.numpy as jnp
import optax
from modelos import forward, init_params
from tarea_cross import gen_cross, gen_intra
from datos import V_E001

L = int(os.environ.get("L_CARGA", "6"))
R = L // 2
B = int(os.environ.get("BATCH", "32"))
PASOS = int(os.environ.get("PASOS", "600"))
LR = 3e-3
AZAR = 1.0 / V_E001.NV


def loss_fn(params, x, y, kind):
    logits = forward(params, x, kind)
    mask = y >= 0
    yl = jnp.where(mask, y, 0)
    ce = optax.softmax_cross_entropy_with_integer_labels(logits, yl)
    return (ce * mask).sum() / mask.sum(), ((logits.argmax(-1) == yl) * mask).sum() / mask.sum()


def entrenar(kind, modo, semilla, pasos=PASOS):
    """modo: 'cross' (el modelo ve solo S3) | 'intra' (todo concatenado)."""
    rng = np.random.default_rng(1000 + semilla)
    params = init_params(semilla, kind)
    sched = optax.warmup_constant_schedule(0.0, LR, 100)
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(sched, weight_decay=0.01))
    state = opt.init(params)

    @partial(jax.jit, static_argnames="kind")
    def paso(params, state, x, y, kind):
        (l, a), g = jax.value_and_grad(loss_fn, has_aux=True)(params, x, y, kind)
        up, state = opt.update(g, state, params)
        return optax.apply_updates(params, up), state, l, a

    t0 = time.time()
    for s in range(1, pasos + 1):
        if modo == "cross":
            _, _, x, y = gen_cross(rng, B, L, R)
        else:
            x, y = gen_intra(rng, B, L, R)
        params, state, l, a = paso(params, state, jnp.array(x), jnp.array(y), kind)
        if s % 200 == 0:
            print(f"    [{kind}/{modo}/s{semilla}] paso {s:4d} loss {float(l):.4f} "
                  f"acc {float(a):.4f} ({time.time()-t0:.0f}s)", flush=True)

    # evaluacion con semilla distinta de la de entrenamiento
    ev = np.random.default_rng(77000 + semilla)
    accs = []
    for _ in range(8):
        if modo == "cross":
            _, _, x, y = gen_cross(ev, B, L, R)
        else:
            x, y = gen_intra(ev, B, L, R)
        _, a = loss_fn(params, jnp.array(x), jnp.array(y), kind)
        accs.append(float(a))
    return float(np.mean(accs)), float(np.std(accs))


def main():
    semillas = (0, 1, 2)
    print(f"E-I0 · L={L} r={R} B={B} · {PASOS} pasos · azar = {AZAR:.4f}\n", flush=True)
    salida = {}
    for modo in ("intra", "cross"):
        for kind in ("delta", "softmax"):
            res = [entrenar(kind, modo, s) for s in semillas]
            medias = [m for m, _ in res]
            salida[f"{modo}_{kind}"] = {"media": float(np.mean(medias)),
                                        "sd": float(np.std(medias, ddof=1)),
                                        "por_semilla": medias}
            print(f"  {modo:5s} {kind:8s} → acc {np.mean(medias):.4f} "
                  f"(sd {np.std(medias, ddof=1):.4f}) · por semilla "
                  f"{[round(m, 4) for m in medias]}\n", flush=True)
            json.dump(salida, open("resultados_ei0.json", "w"), indent=1)

    print("=" * 70)
    piso = max(salida["cross_delta"]["media"], salida["cross_softmax"]["media"])
    ctrl = max(salida["intra_delta"]["media"], salida["intra_softmax"]["media"])
    print(f"  piso (cross, el mejor de los dos):    {piso:.4f}   · azar {AZAR:.4f}")
    print(f"  control (intra, el mejor de los dos): {ctrl:.4f}")
    ok_piso = piso < 0.10
    ok_ctrl = ctrl > 0.60
    print(f"  piso en el azar          {'OK' if ok_piso else 'FALLA — hay fuga'} (exigido < 0,10)")
    print(f"  control resuelve         {'OK' if ok_ctrl else 'FALLA — el modelo no puede'}"
          f" (exigido > 0,60)")
    if ok_piso and ok_ctrl:
        print("\n  ►► E-I0 PASA. La tarea mide memoria persistente y nada mas. Sigue E-I1.")
    else:
        print("\n  ►► E-I0 NO PASA. No se avanza: cualquier resultado de E-I1 seria ilegible.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
