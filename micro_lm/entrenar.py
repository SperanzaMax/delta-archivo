"""MICRO-LM · entrenamiento. Corre igual en CPU (lento) y en GPU de Colab.

    python entrenar.py --nivel 1 --pasos 20000

Metrica: exactitud del UNICO token de respuesta, desagregada en vigente / anterior, mas la tasa de
abstencion (cuando el modelo elige NOSE). Nada de jueces ni parsers: la respuesta es un token y se
compara con el token esperado.

Lo que este script hereda de lo medido en el brazo interno, y esta escrito a proposito:
  · el archivo se inyecta TEMPRANO (bloque 0);
  · la clave archivada lleva sello de orden co-entrenado;
  · el balance de preguntas es un parametro (`--p-vieja`), porque E-I3d mostro que si casi todas las
    preguntas son por la version vigente, el modelo aprende el ATAJO de la recencia y nunca aprende a
    ordenar. Por defecto 0,35, no 0,05.
"""
import argparse
import json
import os
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jax
import jax.numpy as jnp
import optax

import datos as DAT
import idioma as I
import modelo as M

NOSE = I.STOI["NOSE"]


def evaluar(params, rng, n=8, B=64, nivel=4, p_vieja=0.35, p_nose=0.0):
    """Devuelve un dict de metricas. Las dos caras de la abstencion van SEPARADAS:

      `nose`         acierta NOSE cuando la respuesta no esta en el archivo (lo que se quiere);
      `falsa_abst`   dice NOSE cuando la respuesta SI estaba (el costo de conseguirlo).

    Un modelo que contesta NOSE a todo tendria `nose` = 1,000, y por eso la segunda no es opcional.
    """
    col = {k: [] for k in ("vigente", "anterior", "nose", "nose_ent", "nose_rel",
                           "falsa_abst", "abstencion")}
    for _ in range(n):
        ses, cortes, turnos, mask, cons, pos, tgt, tipo = DAT.lote(
            rng, B, nivel=nivel, n_hechos=4, n_sesiones=4, p_vieja=p_vieja, p_nose=p_nose)
        pred = np.array(predecir(params, jnp.array(ses), jnp.array(cortes), jnp.array(turnos),
                                 jnp.array(mask), jnp.array(cons), jnp.array(pos)))
        ok = pred == tgt
        sub = lambda m: ok[m].mean() if m.any() else np.nan
        col["vigente"].append(sub(tipo == 0))
        col["anterior"].append(sub(tipo == 1))
        col["nose"].append(sub(tipo >= 2))
        col["nose_ent"].append(sub(tipo == 2))
        col["nose_rel"].append(sub(tipo == 3))
        hay = tipo < 2
        col["falsa_abst"].append((pred[hay] == NOSE).mean() if hay.any() else np.nan)
        col["abstencion"].append((pred == NOSE).mean())
    return {k: float(np.nanmean(v)) for k, v in col.items()}


def logits_de(params, ses, cortes, turnos, mask, cons, pos):
    archivo = M.escribir(params, ses, cortes)
    lg = M.responder(params, archivo, turnos, cons, mask)
    return jnp.take_along_axis(lg, pos[:, None, None], axis=1)[:, 0, :]


@jax.jit
def predecir(params, ses, cortes, turnos, mask, cons, pos):
    return logits_de(params, ses, cortes, turnos, mask, cons, pos).argmax(-1)


def perdida(params, ses, cortes, turnos, mask, cons, pos, tgt):
    lg = logits_de(params, ses, cortes, turnos, mask, cons, pos)
    ce = optax.softmax_cross_entropy_with_integer_labels(lg, tgt).mean()
    return ce, (lg.argmax(-1) == tgt).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nivel", type=int, default=1)
    ap.add_argument("--pasos", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--d", type=int, default=192)
    ap.add_argument("--capas", type=int, default=6)
    ap.add_argument("--semilla", type=int, default=0)
    ap.add_argument("--p-vieja", type=float, default=0.35)
    ap.add_argument("--p-nose", type=float, default=0.0,
                    help="fraccion de preguntas SIN respuesta en el archivo (respuesta = NOSE)")
    ap.add_argument("--salida", default="resultados_micro.json")
    ap.add_argument("--pesos", default=None, help="ruta .pkl donde dejar los pesos finales")
    a = ap.parse_args()

    print(f"MICRO-LM · nivel {a.nivel} · vocabulario {I.V} tokens · d={a.d} capas={a.capas}",
          flush=True)
    print(f"dispositivos: {jax.devices()}", flush=True)

    params = M.init_params(a.semilla, I.V, D=a.d, NB=a.capas, N_TURNOS=64)
    n = M.contar(params)
    print(f"parametros: {n:,} ({n * 4 / 1e6:.1f} MB en fp32)\n", flush=True)

    warmup = min(500, max(1, a.pasos // 10))
    sched = optax.warmup_cosine_decay_schedule(0.0, a.lr, warmup, a.pasos, a.lr * 0.1)
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(sched, weight_decay=0.01))
    state = opt.init(params)

    @jax.jit
    def paso(params, state, ses, cortes, turnos, mask, cons, pos, tgt):
        (l, acc), g = jax.value_and_grad(perdida, has_aux=True)(
            params, ses, cortes, turnos, mask, cons, pos, tgt)
        up, state = opt.update(g, state, params)
        return optax.apply_updates(params, up), state, l, acc

    rng = np.random.default_rng(1000 + a.semilla)
    DAT.reset_truncados()
    hist, t0 = [], time.time()
    for s in range(1, a.pasos + 1):
        ses, cortes, turnos, mask, cons, pos, tgt, _ = DAT.lote(
            rng, a.batch, nivel=a.nivel, n_hechos=4, n_sesiones=4, p_vieja=a.p_vieja,
            p_nose=a.p_nose)
        params, state, l, acc = paso(params, state, jnp.array(ses), jnp.array(cortes),
                                     jnp.array(turnos), jnp.array(mask), jnp.array(cons),
                                     jnp.array(pos), jnp.array(tgt))
        if s % 500 == 0:
            print(f"  paso {s:6d}  loss {float(l):.4f}  acc {float(acc):.4f}  "
                  f"({time.time()-t0:.0f}s)", flush=True)
        if s % 2000 == 0 or s == a.pasos:
            trunc = DAT.tasa_truncados()            # la compuerta, en el registro permanente
            ev = np.random.default_rng(90000 + a.semilla)
            m = evaluar(params, ev, nivel=a.nivel, p_vieja=a.p_vieja, p_nose=a.p_nose)
            hist.append({"paso": s, "truncados": float(trunc), **m})
            extra = ("" if a.p_nose == 0 else
                     f" · nose {m['nose']:.4f} (ent {m['nose_ent']:.4f}/rel {m['nose_rel']:.4f})"
                     f" · falsa_abst {m['falsa_abst']:.4f}")
            print(f"    ── eval: vigente {m['vigente']:.4f} · anterior {m['anterior']:.4f}"
                  f" · abstencion {m['abstencion']:.4f} · truncados {trunc:.4f}{extra}", flush=True)
            json.dump({"config": vars(a), "params": n, "historia": hist},
                      open(a.salida, "w"), indent=1)
            # Lo del 13-ago no se repite: si se esta truncando, se corta ACA y no despues de 73 min.
            if trunc > 0.01:
                sys.exit(f"ABORTA: truncamiento {trunc:.4f} > 0,01 — se estaria midiendo el padding")

    if a.pesos:
        with open(a.pesos, "wb") as f:
            pickle.dump({"params": jax.device_get(params), "config": vars(a),
                         "vocab": I.ITOS}, f)
        print(f"pesos guardados en {a.pesos} ({os.path.getsize(a.pesos)/1e6:.1f} MB)", flush=True)
    print("\nlisto:", hist[-1] if hist else "sin evaluaciones")


if __name__ == "__main__":
    main()
