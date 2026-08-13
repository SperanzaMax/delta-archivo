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


def evaluar(params, rng, n=8, B=64, nivel=4, p_vieja=0.35):
    acc_v, acc_a, abst = [], [], []
    for _ in range(n):
        ses, cortes, turnos, mask, cons, pos, tgt, tipo = DAT.lote(
            rng, B, nivel=nivel, n_hechos=4, n_sesiones=4)
        pred = predecir(params, jnp.array(ses), jnp.array(cortes), jnp.array(turnos),
                        jnp.array(mask), jnp.array(cons), jnp.array(pos))
        pred = np.array(pred)
        ok = pred == tgt
        acc_v.append(ok[tipo == 0].mean() if (tipo == 0).any() else np.nan)
        acc_a.append(ok[tipo == 1].mean() if (tipo == 1).any() else np.nan)
        abst.append((pred == NOSE).mean())
    return np.nanmean(acc_v), np.nanmean(acc_a), np.mean(abst)


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
    ap.add_argument("--salida", default="resultados_micro.json")
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
    hist, t0 = [], time.time()
    for s in range(1, a.pasos + 1):
        ses, cortes, turnos, mask, cons, pos, tgt, _ = DAT.lote(
            rng, a.batch, nivel=a.nivel, n_hechos=4, n_sesiones=4)
        params, state, l, acc = paso(params, state, jnp.array(ses), jnp.array(cortes),
                                     jnp.array(turnos), jnp.array(mask), jnp.array(cons),
                                     jnp.array(pos), jnp.array(tgt))
        if s % 500 == 0:
            print(f"  paso {s:6d}  loss {float(l):.4f}  acc {float(acc):.4f}  "
                  f"({time.time()-t0:.0f}s)", flush=True)
        if s % 2000 == 0 or s == a.pasos:
            ev = np.random.default_rng(90000 + a.semilla)
            av, aa, ab = evaluar(params, ev, nivel=a.nivel)
            hist.append({"paso": s, "vigente": float(av), "anterior": float(aa),
                         "abstencion": float(ab)})
            print(f"    ── eval: vigente {av:.4f} · anterior {aa:.4f} · abstencion {ab:.4f}",
                  flush=True)
            json.dump({"config": vars(a), "params": n, "historia": hist},
                      open(a.salida, "w"), indent=1)
    print("\nlisto:", hist[-1] if hist else "sin evaluaciones")


if __name__ == "__main__":
    main()
