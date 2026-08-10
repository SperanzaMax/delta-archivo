"""¿Cuanto deriva REALMENTE el espacio de claves durante el entrenamiento?

La curva de tolerancia (gemacion_deriva2.py) dice que la memoria persistente aguanta mientras
cos(marco de hoy, marco de escritura) >= ~0.7. Falta el otro lado de la desigualdad: en que
regimen cae la deriva real. Sin ese numero la prueba de Basu queda a medias.

Medicion: se entrena delta puro en el harness de Ligamento y, sobre un batch FIJO de sondeo, se
comparan las claves k = l2norm(silu(x @ W_k)) del paso t contra las del paso 0 y contra las del
checkpoint anterior. cos=1 significa que el espacio no se movio.

Corrida deliberadamente chica: pocos pasos, batch chico, 2 hilos. No hace falta converger para
medir cuanto se mueven las coordenadas.
"""
import os, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")
sys.path.insert(0, os.path.expanduser("~/Documentos/Nuevo Transformer/telar-ligamento/src"))

import numpy as np, jax, jax.numpy as jnp, optax
import modelos as M
from modelos import split_heads, l2n, ln, conv3
from datos import gen_mqar
from entrenar import loss_fn

PASOS, BATCH, CARGA, LR, CADA = 400, 16, 8, 3e-3, 25


def claves(params, x):
    """Claves de la capa 0, tal como las ve la regla delta."""
    blk = params["blocks"][0]
    hx = params["emb"][x]
    xin = conv3(blk["conv"], ln(blk["ln1"], hx))
    return l2n(jax.nn.silu(split_heads(xin @ blk["k"])))     # (B,H,T,DH)


def main():
    rng = np.random.default_rng(0)
    xs, ys = gen_mqar(rng, BATCH, CARGA)
    xs, ys = jnp.asarray(xs), jnp.asarray(ys)
    xp, _ = gen_mqar(np.random.default_rng(999), 8, CARGA)    # batch de sondeo, fijo
    xp = jnp.asarray(xp)

    params = M.init_params(0, "delta")
    opt = optax.adam(LR); st = opt.init(params)
    gv = jax.jit(jax.value_and_grad(loss_fn, has_aux=True), static_argnums=3)

    k0 = claves(params, xp); kprev = k0
    print(f"deriva del espacio de claves — delta puro, {PASOS} pasos, batch {BATCH}, carga {CARGA}")
    print(f"{'paso':>6} {'loss':>7} {'acc':>6} {'cos vs paso 0':>14} "
          f"{'cos vs anterior':>16} {'theta_equiv':>12}")
    t0 = time.time()
    for s in range(PASOS + 1):
        if s % CADA == 0:
            k = claves(params, xp)
            c0 = float(jnp.mean(jnp.sum(k * k0, -1)))
            cp = float(jnp.mean(jnp.sum(k * kprev, -1)))
            kprev = k
            (l, a), _ = gv(params, xs, ys, "delta")
            # theta equivalente: rotacion por paso que produciria este cos acumulado
            th = float(np.arccos(np.clip(c0, -1, 1)) / max(s, 1))
            print(f"{s:6d} {float(l):7.3f} {float(a):6.3f} {c0:14.3f} {cp:16.3f} {th:12.5f}")
        (l, a), g = gv(params, xs, ys, "delta")
        upd, st = opt.update(g, st, params)
        params = optax.apply_updates(params, upd)
        rng2 = np.random.default_rng(1000 + s)
        xs2, ys2 = gen_mqar(rng2, BATCH, CARGA)
        xs, ys = jnp.asarray(xs2), jnp.asarray(ys2)
    print(f"\n{time.time()-t0:.0f} s")


if __name__ == "__main__":
    main()
