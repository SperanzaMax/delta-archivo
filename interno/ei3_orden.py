"""E-I3 — METADATO DE ORDEN CO-ENTRENADO: la prueba directa del dictamen de R4.

E-I2 dejo el cuello de botella MEDIDO, no supuesto: 0,9974 cuando el archivo tiene una sola version
de la clave consultada y 0,4576 cuando compiten la vieja y la nueva. 0,4576 ~ 0,5 = elegir al azar
entre las dos. El modelo encuentra el hecho correcto y no sabe cual version rige -- exactamente el
modo de falla que R1/R4 midieron con un indice NO parametrico sobre encoder congelado ("la geometria
agrupa perfecto pero no ordena"), reproducido acá por un indice CO-ENTRENADO dentro de la red.

El dictamen que salio de R4 fue: GEOMETRIA PARA AGRUPAR, METADATO PARA ORDENAR. Hasta hoy es una
recomendacion de diseño derivada de dos negativos. Esto la prueba de frente: se le da al archivo un
sello de orden aprendido y se mide si el conflicto de versiones se levanta.

MECANISMO. El archivo son L+r vectores producidos por el modelo (L de S1, r de S2). Se le suma a la
CLAVE archivada un embedding de orden co-entrenado E_ord[t], donde t es el turno de escritura de esa
entrada. Entra en la clave y no en el valor porque tiene que sesgar la SELECCION.

EL ARCHIVO SE BARAJA. Sin barajar, el turno de escritura coincide siempre con la posicion en el
tensor y la prueba mediria "aprender a mirar el final del archivo", que es posicion, no orden. Se
permuta la dimension del archivo por muestra y el sello viaja pegado a su entrada. La lectura es un
softmax sobre las entradas, o sea invariante a la permutacion: barajar no cambia nada en la condicion
sin sello (y eso la deja identica a E-I2, que funciona como control de replica de la implementacion).

CONDICIONES (las tres con inyeccion temprana -- bloque 0 --, lr = 1e-3, 4000 pasos, semillas 0..4):
  ninguno   sin sello. Baseline. Debe reproducir E-I2: revisadas ~ 0,46.
  sello     ak = hw @ Wk + E_ord[turno real de escritura].
  barajado  ak = hw @ Wk + E_ord[turno ALEATORIO], sin relacion con el orden real. Control de
            falsacion: misma capacidad, mismos parametros, señal destruida. Sin esta celda, una
            mejora en `sello` se explicaria igual de bien por los parametros extra.

PREDICCIONES, comprometidas antes del dato:
  P-1  (bloqueante) revisadas(sello) - revisadas(ninguno) >= +0,20. Si no se cumple, el metadato de
       orden co-entrenado NO levanta el conflicto de versiones y el dictamen de R4 se queda sin
       apoyo justo del lado parametrico, que es el que faltaba.
  P-2  revisadas(sello) - revisadas(barajado) >= +0,15. Aisla el ORDEN de la CAPACIDAD.
  P-3  no_revisadas(sello) >= 0,95. El sello no puede pagarse rompiendo la identificacion del item:
       agrupar tiene que seguir funcionando igual de bien.

LIMITES DECLARADOS ANTES DE CORRER. En esta tarea la version vigente es siempre la de turno mayor,
asi que la politica que el modelo puede aprender es "gana el sello mas alto". Eso hace la prueba de
SUFICIENCIA del mecanismo -- alcanza un metadato de orden para levantar el conflicto --, no de que el
modelo infiera un orden que no se le da. Un P-1 que cumple no dice "el modelo razona sobre el
tiempo": dice que la informacion que a la geometria le falta cabe en un sello y que el lector la sabe
usar. Sigue siendo un modelo de 64 dimensiones sobre tarea sintetica, con archivo de 9 entradas.
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
from modelos import D, glorot, init_params, ln

L = E.L
R = E.R
N_ARCH = L + R                       # 9 entradas: 6 de S1 + 3 de S2
B = E.B
PASOS = int(os.environ.get("PASOS", "4000"))
LR = 1e-3
BLOQUE = 0                           # inyeccion temprana: la que funciona (E-I2, +0,3448)
SEMILLAS = (0, 1, 2, 3, 4)
MODOS = ("ninguno", "sello", "barajado")


def init_extra(seed):
    """Los mismos parametros que E-I2 mas la tabla de orden."""
    ex = E.init_extra(seed)
    ex["ord"] = glorot(jax.random.PRNGKey(7000 + seed), (N_ARCH, D))
    return ex


def gen_lote(rng):
    """El lote de E-I2 mas la permutacion del archivo y los sellos aleatorios del control."""
    s1, pos1, s2, pos2, s3, y3, rev = E.gen_lote(rng, B, L, R)
    perm = np.argsort(rng.random((B, N_ARCH)), axis=1).astype(np.int32)
    falso = np.argsort(rng.random((B, N_ARCH)), axis=1).astype(np.int32)
    return (jnp.array(s1), pos1, jnp.array(s2), pos2, jnp.array(s3), jnp.array(y3),
            jnp.array(rev), jnp.array(perm), jnp.array(falso))


def forward(params, lote, modo):
    s1, pos1, s2, pos2, s3, _, _, perm, falso = lote
    ex = params["extra"]

    h1 = E.tronco(params, s1)[:, pos1, :]
    h2 = E.tronco(params, s2)[:, pos2, :]
    hw = jnp.concatenate([h1, h2], axis=1)              # en ORDEN de escritura: viejas y despues nuevas

    # el archivo se baraja; el turno de escritura de cada entrada viaja con ella
    idx = perm[:, :, None]
    hw = jnp.take_along_axis(hw, jnp.broadcast_to(idx, (hw.shape[0], N_ARCH, D)), axis=1)

    ak, av = hw @ ex["kw"], hw @ ex["vw"]
    if modo == "sello":
        ak = ak + ex["ord"][perm]                       # el turno REAL de escritura
    elif modo == "barajado":
        ak = ak + ex["ord"][falso]                      # un turno sin relacion con el orden

    def lectura(h):
        q = h @ ex["qr"]
        sim = jnp.einsum("btd,bnd->btn", q, ak) / jnp.sqrt(D)
        w = jax.nn.softmax(sim, axis=-1)
        return jnp.einsum("btn,bnd->btd", w, av) @ ex["wo"]

    h3 = E.tronco(params, s3, lectura, BLOQUE)
    return ln(params["ln_f"], h3) @ params["head"]["w"] + params["head"]["b"]


def loss_fn(params, lote, modo):
    y3, rev = lote[5], lote[6]
    logits = forward(params, lote, modo)
    mask = y3 >= 0
    yl = jnp.where(mask, y3, 0)
    ce = optax.softmax_cross_entropy_with_integer_labels(logits, yl)
    ok = (logits.argmax(-1) == yl) * mask
    acc = ok.sum() / mask.sum()
    okq = ok[:, 2:]
    acc_rev = (okq * rev).sum() / jnp.maximum(rev.sum(), 1)
    acc_no = (okq * (1 - rev)).sum() / jnp.maximum((1 - rev).sum(), 1)
    return (ce * mask).sum() / mask.sum(), (acc, acc_rev, acc_no)


def entrenar(modo, semilla, pasos=PASOS):
    params = init_params(semilla, E.KIND)
    params["extra"] = init_extra(semilla)
    rng = np.random.default_rng(5000 + semilla)         # el mismo flujo de datos que E-I2
    sched = optax.warmup_constant_schedule(0.0, LR, 100)
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(sched, weight_decay=0.01))
    state = opt.init(params)

    @partial(jax.jit, static_argnames="m")
    def paso(params, state, lote, m):
        (l, aux), g = jax.value_and_grad(loss_fn, has_aux=True)(params, lote, m)
        up, state = opt.update(g, state, params)
        return optax.apply_updates(params, up), state, l, aux

    t0 = time.time()
    for s in range(1, pasos + 1):
        params, state, l, aux = paso(params, state, gen_lote(rng), modo)
        if s % 1000 == 0:
            a, ar, an = (float(v) for v in aux)
            print(f"    [{modo}/s{semilla}] paso {s:4d} loss {float(l):.4f} acc {a:.4f} "
                  f"(rev {ar:.4f} · no-rev {an:.4f}) ({time.time()-t0:.0f}s)", flush=True)

    ev = np.random.default_rng(99000 + semilla)
    res = [[float(v) for v in loss_fn(params, gen_lote(ev), modo)[1]] for _ in range(8)]
    return np.mean(res, axis=0)


def main():
    print(f"E-I3 · archivo de {N_ARCH} entradas BARAJADO · bloque {BLOQUE} · lr {LR} · "
          f"{PASOS} pasos · {len(SEMILLAS)} semillas\n"
          f"referencia E-I2 (replica, 5 semillas): revisadas 0,4576 · no revisadas 0,9974\n", flush=True)
    salida = {}
    for modo in MODOS:
        rs = []
        for s in SEMILLAS:
            r = entrenar(modo, s)
            rs.append(r)
            print(f"  {modo:9s} s{s} → acc {r[0]:.4f} (rev {r[1]:.4f} · no-rev {r[2]:.4f})", flush=True)
            json.dump(dict(salida, **{modo: {"parcial": np.array(rs).tolist()}}),
                      open("resultados_ei3.json", "w"), indent=1)
        a = np.array(rs)
        salida[modo] = {"acc": float(a[:, 0].mean()), "sd": float(a[:, 0].std(ddof=1)),
                        "revisadas": float(a[:, 1].mean()),
                        "sd_revisadas": float(a[:, 1].std(ddof=1)),
                        "no_revisadas": float(a[:, 2].mean()),
                        "por_semilla": a[:, 0].tolist(),
                        "revisadas_por_semilla": a[:, 1].tolist()}
        print(f"\n  ►► {modo}: acc {a[:,0].mean():.4f} (sd {a[:,0].std(ddof=1):.4f}) · "
              f"revisadas {a[:,1].mean():.4f} (sd {a[:,1].std(ddof=1):.4f}) · "
              f"no revisadas {a[:,2].mean():.4f}\n", flush=True)
        json.dump(salida, open("resultados_ei3.json", "w"), indent=1)

    p1 = salida["sello"]["revisadas"] - salida["ninguno"]["revisadas"]
    p2 = salida["sello"]["revisadas"] - salida["barajado"]["revisadas"]
    p3 = salida["sello"]["no_revisadas"]
    ctrl = salida["ninguno"]["revisadas"]
    print("=" * 74)
    print(f"  control de replica: `ninguno` revisadas {ctrl:.4f} vs 0,4576 de E-I2  "
          f"{'coherente' if abs(ctrl - 0.4576) < 0.08 else '*** DISCREPA: revisar implementacion ***'}")
    print(f"  P-1 bloqueante: sello − ninguno  = {p1:+.4f}  "
          f"{'CUMPLE' if p1 >= 0.20 else 'NO CUMPLE'} (exigido >= +0,20)")
    print(f"  P-2 orden vs capacidad: sello − barajado = {p2:+.4f}  "
          f"{'CUMPLE' if p2 >= 0.15 else 'NO CUMPLE'} (exigido >= +0,15)")
    print(f"  P-3 no degrada agrupar: no_revisadas(sello) = {p3:.4f}  "
          f"{'CUMPLE' if p3 >= 0.95 else 'NO CUMPLE'} (exigido >= 0,95)")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
