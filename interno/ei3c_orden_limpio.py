"""E-I3c — LA PREGUNTA DE E-I3b, CON LA FUGA TAPADA.

E-I3b se rompio solo, y conviene decir exactamente como. Para tener tres versiones de una clave puse
las dos revisiones en la MISMA secuencia (S2: primero los v2, despues los v3). Con eso, el estado que
el modelo produce en la posicion de v3 ya vio pasar a v2 y arrastra el rastro de ser posterior: el
ORDEN quedo escrito en el CONTENIDO del vector archivado, sin que ningun metadato lo pusiera ahi.

A 4000 pasos eso no se veia, porque a 4000 pasos nadie resolvia la pregunta por la version anterior
(sello 0,1109 · barajado 0,0771 · ninguno 0,0443) y parecia un negativo limpio. Con 12000 pasos la
condicion SIN SELLO llega a ~0,97 en esa misma pregunta. O sea que la tarea se puede resolver sin
metadato, y por lo tanto E-I3b no puede decir nada sobre si el sello aporta el orden. No refuta a
E-I3: es incapaz de evaluarlo.

LA CORRECCION. Cada version se escribe en su PROPIA secuencia, con el estado reseteado entre una y
otra:

    S1  BOS (k v1)*L SEP     las L claves, primera version
    S2  BOS (k v2)*R SEP     las R revisadas, segunda version
    S3  BOS (k v3)*R SEP     las R revisadas, tercera version
    S4  BOS marcador (k)*L   la consulta: CTX_A = cual RIGE · CTX_B = cual era la ANTERIOR

Ninguna secuencia ve a las otras. Para el modelo, v2 y v3 son las dos "primera y unica mencion de su
secuencia", exactamente igual que v1: sus estados son indistinguibles en cuanto a recencia. La unica
cosa en todo el sistema que dice cual vino antes es el sello de orden. Si con eso el lector contesta
"cual era la anterior", usa el orden; si no contesta, no lo usa. No queda tercera explicacion.

PRESUPUESTO. 12000 pasos desde el principio, no 4000: E-I3b ya mostro que la pregunta por la anterior
recien despega entre 4000 y 5000 pasos, mucho despues de que la vigente satura. Correrlo corto seria
repetir a proposito el error de hoy.

CONDICIONES: `ninguno`, `sello`, `barajado`, las tres con el mismo presupuesto. 3 semillas en vez de
5, y esto es una decision de COSTO declarada: cada corrida son ~20 min y son 9. Si el resultado sale
apretado, se agregan semillas antes de escribir nada.

PREDICCIONES, comprometidas antes del dato:
  P-1  (bloqueante, y es la que faltaba en el chequeo de presupuesto de E-I3b) ANTERIOR(ninguno) <=
       0,40. Es el control de que la fuga esta tapada. Si el baseline vuelve a resolver la pregunta
       sin metadato, hay OTRA via de informacion que no vi, el experimento sigue sin poder medir lo
       que quiere medir, y hay que encontrarla antes de correr nada mas.
  P-2  ANTERIOR(sello) >= 0,80.
  P-3  ANTERIOR(sello) - ANTERIOR(barajado) >= +0,30.
  Las tres juntas son la afirmacion completa: el orden se puede usar, y viene del sello.

SI P-1 CUMPLE Y P-2 NO: es el negativo genuino que E-I3b creyo tener a 4000 pasos, y entonces el
metadato sirve para "cual rige" y no para ordenar. Se escribe asi, y el informe de E-I3 se corrige.
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
from datos import IGNORE, V_E001
from modelos import D, glorot, init_params, ln

L = E.L
R = E.R
N_ARCH = L + 2 * R
B = E.B
PASOS = int(os.environ.get("PASOS", "12000"))
LR = 1e-3
BLOQUE = 0
SEMILLAS = (0, 1, 2)
MODOS = ("ninguno", "sello", "barajado")
VOC = V_E001


def _sec(keys, vals):
    """BOS (k v)* SEP, y las posiciones de los valores."""
    n = keys.shape[1]
    x = np.full((B, 2 * n + 2), VOC.PAD, dtype=np.int32)
    x[:, 0] = VOC.BOS
    x[:, 1:2*n+1:2] = VOC.K0 + keys
    x[:, 2:2*n+2:2] = VOC.V0 + vals
    x[:, -1] = VOC.SEP
    return x, np.arange(2, 2 * n + 2, 2)


def gen_lote(rng):
    keys = np.argsort(rng.random((B, VOC.NK)), axis=1)[:, :L]
    v1 = rng.integers(0, VOC.NV, size=(B, L))
    v2 = (v1[:, :R] + 1 + rng.integers(0, VOC.NV - 1, size=(B, R))) % VOC.NV
    v3 = (v2 + 1 + rng.integers(0, VOC.NV - 1, size=(B, R))) % VOC.NV

    s1, pos1 = _sec(keys, v1)                    # turnos 0..L-1
    s2, pos2 = _sec(keys[:, :R], v2)             # turnos L..L+R-1   -- secuencia propia
    s3, pos3 = _sec(keys[:, :R], v3)             # turnos L+R..L+2R-1 -- secuencia propia

    vigente = v1.copy(); vigente[:, :R] = v3
    ant_full = np.zeros_like(vigente); ant_full[:, :R] = v2

    perm = np.argsort(rng.random((B, L)), axis=1)
    q = np.take_along_axis(keys, perm, axis=1)
    revisada = np.take_along_axis((np.arange(L)[None, :] < R).repeat(B, 0), perm, axis=1)
    tgt_vig = np.take_along_axis(vigente, perm, axis=1)
    tgt_ant = np.take_along_axis(ant_full, perm, axis=1)
    tipo_b = (np.arange(B) % 2 == 1)

    s4 = np.full((B, L + 2), VOC.PAD, dtype=np.int32)
    y4 = np.full((B, L + 2), IGNORE, dtype=np.int32)
    s4[:, 0] = VOC.BOS
    s4[:, 1] = np.where(tipo_b, VOC.CTX_B, VOC.CTX_A)
    s4[:, 2:] = VOC.K0 + q
    y4[:, 2:] = VOC.V0 + np.where(tipo_b[:, None], tgt_ant, tgt_vig)
    y4[:, 2:] = np.where(tipo_b[:, None] & ~revisada, IGNORE, y4[:, 2:])

    arch = np.argsort(rng.random((B, N_ARCH)), axis=1).astype(np.int32)
    falso = np.argsort(rng.random((B, N_ARCH)), axis=1).astype(np.int32)
    return (jnp.array(s1), pos1, jnp.array(s2), pos2, jnp.array(s3), pos3,
            jnp.array(s4), jnp.array(y4), jnp.array(revisada), jnp.array(tipo_b),
            jnp.array(arch), jnp.array(falso))


def init_extra(seed):
    ex = E.init_extra(seed)
    ex["ord"] = glorot(jax.random.PRNGKey(7000 + seed), (N_ARCH, D))
    return ex


def forward(params, lote, modo):
    s1, pos1, s2, pos2, s3, pos3, s4 = lote[:7]
    arch, falso = lote[10], lote[11]
    ex = params["extra"]

    # tres escrituras independientes: el estado no cruza de una secuencia a otra
    h1 = E.tronco(params, s1)[:, pos1, :]
    h2 = E.tronco(params, s2)[:, pos2, :]
    h3 = E.tronco(params, s3)[:, pos3, :]
    hw = jnp.concatenate([h1, h2, h3], axis=1)

    idx = jnp.broadcast_to(arch[:, :, None], (hw.shape[0], N_ARCH, D))
    hw = jnp.take_along_axis(hw, idx, axis=1)

    ak, av = hw @ ex["kw"], hw @ ex["vw"]
    if modo == "sello":
        ak = ak + ex["ord"][arch]
    elif modo == "barajado":
        ak = ak + ex["ord"][falso]

    def lectura(h):
        q = h @ ex["qr"]
        sim = jnp.einsum("btd,bnd->btn", q, ak) / jnp.sqrt(D)
        return jnp.einsum("btn,bnd->btd", jax.nn.softmax(sim, axis=-1), av) @ ex["wo"]

    h4 = E.tronco(params, s4, lectura, BLOQUE)
    return ln(params["ln_f"], h4) @ params["head"]["w"] + params["head"]["b"]


def loss_fn(params, lote, modo):
    y4, rev, tipo_b = lote[7], lote[8], lote[9]
    logits = forward(params, lote, modo)
    mask = y4 >= 0
    yl = jnp.where(mask, y4, 0)
    ce = optax.softmax_cross_entropy_with_integer_labels(logits, yl)
    ok = (logits.argmax(-1) == yl) * mask
    okq, mq = ok[:, 2:], mask[:, 2:]
    es_b = tipo_b[:, None]
    tri = lambda m: (okq * m).sum() / jnp.maximum(m.sum(), 1)
    return ((ce * mask).sum() / mask.sum(),
            (ok.sum() / mask.sum(), tri(mq * (~es_b) * rev), tri(mq * es_b * rev),
             tri(mq * (~es_b) * (1 - rev))))


def entrenar(modo, semilla, pasos=PASOS):
    params = init_params(semilla, E.KIND)
    params["extra"] = init_extra(semilla)
    rng = np.random.default_rng(5000 + semilla)
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
        if s % 2000 == 0:
            a, av, an, au = (float(v) for v in aux)
            print(f"    [{modo}/s{semilla}] paso {s:5d} vig {av:.4f} · ANT {an:.4f} · "
                  f"una {au:.4f} ({time.time()-t0:.0f}s)", flush=True)

    ev = np.random.default_rng(99000 + semilla)
    return np.mean([[float(v) for v in loss_fn(params, gen_lote(ev), modo)[1]]
                    for _ in range(8)], axis=0)


def main():
    print(f"E-I3c · CADA VERSION EN SU PROPIA SECUENCIA · archivo {N_ARCH} entradas barajado · "
          f"{PASOS} pasos · {len(SEMILLAS)} semillas\n"
          f"el unico dato de orden en todo el sistema es el sello\n", flush=True)
    salida = {}
    for modo in MODOS:
        rs = []
        for s in SEMILLAS:
            r = entrenar(modo, s)
            rs.append(r)
            print(f"  {modo:9s} s{s} → vig {r[1]:.4f} · ANT {r[2]:.4f} · una {r[3]:.4f}", flush=True)
            json.dump(dict(salida, **{modo: {"parcial": np.array(rs).tolist()}}),
                      open("resultados_ei3c.json", "w"), indent=1)
        a = np.array(rs)
        salida[modo] = {"vigente": float(a[:, 1].mean()), "anterior": float(a[:, 2].mean()),
                        "sd_anterior": float(a[:, 2].std(ddof=1)),
                        "una_version": float(a[:, 3].mean()),
                        "anterior_por_semilla": a[:, 2].tolist()}
        print(f"\n  ►► {modo}: vigente {a[:,1].mean():.4f} · ANTERIOR {a[:,2].mean():.4f} "
              f"(sd {a[:,2].std(ddof=1):.4f}) · una version {a[:,3].mean():.4f}\n", flush=True)
        json.dump(salida, open("resultados_ei3c.json", "w"), indent=1)

    p1 = salida["ninguno"]["anterior"]
    p2 = salida["sello"]["anterior"]
    p3 = p2 - salida["barajado"]["anterior"]
    print("=" * 74)
    print(f"  P-1 la fuga esta tapada: ANTERIOR(ninguno) = {p1:.4f}  "
          f"{'CUMPLE' if p1 <= 0.40 else '*** NO CUMPLE: hay otra via de informacion, buscarla ***'}")
    print(f"  P-2 el orden se usa: ANTERIOR(sello) = {p2:.4f}  "
          f"{'CUMPLE' if p2 >= 0.80 else 'NO CUMPLE → el metadato ordena la vigente y nada mas'}")
    print(f"  P-3 viene del sello: sello − barajado = {p3:+.4f}  "
          f"{'CUMPLE' if p3 >= 0.30 else 'NO CUMPLE'}")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
