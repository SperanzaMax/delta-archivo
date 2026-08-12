"""E-I2 — LA CONSULTA DIFICIL: que el modelo tenga que ALINEAR dos representaciones propias.

El limite grande de E-I1: la clave del archivo era el mismo token que se consultaba, asi que al
lector le bastaba una proyeccion casi-identidad. Eso mostro que el modelo sabe USAR un archivo, no
que sepa CONSULTARLO. La pregunta central del protocolo seguia sin tocar.

Acá el archivo NO guarda tokens. Guarda vectores que el propio modelo produce al leer las secuencias
de escritura:

  S1 (escribe)  el modelo la procesa; en la posicion de cada valor toma su estado h
                clave archivada  k_w = h @ Wk_write      valor archivado  v_w = h @ Wv_write
  S2 (revisa)   idem, para las r claves revisadas -> entran como entradas NUEVAS, no reemplazan
  S3 (consulta) el modelo la procesa; en la posicion de cada query forma  q = h @ Wq_read
                y recupera del archivo por similitud

Para acertar, el modelo tiene que hacer que `q` (formada leyendo una consulta, en un contexto) caiga
cerca de `k_w` (formada leyendo una afirmacion, en OTRO contexto). Eso es formar una consulta util, y
no hay ninguna identidad que se lo regale.

Y hay una segunda dificultad, que es la tarea canonica del programa: para las claves revisadas el
archivo contiene DOS entradas -- la vieja y la nueva -- y ambas responden a la misma consulta. El
modelo tiene que recuperar la VIGENTE. La memoria no le dice cual es: solo estan las dos.

PREDICCIONES, comprometidas antes del dato:
  P-1  (bloqueante) acc global > 0,30, contra el piso 0,0215 de E-I0. Si no, el modelo no aprende a
       alinear escritura y consulta, y esa es la respuesta a la pregunta central del brazo.
  P-2  acc(no revisadas) - acc(revisadas) >= +0,10. La competencia entre versiones tiene que costar
       algo; si no cuesta nada, la tarea no esta midiendo el conflicto de versiones.
  P-3  temprano > tardio, como en E-I1.
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
from datos import IGNORE, V_E001
from modelos import D, NB, conv3, glorot, init_params, ln, mixer

L = int(os.environ.get("L_CARGA", "6"))
R = L // 2
B = int(os.environ.get("BATCH", "32"))
PASOS = int(os.environ.get("PASOS", "4000"))
LR = 3e-3
KIND = "delta"
AZAR = 1.0 / V_E001.NV


def gen_lote(rng, B, L, r, voc=V_E001):
    """S1, S2, S3, targets y la mascara de cuales claves fueron revisadas."""
    keys = np.argsort(rng.random((B, voc.NK)), axis=1)[:, :L]
    v1 = rng.integers(0, voc.NV, size=(B, L))
    v2 = (v1[:, :r] + 1 + rng.integers(0, voc.NV - 1, size=(B, r))) % voc.NV
    final = v1.copy(); final[:, :r] = v2

    s1 = np.full((B, 2 * L + 2), voc.PAD, dtype=np.int32)
    s1[:, 0] = voc.BOS; s1[:, 1:2*L+1:2] = voc.K0 + keys; s1[:, 2:2*L+2:2] = voc.V0 + v1
    s1[:, -1] = voc.SEP
    pos1 = np.arange(2, 2 * L + 2, 2)                 # posiciones de los valores en S1

    s2 = np.full((B, 2 * r + 2), voc.PAD, dtype=np.int32)
    s2[:, 0] = voc.BOS; s2[:, 1:2*r+1:2] = voc.K0 + keys[:, :r]; s2[:, 2:2*r+2:2] = voc.V0 + v2
    s2[:, -1] = voc.SEP
    pos2 = np.arange(2, 2 * r + 2, 2)

    perm = np.argsort(rng.random((B, L)), axis=1)
    q = np.take_along_axis(keys, perm, axis=1)
    tgt = np.take_along_axis(final, perm, axis=1)
    revisada = np.take_along_axis((np.arange(L)[None, :] < r).repeat(B, 0), perm, axis=1)

    s3 = np.full((B, L + 2), voc.PAD, dtype=np.int32)
    y3 = np.full((B, L + 2), IGNORE, dtype=np.int32)
    s3[:, 0] = voc.BOS; s3[:, 1] = voc.SEP; s3[:, 2:] = voc.K0 + q
    y3[:, 2:] = voc.V0 + tgt
    return s1, pos1, s2, pos2, s3, y3, revisada


def init_extra(seed):
    ks = jax.random.split(jax.random.PRNGKey(4000 + seed), 4)
    return {"kw": glorot(ks[0], (D, D)), "vw": glorot(ks[1], (D, D)),
            "qr": glorot(ks[2], (D, D)), "wo": glorot(ks[3], (D, D))}


def tronco(params, x, lectura=None, bloque=None):
    """Pasa x por los bloques. Si `lectura` esta, la inyecta en `bloque`. Devuelve estados."""
    hx = params["emb"][x]
    for i, blk in enumerate(params["blocks"]):
        if lectura is not None and i == bloque:
            hx = hx + lectura(ln(blk["ln1"], hx))
        hx = hx + mixer(blk, conv3(blk["conv"], ln(blk["ln1"], hx)), KIND)
        h2 = ln(blk["ln2"], hx)
        hx = hx + jax.nn.gelu(h2 @ blk["m1"]["w"] + blk["m1"]["b"]) @ blk["m2"]["w"] + blk["m2"]["b"]
    return hx


def forward(params, s1, pos1, s2, pos2, s3, cfg):
    ex = params["extra"]
    # --- escritura: el archivo son vectores que produce el propio modelo ---
    h1 = tronco(params, s1)[:, pos1, :]                 # (B, L, D)
    h2 = tronco(params, s2)[:, pos2, :]                 # (B, r, D)
    hw = jnp.concatenate([h1, h2], axis=1)              # L+r entradas: las viejas Y las nuevas
    ak, av = hw @ ex["kw"], hw @ ex["vw"]

    def lectura(h):
        q = h @ ex["qr"]
        sim = jnp.einsum("btd,bnd->btn", q, ak) / jnp.sqrt(D)
        w = jax.nn.softmax(sim, axis=-1)
        return jnp.einsum("btn,bnd->btd", w, av) @ ex["wo"]

    h3 = tronco(params, s3, lectura, cfg["bloque"])
    return ln(params["ln_f"], h3) @ params["head"]["w"] + params["head"]["b"]


def loss_fn(params, lote, cfg):
    s1, pos1, s2, pos2, s3, y3, rev = lote
    logits = forward(params, s1, pos1, s2, pos2, s3, cfg)
    mask = y3 >= 0
    yl = jnp.where(mask, y3, 0)
    ce = optax.softmax_cross_entropy_with_integer_labels(logits, yl)
    ok = (logits.argmax(-1) == yl) * mask
    acc = ok.sum() / mask.sum()
    # desglose por revisada / no revisada (las queries viven en las columnas 2:)
    okq, revq = ok[:, 2:], rev
    acc_rev = (okq * revq).sum() / jnp.maximum(revq.sum(), 1)
    acc_no = (okq * (1 - revq)).sum() / jnp.maximum((1 - revq).sum(), 1)
    return (ce * mask).sum() / mask.sum(), (acc, acc_rev, acc_no)


def entrenar(bloque, semilla, pasos=PASOS):
    cfg = {"bloque": bloque}
    params = init_params(semilla, KIND)
    params["extra"] = init_extra(semilla)
    rng = np.random.default_rng(5000 + semilla)
    sched = optax.warmup_constant_schedule(0.0, LR, 100)
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(sched, weight_decay=0.01))
    state = opt.init(params)

    @partial(jax.jit, static_argnames="bl")
    def paso(params, state, lote, bl):
        (l, aux), g = jax.value_and_grad(loss_fn, has_aux=True)(params, lote, {"bloque": bl})
        up, state = opt.update(g, state, params)
        return optax.apply_updates(params, up), state, l, aux

    def lote_jnp(r):
        s1, p1, s2, p2, s3, y3, rev = gen_lote(r, B, L, R)
        return (jnp.array(s1), p1, jnp.array(s2), p2, jnp.array(s3),
                jnp.array(y3), jnp.array(rev))

    t0 = time.time()
    for s in range(1, pasos + 1):
        params, state, l, aux = paso(params, state, lote_jnp(rng), bloque)
        if s % 500 == 0:
            a, ar, an = (float(v) for v in aux)
            print(f"    [b{bloque}/s{semilla}] paso {s:4d} loss {float(l):.4f} acc {a:.4f} "
                  f"(rev {ar:.4f} · no-rev {an:.4f}) ({time.time()-t0:.0f}s)", flush=True)

    ev = np.random.default_rng(99000 + semilla)
    res = []
    for _ in range(8):
        _, aux = loss_fn(params, lote_jnp(ev), cfg)
        res.append([float(v) for v in aux])
    return np.mean(res, axis=0)


def main():
    semillas = (0, 1, 2)
    print(f"E-I2 · L={L} r={R} · archivo = {L+R} vectores PRODUCIDOS POR EL MODELO · "
          f"{PASOS} pasos · piso 0,0215\n", flush=True)
    salida = {}
    for bloque, etiqueta in ((0, "temprano"), (NB - 1, "tardio")):
        rs = np.array([entrenar(bloque, s) for s in semillas])
        salida[etiqueta] = {"acc": float(rs[:, 0].mean()), "sd": float(rs[:, 0].std(ddof=1)),
                            "acc_revisadas": float(rs[:, 1].mean()),
                            "acc_no_revisadas": float(rs[:, 2].mean()),
                            "por_semilla": rs[:, 0].tolist()}
        print(f"  {etiqueta:8s} → acc {rs[:,0].mean():.4f} (sd {rs[:,0].std(ddof=1):.4f}) · "
              f"revisadas {rs[:,1].mean():.4f} · no revisadas {rs[:,2].mean():.4f}\n", flush=True)
        json.dump(salida, open("resultados_ei2.json", "w"), indent=1)

    mejor = max(v["acc"] for v in salida.values())
    brecha = salida["temprano"]["acc_no_revisadas"] - salida["temprano"]["acc_revisadas"]
    dpos = salida["temprano"]["acc"] - salida["tardio"]["acc"]
    print("=" * 72)
    print(f"  P-1 bloqueante: mejor {mejor:.4f}  {'CUMPLE' if mejor > 0.30 else 'NO CUMPLE'} "
          f"(exigido > 0,30 vs piso 0,0215)")
    print(f"  P-2 conflicto de versiones: no-rev − rev = {brecha:+.4f}  "
          f"{'CUMPLE' if brecha >= 0.10 else 'NO CUMPLE'} (exigido >= +0,10)")
    print(f"  P-3 posicion: temprano − tardio = {dpos:+.4f}  "
          f"{'CUMPLE' if dpos > 0 else 'NO CUMPLE'} (direccional)")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
