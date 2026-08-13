"""E-I3b — ¿USA EL ORDEN O PREFIERE LO ULTIMO? El limite declarado de E-I3, cerrado.

E-I3 dio 0,9956 en claves revisadas con un sello de orden co-entrenado, contra 0,4570 sin el y
0,4768 con el sello barajado. Pero en esa tarea la version vigente es SIEMPRE la de turno mayor, asi
que la politica aprendible mas barata es "gana el sello mas alto". El resultado probaba SUFICIENCIA
del mecanismo, no que el lector use el orden como orden. Es la primera objecion que le haria un
revisor, y es correcta.

COMO SE SEPARA. Se archivan TRES versiones de cada clave revisada (v1 en S1; v2 y v3 en S2, en dos
rondas, con turnos crecientes) y la secuencia de consulta lleva adelante un marcador de que se
pregunta:

    CTX_A  ->  cual es la version VIGENTE   (target v3)
    CTX_B  ->  cual era la version ANTERIOR (target v2)

Un sesgo monotono fijo sobre el sello -- "mas peso al turno mas alto" -- resuelve CTX_A y falla CTX_B
por construccion, porque el mismo modelo, con los mismos pesos, tiene que dar dos respuestas
distintas sobre el mismo archivo segun el marcador. Solo puede hacerlo si el sello entra como una
coordenada que la consulta puede APUNTAR, no como una preferencia constante.

En las consultas CTX_B la perdida se calcula unicamente sobre las claves revisadas: para una clave
con una sola version, "la anterior" no existe, y pedirla seria una pregunta sin respuesta.

CONDICIONES (5 semillas, inyeccion temprana, lr = 1e-3, 4000 pasos), las mismas de E-I3:
  ninguno   sin sello. Con tres versiones compitiendo el azar baja a ~1/3.
  sello     ak = hw @ Wk + E_ord[turno real de escritura].
  barajado  sello sin relacion con el orden. Control de falsacion.

PREDICCIONES, comprometidas antes del dato:
  P-1  (bloqueante) acc(ANTERIOR | revisadas, sello) >= 0,80. Es la prediccion que separa las dos
       hipotesis: la recencia pura da ~0 en esta celda, no 1/3, porque devolver v3 cuando se pide v2
       es un error sistematico, no un sorteo. Si NO cumple, lo que E-I3 midio fue recencia y hay que
       decirlo asi en el informe y en el paper.
  P-2  acc(VIGENTE | revisadas, sello) >= 0,95. Aprender a responder las dos preguntas no puede
       costarle la que ya sabia hacer.
  P-3  acc(ANTERIOR | revisadas, sello) - acc(ANTERIOR | revisadas, barajado) >= +0,30. Otra vez:
       el orden, no los parametros.

LIMITE QUE ESTE EXPERIMENTO NO LEVANTA. Sigue habiendo un unico eje temporal y las versiones se
archivan en orden. No se prueba nada sobre ordenes parciales, escrituras concurrentes ni sellos
ruidosos. Y sigue siendo d=64 sobre tarea sintetica.
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

L = E.L                              # 6 claves
R = E.R                              # 3 de ellas se revisan, dos veces cada una
N_ARCH = L + 2 * R                   # 12 entradas: 6 iniciales + 3 segundas + 3 terceras
B = E.B
PASOS = int(os.environ.get("PASOS", "4000"))
LR = 1e-3
BLOQUE = 0
SEMILLAS = (0, 1, 2, 3, 4)
MODOS = ("ninguno", "sello", "barajado")
VOC = V_E001


def gen_lote(rng):
    """Tres versiones por clave revisada y dos tipos de consulta segun el marcador."""
    keys = np.argsort(rng.random((B, VOC.NK)), axis=1)[:, :L]
    v1 = rng.integers(0, VOC.NV, size=(B, L))
    # v2 != v1 y v3 != v2: cada revision cambia algo, si no "anterior" y "vigente" coinciden
    v2 = (v1[:, :R] + 1 + rng.integers(0, VOC.NV - 1, size=(B, R))) % VOC.NV
    v3 = (v2 + 1 + rng.integers(0, VOC.NV - 1, size=(B, R))) % VOC.NV

    s1 = np.full((B, 2 * L + 2), VOC.PAD, dtype=np.int32)
    s1[:, 0] = VOC.BOS; s1[:, 1:2*L+1:2] = VOC.K0 + keys; s1[:, 2:2*L+2:2] = VOC.V0 + v1
    s1[:, -1] = VOC.SEP
    pos1 = np.arange(2, 2 * L + 2, 2)

    # S2: primero la ronda de segundas versiones, despues la de terceras -> el turno crece con la version
    s2 = np.full((B, 4 * R + 2), VOC.PAD, dtype=np.int32)
    s2[:, 0] = VOC.BOS
    s2[:, 1:2*R+1:2] = VOC.K0 + keys[:, :R]; s2[:, 2:2*R+2:2] = VOC.V0 + v2
    s2[:, 2*R+1:4*R+1:2] = VOC.K0 + keys[:, :R]; s2[:, 2*R+2:4*R+2:2] = VOC.V0 + v3
    s2[:, -1] = VOC.SEP
    pos2 = np.concatenate([np.arange(2, 2*R+2, 2), np.arange(2*R+2, 4*R+2, 2)])

    vigente = v1.copy(); vigente[:, :R] = v3
    anterior = v2                                        # solo definida para las revisadas

    perm = np.argsort(rng.random((B, L)), axis=1)
    q = np.take_along_axis(keys, perm, axis=1)
    revisada = np.take_along_axis((np.arange(L)[None, :] < R).repeat(B, 0), perm, axis=1)
    tgt_vig = np.take_along_axis(vigente, perm, axis=1)
    # para las revisadas, su version anterior; el resto se enmascara en el modo B
    ant_full = np.zeros_like(vigente); ant_full[:, :R] = anterior
    tgt_ant = np.take_along_axis(ant_full, perm, axis=1)

    # la mitad del lote pregunta VIGENTE, la otra mitad ANTERIOR
    tipo_b = (np.arange(B) % 2 == 1)

    s3 = np.full((B, L + 2), VOC.PAD, dtype=np.int32)
    y3 = np.full((B, L + 2), IGNORE, dtype=np.int32)
    s3[:, 0] = VOC.BOS
    s3[:, 1] = np.where(tipo_b, VOC.CTX_B, VOC.CTX_A)    # el marcador de que se pregunta
    s3[:, 2:] = VOC.K0 + q
    y3[:, 2:] = VOC.V0 + np.where(tipo_b[:, None], tgt_ant, tgt_vig)
    # en modo ANTERIOR solo se pide lo que existe: las claves con mas de una version
    y3[:, 2:] = np.where(tipo_b[:, None] & ~revisada, IGNORE, y3[:, 2:])

    arch = np.argsort(rng.random((B, N_ARCH)), axis=1).astype(np.int32)
    falso = np.argsort(rng.random((B, N_ARCH)), axis=1).astype(np.int32)
    return (jnp.array(s1), pos1, jnp.array(s2), pos2, jnp.array(s3), jnp.array(y3),
            jnp.array(revisada), jnp.array(tipo_b), jnp.array(arch), jnp.array(falso))


def init_extra(seed):
    ex = E.init_extra(seed)
    ex["ord"] = glorot(jax.random.PRNGKey(7000 + seed), (N_ARCH, D))
    return ex


def forward(params, lote, modo):
    s1, pos1, s2, pos2, s3, _, _, _, arch, falso = lote
    ex = params["extra"]

    h1 = E.tronco(params, s1)[:, pos1, :]
    h2 = E.tronco(params, s2)[:, pos2, :]
    hw = jnp.concatenate([h1, h2], axis=1)               # orden de escritura: v1, luego v2, luego v3

    idx = jnp.broadcast_to(arch[:, :, None], (hw.shape[0], N_ARCH, D))
    hw = jnp.take_along_axis(hw, idx, axis=1)            # el archivo se baraja

    ak, av = hw @ ex["kw"], hw @ ex["vw"]
    if modo == "sello":
        ak = ak + ex["ord"][arch]
    elif modo == "barajado":
        ak = ak + ex["ord"][falso]

    def lectura(h):
        q = h @ ex["qr"]
        sim = jnp.einsum("btd,bnd->btn", q, ak) / jnp.sqrt(D)
        return jnp.einsum("btn,bnd->btd", jax.nn.softmax(sim, axis=-1), av) @ ex["wo"]

    h3 = E.tronco(params, s3, lectura, BLOQUE)
    return ln(params["ln_f"], h3) @ params["head"]["w"] + params["head"]["b"]


def loss_fn(params, lote, modo):
    y3, rev, tipo_b = lote[5], lote[6], lote[7]
    logits = forward(params, lote, modo)
    mask = y3 >= 0
    yl = jnp.where(mask, y3, 0)
    ce = optax.softmax_cross_entropy_with_integer_labels(logits, yl)
    ok = (logits.argmax(-1) == yl) * mask

    okq, mq = ok[:, 2:], mask[:, 2:]
    es_b = tipo_b[:, None]
    m_vig = mq * (~es_b) * rev                           # vigente, claves revisadas
    m_ant = mq * es_b * rev                              # anterior, claves revisadas
    m_una = mq * (~es_b) * (1 - rev)                     # una sola version (control de agrupar)
    tri = lambda m: (okq * m).sum() / jnp.maximum(m.sum(), 1)
    return ((ce * mask).sum() / mask.sum(),
            (ok.sum() / mask.sum(), tri(m_vig), tri(m_ant), tri(m_una)))


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
        if s % 1000 == 0:
            a, av, an, au = (float(v) for v in aux)
            print(f"    [{modo}/s{semilla}] paso {s:4d} loss {float(l):.4f} acc {a:.4f} "
                  f"(vig {av:.4f} · ANT {an:.4f} · una {au:.4f}) ({time.time()-t0:.0f}s)", flush=True)

    ev = np.random.default_rng(99000 + semilla)
    res = [[float(v) for v in loss_fn(params, gen_lote(ev), modo)[1]] for _ in range(8)]
    return np.mean(res, axis=0)


def main():
    print(f"E-I3b · archivo de {N_ARCH} entradas BARAJADO · 3 versiones por clave revisada · "
          f"consulta VIGENTE (CTX_A) vs ANTERIOR (CTX_B)\n"
          f"la recencia pura predice ~0 en ANTERIOR, no 1/3\n", flush=True)
    salida = {}
    for modo in MODOS:
        rs = []
        for s in SEMILLAS:
            r = entrenar(modo, s)
            rs.append(r)
            print(f"  {modo:9s} s{s} → acc {r[0]:.4f} (vig {r[1]:.4f} · ANT {r[2]:.4f} · "
                  f"una {r[3]:.4f})", flush=True)
            json.dump(dict(salida, **{modo: {"parcial": np.array(rs).tolist()}}),
                      open("resultados_ei3b.json", "w"), indent=1)
        a = np.array(rs)
        salida[modo] = {"acc": float(a[:, 0].mean()), "sd": float(a[:, 0].std(ddof=1)),
                        "vigente": float(a[:, 1].mean()), "sd_vigente": float(a[:, 1].std(ddof=1)),
                        "anterior": float(a[:, 2].mean()), "sd_anterior": float(a[:, 2].std(ddof=1)),
                        "una_version": float(a[:, 3].mean()),
                        "anterior_por_semilla": a[:, 2].tolist()}
        print(f"\n  ►► {modo}: vigente {a[:,1].mean():.4f} · ANTERIOR {a[:,2].mean():.4f} "
              f"(sd {a[:,2].std(ddof=1):.4f}) · una version {a[:,3].mean():.4f}\n", flush=True)
        json.dump(salida, open("resultados_ei3b.json", "w"), indent=1)

    p1 = salida["sello"]["anterior"]
    p2 = salida["sello"]["vigente"]
    p3 = p1 - salida["barajado"]["anterior"]
    print("=" * 74)
    print(f"  P-1 bloqueante: ANTERIOR con sello = {p1:.4f}  "
          f"{'CUMPLE → usa el orden' if p1 >= 0.80 else 'NO CUMPLE → era recencia'} (exigido >= 0,80)")
    print(f"  P-2 no se paga con la vigente: {p2:.4f}  "
          f"{'CUMPLE' if p2 >= 0.95 else 'NO CUMPLE'} (exigido >= 0,95)")
    print(f"  P-3 orden vs capacidad: sello − barajado = {p3:+.4f}  "
          f"{'CUMPLE' if p3 >= 0.30 else 'NO CUMPLE'} (exigido >= +0,30)")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
