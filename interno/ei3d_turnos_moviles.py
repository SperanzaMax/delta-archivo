"""E-I3d — ¿ORDEN, O UNA TABLA DE SLOTS? El limite de E-I3c, cerrado.

E-I3c tapa la fuga de contenido de E-I3b y muestra que con sello el lector contesta tanto "cual
rige" como "cual era la anterior". Pero tiene un agujero que aparece al preguntarse que OTRA cosa
produciria ese mismo numero: los turnos son fijos por rol. v1 se lleva siempre los turnos 0..5, v2
los 6..8, v3 los 9..11. Con eso al modelo le alcanza con aprender una tabla

    "los embeddings 9,10,11 son la version vigente; 6,7,8 son la anterior"

sin ninguna nocion de que 9 viene DESPUES de 6. Es una correspondencia slot->rol, no un orden. Y el
control `barajado` no separa las dos lecturas: al aleatorizar el sello rompe la tabla exactamente
igual que rompe el orden, asi que su caida es compatible con ambas.

LA CORRECCION: LOS TURNOS SE MUEVEN. Por cada muestra se sortean N_ARCH turnos distintos de un rango
mas grande (0..N_TURNOS-1, con N_TURNOS = 32 y N_ARCH = 12), se ordenan, y se reparten respetando el
orden real de escritura: los L mas chicos para v1, los R siguientes para v2, los R mas grandes para
v3. El orden relativo se mantiene siempre; los valores absolutos cambian en cada muestra.

Asi el turno 9 puede ser una primera version en una muestra y una tercera en la siguiente, y ninguna
tabla fija de slots resuelve la tarea. Lo unico que sigue funcionando es COMPARAR: "de las entradas
que responden a esta clave, la de turno mayor es la vigente y la anterior es la que le sigue hacia
abajo". Eso es usar el orden como orden, y es lo que el experimento va a poder afirmar o negar.

PREDICCIONES, comprometidas antes del dato:
  P-1  (bloqueante) ANTERIOR(sello) >= 0,80 con turnos moviles. Si CUMPLE, el lector compara turnos
       y la afirmacion "usa el orden" queda sostenida. Si NO cumple -- y muy en particular si cae
       cerca del 0,29 de la condicion sin sello --, entonces lo que E-I3c midio era la tabla de
       slots, y hay que escribirlo asi: el metadato funciona como ETIQUETA DE ROL, no como orden.
  P-2  VIGENTE(sello) >= 0,80. Se mide aparte a proposito: es posible que "la mas nueva" sobreviva a
       los turnos moviles (comparar un maximo es mas facil) y que se caiga solo la anterior. Ese
       resultado partido seria informativo, no un empate.
  P-3  ANTERIOR(sello) - ANTERIOR(barajado) >= +0,30, con barajado sorteando turnos SIN respetar el
       orden relativo (misma tabla, misma cantidad de parametros, sin la relacion).

CONTROL DE INTERPRETACION, declarado antes: si P-1 y P-2 cumplen, la afirmacion es "el lector compara
turnos"; no es "el lector entiende el tiempo", que no lo mide ningun experimento de esta serie.

3 semillas, 12000 pasos, por el mismo motivo de costo que E-I3c.
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
import ei3c_orden_limpio as C
from modelos import D, glorot, init_params, ln

L, R = C.L, C.R
N_ARCH = C.N_ARCH                    # 12 entradas
N_TURNOS = 32                        # el rango del que se sortean los turnos
B = C.B
PASOS = int(os.environ.get("PASOS", "12000"))
LR = 1e-3
BLOQUE = 0
SEMILLAS = (0, 1, 2)
MODOS = ("sello", "barajado", "ninguno")
VOC = C.VOC


def turnos_moviles(rng):
    """N_ARCH turnos distintos por muestra, ordenados: L para v1, R para v2, R para v3.

    El orden relativo es siempre el real; los valores absolutos cambian en cada muestra, asi que
    ninguna tabla fija slot->rol resuelve la tarea.
    """
    t = np.argsort(rng.random((B, N_TURNOS)), axis=1)[:, :N_ARCH]
    return np.sort(t, axis=1).astype(np.int32)


def gen_lote(rng):
    """El lote de E-I3c, con turnos que se mueven y un barajado que rompe SOLO la relacion.

    Devuelve, ademas de lo de E-I3c: `arch` (la permutacion con que se baraja el archivo, indice 10),
    el sello REAL ya alineado a esa permutacion (12) y el sello del control (13). El control lleva
    los MISMOS valores de turno que el real, desordenados entre si: misma tabla, mismos parametros,
    misma distribucion, sin la relacion con el orden de escritura.
    """
    base = list(C.gen_lote(rng))
    orden = turnos_moviles(rng)                       # (B, N_ARCH) creciente, en orden de escritura
    arch = np.array(base[10])                         # como se baraja el archivo
    sello_real = np.take_along_axis(orden, arch, axis=1)
    falso = np.argsort(rng.random((B, N_ARCH)), axis=1).astype(np.int32)
    sello_falso = np.take_along_axis(sello_real, falso, axis=1)
    return tuple(base) + (jnp.array(sello_real), jnp.array(sello_falso))


def init_extra(seed):
    ex = E.init_extra(seed)
    ex["ord"] = glorot(jax.random.PRNGKey(7000 + seed), (N_TURNOS, D))
    return ex


def forward(params, lote, modo):
    s1, pos1, s2, pos2, s3, pos3, s4 = lote[:7]
    arch, sello_real, sello_falso = lote[10], lote[12], lote[13]
    ex = params["extra"]

    h1 = E.tronco(params, s1)[:, pos1, :]
    h2 = E.tronco(params, s2)[:, pos2, :]
    h3 = E.tronco(params, s3)[:, pos3, :]
    hw = jnp.concatenate([h1, h2, h3], axis=1)

    # el archivo se baraja y el sello viaja pegado a su entrada: sin esto, la POSICION en el tensor
    # volveria a codificar el rol y la tabla de slots entraria por la ventana
    idx = jnp.broadcast_to(arch[:, :, None], (hw.shape[0], N_ARCH, D))
    hw = jnp.take_along_axis(hw, idx, axis=1)

    ak, av = hw @ ex["kw"], hw @ ex["vw"]
    if modo == "sello":
        ak = ak + ex["ord"][sello_real]
    elif modo == "barajado":
        ak = ak + ex["ord"][sello_falso]

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
    print(f"E-I3d · TURNOS MOVILES · {N_ARCH} entradas sorteadas de {N_TURNOS} turnos posibles · "
          f"{PASOS} pasos · {len(SEMILLAS)} semillas\n"
          f"ninguna tabla fija slot->rol resuelve la tarea: hay que comparar\n", flush=True)
    salida = {}
    for modo in MODOS:
        rs = []
        for s in SEMILLAS:
            r = entrenar(modo, s)
            rs.append(r)
            print(f"  {modo:9s} s{s} → vig {r[1]:.4f} · ANT {r[2]:.4f} · una {r[3]:.4f}", flush=True)
            json.dump(dict(salida, **{modo: {"parcial": np.array(rs).tolist()}}),
                      open("resultados_ei3d.json", "w"), indent=1)
        a = np.array(rs)
        salida[modo] = {"vigente": float(a[:, 1].mean()), "anterior": float(a[:, 2].mean()),
                        "sd_anterior": float(a[:, 2].std(ddof=1)),
                        "una_version": float(a[:, 3].mean()),
                        "anterior_por_semilla": a[:, 2].tolist()}
        print(f"\n  ►► {modo}: vigente {a[:,1].mean():.4f} · ANTERIOR {a[:,2].mean():.4f} "
              f"(sd {a[:,2].std(ddof=1):.4f}) · una version {a[:,3].mean():.4f}\n", flush=True)
        json.dump(salida, open("resultados_ei3d.json", "w"), indent=1)

    p1 = salida["sello"]["anterior"]
    p2 = salida["sello"]["vigente"]
    p3 = p1 - salida["barajado"]["anterior"]
    print("=" * 74)
    print(f"  P-1 compara turnos: ANTERIOR(sello) = {p1:.4f}  "
          f"{'CUMPLE → usa el orden' if p1 >= 0.80 else 'NO CUMPLE → era tabla de slots'}")
    print(f"  P-2 la vigente aguanta: {p2:.4f}  {'CUMPLE' if p2 >= 0.80 else 'NO CUMPLE'}")
    print(f"  P-3 viene de la relacion: sello − barajado = {p3:+.4f}  "
          f"{'CUMPLE' if p3 >= 0.30 else 'NO CUMPLE'}")
    print(f"  referencia E-I3c (turnos fijos): ANTERIOR sello 0,97 · ninguno 0,2917")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
