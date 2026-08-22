"""E-I4c — forzar la deriva con CAMBIO DE DISTRIBUCION.

`PREREG_EI4C_DISTRIBUCION.md`, SHA 8e051d74..., congelado antes de este archivo.

E-I4 y E-I4b midieron el envejecimiento por ANTIGUEDAD y las dos veces el coseno se quedo arriba de
0,70, o sea del lado donde la teoria del proyecto ya predice que no pasa nada. P-2 —la pregunta de si
el indice CO-ENTRENADO tolera lo que mata al no parametrico— quedo no evaluable las dos veces.

Aca la deriva se produce afinando en otra distribucion, que es lo que hizo R6 afuera y lo que le pasa
de verdad a un modelo desplegado: no envejece por pasos de gradiente, envejece porque lo siguen
entrenando en datos nuevos.

    fase A   6000 pasos con claves sorteadas de [0, 64)
    fase B   se sigue entrenando con claves de [64, 128), y se mide a las edades 0/500/2000/6000

El archivo se escribe con los pesos del final de la fase A y se lee con los de la fase B. La
evaluacion usa hechos de la distribucion A: el archivo es viejo, que es justo la situacion que
interesa. El control de eso es la edad 0 —mismo desajuste de distribucion, cero deriva acumulada—.

POR QUE PARTIR EL VOCABULARIO Y NO CAMBIAR LA CARGA: cambiar `L` mueve `N_ARCH = L + R`, o sea el
tamaño del archivo y el uso de `ord`. Un negativo ahi seria ininterpretable —deriva o cambio de
forma—. Partir las claves deja todas las formas intactas.
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
import ei2_consulta as E
import ei3_orden as E3
import ei4_envejecimiento as E4
from datos import V_E001
from modelos import init_params

PASOS_A = int(os.environ.get("PASOS_A", "6000"))
EDADES = (0, 500, 2000, 6000)
SEMILLAS = (0, 1, 2)
LR = 1e-3
CORTE = V_E001.NK // 2                 # 64: claves [0,64) en la fase A, [64,128) en la fase B


def gen_lote_particion(rng, lo, hi):
    """El lote de E-I3 con las claves restringidas a [lo, hi).

    Se replica `ei3_orden.gen_lote` en vez de importarlo porque lo unico que cambia es de donde salen
    las claves, y hace falta que el resto —valores, revisiones, permutacion del archivo, sellos— siga
    identico para que la unica diferencia entre fases sea la distribucion de claves.
    """
    B, L, R = E3.B, E3.L, E3.R
    voc = V_E001
    n = hi - lo
    assert L <= n, f"L={L} no entra en el tramo [{lo},{hi})"
    keys = lo + np.argsort(rng.random((B, n)), axis=1)[:, :L]
    v1 = rng.integers(0, voc.NV, size=(B, L))
    v2 = (v1[:, :R] + 1 + rng.integers(0, voc.NV - 1, size=(B, R))) % voc.NV
    final = v1.copy(); final[:, :R] = v2

    s1 = np.full((B, 2 * L + 2), voc.PAD, dtype=np.int32)
    s1[:, 0] = voc.BOS; s1[:, 1:2*L+1:2] = voc.K0 + keys; s1[:, 2:2*L+2:2] = voc.V0 + v1
    s1[:, -1] = voc.SEP
    pos1 = np.arange(2, 2 * L + 2, 2)

    s2 = np.full((B, 2 * R + 2), voc.PAD, dtype=np.int32)
    s2[:, 0] = voc.BOS; s2[:, 1:2*R+1:2] = voc.K0 + keys[:, :R]
    s2[:, 2:2*R+2:2] = voc.V0 + v2
    s2[:, -1] = voc.SEP
    pos2 = np.arange(2, 2 * R + 2, 2)

    s3 = np.full((B, L + 2), voc.PAD, dtype=np.int32)
    s3[:, 0] = voc.BOS; s3[:, 1:L+1] = voc.K0 + keys
    y3 = np.full((B, L + 2), -1, dtype=np.int32)
    y3[:, 2:L+2] = voc.V0 + final
    rev = np.zeros((B, L), dtype=np.float32); rev[:, :R] = 1.0

    perm = np.argsort(rng.random((B, E3.N_ARCH)), axis=1).astype(np.int32)
    falso = np.argsort(rng.random((B, E3.N_ARCH)), axis=1).astype(np.int32)
    return (jnp.array(s1), pos1, jnp.array(s2), pos2, jnp.array(s3), jnp.array(y3),
            jnp.array(rev), jnp.array(perm), jnp.array(falso))


def correr(semilla):
    params = init_params(semilla, E.KIND)
    params["extra"] = E3.init_extra(semilla)
    rng = np.random.default_rng(5000 + semilla)
    sched = optax.warmup_constant_schedule(0.0, LR, 100)
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(sched, weight_decay=0.01))
    state = opt.init(params)

    @partial(jax.jit, static_argnames="m")
    def paso(params, state, lote, m):
        (l, aux), g = jax.value_and_grad(E3.loss_fn, has_aux=True)(params, lote, m)
        up, state = opt.update(g, state, params)
        return optax.apply_updates(params, up), state, l, aux

    t0 = time.time()
    for s in range(1, PASOS_A + 1):
        params, state, l, aux = paso(params, state, gen_lote_particion(rng, 0, CORTE), "sello")
        if s % 2000 == 0:
            a, ar, an = (float(v) for v in aux)
            print(f"    [s{semilla}] fase A paso {s} loss {float(l):.4f} acc {a:.4f} "
                  f"(rev {ar:.4f}) ({time.time()-t0:.0f}s)", flush=True)

    params_A = jax.tree_util.tree_map(lambda x: x.copy(), params)
    fotos = {0: params_A}

    for s in range(1, max(EDADES) + 1):
        params, state, l, aux = paso(params, state, gen_lote_particion(rng, CORTE, V_E001.NK),
                                     "sello")
        if s in EDADES:
            fotos[s] = jax.tree_util.tree_map(lambda x: x.copy(), params)
        if s % 2000 == 0:
            print(f"    [s{semilla}] fase B paso {s} loss {float(l):.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    # Evaluacion: archivo escrito con los pesos de la fase A, leido con los de cada edad, sobre
    # hechos de la distribucion A.
    rng_ev = np.random.default_rng(90000 + semilla)
    lotes = [gen_lote_particion(rng_ev, 0, CORTE) for _ in range(8)]
    out = {}
    for edad in EDADES:
        accs, revs, nos, coss = [], [], [], []
        for lote in lotes:
            a, ar, an, cos = E4.evaluar(fotos[edad], params_A, lote)
            accs.append(a); revs.append(ar); nos.append(an); coss.append(cos)
        out[edad] = {"acc": float(np.mean(accs)), "rev": float(np.mean(revs)),
                     "no_rev": float(np.mean(nos)), "cos": float(np.mean(coss))}
        print(f"    [s{semilla}] edad {edad:>5}  cos {out[edad]['cos']:.4f}  "
              f"rev {out[edad]['rev']:.4f}  no-rev {out[edad]['no_rev']:.4f}", flush=True)
    return out


def main():
    print(f"E-I4c · deriva por CAMBIO DE DISTRIBUCION · claves [0,{CORTE}) -> [{CORTE},"
          f"{V_E001.NK}) · fase A {PASOS_A} pasos · edades {EDADES} · {len(SEMILLAS)} semillas\n")
    todo = {}
    for semilla in SEMILLAS:
        todo[semilla] = correr(semilla)
        json.dump(todo, open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                          "resultados_ei4c.json"), "w"), indent=1)

    print("\n" + "=" * 70)
    print(f"{'edad':>6} {'cos':>8} {'rev':>8} {'no_rev':>8}   (media de las semillas)")
    med = {}
    for edad in EDADES:
        med[edad] = {k: float(np.mean([todo[s][edad][k] for s in SEMILLAS]))
                     for k in ("cos", "rev", "no_rev")}
        print(f"{edad:>6} {med[edad]['cos']:>8.4f} {med[edad]['rev']:>8.4f} "
              f"{med[edad]['no_rev']:>8.4f}")

    print("\nPor semilla, en la edad maxima (la bimodalidad es parte del fenomeno):")
    for s in SEMILLAS:
        e = todo[s][max(EDADES)]
        print(f"  s{s}: cos {e['cos']:.4f} · rev {e['rev']:.4f}")

    cos_max = med[max(EDADES)]["cos"]
    caida = med[0]["rev"] - med[max(EDADES)]["rev"]
    print("\nPREDICCIONES")
    print(f"  P-1 (bloqueante) cos <= 0,70 en la edad maxima: {cos_max:.4f}  "
          f"{'CUMPLE' if cos_max <= 0.70 else 'NO CUMPLE -> sin poder de resolucion'}")
    if cos_max <= 0.70:
        print(f"  P-2  caida en revisadas = {caida:+.4f}  "
              f"{'CUMPLE (cae >= 0,10)' if caida >= 0.10 else 'NO CUMPLE -> el co-entrenado TOLERA'}")
    else:
        print("  P-2  NO EVALUABLE (mismo desenlace que E-I4 y E-I4b)")
    c2000 = med.get(2000, {}).get("cos", float("nan"))
    print(f"  P-3  cos a 2000 pasos = {c2000:.4f} contra 0,9067 de E-I4b  "
          f"{'CUMPLE' if c2000 < 0.9067 else 'NO CUMPLE'}")
    print("\nEl veredicto lo escribe una persona.")


if __name__ == "__main__":
    main()
