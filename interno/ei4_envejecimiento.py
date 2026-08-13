"""E-I4 — EL ARCHIVO ENVEJECE: la deriva de R5.2, ahora adentro de la red.

Todo el brazo interno tiene el mismo agujero, declarado en los limites de E-I2 y E-I3: el archivo se
recomputa fresco en cada paso. Eso lo vuelve una memoria de trabajo elegante, no una memoria
persistente -- las claves archivadas siempre estan escritas con los mismos pesos que las leen, y por
construccion no puede haber stale index.

El brazo NO parametrico ya midio el fenomeno que falta, y es el que decide si todo esto sirve:
  R5.1  la memoria persistente funciona mientras cos(marco de hoy, marco de escritura) >~ 0,7;
        degrada entre 0,7 y 0,4; muere debajo.
  R5.2  entrenando delta puro desde cero, el coseno cae 1,000 -> 0,727 en 25 pasos.
  R6    sobre un modelo YA entrenado que se afina, a 400 pasos el coseno sigue en 0,882.
La conclusion de R6 fue que la memoria persistente es viable sobre un modelo entrenado. Nunca se
comprobo DENTRO de la red, con el indice co-entrenado que ahora sabemos que funciona.

DISEÑO. No se cambia el entrenamiento: se entrena exactamente como E-I3 (condicion `sello`, que es la
que resuelve la tarea) y se guardan fotos de los pesos en el camino. Despues, en EVALUACION, se
escribe el archivo con los pesos VIEJOS y se lee con los pesos de hoy:

    edad 0     escribe y lee el mismo modelo             (control: es E-I3)
    edad 25    escribe el modelo de hace 25 pasos        (donde R5.2 medio 0,727 desde cero)
    edad 100   ...
    edad 400   escribe el modelo de hace 400 pasos       (donde R6 medio 0,882 afinando)

Y se mide, en la misma corrida, el coseno medio entre la clave que produce el modelo viejo y la que
produce el modelo de hoy para la misma entrada. Eso permite poner cada celda de accuracy sobre la
curva de tolerancia de R5.1 y ver si el umbral ~0,7 -- medido afuera, con encoder congelado y kNN --
tambien manda adentro, con un lector co-entrenado y softmax sobre 9 entradas.

PREDICCIONES, comprometidas antes del dato:
  P-1  (bloqueante) el coseno cae con la edad de forma monotona: cos(400) < cos(25) < cos(0) = 1.
       Es control de sanidad del instrumento, y PUEDE fallar: si el modelo ya converge y sus pesos
       casi no se mueven en la ventana final, los cuatro cosenos dan ~1 y el experimento no mide
       nada. Si pasa eso, hay que rehacerlo tomando las fotos temprano en el entrenamiento.
  P-2  la accuracy en claves revisadas cae de forma monotona con la edad, y la caida entre edad 0 y
       edad 400 es >= 0,05. Si NO cae, el resultado es mejor de lo esperado y hay que decirlo asi:
       el indice co-entrenado tolera el envejecimiento que el no parametrico no toleraba.
  P-3  la degradacion aparece ANTES en las claves revisadas que en las de una sola version. Razon
       mecanica: distinguir dos versiones de la misma clave exige mas precision en la clave que
       distinguir claves distintas, asi que el mismo desplazamiento del marco rompe primero lo mas
       fino. Si se cumple, el envejecimiento no degrada parejo -- se come primero la funcion que le
       costo mas conseguir al programa.

LO QUE NO PRUEBA. Las fotos se toman al final del entrenamiento, donde el modelo ya converge: es el
regimen de R6 (modelo entrenado que se afina), no el de R5.2 (aprendizaje inicial). Es el regimen
realista para una memoria desplegada, pero es UN regimen. Tampoco hay reindexado ni correccion de
deriva: se mide el daño crudo.
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
import ei3_orden as E3
from modelos import D, init_params, ln

PASOS = int(os.environ.get("PASOS", "4000"))
LR = 1e-3
BLOQUE = 0
SEMILLAS = (0, 1, 2, 3, 4)
EDADES = (0, 25, 100, 400)
N_ARCH = E3.N_ARCH


def archivo(params_w, lote):
    """Las claves y valores archivados, escritos con los pesos que se le pasen."""
    s1, pos1, s2, pos2 = lote[0], lote[1], lote[2], lote[3]
    perm = lote[7]
    ex = params_w["extra"]
    h1 = E.tronco(params_w, s1)[:, pos1, :]
    h2 = E.tronco(params_w, s2)[:, pos2, :]
    hw = jnp.concatenate([h1, h2], axis=1)
    idx = jnp.broadcast_to(perm[:, :, None], (hw.shape[0], N_ARCH, D))
    hw = jnp.take_along_axis(hw, idx, axis=1)
    return hw @ ex["kw"] + ex["ord"][perm], hw @ ex["vw"]


def evaluar(params_r, params_w, lote):
    """Escribe el archivo con params_w, lo lee con params_r. Devuelve accs y el coseno del marco."""
    ak, av = archivo(params_w, lote)
    ex = params_r["extra"]

    def lectura(h):
        q = h @ ex["qr"]
        sim = jnp.einsum("btd,bnd->btn", q, ak) / jnp.sqrt(D)
        return jnp.einsum("btn,bnd->btd", jax.nn.softmax(sim, axis=-1), av) @ ex["wo"]

    s3, y3, rev = lote[4], lote[5], lote[6]
    h3 = E.tronco(params_r, s3, lectura, BLOQUE)
    logits = ln(params_r["ln_f"], h3) @ params_r["head"]["w"] + params_r["head"]["b"]

    mask = y3 >= 0
    yl = jnp.where(mask, y3, 0)
    ok = (logits.argmax(-1) == yl) * mask
    okq = ok[:, 2:]
    acc = ok.sum() / mask.sum()
    acc_rev = (okq * rev).sum() / jnp.maximum(rev.sum(), 1)
    acc_no = (okq * (1 - rev)).sum() / jnp.maximum((1 - rev).sum(), 1)

    ak_hoy, _ = archivo(params_r, lote)
    cos = jnp.mean(jnp.sum(ak * ak_hoy, -1) /
                   (jnp.linalg.norm(ak, axis=-1) * jnp.linalg.norm(ak_hoy, axis=-1) + 1e-8))
    return float(acc), float(acc_rev), float(acc_no), float(cos)


def entrenar_con_fotos(semilla, pasos=PASOS):
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

    fotos = {}
    cortes = {pasos - e: e for e in EDADES}
    t0 = time.time()
    for s in range(1, pasos + 1):
        params, state, l, aux = paso(params, state, E3.gen_lote(rng), "sello")
        if s in cortes:
            fotos[cortes[s]] = jax.tree_util.tree_map(lambda x: x.copy(), params)
        if s % 1000 == 0:
            a, ar, an = (float(v) for v in aux)
            print(f"    [s{semilla}] paso {s:4d} loss {float(l):.4f} acc {a:.4f} "
                  f"(rev {ar:.4f} · no-rev {an:.4f}) ({time.time()-t0:.0f}s)", flush=True)
    return params, fotos


def main():
    print(f"E-I4 · el archivo se escribe con pesos viejos y se lee con los de hoy · "
          f"edades {EDADES} · {len(SEMILLAS)} semillas\n"
          f"referencia externa: R5.1 umbral cos ~0,7 · R5.2 0,727 a 25 pasos desde cero · "
          f"R6 0,882 a 400 pasos afinando\n", flush=True)
    acum = {e: [] for e in EDADES}
    for semilla in SEMILLAS:
        params, fotos = entrenar_con_fotos(semilla)
        ev = np.random.default_rng(99000 + semilla)
        for edad in EDADES:
            rs = [evaluar(params, fotos[edad], E3.gen_lote(ev)) for _ in range(8)]
            m = np.mean(rs, axis=0)
            acum[edad].append(m)
            print(f"  s{semilla} edad {edad:3d} → acc {m[0]:.4f} (rev {m[1]:.4f} · "
                  f"no-rev {m[2]:.4f}) · cos {m[3]:.4f}", flush=True)
        json.dump({str(e): np.array(v).tolist() for e, v in acum.items()},
                  open("resultados_ei4.json", "w"), indent=1)

    print("\n" + "=" * 74)
    res = {}
    for edad in EDADES:
        a = np.array(acum[edad])
        res[edad] = {"acc": a[:, 0].mean(), "rev": a[:, 1].mean(), "no_rev": a[:, 2].mean(),
                     "cos": a[:, 3].mean(), "sd_rev": a[:, 1].std(ddof=1)}
        print(f"  edad {edad:3d} → cos {res[edad]['cos']:.4f} · revisadas {res[edad]['rev']:.4f} "
              f"(sd {res[edad]['sd_rev']:.4f}) · una version {res[edad]['no_rev']:.4f}")
    print("-" * 74)
    cos_mono = all(res[a]["cos"] >= res[b]["cos"] - 1e-6 for a, b in zip(EDADES, EDADES[1:]))
    caida_rev = res[0]["rev"] - res[EDADES[-1]]["rev"]
    caida_no = res[0]["no_rev"] - res[EDADES[-1]]["no_rev"]
    print(f"  P-1 el instrumento mide: coseno monotono decreciente  "
          f"{'CUMPLE' if cos_mono else 'NO CUMPLE — fotos demasiado juntas, rehacer temprano'}")
    print(f"  P-2 caida en revisadas (edad 0 → {EDADES[-1]}) = {caida_rev:+.4f}  "
          f"{'CUMPLE' if caida_rev >= 0.05 else 'NO CUMPLE — el indice co-entrenado TOLERA envejecer'}")
    print(f"  P-3 pega primero en lo fino: caida rev {caida_rev:+.4f} vs una version {caida_no:+.4f}  "
          f"{'CUMPLE' if caida_rev > caida_no else 'NO CUMPLE'}")
    print("=" * 74)
    json.dump({str(k): {kk: float(vv) for kk, vv in v.items()} for k, v in res.items()},
              open("resumen_ei4.json", "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
