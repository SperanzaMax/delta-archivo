#!/usr/bin/env python3
"""Chequeo de instrumento de `--blanco` (A5), ANTES de gastar una sola unidad de pool.

  B-1  con `--blanco ausencia` (el default) la perdida da EXACTAMENTE lo mismo que antes del cambio.
       Es la guarda que protege los controles ya corridos, igual que K-5 en lat2 y A-2 en el slot.
  B-2  con `--blanco error` la perdida es DISTINTA. Si diera igual, el flag no hace nada.
  B-3  el blanco `error` vale 1 SIEMPRE que tgt==NOSE, que es su definicion.
  B-4  el gradiente llega a la cabeza con los dos blancos (el chequeo que en lat2 destapo el decay).
"""
import pickle, sys, numpy as np, jax, jax.numpy as jnp, optax
sys.path.insert(0, "/home/maxi/Documentos/Nuevo Transformer/delta-archivo/micro_lm")
import entrenar as E, datos as DAT, modelo as M

NOSE = E.NOSE
ck = pickle.load(open("ckpts/p3_s1.pkl", "rb"))
params = jax.tree_util.tree_map(jnp.asarray, ck["params"])
E._DONDE, E._ABST = ck["config"].get("donde", "pre"), ck["config"].get("abst", "cabeza")
rng = np.random.default_rng(999)
ses, cor, tur, mask, cons, pos, tgt, tipo = DAT.lote(rng, 64, nivel=ck["config"]["nivel"],
                                                     n_hechos=4, n_sesiones=4, p_nose=0.4)
A = [jnp.array(x) for x in (ses, cor, tur, mask, cons, pos, tgt)]

def corre(blanco):
    E._BLANCO = blanco
    (l, acc), g = jax.value_and_grad(E.perdida_cabeza, has_aux=True)(params, *A)
    gn = float(jnp.linalg.norm(g["abst"]["w"]))
    return float(l), float(acc), gn

# referencia: la formula ANTERIOR al cambio, escrita a mano
def referencia():
    lg, a = E._partes(params, *A[:-1])
    es_nose = (A[-1] == NOSE).astype(jnp.float32)
    bce = optax.sigmoid_binary_cross_entropy(a, es_nose).mean()
    lg_v = lg.at[:, NOSE].set(-1e9)
    ce = optax.softmax_cross_entropy_with_integer_labels(lg_v, A[-1])
    hay = 1.0 - es_nose
    return float(bce + (ce * hay).sum() / jnp.maximum(hay.sum(), 1.0))

ref = referencia()
la, aa, ga = corre("ausencia")
le, ae, ge = corre("error")
print(f"B-1  ausencia {la:.10f}  vs referencia {ref:.10f}   "
      f"maxabs {abs(la-ref):.2e}   {'OK' if abs(la-ref) < 1e-6 else 'FALLA'}")
print(f"B-2  error    {le:.10f}   distinta de ausencia: {'OK' if abs(le-la) > 1e-4 else 'FALLA'}")

E._BLANCO = "error"
lg, a = E._partes(params, *A[:-1])
arg = np.asarray(lg.at[:, NOSE].set(-1e9).argmax(-1)); t = np.asarray(A[-1])
mal = (arg != t); sinr = (t == NOSE)
print(f"B-3  blanco=1 en todas las tgt==NOSE: {'OK' if mal[sinr].all() else 'FALLA'}"
      f"   ({sinr.sum()} casos)   ·  tasa global del blanco {mal.mean():.4f}"
      f"   ·  tasa 'ausencia' {sinr.mean():.4f}")
print(f"B-4  |grad cabeza|  ausencia {ga:.6f}  error {ge:.6f}   "
      f"{'OK' if ga > 1e-9 and ge > 1e-9 else 'FALLA'}")
