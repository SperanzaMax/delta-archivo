"""¿Qué aprendieron las proyecciones de `attnp`? · 2026-09-08

Nace de un resultado inesperado. En el paso 6.000 `ap3` va en `nose_rel` 0,3322 y PLANO desde el
3.000, contra 0,7642 de `at3` y 0,6045 del kernel 3. Es peor que todos, y había que separar dos
causas antes de dejar correr 20.000 pasos más.

  (a) el weight decay se comió las proyecciones. El §6 del prereg lo anticipó: el atractor del decay
      sobre una identidad es la matriz NULA, y con `wqr = wkr = 0` la similitud es cero, el softmax
      queda uniforme y la lectura recibe el promedio del pasado, sin información.
  (b) el gradiente las movió de verdad, y a algún lado que no ayuda.

El control que las separa no hay que construirlo. Las proyecciones de los bloques 1 a 3 NO reciben
gradiente, porque la lectura entra sólo en el bloque 0, así que son la trayectoria del decay puro.
Cualquier diferencia entre el bloque 0 y ellos es gradiente.

    python salud_attnp.py ckpts/ap3_s0.pkl
"""
import os
import pickle
import sys

import numpy as np
import jax, jax.numpy as jnp

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
import modelo as M
import conf_ckpt

N, T = int(os.environ.get("N", "64")), 24


def analizar(ruta):
    b = pickle.load(open(ruta, "rb"))
    p, cfg = b["params"], b["config"]
    conf_ckpt.aplicar(cfg)
    D = p["emb"].shape[1]
    I = np.eye(D)
    print(f"\n{'=' * 78}\n{os.path.basename(ruta)} · {conf_ckpt.descripcion(cfg)}")
    print(f"\n{'bloque':>7}{'|wqr|':>10}{'|wkr|':>10}{'|wvr|':>10}{'dist a I':>11}{'diag med':>10}")
    for i, blk in enumerate(p["blocks"]):
        wq, wk, wv = (np.asarray(blk[k]) for k in ("wqr", "wkr", "wvr"))
        print(f"{i:>7}{np.linalg.norm(wq):10.4f}{np.linalg.norm(wk):10.4f}"
              f"{np.linalg.norm(wv):10.4f}{np.linalg.norm(wq - I):11.4f}{np.diag(wq).mean():10.4f}")
    print(f"{'identidad':>7}{np.linalg.norm(I):10.4f}{np.linalg.norm(I):10.4f}"
          f"{np.linalg.norm(I):10.4f}{0.0:11.4f}{1.0:10.4f}")

    d01 = np.abs(np.asarray(p["blocks"][0]["wqr"]) - np.asarray(p["blocks"][1]["wqr"])).max()
    print(f"\n  bloque 0 contra bloque 1 (decay puro), diferencia maxima {d01:.6f}")
    print("  -> " + ("GRADIENTE, el diseño no esta roto" if d01 > 1e-4 else
                     "SOLO DECAY, las proyecciones nunca se entrenaron"))

    rng = np.random.default_rng(5)
    x = M.ln(p["blocks"][0]["ln1"], p["emb"][jnp.array(rng.integers(0, p["emb"].shape[0], (N, T)))])
    blk = p["blocks"][0]
    q, k = x @ blk["wqr"], x @ blk["wkr"]
    sim = jnp.einsum("btd,bsd->bts", q, k) / jnp.sqrt(D)
    sim = jnp.where(jnp.tril(jnp.ones((T, T), bool))[None], sim, -1e9)
    pr = np.asarray(jax.nn.softmax(sim, -1))[:, T - 1, :]
    propia = float(pr[:, T - 1].mean())
    efec = float(np.exp(-(pr * np.log(pr + 1e-12)).sum(-1)).mean())
    print(f"\n  ATENCION RESULTANTE en la posicion de lectura")
    print(f"    peso en la propia posicion  {propia:.6f}   (`attn` sin proyecciones da 0,42-0,47)")
    print(f"    posiciones efectivas        {efec:.2f} de {T}   (`attn` da 9,6-11,3)")
    veredicto = ("COLAPSO A LO LOCAL, el modelo aprendio a NO mirar el contexto" if efec < 3
                 else "atiende al contexto")
    print(f"    -> {veredicto}")
    return {"dist_grad": d01, "propia": propia, "efectivas": efec}


if __name__ == "__main__":
    for r in (sys.argv[1:] or ["ckpts/ap3_s0.pkl"]):
        analizar(r)
