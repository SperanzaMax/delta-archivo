"""¿La query en la posicion de la ENTIDAD depende del token de la RELACION? (2026-08-22)

El eslabon mecanico del camino lateral, medido sobre consultas REALES del idioma en vez de sobre
tokens al azar como en `chequeo_query_conjunta.py`.

`idioma.pregunta()` genera un solo formato: `cual es {art} {sust} de {ent} ?`. Ahi el sustantivo de
la relacion y la entidad quedan **a distancia 2**:

    cual   es   la   clave   de   tienda   ?
     0     1    2      3      4     5      6
                       ^relacion         ^entidad, dos mas adelante

La conv de kernel 3 mira `t`, `t−1` y `t−2`, asi que en la posicion de `tienda` alcanza justo a ver
`clave`. Esa es la razon por la que el camino lateral podria formar una query conjunta entidad x
relacion sin tocar el punto de inyeccion.

La prueba: se **cambia el token de la relacion** en la consulta y se mide cuanto se mueve la query en
la posicion de la entidad.

  · en `pre` tiene que dar **cero exacto** — la query es funcion pura del token de su posicion, y ahi
    el token sigue siendo `tienda`;
  · en `lat` tiene que **moverse**;
  · y como control de alcance, cambiar un token que quede FUERA de la ventana (`cual`, a distancia 5)
    no tiene que mover nada en ninguna de las dos.

El tercero es el que hace que la medicion signifique algo: sin el, un `lat` que dependiera de todo el
contexto daria el mismo resultado y seria `post` con otro nombre.
"""
import argparse
import json
import os
import pickle

import jax
import jax.numpy as jnp
import numpy as np

import datos as DAT
import idioma as I
import modelo as M

AQUI = os.path.dirname(os.path.abspath(__file__))


def query_en(params, cons, donde):
    """(B, T, D) — la query de lectura en cada posicion, con la arquitectura pedida."""
    guardado = {}

    def lectura(h):
        guardado["h"] = h
        return jnp.zeros_like(h)

    M.tronco(params, cons, lectura, 0, donde)
    return guardado["h"] @ params["arch"]["qr"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir-ckpt", default=os.path.join(AQUI, "ckpts", "qc_congelados"))
    ap.add_argument("--unidades", default="p3_s0")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--salida", default=os.path.join(AQUI, "query_conjunta_real_20260822.json"))
    A = ap.parse_args()

    # Los tokens de relacion: el sustantivo de cada una. Son los que se intercambian.
    susts = [I.STOI[s] for s, _v, _a in I.RELACIONES.values()]

    print("¿LA QUERY EN LA POSICION DE LA ENTIDAD MIRA EL TOKEN DE LA RELACION?")
    print("consultas reales · se cambia la relacion y se mide la query en la entidad\n")
    print(f"{'unidad':<8} {'donde':<5} | {'cambiar RELACION':>17} {'cambiar token LEJANO':>21} | {'n':>5}")
    print("-" * 68)

    res = {}
    for uni in A.unidades.split(","):
        ck = os.path.join(A.dir_ckpt, f"{uni}.pkl")
        if not os.path.exists(ck):
            print(f"{uni:<8} sin checkpoint")
            continue
        with open(ck, "rb") as f:
            d = pickle.load(f)
        params = jax.tree_util.tree_map(jnp.asarray, d["params"])
        cfg = d["config"]
        donde = cfg.get("donde", "pre")
        rng = np.random.default_rng(4242 + cfg["semilla"])
        fn = jax.jit(lambda p, c: query_en(p, c, donde))

        d_rel, d_lej, n = [], [], 0
        for _ in range(A.n):
            _s, _c, _t, _m, cons, _p, _tg, _ti = DAT.lote(
                rng, A.batch, nivel=cfg["nivel"], n_hechos=4, n_sesiones=4, p_nose=0.0)
            cons = np.asarray(cons)
            q0 = np.asarray(fn(params, jnp.array(cons)))
            for b in range(len(cons)):
                fila = [t for t in cons[b] if t != DAT.PAD]
                # posicion de la entidad: el ultimo token antes del '?'
                try:
                    iq = fila.index(I.STOI["?"])
                except ValueError:
                    continue
                ient = iq - 1
                irel = ient - 2                       # el sustantivo de la relacion
                if irel < 0 or int(cons[b, irel]) not in susts:
                    continue
                # (a) cambiar la RELACION por otra distinta
                c1 = cons[b].copy()
                otros = [s for s in susts if s != int(c1[irel])]
                c1[irel] = otros[int(rng.integers(0, len(otros)))]
                # (b) control de alcance: cambiar un token FUERA de la ventana de la conv
                c2 = cons[b].copy()
                ilej = ient - 5
                if ilej < 0:
                    continue
                c2[ilej] = (int(c2[ilej]) + 3) % I.V
                qq = np.asarray(fn(params, jnp.array(np.stack([c1, c2]))))
                base = q0[b, ient]
                nb = np.linalg.norm(base) + 1e-9
                d_rel.append(float(np.linalg.norm(qq[0, ient] - base) / nb))
                d_lej.append(float(np.linalg.norm(qq[1, ient] - base) / nb))
                n += 1

        r = {"donde": donde, "n": n,
             "delta_por_relacion": float(np.mean(d_rel)) if d_rel else float("nan"),
             "delta_por_lejano": float(np.mean(d_lej)) if d_lej else float("nan")}
        res[uni] = r
        print(f"{uni:<8} {donde:<5} | {r['delta_por_relacion']:>17.8f} "
              f"{r['delta_por_lejano']:>21.8f} | {n:>5}")

    print("\n" + "-" * 68)
    print("Esperado · pre: 0 en las dos columnas (la query es funcion pura del token de su posicion)")
    print("           lat: > 0 en «relacion» y 0 en «lejano» (ventana de la conv, dos tokens atras)")
    with open(A.salida, "w") as f:
        json.dump({"que_es": "eslabon mecanico del camino lateral", "unidades": res}, f, indent=1)
    print(f"\n-> {A.salida}")


if __name__ == "__main__":
    main()
