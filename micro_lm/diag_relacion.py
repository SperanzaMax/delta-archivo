#!/usr/bin/env python3
"""¿El modelo usa la RELACION como atajo y por eso ignora la entidad?

Sale del diagnostico del round-trip: en nivel 3 el modelo contesta lo MISMO para las 4 entidades
(igual_todas 0,97). Hay dos lecturas y hay que separarlas antes de interpretar nada:

  (A) marginaliza sobre la entidad -> `P(V)` dejo de depender de `E`, como dice la lectura de la
      disyuncion;
  (A') usa la RELACION como atajo -> la pregunta lleva relacion + entidad, y con 4 hechos sobre 6
      relaciones la relacion sola casi siempre alcanza para encontrar el hecho. Sustituir la entidad
      no cambia nada porque la entidad nunca entro en la decision, pero NO porque se promediaran
      entradas: porque hay una pista mas barata.

La cuenta que hace sospechar (A'): P(la relacion preguntada este repetida) = 1 - (5/6)^3 = 0,4213, y
si ahi elige a ciegas entre las dos candidatas el error seria 0,4213/2 = 0,2107 — que es el
`err_identidad` medido (0,19-0,21).

La medicion: separar acierto y `err_identidad` segun si la relacion preguntada es UNICA en el
episodio o esta REPETIDA. Si (A') gobierna, el error tiene que concentrarse casi entero en las
repetidas y casi desaparecer en las unicas.
"""
import os, sys, json, pickle, argparse
from functools import partial
import numpy as np
import jax, jax.numpy as jnp

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
import idioma as I, datos as DAT, modelo as M
import sonda_roundtrip as RT

NOSE = I.STOI["NOSE"]


def responder(params, ses, cortes, turnos, cons, mask, pos, donde="pre"):
    archivo = M.escribir(params, ses, cortes)
    lg, _ = M.responder_con_abst(params, archivo, turnos, cons, mask, donde=donde)
    lg = jnp.take_along_axis(lg, pos[:, None, None], axis=1)[:, 0, :]
    return lg.at[:, NOSE].set(-jnp.inf).argmax(-1)


ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=16)
ap.add_argument("--batch", type=int, default=64)
ap.add_argument("--p-nose", type=float, default=0.0)   # solo preguntas CON respuesta
ap.add_argument("--unidades", default="1_s0,2_s0,3_s0,3_s1,3_s2,4_s0,4_s1,4_s2")
ap.add_argument("--dir-ckpt", default=os.path.join(AQUI, "ckpts", "rt_congelados"))
ap.add_argument("--prefijo", default="c",
                help="familia de la corrida. Estaba escrito a mano como 'c' (la campania de la "
                     "cabeza); la campania de la query conjunta usa 'p' (pre) y 'q' (post)")
ap.add_argument("--salida", default=os.path.join(AQUI, "diag_relacion_20260820.json"))
A = ap.parse_args()

print("¿ES LA RELACION EL ATAJO? · acierto y err_identidad segun relacion unica o repetida")
print(f"{A.n * A.batch} muestras por unidad · p_nose={A.p_nose}\n")
print(f"{'unidad':<7} | {'n_unica':>7} {'ac_unica':>9} {'ident_unica':>12} | "
      f"{'n_rep':>6} {'ac_rep':>7} {'ident_rep':>10} | {'P(rep)':>7}")
print("-" * 82)

res = {}
for u in A.unidades.split(","):
    ck = os.path.join(A.dir_ckpt, f"{A.prefijo}{u}.pkl")
    if not os.path.exists(ck):
        print(f"c{u}: sin checkpoint"); continue
    with open(ck, "rb") as f:
        d = pickle.load(f)
    params = jax.tree_util.tree_map(jnp.asarray, d["params"])
    # La posicion de la lectura se lee DEL CHECKPOINT, no de un flag: la sonda tiene que correr la
    # arquitectura con la que la unidad se entreno, y un flag a mano es justo la clase de cosa que se
    # olvida al medir media campania. Los ckpts anteriores al 22-ago no la traen y todos son `pre`.
    donde = d.get("config", {}).get("donde", "pre")
    # el nombre puede traer sufijo de paso (`4_s0_p20000`): la semilla es el numero pegado a `_s`
    nivel, semilla = int(u[0]), int(u.split("_s")[1].split("_")[0])
    rng = np.random.default_rng(RT.SEM_PRUEBA + semilla)
    fn = jax.jit(partial(responder, donde=donde))

    REP, OK, ID = [], [], []
    for _ in range(A.n):
        sal = DAT.lote(rng, A.batch, nivel=nivel, n_hechos=4, n_sesiones=4, p_vieja=0.35,
                       p_nose=A.p_nose, con_meta=True, con_origen=True)
        ses, cortes, turnos, mask, cons, pos, tgt, tipo, meta, _o, _h = sal
        X = np.asarray(fn(params, jnp.array(ses), jnp.array(cortes), jnp.array(turnos),
                          jnp.array(cons), jnp.array(np.asarray(mask)), jnp.array(pos)))
        tgt = np.asarray(tgt); tipo = np.asarray(tipo)
        for b in range(len(X)):
            m = meta[b]
            if tipo[b] >= 2 or not m["hecho"]:
                continue
            rel = m["hecho"]["rel"]
            rep = any(o["rel"] == rel for o in m["otros"])
            x = I.ITOS[X[b]]
            ident = (X[b] != tgt[b]) and any(x in o["versiones"] for o in m["otros"])
            REP.append(rep); OK.append(X[b] == tgt[b]); ID.append(ident)

    REP, OK, ID = np.array(REP), np.array(OK), np.array(ID)
    r = {"n": int(len(REP)), "P_rep": float(REP.mean()),
         "n_unica": int((~REP).sum()), "ac_unica": float(OK[~REP].mean()),
         "ident_unica": float(ID[~REP].mean()),
         "n_rep": int(REP.sum()), "ac_rep": float(OK[REP].mean()),
         "ident_rep": float(ID[REP].mean())}
    res[u] = r
    # El prefijo salia escrito a mano como "c" tambien aca, asi que la familia `q` se imprimia como
    # `c3_s0`. El checkpoint leido era el correcto —eso ya usa `A.prefijo`—, pero una tabla que
    # miente el nombre de la unidad es justo lo que despues se copia a un informe.
    print(f"{A.prefijo}{u:<6} | {r['n_unica']:>7} {r['ac_unica']:>9.4f} {r['ident_unica']:>12.4f} | "
          f"{r['n_rep']:>6} {r['ac_rep']:>7.4f} {r['ident_rep']:>10.4f} | {r['P_rep']:>7.4f}")

json.dump({"que_es": "diagnostico post-hoc, sin criterio de decision", "unidades": res},
          open(A.salida, "w"), indent=1, default=float)
print(f"\n-> {A.salida}")
print("Si (A') gobierna: `ident_unica` ~ 0 y `ident_rep` ~ 0,5. Si el error esta repartido parejo,")
print("la relacion NO es el atajo y la lectura de la marginalizacion sigue en pie.")
