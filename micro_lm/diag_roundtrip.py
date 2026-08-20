#!/usr/bin/env python3
"""DIAGNÓSTICO del round-trip — no es una prediccion, no decide nada.

`PREREG_ROUNDTRIP.md` archivo el experimento sin interpretar porque RT-3 fallo 8/8. Antes de
escribir una sola linea de interpretacion hay que separar dos lecturas del mismo numero (H = ln 4
exacto, o sea posterior UNIFORME sobre las candidatas, en las tres unidades de nivel 3):

  (A) el modelo no condiciona en la entidad -> la marginalizacion sobre la entidad de origen es real
      y `P(V)` dejo de depender de `E`. Es propiedad del MODELO.
  (B) sustituir la entidad saca la consulta de distribucion en los niveles dificiles y el logit deja
      de ser comparable. Es propiedad del INSTRUMENTO.

La medicion que las separa: **¿cambia la RESPUESTA (el argmax) al preguntar por otra entidad?**

  · si contesta lo MISMO para todas las candidatas -> (A): la entidad no entra en la decision.
  · si contesta DISTINTO segun la entidad pero el logit de `X` no ordena -> (B): la respuesta si
    depende de `E`, y lo que falla es leer esa dependencia por el logit de un token fijo.

Se reporta ademas cuantas veces la respuesta a la consulta contrafactual es **el valor de esa otra
entidad**, que es lo que haria un modelo que si separa las entradas.
"""
import os, sys, json, pickle, argparse
import numpy as np
import jax, jax.numpy as jnp

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
import idioma as I, datos as DAT, modelo as M
import sonda_roundtrip as RT

NOSE = I.STOI["NOSE"]


def respuestas(params, ses, cortes, turnos, cons_var, mask, pos):
    """El argmax (sin NOSE) de CADA variante de la consulta. (J, B)."""
    archivo = M.escribir(params, ses, cortes)

    def una(cj):
        lg, _ = M.responder_con_abst(params, archivo, turnos, cj, mask)
        lg = jnp.take_along_axis(lg, pos[:, None, None], axis=1)[:, 0, :]
        return lg.at[:, NOSE].set(-jnp.inf).argmax(-1)

    return jax.vmap(una)(cons_var)


ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=8)
ap.add_argument("--batch", type=int, default=64)
ap.add_argument("--p-nose", type=float, default=0.4)
ap.add_argument("--unidades", default="1_s0,2_s0,3_s0,3_s1,4_s0,4_s2")
ap.add_argument("--dir-ckpt", default=os.path.join(AQUI, "ckpts", "rt_congelados"))
ap.add_argument("--salida", default=os.path.join(AQUI, "diag_roundtrip_20260820.json"))
A = ap.parse_args()

print("DIAGNÓSTICO del round-trip · ¿cambia la RESPUESTA al cambiar la entidad?")
print(f"{A.n * A.batch} muestras por unidad · p_nose={A.p_nose}\n")
print(f"{'unidad':<7} {'igual_todas':>12} {'distintas':>10} {'resp=valor de E\'':>17} {'H_media':>8}")
print("-" * 60)

res = {}
for u in A.unidades.split(","):
    ck = os.path.join(A.dir_ckpt, f"c{u}.pkl")
    if not os.path.exists(ck):
        print(f"c{u}: sin checkpoint"); continue
    with open(ck, "rb") as f:
        d = pickle.load(f)
    params = jax.tree_util.tree_map(jnp.asarray, d["params"])
    nivel, semilla = int(u[0]), int(u.split("_s")[1])
    rng = np.random.default_rng(RT.SEM_PRUEBA + semilla)
    fn = jax.jit(respuestas)

    IG, DIS, PROP, NV = [], [], [], []
    for _ in range(A.n):
        sal = DAT.lote(rng, A.batch, nivel=nivel, n_hechos=4, n_sesiones=4, p_vieja=0.35,
                       p_nose=A.p_nose, con_meta=True, con_origen=True)
        ses, cortes, turnos, mask, cons, pos, tgt, tipo, meta, _o, _h = sal
        pe = RT.pos_entidad(cons)
        tok, val = RT.candidatas(cons, pe, meta)
        cv = RT.variantes(cons, pe, tok)
        R = np.asarray(fn(params, jnp.array(ses), jnp.array(cortes), jnp.array(turnos),
                          jnp.array(cv), jnp.array(np.asarray(mask)), jnp.array(pos))).T  # (B, J)

        for b in range(len(pe)):
            v = val[b]
            rs = R[b][v]
            IG.append(np.all(rs == rs[0]))               # contesta lo mismo a todas
            NV.append(len(np.unique(rs)))                # cuantas respuestas distintas da
            # ¿la respuesta a la consulta por E' es un valor DE E'?
            ents = {}
            m = meta[b]
            if m["hecho"]:
                ents[I.STOI[m["hecho"]["ent"]]] = set(m["hecho"]["versiones"])
            for o in m["otros"]:
                ents[I.STOI[o["ent"]]] = set(o["versiones"])
            ok = tot = 0
            for j in range(1, len(v)):
                if not v[j]:
                    continue
                e = int(tok[b, j])
                if e in ents:
                    tot += 1
                    ok += I.ITOS[R[b, j]] in ents[e]
            if tot:
                PROP.append(ok / tot)

    ig = float(np.mean(IG)); nv = float(np.mean(NV)); pr = float(np.mean(PROP)) if PROP else np.nan
    res[u] = {"igual_todas": ig, "respuestas_distintas_media": nv, "resp_es_valor_de_Ep": pr,
              "n": len(IG)}
    print(f"c{u:<6} {ig:>12.4f} {nv:>10.2f} {pr:>17.4f} {'':>8}")

json.dump({"que_es": "diagnostico post-hoc del PREREG_ROUNDTRIP, sin criterio de decision",
           "unidades": res}, open(A.salida, "w"), indent=1, default=float)
print(f"\n-> {A.salida}")
print("Lectura: `igual_todas` alto => el modelo NO condiciona en la entidad (A).")
print("         `igual_todas` bajo con RT-3 fallando => el instrumento no lee esa dependencia (B).")
