"""¿La entrada del hecho preguntado ESTA en el archivo, y en que puesto queda?

    python rank_hecho.py ckpts/n4_s0.pkl --n 4000

Cierra el limite declarado el 16-ago: las sondas mostraron que el hecho propio no se recupera en
NINGUNA version, y desde afuera «no se escribio» y «se escribio y la lectura no lo alcanza» se ven
igual. Con el mapeo enunciado->hecho (`con_origen`) se puede mirar directamente en que posicion del
ranking de lectura quedo la entrada de ese hecho.

Se mide en la posicion de MAXIMO FOCO, no en la de respuesta: el 16-ago se comprobo que el modelo
concentra en posiciones intermedias y que en `pos_q` la distribucion ya esta difusa.

  rank 0  -> la entrada del hecho preguntado GANA la lectura
  rank alto -> esta en el archivo pero pierde  -> problema de LECTURA
  ausente -> nunca se escribio                 -> problema de ESCRITURA
"""
import argparse, pickle
import numpy as np, jax, jax.numpy as jnp
import datos as DAT, idioma as I, modelo as M, entrenar as E
from ser import clasificar

ap = argparse.ArgumentParser(); ap.add_argument("pesos")
ap.add_argument("--n", type=int, default=4000); ap.add_argument("--B", type=int, default=64)
a = ap.parse_args()
bulto = pickle.load(open(a.pesos, "rb"))
params = jax.tree_util.tree_map(jnp.asarray, bulto["params"]); nivel = bulto["config"]["nivel"]
rng = np.random.default_rng(31415)
acc = {g: {"n": 0, "ausente": 0, "rank0": 0, "ranks": []} for g in ("err_identidad", "acierto")}
vistos = 0
while vistos < a.n:
    B = min(a.B, a.n - vistos)
    ses, cortes, turnos, mask, cons, pos, tgt, tipo, meta, oarch, hq = DAT.lote(
        rng, B, nivel=nivel, n_hechos=4, n_sesiones=4, p_nose=0.0, con_meta=True, con_origen=True)
    jses, jcor, jtur = jnp.array(ses), jnp.array(cortes), jnp.array(turnos)
    jmask, jcons, jpos = jnp.array(mask), jnp.array(cons), jnp.array(pos)
    pred = np.asarray(E.predecir(params, jses, jcor, jtur, jmask, jcons, jpos))
    ar = params["arch"]
    ak = M.escribir(params, jses, jcor) @ ar["kw"] + ar["ord"][jtur]
    h = params["emb"][jcons]
    q = M.ln(params["blocks"][0]["ln1"], h) @ ar["qr"]
    sim = np.asarray(jnp.einsum("btd,bnd->btn", q, ak) / jnp.sqrt(h.shape[-1])
                     + jnp.where(jmask, 0.0, -1e9)[:, None, :])
    p = np.asarray(jax.nn.softmax(jnp.array(sim), -1))
    ent = -(p * np.log(p + 1e-12)).sum(-1)
    for i in range(B):
        cat = clasificar(I.ITOS[int(pred[i])], I.ITOS[int(tgt[i])], meta[i])
        if cat not in acc or hq[i] < 0:
            continue
        c = acc[cat]; c["n"] += 1
        mios = np.where(oarch[i] == hq[i])[0]           # slots del hecho preguntado
        if len(mios) == 0:
            c["ausente"] += 1; continue
        f = int(ent[i, :pos[i] + 1].argmin())            # posicion de maximo foco
        orden = np.argsort(-sim[i, f])                   # ranking de entradas
        rk = int(min(np.where(np.isin(orden, mios))[0]))
        c["ranks"].append(rk)
        if rk == 0: c["rank0"] += 1
    vistos += B
print(f"pesos: {a.pesos} · nivel {nivel} · n={vistos}\n")
for g in ("err_identidad", "acierto"):
    c = acc[g]; n = max(1, c["n"]); r = np.array(c["ranks"]) if c["ranks"] else np.array([0])
    print(f"{g}  (n={c['n']})")
    print(f"   entrada AUSENTE del archivo   {c['ausente']/n:.4f}   -> nunca se escribio")
    print(f"   rank 0 (gana la lectura)      {c['rank0']/n:.4f}")
    print(f"   rank mediano                  {np.median(r):.1f}   medio {r.mean():.2f}\n")
