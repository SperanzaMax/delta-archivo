"""¿El AUC de la Fase 0 cambia si el score se toma en la posicion de FOCO y no en pos_q?

    python score_pos_foco.py ckpts/n4_s0.pkl --n 4000

`foco_posiciones.py` mostro que el modelo concentra la lectura en posiciones INTERMEDIAS de la
consulta (top-1 llega a 0,65) y que en `pos_q` la distribucion ya esta difusa. La Fase 0 midio el
score en `pos_q`. Si la señal de ausencia vive donde vive el foco, el AUC 0,4984 seria un artefacto
de la posicion elegida — y hay que saberlo antes de asentar el resultado.

Se recomputan las mismas señales en tres lugares: pos_q (replica), la posicion de MINIMA entropia
(donde el modelo mas enfoca) y el maximo sobre todas las posiciones.
"""
import argparse, pickle
import numpy as np, jax, jax.numpy as jnp
import datos as DAT, modelo as M
from score_archivo import auc

ap = argparse.ArgumentParser(); ap.add_argument("pesos")
ap.add_argument("--n", type=int, default=4000); ap.add_argument("--B", type=int, default=64)
ap.add_argument("--p-nose", type=float, default=0.4)
a = ap.parse_args()
bulto = pickle.load(open(a.pesos, "rb"))
params = jax.tree_util.tree_map(jnp.asarray, bulto["params"]); nivel = bulto["config"]["nivel"]
rng = np.random.default_rng(31415)
col = {k: [] for k in ("max_posq", "max_foco", "max_global", "marg_foco")}
tipos, vistos = [], 0
while vistos < a.n:
    B = min(a.B, a.n - vistos)
    ses, cortes, turnos, mask, cons, pos, tgt, tipo = DAT.lote(
        rng, B, nivel=nivel, n_hechos=4, n_sesiones=4, p_nose=a.p_nose)
    arch = M.escribir(params, jnp.array(ses), jnp.array(cortes))
    ar = params["arch"]
    ak = arch @ ar["kw"] + ar["ord"][jnp.array(turnos)]
    penal = jnp.where(jnp.array(mask), 0.0, -1e9)[:, None, :]
    h = params["emb"][jnp.array(cons)]
    q = M.ln(params["blocks"][0]["ln1"], h) @ ar["qr"]
    sim = np.asarray(jnp.einsum("btd,bnd->btn", q, ak) / jnp.sqrt(h.shape[-1]) + penal)
    p = np.asarray(jax.nn.softmax(jnp.array(sim), -1))
    e = -(p * np.log(p + 1e-12)).sum(-1)
    for i in range(B):
        T = pos[i] + 1
        f = int(e[i, :T].argmin())                       # posicion donde MAS enfoca
        s_f = np.sort(sim[i, f]); 
        col["max_posq"].append(sim[i, pos[i]].max())
        col["max_foco"].append(sim[i, f].max())
        col["marg_foco"].append(s_f[-1] - s_f[-2])
        col["max_global"].append(sim[i, :T].max())
    tipos.extend(np.asarray(tipo).tolist()); vistos += B
tipos = np.array(tipos); col = {k: np.array(v) for k, v in col.items()}
con = tipos <= 1; ent = tipos == 2; rel = tipos == 3; sin = ent | rel
print(f"pesos: {a.pesos} · nivel {nivel} · n={vistos} · con {con.sum()} / sin {sin.sum()}")
print("AUC(con respuesta, sin respuesta) segun DONDE se tome el score:")
for k in ("max_posq", "max_foco", "max_global", "marg_foco"):
    v = col[k]
    print(f"  {k:<12} {auc(v[con], v[sin]):.4f}   ent {auc(v[con], v[ent]):.4f}  rel {auc(v[con], v[rel]):.4f}")
