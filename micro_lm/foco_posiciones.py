"""Verificacion de §4.5: ¿el modelo concentra en ALGUNA posicion de la consulta?

    python foco_posiciones.py ckpts/n4_s0.pkl --n 1024

El foco se midio en `pos_q`, la posicion desde la que se responde. Si el modelo concentrara en
posiciones intermedias y `pos_q` viera el estado ya integrado, la conclusion «no enfoca» seria un
artefacto de donde mire. Esto recorre TODAS las posiciones reales de la consulta y reporta el minimo
de entropia alcanzado en cada muestra: si en ninguna posicion baja, no enfoca en ninguna parte.
"""
import argparse, pickle
import numpy as np, jax, jax.numpy as jnp
import datos as DAT, modelo as M

ap = argparse.ArgumentParser(); ap.add_argument("pesos")
ap.add_argument("--n", type=int, default=1024); ap.add_argument("--B", type=int, default=64)
a = ap.parse_args()
bulto = pickle.load(open(a.pesos, "rb"))
params = jax.tree_util.tree_map(jnp.asarray, bulto["params"]); nivel = bulto["config"]["nivel"]
rng = np.random.default_rng(31415)
ent_min, ent_pos, top_max, vistos = [], [], [], 0
while vistos < a.n:
    B = min(a.B, a.n - vistos)
    ses, cortes, turnos, mask, cons, pos, tgt, tipo = DAT.lote(
        rng, B, nivel=nivel, n_hechos=4, n_sesiones=4, p_nose=0.0)
    arch = M.escribir(params, jnp.array(ses), jnp.array(cortes))
    ar = params["arch"]
    ak = arch @ ar["kw"] + ar["ord"][jnp.array(turnos)]
    penal = jnp.where(jnp.array(mask), 0.0, -1e9)[:, None, :]
    h = params["emb"][jnp.array(cons)]
    q = M.ln(params["blocks"][0]["ln1"], h) @ ar["qr"]
    sim = jnp.einsum("btd,bnd->btn", q, ak) / jnp.sqrt(h.shape[-1]) + penal
    p = np.asarray(jax.nn.softmax(sim, -1))                    # (B, Tq, N)
    e = -(p * np.log(p + 1e-12)).sum(-1)                       # (B, Tq)
    for i in range(B):
        real = e[i, :pos[i] + 1]
        ent_min.append(real.min()); ent_pos.append(e[i, pos[i]])
        top_max.append(p[i, :pos[i] + 1].max())
    vistos += B
ent_min, ent_pos, top_max = map(np.array, (ent_min, ent_pos, top_max))
print(f"pesos: {a.pesos} · nivel {nivel} · n={vistos}")
print(f"  entropia en pos_q              {ent_pos.mean():.4f}")
print(f"  entropia MINIMA sobre todas    {ent_min.mean():.4f}")
print(f"  masa top-1 MAXIMA sobre todas  {top_max.mean():.4f}")
print(f"  -> {'ENFOCA en alguna posicion' if top_max.mean() > 0.60 else 'NO enfoca en NINGUNA posicion'}")
