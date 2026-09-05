"""La MISMA muestra de la pelicula, leida por un checkpoint YA ENTRENADO. 2026-09-05

La pelicula muestra el arranque. Esto muestra el final: `kq3_s0`, 26.000 pasos, kernel 5, la misma
arquitectura y el mismo nivel del idioma. La muestra se regenera con la misma semilla (77) y el mismo
procedimiento que `pelicula.py`, asi que es literalmente el mismo episodio y la misma pregunta.
"""
import json, pickle
import numpy as np, jax.numpy as jnp

import datos as DAT, idioma as I, modelo as M
from pelicula import atencion_archivo

CK = "ckpts/kq3_s0.pkl"
b = pickle.load(open(CK, "rb"))
cfg, params = b["config"], b["params"]
M.KQ = cfg.get("kernel_q", 3)
print(f"{CK} · paso {cfg.get('pasos')} · d={cfg['d']} capas={cfg['capas']} "
      f"kernel_q={M.KQ} donde={cfg['donde']}")

mrng = np.random.default_rng(77)
while True:
    s, c, t, mk, q, pq, tg, tp, meta, orig, hq = DAT.lote(
        mrng, 8, nivel=cfg["nivel"], p_vieja=0.0, p_nose=0.0, con_meta=True, con_origen=True)
    i = int(np.argmax(tp == DAT.TIPOS["vigente"]))
    if tp[i] == DAT.TIPOS["vigente"] and hq[i] >= 0:
        break
muestra = (jnp.array(s[i:i+1]), jnp.array(c[i:i+1]), jnp.array(t[i:i+1]),
           jnp.array(mk[i:i+1]), jnp.array(q[i:i+1]), np.array(pq[i:i+1]))
correctos = [int(x) for x in np.where((orig[i] == hq[i]) & mk[i])[0]]
att, lg = atencion_archivo(params, muestra, cfg["donde"])
cq = np.array(params["blocks"][0]["convq"])

ref = {"ckpt": CK, "pasos": int(cfg.get("pasos", 0)), "kernel_q": int(M.KQ),
       "taps": [round(float(np.abs(x).mean()), 5) for x in cq],
       "atencion": [round(float(x), 5) for x in att],
       "pred": I.ITOS[int(np.argmax(lg))],
       "correcta": I.ITOS[int(tg[i])],
       "correctos": correctos,
       "masa_correcta": round(float(sum(att[j] for j in correctos)), 5)}
json.dump(ref, open("referencia_pelicula.json", "w"), separators=(",", ":"))
print(f"pregunta: {' '.join(I.ITOS[int(z)] for z in q[i] if I.ITOS[int(z)] != '.')}")
print(f"correcta {ref['correcta']} · contesta {ref['pred']} · "
      f"masa en la correcta {ref['masa_correcta']:.4f}")
print(f"taps {ref['taps']}")
