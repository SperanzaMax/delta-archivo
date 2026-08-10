"""La frontera: interpolar entre representacion SIN contexto (trivial pero identifica) y CON
contexto (interesante pero no identifica). xin = (1-a)*ln1 + a*conv3(ln1).

a=0   -> funcion pura del token: el archivo es un diccionario, la tarea se trivializa.
a=1   -> lo que fallo: AUC 0.789, no hay identificador.
La pregunta es si existe un a intermedio donde la representacion lleve contexto Y siga
identificando. Si no existe, este modelo no admite memoria persistente no trivial.
"""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "2")
sys.path.insert(0, os.path.expanduser("~/Documentos/Nuevo Transformer/telar-ligamento/src"))
import numpy as np, jax, jax.numpy as jnp
import exp_gemacion as E
from modelos import ln, conv3, split_heads, l2n, H
from tarea_cross import gen_cross
from paso_a_identificador import auc

p = E.preentrenar()
blk = p["blocks"][0]

def hacer_claves(alpha):
    def f(params, x):
        b = params["blocks"][0]
        l1 = ln(b["ln1"], params["emb"][x])
        xin = (1 - alpha) * l1 + alpha * conv3(b["conv"], l1)
        z = l2n(jax.nn.silu(split_heads(xin @ b["k"])))
        return np.asarray(z.transpose(0, 2, 1, 3))
    return f

d = gen_cross(np.random.default_rng(7), 48, 8, 4)
print(f"{'alpha':>6} {'AUC ident':>10} {'sobrescr V':>11} {'sobrescr A':>11} "
      f"{'gemacion V':>11} {'gemacion A':>11}")
for a in (0.0, 0.25, 0.5, 0.75, 1.0):
    E.claves_latentes = hacer_claves(a)
    pos, neg = [], []
    for b in range(16):
        K1 = E.claves_latentes(p, jnp.asarray(d["x1"][b:b+1]))[0]
        K3 = E.claves_latentes(p, jnp.asarray(d["x3"][b:b+1]))[0]
        for t1 in range(1, d["x1"].shape[1]-1, 2):
            tok = d["x1"][b, t1]
            for t3 in range(2, d["x3"].shape[1]):
                s = float(np.einsum("hd,hd->", K1[t1], K3[t3]) / H)
                (pos if d["x3"][b, t3] == tok else neg).append(s)
    A = auc(np.array(pos), np.array(neg))
    res = {}
    for modo in ("sobrescritura", "gemacion"):
        V = Vn = C = Cn = 0
        for b in range(48):
            v, vn, c, cn = E.episodio(p, d, b, modo)
            V += v; Vn += vn; C += c; Cn += cn
        res[modo] = (V/Vn, C/max(Cn,1))
    print(f"{a:6.2f} {A:10.3f} {res['sobrescritura'][0]:11.3f} {res['sobrescritura'][1]:11.3f} "
          f"{res['gemacion'][0]:11.3f} {res['gemacion'][1]:11.3f}")
