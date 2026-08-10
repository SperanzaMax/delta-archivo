"""PASO A — ¿existe una representacion que identifique una clave ENTRE secuencias?

Es la compuerta del experimento: si ninguna representacion separa "misma clave" de "clave
distinta", no hay archivo persistente posible sobre este modelo, y B y C no tienen sentido.

Se comparan candidatos a lo largo del forward del bloque 0. La cadena real es:
    emb -> ln1 -> conv3 -> @W_k -> silu -> l2n
y `conv3` es el sospechoso: mezcla cada token con sus dos vecinos, asi que la misma clave
rodeada de pares (S1) y rodeada de consultas (S3) queda distinta.

Metrica: AUC de discriminacion entre las dos distribuciones de coseno. 0.5 = indistinguible,
1.0 = separacion perfecta. Se reporta ademas d' y el mejor umbral con su exactitud.
"""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "2")
sys.path.insert(0, os.path.expanduser("~/Documentos/Nuevo Transformer/telar-ligamento/src"))

import numpy as np, jax, jax.numpy as jnp
import modelos as M
from modelos import split_heads, l2n, ln, conv3, H, DH
from exp_gemacion import preentrenar
from tarea_cross import gen_cross


def reps(params, x):
    """Devuelve dict nombre -> (B,T,H,DH) normalizado por cabeza."""
    blk = params["blocks"][0]
    emb = params["emb"][x]
    l1 = ln(blk["ln1"], emb)
    cv = conv3(blk["conv"], l1)
    out = {}

    def guardar(nombre, z):
        z = l2n(split_heads(z))                      # (B,H,T,DH)
        out[nombre] = np.asarray(z.transpose(0, 2, 1, 3))

    guardar("emb crudo", emb)
    guardar("ln1 (sin conv)", l1)
    guardar("conv3", cv)
    guardar("W_k sobre emb", emb @ blk["k"])
    guardar("W_k sobre ln1", l2n(jax.nn.silu(split_heads(l1 @ blk["k"]))).transpose(0, 2, 1, 3)
            .reshape(x.shape[0], x.shape[1], -1))
    # el que se uso en el experimento fallido:
    z = l2n(jax.nn.silu(split_heads(cv @ blk["k"])))
    out["W_k sobre conv3 (usado)"] = np.asarray(z.transpose(0, 2, 1, 3))
    return out


def auc(pos, neg):
    """AUC por conteo de pares (equivalente a Mann-Whitney)."""
    todo = np.concatenate([pos, neg])
    r = np.argsort(np.argsort(todo)) + 1
    rp = r[:len(pos)].sum()
    return float((rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def evaluar(params, B=32, L=8, r=4, seed=7):
    d = gen_cross(np.random.default_rng(seed), B, L, r)
    acum = {}
    for b in range(B):
        R1 = reps(params, jnp.asarray(d["x1"][b:b + 1]))
        R3 = reps(params, jnp.asarray(d["x3"][b:b + 1]))
        for nom in R1:
            A, C = R1[nom][0], R3[nom][0]            # (T,H,DH)
            p, n = acum.setdefault(nom, ([], []))
            for t1 in range(1, d["x1"].shape[1] - 1, 2):
                tok = d["x1"][b, t1]
                for t3 in range(2, d["x3"].shape[1]):
                    s = float(np.einsum("hd,hd->", A[t1], C[t3]) / H)
                    (p if d["x3"][b, t3] == tok else n).append(s)
    return acum


def main():
    p = preentrenar()
    acum = evaluar(p)
    print(f"{'representacion':>26} {'misma':>8} {'distinta':>9} {'d-prima':>8} "
          f"{'AUC':>7} {'acc@mejor umbral':>17}")
    filas = []
    for nom, (pos, neg) in acum.items():
        pos, neg = np.array(pos), np.array(neg)
        sd = np.sqrt((pos.var() + neg.var()) / 2) + 1e-9
        dp = (pos.mean() - neg.mean()) / sd
        a = auc(pos, neg)
        us = np.linspace(min(pos.min(), neg.min()), max(pos.max(), neg.max()), 200)
        acc = max((np.mean(pos > u) * len(pos) + np.mean(neg <= u) * len(neg))
                  / (len(pos) + len(neg)) for u in us)
        filas.append((a, nom, pos.mean(), neg.mean(), dp, acc))
        print(f"{nom:>26} {pos.mean():8.3f} {neg.mean():9.3f} {dp:8.2f} {a:7.3f} {acc:17.3f}")
    filas.sort(reverse=True)
    mejor = filas[0]
    print(f"\nMEJOR: '{mejor[1]}' con AUC {mejor[0]:.3f}")
    print("COMPUERTA:", "PASA — hay identificador utilizable" if mejor[0] > 0.90
          else "NO PASA — sin identificador estable, el archivo persistente es imposible aqui")
    return filas


if __name__ == "__main__":
    main()
