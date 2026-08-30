"""Fase 0 de `PREREG_CLAVE_DISCRETA.md` (SHA 3c89348b). CPU, checkpoints en disco, cero GPU.

Cuantiza POST-HOC la clave del archivo con product quantization (el vector se parte en `m` tramos y
cada tramo se manda a uno de `k` centroides) y mide las cuatro predicciones.

La pregunta: con claves continuas «ninguna coincidencia» no existe —`modelo.py:238` hace softmax
sobre el archivo y suma 1 siempre—, y el 16-ago eso dio `s_max` 0,4984, el azar exacto. Con simbolos,
«no matchea ninguno» pasa a ser un evento observable. ¿Aparece la señal que el continuo no tenia?

Uso:  python3 clave_discreta_fase0.py v3_s0 v3_s1 v3_s2 --n 2000
"""

import argparse
import os
import pickle
import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.getcwd())
import datos as DAT
import entrenar as E
import modelo as M

RNG_CB = np.random.default_rng(31415)   # codebooks: semilla fija y distinta de la de datos


# ------------------------------------------------------------------ product quantization
def entrenar_codebooks(K, m, k, iters=25):
    """k-means por tramo sobre las claves observadas. Devuelve lista de (k, d/m)."""
    d = K.shape[1]
    assert d % m == 0, f"D={d} no es divisible por m={m}"
    dm = d // m
    libros = []
    for t in range(m):
        X = K[:, t * dm:(t + 1) * dm]
        C = X[RNG_CB.choice(len(X), k, replace=False)].copy()
        for _ in range(iters):
            d2 = ((X[:, None, :] - C[None, :, :]) ** 2).sum(-1)
            asg = d2.argmin(1)
            for j in range(k):
                sel = asg == j
                if sel.any():
                    C[j] = X[sel].mean(0)
        libros.append(C)
    return libros


def codificar(X, libros):
    """Devuelve (codigos enteros (n, m), reconstruccion (n, d))."""
    m = len(libros)
    dm = X.shape[1] // m
    cods, recon = [], []
    for t, C in enumerate(libros):
        Z = X[:, t * dm:(t + 1) * dm]
        d2 = ((Z[:, None, :] - C[None, :, :]) ** 2).sum(-1)
        a = d2.argmin(1)
        cods.append(a)
        recon.append(C[a])
    return np.stack(cods, 1), np.concatenate(recon, 1)


# ------------------------------------------------------------------ extraccion desde el modelo
def extraer(ruta, n, B, semilla):
    """Devuelve por muestra: claves del archivo, query en pos_q, mascara, y si habia respuesta."""
    with open(ruta, "rb") as f:
        bulto = pickle.load(f)
    params, cfg = bulto["params"], bulto["config"]
    E._DONDE = cfg.get("donde", "pre")
    E._ABST = cfg.get("abst", "token")
    if "abst" not in params:
        d = params["ln_f"]["g"].shape[-1]
        params = dict(params)
        params["abst"] = {"w": jnp.zeros((d, 1)), "b": jnp.zeros((1,))}
        E._ABST = "token"

    a_p = params["arch"]

    @jax.jit
    def piezas(p, ses, cortes, turnos, mask, cons, pos):
        archivo = M.escribir(p, ses, cortes)
        ak = archivo @ p["arch"]["kw"] + p["arch"]["ord"][turnos]      # la CLAVE, con su sello
        # la query se forma en el tronco igual que en `responder`, y se lee en la posicion de consulta
        h = M.tronco(p, cons)
        q = h @ p["arch"]["qr"]
        q = jnp.take_along_axis(q, pos[:, None, None], axis=1)[:, 0, :]
        return ak, q

    rng = np.random.default_rng(semilla)
    AK, Q, MK, HAY, TIPO = [], [], [], [], []
    vistos = 0
    while vistos < n:
        b = min(B, n - vistos)
        ses, cortes, turnos, mask, cons, pos, tgt, tipo, meta = DAT.lote(
            rng, b, nivel=cfg["nivel"], n_hechos=4, n_sesiones=4,
            p_nose=cfg.get("p_nose", 0.4), con_meta=True)
        ak, q = piezas(params, jnp.array(ses), jnp.array(cortes), jnp.array(turnos),
                       jnp.array(mask), jnp.array(cons), jnp.array(pos))
        AK.append(np.asarray(ak, dtype=np.float64))
        Q.append(np.asarray(q, dtype=np.float64))
        MK.append(np.asarray(mask))
        HAY.append(np.asarray(tgt) != E.NOSE)
        TIPO.append(np.asarray(tipo))
        vistos += b
    return (np.concatenate(AK), np.concatenate(Q), np.concatenate(MK),
            np.concatenate(HAY), np.concatenate(TIPO), params, cfg)


def auc(pos, neg):
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    x = np.concatenate([pos, neg])
    r = np.argsort(np.argsort(x)) + 1
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("unidades", nargs="+")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--lote", type=int, default=64)
    ap.add_argument("--semilla", type=int, default=54321)
    ap.add_argument("--m", type=int, default=8, help="tramos del product quantizer")
    a = ap.parse_args()

    print("=" * 92)
    print(f"FASE 0 · CLAVE DISCRETA · PREREG 3c89348b · n={a.n} · semilla {a.semilla} · m={a.m}")
    print("Referencia MEDIDA del continuo (16-ago): AUC s_max = 0,4984 / 0,5022 = el azar exacto")
    print("=" * 92)

    for u in a.unidades:
        ruta = f"ckpts/{u}.pkl"
        if not os.path.exists(ruta):
            print(f"\n{u}: sin checkpoint")
            continue
        AK, Q, MK, HAY, TIPO, params, cfg = extraer(ruta, a.n, a.lote, a.semilla)
        n, N, D = AK.shape
        print(f"\n--- {u} · archivo {N} entradas · D={D} · con respuesta {HAY.mean():.3f} ---")

        # codebooks entrenados sobre las claves VALIDAS de una parte, y aplicados a todo
        planas = AK.reshape(-1, D)[MK.reshape(-1)]
        muestra = planas[RNG_CB.choice(len(planas), min(20000, len(planas)), replace=False)]

        # `res` = valores distintos que toma el estadistico. NO estaba en el prereg y se agrega
        # como DESVIACION declarada: con k alto el maximo de coincidencias satura en 0 y el AUC pasa
        # a ser ruido de empates, que es el instrumento vacio del monitor v1 del 20-ago. Una fila con
        # res <= 2 no se lee.
        print(f"{'k':>5} {'bits/clave':>11} | {'Q-0 recon':>10} {'res':>4} | {'Q-1 AUC':>8} {'Q-2 nulo':>9} | "
              f"{'Q-3 ent':>8} {'Q-3 rel':>8}")
        print("-" * 92)
        for k in (4, 16, 64, 256):
            libros = entrenar_codebooks(muestra, a.m, k)
            cod_k, rec_k = codificar(planas, libros)

            # Q-0 · ¿la cuantizacion preserva la clave? (coseno de la reconstruccion)
            cos = ((planas * rec_k).sum(1) /
                   (np.linalg.norm(planas, axis=1) * np.linalg.norm(rec_k, axis=1) + 1e-9)).mean()

            # codigos por muestra, y codigo de la query con los MISMOS libros
            cod_full = np.zeros((n, N, a.m), dtype=np.int32)
            cod_full[MK] = cod_k
            cod_q, _ = codificar(Q, libros)

            # estadistico: maximo de sub-codigos coincidentes sobre las entradas VALIDAS
            coinc = (cod_full == cod_q[:, None, :]).sum(-1).astype(np.float64)
            coinc[~MK] = -1
            s = coinc.max(1)

            # Q-2 · nulo: los MISMOS codigos, barajados entre entradas (rompe la asignacion,
            # conserva la marginal). Si esto tambien separa, mide la escala y no la coincidencia.
            perm = RNG_CB.permutation(n)
            coinc_n = (cod_full[perm] == cod_q[:, None, :]).sum(-1).astype(np.float64)
            coinc_n[~MK[perm]] = -1
            s_nulo = coinc_n.max(1)

            es_ent = TIPO == 2      # datos.py:42 · TIPOS = {vigente:0, anterior:1, nose_ent:2, nose_rel:3}
            es_rel = TIPO == 3
            bits = a.m * np.log2(k)
            res = len(np.unique(s))
            marca = "" if res > 2 else "   <- SIN RESOLUCION, no se lee"
            print(f"{k:>5} {bits:>11.0f} | {cos:>10.4f} {res:>4} | {auc(s[HAY], s[~HAY]):>8.4f} "
                  f"{auc(s_nulo[HAY], s_nulo[~HAY]):>9.4f} | "
                  f"{auc(s[HAY], s[es_ent]):>8.4f} {auc(s[HAY], s[es_rel]):>8.4f}{marca}")

    print("\nQ-0 bloqueante: la reconstruccion no puede destruir la clave.")
    print("Q-1 principal: AUC >= 0,70 contra la referencia 0,4984 del continuo.")
    print("Q-2 nulo: tiene que quedar en 0,45-0,55, si no mide escala y no coincidencia.")
    print("Q-3: la separacion debe ser MAYOR en nose_ent que en nose_rel.")


if __name__ == "__main__":
    main()
