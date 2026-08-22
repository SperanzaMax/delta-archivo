#!/usr/bin/env python3
"""SONDA DEL EMPATE DE CLAVE · `PREREG_EMPATE_CLAVE.md` SHA b78b2141..., congelado antes de esto.

    python sonda_empate.py --n 32 --salida empate_20260821.json

Mide E-1..E-6. La justificacion, el mecanismo y los limites estan en el prereg y no se repiten aca;
lo que va abajo es solo lo que hace falta para leer el codigo.

METRICAS (sobre los scores crudos del bloque 0, por posicion de la consulta, entradas validas):
  z_foco    (s1-s2)/std en la posicion de maximo matcheo
  z_min     el menor (s1-s2)/std entre las posiciones hasta la de la respuesta
  consenso  solapamiento entre las distribuciones de lectura de las DOS posiciones de mayor
            matcheo. Menos solapamiento = las dos consultas apuntan a entradas distintas = la
            conjuncion entidad x relacion es ambigua.

En las tres, EMPATE = valor BAJO. Las AUC se calculan sobre el valor negado para que se lean todas
en la misma direccion: > 0,5 significa «el grupo colisionado se ve mas empatado».

DOS PRECISIONES DE IMPLEMENTACION, declaradas (van al informe como D-1 y D-2, no como edicion del
prereg congelado):

  D-1 · `consenso` se implementa como suma de minimos entre las dos distribuciones softmax
        (solapamiento continuo en [0,1]) y no como el conteo de coincidencias del top-2. El conteo
        toma tres valores (0 · 0,5 · 1) y un AUC sobre tres valores es casi todo empates, que es
        justo lo que E-5 necesita resolver con 0,05 de margen. Es la version continua de lo mismo.

  D-2 · el nulo N-2 del §3 —«barajar que entrada corresponde a que hecho»— se implementa como
        permutacion de la etiqueta de colision DENTRO de estratos de (entradas validas, revisado).
        Barajar la asignacion entrada->hecho no cambiaria los scores, que es de donde sale la
        etiqueta de todos modos; lo que hay que romper es el pareo entre el empate y la colision
        SIN romper la estructura del episodio. Estratificar es lo que hace que el nulo pueda
        FALLAR: si el detector estuviera leyendo el tamaño del episodio en vez de la colision, el
        nulo estratificado seguiria discriminando y lo delata. Sin estratos seria «permutar
        etiquetas» a secas, el nulo flojo que el monitor v1 enseño a no usar.
"""
import argparse
import json
import os
import pickle
import sys

import numpy as np
import jax
import jax.numpy as jnp

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
import datos as DAT
import idioma as I
import modelo as M
from score_archivo import auc

NOSE = I.STOI["NOSE"]
SEM_PRUEBA = 77000
DIR_CK = os.path.join(AQUI, "ckpts", "rt_congelados")

# (nombre de la celda, archivo del checkpoint, nivel, semilla, paso declarado)
CELDAS = [
    ("c1_s0@14000", "c1_s0.pkl", 1, 0, 14000),
    ("c2_s0@14000", "c2_s0.pkl", 2, 0, 14000),
    ("c3_s0@14000", "c3_s0.pkl", 3, 0, 14000),
    ("c3_s1@14000", "c3_s1.pkl", 3, 1, 14000),
    ("c3_s2@14000", "c3_s2.pkl", 3, 2, 14000),
    ("c4_s0@14000", "c4_s0.pkl", 4, 0, 14000),
    ("c4_s1@14000", "c4_s1.pkl", 4, 1, 14000),
    ("c4_s2@14000", "c4_s2.pkl", 4, 2, 14000),
    ("c4_s0@20000", "c4_s0_p20000.pkl", 4, 0, 20000),
    ("c4_s1@20000", "c4_s1_p20000.pkl", 4, 1, 20000),
    ("c4_s2@20000", "c4_s2_p20000.pkl", 4, 2, 20000),
]


def responder(params, ses, cortes, turnos, cons, mask, pos):
    archivo = M.escribir(params, ses, cortes)
    lg, _ = M.responder_con_abst(params, archivo, turnos, cons, mask)
    lg = jnp.take_along_axis(lg, pos[:, None, None], axis=1)[:, 0, :]
    return lg.at[:, NOSE].set(-jnp.inf).argmax(-1)


def scores_todas(params, ses, cortes, turnos, mask, consulta):
    """(B, Tq, N) — los scores de lectura del bloque 0 en todas las posiciones de la consulta.

    La query es `ln(emb[token]) @ qr`: funcion pura del token de su posicion, porque la lectura se
    inyecta antes de la conv y del mixer (`modelo.tronco`). Por eso la medicion es por posicion y no
    en la de la respuesta, que es donde el smoke de esta mañana encontraba todo plano.
    """
    a = params["arch"]
    archivo = M.escribir(params, ses, cortes)
    ak = archivo @ a["kw"] + a["ord"][turnos]
    penal = jnp.where(mask, 0.0, -1e9)
    h = params["emb"][consulta]
    q = M.ln(params["blocks"][0]["ln1"], h) @ a["qr"]
    return jnp.einsum("btd,bnd->btn", q, ak) / jnp.sqrt(h.shape[-1]) + penal[:, None, :]


def metricas(st, nval):
    """z_foco, z_min y consenso a partir de (T util, N validas) de scores crudos."""
    if st.shape[0] == 0 or st.shape[1] < 2:
        return np.nan, np.nan, np.nan
    ordt = np.sort(st, axis=1)[:, ::-1]
    sdt = st.std(axis=1) + 1e-12
    zt = (ordt[:, 0] - ordt[:, 1]) / sdt
    fuerza = st.max(axis=1)
    i1 = int(np.argmax(fuerza))
    z_foco, z_min = float(zt[i1]), float(zt.min())
    if st.shape[0] < 2:
        return z_foco, z_min, np.nan
    i2 = int(np.argsort(fuerza)[-2])
    pa = np.exp(st[i1] - st[i1].max()); pa /= pa.sum()
    pb = np.exp(st[i2] - st[i2].max()); pb /= pb.sum()
    return z_foco, z_min, float(np.minimum(pa, pb).sum())      # D-1


def nulo_gauss(st, rng):
    """N-1 · scores reemplazados por gaussianas de igual media y desvio, por posicion."""
    mu = st.mean(axis=1, keepdims=True)
    sd = st.std(axis=1, keepdims=True)
    return mu + sd * rng.standard_normal(st.shape)


def permutar_en_estratos(etiq, estratos, rng):
    """N-2 · permuta la etiqueta de colision dentro de cada estrato (D-2)."""
    out = etiq.copy()
    for e in np.unique(estratos):
        sel = np.where(estratos == e)[0]
        out[sel] = etiq[rng.permutation(sel)]
    return out


def auc_seguro(x, y):
    x = x[~np.isnan(x)]; y = y[~np.isnan(y)]
    return auc(x, y) if len(x) and len(y) else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=32, help="lotes por celda (batch 64)")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--celdas", default="todas")
    ap.add_argument("--salida", default=os.path.join(AQUI, "empate_20260821.json"))
    A = ap.parse_args()

    celdas = CELDAS if A.celdas == "todas" else [
        c for c in CELDAS if c[0] in A.celdas.split(",")]

    print("SONDA · EMPATE DE CLAVE · prereg SHA b78b2141...")
    print(f"{A.n * A.batch} muestras por celda · {len(celdas)} celdas\n")
    res = {}

    for nom, arch, nivel, semilla, paso_decl in celdas:
        ck = os.path.join(DIR_CK, arch)
        if not os.path.exists(ck):
            print(f"{nom}: sin checkpoint, se saltea")
            continue
        with open(ck, "rb") as f:
            d = pickle.load(f)
        paso_real = int(d.get("paso", -1))
        if paso_real != paso_decl:
            print(f"{nom}: el checkpoint dice paso {paso_real} y el prereg declara {paso_decl} "
                  f"-> se saltea (regla D-1 del 20-ago)")
            continue
        params = jax.tree_util.tree_map(jnp.asarray, d["params"])
        rng = np.random.default_rng(SEM_PRUEBA + semilla)
        rng_nulo = np.random.default_rng(990000 + semilla)
        fn = jax.jit(responder)

        Z, ZM, CO, ZN, REP, REV, OK, ID, NV = [], [], [], [], [], [], [], [], []
        for _ in range(A.n):
            sal = DAT.lote(rng, A.batch, nivel=nivel, n_hechos=4, n_sesiones=4,
                           p_vieja=0.35, p_nose=0.0, con_meta=True)
            ses, cortes, turnos, mask, cons, pos, tgt, tipo, meta = sal
            jses, jcor = jnp.array(ses), jnp.array(cortes)
            jtur, jmask = jnp.array(turnos), jnp.array(np.asarray(mask))
            jcons, jpos = jnp.array(cons), jnp.array(pos)
            X = np.asarray(fn(params, jses, jcor, jtur, jcons, jmask, jpos))
            stodas = np.asarray(scores_todas(params, jses, jcor, jtur, jmask, jcons))
            tgt = np.asarray(tgt); tipo = np.asarray(tipo); mk = np.asarray(mask)

            for b in range(len(X)):
                m = meta[b]
                if tipo[b] >= 2 or not m["hecho"]:
                    continue
                val = mk[b].astype(bool)
                st = stodas[b][: int(pos[b]) + 1][:, val]
                nval = int(val.sum())
                zf, zm, co = metricas(st, nval)
                zfn, _, _ = metricas(nulo_gauss(st, rng_nulo), nval)
                x = I.ITOS[X[b]]
                rel = m["hecho"]["rel"]
                Z.append(zf); ZM.append(zm); CO.append(co); ZN.append(zfn)
                REP.append(any(o["rel"] == rel for o in m["otros"]))
                REV.append(len(m["hecho"]["versiones"]) > 1)
                OK.append(bool(X[b] == tgt[b]))
                ID.append(bool(X[b] != tgt[b]) and any(x in o["versiones"] for o in m["otros"]))
                NV.append(nval)

        Z = np.array(Z); ZM = np.array(ZM); CO = np.array(CO); ZN = np.array(ZN)
        REP = np.array(REP); REV = np.array(REV); OK = np.array(OK); ID = np.array(ID)
        NV = np.array(NV)

        estratos = np.array([f"{a}_{int(b)}" for a, b in zip(NV, REV)])
        REPn = permutar_en_estratos(REP, estratos, rng_nulo)

        r = {
            "paso": paso_real, "nivel": nivel, "semilla": semilla, "n": int(len(Z)),
            "acierto": float(OK.mean()), "err_identidad": float(ID.mean()),
            "P_rep": float(REP.mean()), "P_rev": float(REV.mean()),
            # E-1 / E-2 · detectar la condicion
            "e1_zfoco": auc_seguro(-Z[REP], -Z[~REP]),
            "e1_zmin": auc_seguro(-ZM[REP], -ZM[~REP]),
            "e1_cons": auc_seguro(-CO[REP], -CO[~REP]),
            "e2_zfoco": auc_seguro(-Z[REP & ~REV], -Z[~REP & ~REV]),
            # E-3 · los dos nulos, sobre la misma metrica principal
            "n1_gauss": auc_seguro(-ZN[REP], -ZN[~REP]),
            "n2_perm": auc_seguro(-Z[REPn], -Z[~REPn]),
            # E-4 / E-5 · convertir en abstencion
            "e4_zfoco": auc_seguro(-Z[ID], -Z[OK]),
            "e5_cons": auc_seguro(-CO[ID], -CO[OK]),
            "med_zfoco_rep": float(np.nanmean(Z[REP])),
            "med_zfoco_uni": float(np.nanmean(Z[~REP])),
        }
        res[nom] = r
        print(f"=== {nom} · n={r['n']} · acierto {r['acierto']:.4f} · "
              f"err_id {r['err_identidad']:.4f} · P(rep) {r['P_rep']:.4f}")
        print(f"    E-1 z_foco {r['e1_zfoco']:.4f} · z_min {r['e1_zmin']:.4f} · "
              f"cons {r['e1_cons']:.4f}   |   E-2 {r['e2_zfoco']:.4f}")
        print(f"    NULOS  N-1 gauss {r['n1_gauss']:.4f} · N-2 perm {r['n2_perm']:.4f}")
        print(f"    E-4 z_foco {r['e4_zfoco']:.4f} · E-5 cons {r['e5_cons']:.4f}\n")
        json.dump(res, open(A.salida, "w"), indent=1, default=float)

    # ---- veredicto por prediccion, con el criterio del prereg ------------------------------------
    base = {k: v for k, v in res.items() if v["paso"] == 14000}
    print("=" * 78)
    print("VEREDICTO (brazo principal, las 8 unidades a 14000)")
    e1 = sum(v["e1_zfoco"] >= 0.60 for v in base.values())
    e2 = sum(v["e2_zfoco"] >= 0.60 for v in base.values())
    n1 = sum(v["n1_gauss"] >= 0.60 for v in base.values())
    n2 = sum(v["n2_perm"] >= 0.60 for v in base.values())
    e4 = sum(v["e4_zfoco"] >= 0.65 for v in base.values())
    e5 = sum(v["e5_cons"] - v["e4_zfoco"] >= 0.05 for v in base.values())
    n = len(base)
    print(f"  E-1  AUC(z_foco) >= 0,60          {e1} de {n}   (pide >= 6 de 8)")
    print(f"  E-2  idem sin revisados           {e2} de {n}   (pide >= 6 de 8)")
    print(f"  E-3  N-1 pasa                     {n1} de {n}   (pide <= 2 de 8, BLOQUEANTE)")
    print(f"       N-2 pasa                     {n2} de {n}   (pide <= 2 de 8, BLOQUEANTE)")
    print(f"  E-4  AUC(err vs acierto) >= 0,65  {e4} de {n}   (pide >= 5 de 8)")
    print(f"  E-5  consenso mejora >= 0,05      {e5} de {n}")
    for nom in ("c4_s0", "c4_s1", "c4_s2"):
        a, b = res.get(f"{nom}@14000"), res.get(f"{nom}@20000")
        if a and b:
            print(f"  E-6  {nom}: {a['e1_zfoco']:.4f} (14000) -> {b['e1_zfoco']:.4f} (20000)   "
                  f"{'baja' if b['e1_zfoco'] < a['e1_zfoco'] else 'NO BAJA'}")
    print(f"\n-> {A.salida}")


if __name__ == "__main__":
    main()
