#!/usr/bin/env python3
"""El monitor de desacuerdo interno v2 — PREREG_MONITOR_DESACUERDO_V2.md (SHA b8e187b4...).

K pasadas, cada una con una fraccion f=0,25 de las entradas vivas del archivo TAPADAS (mask a 0), y
se mide la `consistencia`: la fraccion de las K que coincide con la respuesta modal.

Por que tapar y no permutar: permutar no cambia nada. La lectura es una atencion softmax, invariante
a permutar por construccion algebraica —medido: max|Δlogit| = 5,7e-06—, asi que el v1 media un
estadistico identicamente 1. Tapar cambia el CONJUNTO y ninguna identidad lo anula. Ver D-1 de
DESVIACIONES_MONITOR.md.

Y trae una prediccion a priori que es toda la gracia: una respuesta anclada en UNA entrada sobrevive
exactamente cuando esa entrada no fue tapada, o sea en 1-f = 0,75 de las pasadas. De ahi sale el
corte ESTRUCTURAL —abstenerse si consistencia < 1-f— sin ajustar un solo umbral contra etiquetas,
que es lo que al logit no se le pudo sacar el 20-ago.
"""
import os, sys, pickle, argparse, json
import numpy as np
import jax, jax.numpy as jnp

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
import idioma as I, datos as DAT, modelo as M

NOSE = I.STOI["NOSE"]
UNIDADES = ["1_s0", "2_s0", "3_s0", "3_s1", "3_s2", "4_s0", "4_s1", "4_s2"]
SEM_PRUEBA = 77000
K, F = 16, 0.25          # PREREG §1-§2, fijados antes de correr


def responder(params, archivo, turnos, mask, cons, pos):
    lg, _ = M.responder_con_abst(params, archivo, turnos, cons, mask)
    lg = jnp.take_along_axis(lg, pos[:, None, None], axis=1)[:, 0, :]
    return lg.at[:, NOSE].set(-jnp.inf).argmax(-1)


def pasadas(params, ses, cortes, turnos, masks, cons, pos):
    """Una respuesta por mascara. `masks` es (K, B, N) — cada pasada tapa entradas distintas."""
    archivo = M.escribir(params, ses, cortes)
    return jax.vmap(lambda m: responder(params, archivo, turnos, m, cons, pos))(masks)


def consistencia(R):
    K_, B = R.shape
    out = np.empty(B)
    modo = np.empty(B, dtype=R.dtype)
    for i in range(B):
        v, c = np.unique(R[:, i], return_counts=True)
        j = c.argmax()
        out[i] = c[j] / K_
        modo[i] = v[j]
    return out, modo


def tapar(mask, rp, f):
    """(K,B,N): cada pasada tapa una fraccion f de las entradas VIVAS, independiente por muestra."""
    B, N = mask.shape
    out = np.repeat(mask[None], K, axis=0).copy()
    for k in range(K):
        for b in range(B):
            viv = np.flatnonzero(mask[b])
            q = int(round(f * len(viv)))
            if q:
                out[k, b, rp.choice(viv, size=q, replace=False)] = 0
    return out


def auc(x, pos_mask):
    p, n = x[pos_mask], x[~pos_mask]
    if len(p) == 0 or len(n) == 0:
        return np.nan
    r = np.argsort(np.argsort(np.concatenate([p, n]))) + 1
    return float((r[:len(p)].sum() - len(p) * (len(p) + 1) / 2) / (len(p) * len(n)))


def pasa(nose, falsa):
    return bool(falsa <= 0.10 and nose >= 0.50)


ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=8)
ap.add_argument("--batch", type=int, default=64)
ap.add_argument("--p-nose", type=float, default=0.4)
ap.add_argument("--unidades", default="")
ap.add_argument("--ckpt-alt", action="append", default=[], metavar="UNIDAD=RUTA",
                help="fija el checkpoint de una unidad. Las tres de nivel 4 se leen de su copia "
                     ".p14000: el prereg pide las 8 a 14000 pasos, y ademas c4_s0/c4_s1 se estan "
                     "entrenando ahora mismo — leer el .pkl vivo es exactamente la D-1 del dia.")
ap.add_argument("--salida", default=os.path.join(AQUI, "desacuerdo_20260820.json"))
a_ = ap.parse_args()
UNI = a_.unidades.split(",") if a_.unidades else UNIDADES
ALT = dict(x.split("=", 1) for x in a_.ckpt_alt)

print(f"MONITOR DE DESACUERDO v2 — PREREG_MONITOR_DESACUERDO_V2.md (SHA b8e187b4...)")
print(f"K={K} pasadas · f={F} de las entradas tapadas · corte estructural consistencia < {1-F}")
print(f"rng {SEM_PRUEBA}+s · {a_.n}x{a_.batch} = {a_.n * a_.batch} muestras por unidad\n")
print(f"{'unidad':<8} {'AUC':>7} {'M-1':>5} | {'f_abst':>8} {'nose':>8} {'M-2':>5} | "
       f"{'M-3 cambia':>11} {'M-3':>5} | {'M-4':>6}")
print("-" * 78)

res, n1, n2, n3, n4 = {}, 0, 0, 0, 0
for u in UNI:
    ck = ALT.get(u, os.path.join(AQUI, "ckpts", f"c{u}.pkl"))
    if not os.path.isabs(ck):
        ck = os.path.join(AQUI, ck)
    if not os.path.exists(ck):
        print(f"c{u}: sin checkpoint")
        continue
    with open(ck, "rb") as f:
        _d = pickle.load(f)
    print(f"c{u}: {os.path.basename(ck)} · paso {_d.get('paso','?')}", flush=True)
    params = jax.tree_util.tree_map(jnp.asarray, _d["params"])
    if "abst" not in params:
        continue
    nivel, semilla = int(u[0]), int(u.split("_s")[1])
    rng = np.random.default_rng(SEM_PRUEBA + semilla)
    rp = np.random.default_rng(1234 + semilla)
    fn = jax.jit(pasadas)

    CO, OKV, TIPO, ID, CAMBIA = [], [], [], [], []
    for _ in range(a_.n):
        sal = DAT.lote(rng, a_.batch, nivel=nivel, n_hechos=4, n_sesiones=4,
                       p_vieja=0.35, p_nose=a_.p_nose, con_meta=True, con_origen=True)
        ses, cortes, turnos, mask, cons, pos, tgt, tipo, meta, origen, hecho_q = sal
        mask = np.asarray(mask)
        args = (jnp.array(ses), jnp.array(cortes), jnp.array(turnos))
        rest = (jnp.array(cons), jnp.array(pos))

        masks = tapar(mask, rp, F)
        R = np.asarray(fn(params, *args, jnp.array(masks), *rest))
        co, modo = consistencia(R)
        CO.append(co); OKV.append(modo == tgt); TIPO.append(tipo)

        # M-4 · el nulo: f = 0, o sea K mascaras identicas. Debe dar 1,000 exacto.
        m0 = np.repeat(mask[None], K, axis=0)
        ID.append(consistencia(np.asarray(fn(params, *args, jnp.array(m0), *rest)))[0])

        # M-3 · el control fuerte: tapar EXACTAMENTE las entradas que originaron el hecho
        # preguntado, identificadas por `origen`. Si la respuesta no cambia, el modelo no las lee.
        org = np.asarray(origen)
        hq = np.asarray(hecho_q)
        mq = mask.copy()
        for b in range(mask.shape[0]):
            if hq[b] >= 0:
                mq[b, org[b] == hq[b]] = 0
        Rq = np.asarray(fn(params, *args, jnp.array(np.repeat(mq[None], K, axis=0)), *rest))
        base = np.asarray(fn(params, *args, jnp.array(m0), *rest))[0]
        ok_base = (base == tgt) & (tipo < 2)
        CAMBIA.append(np.where(ok_base, Rq[0] != base, np.nan))

    co = np.concatenate(CO); okv = np.concatenate(OKV); tipo = np.concatenate(TIPO)
    cid = np.concatenate(ID); cam = np.concatenate(CAMBIA)

    a1 = auc(1.0 - co, tipo >= 2)
    m1 = bool(a1 >= 0.70)
    abst = co < (1 - F)
    sin_r, hay, vig = tipo >= 2, tipo < 2, tipo == 0
    nose = float(abst[sin_r].mean()); falsa = float(abst[hay].mean())
    vigente = float((okv[vig] & ~abst[vig]).mean())
    m2 = pasa(nose, falsa)
    frac_cambia = float(np.nanmean(cam)) if np.any(~np.isnan(cam)) else np.nan
    m3 = bool(frac_cambia >= 0.50)
    m4 = bool(np.all(cid == 1.0))
    n1 += m1; n2 += m2; n3 += m3; n4 += m4

    res[u] = {"auc": a1, "M1": m1, "consist_media": float(co.mean()), "falsa": falsa,
              "nose": nose, "vigente": vigente, "M2": m2, "M3_frac_cambia": frac_cambia,
              "M3": m3, "M4": m4}
    print(f"c{u:<7} {a1:>7.3f} {('SI' if m1 else 'no'):>5} | {falsa:>8.4f} {nose:>8.4f} "
          f"{('SI' if m2 else 'no'):>5} | {frac_cambia:>11.4f} {('SI' if m3 else 'no'):>5} | "
          f"{('OK' if m4 else 'FALLA'):>6}")

n = len(res)
print("-" * 78)
print(f"M-1 · AUC >= 0,70 en {n1}/{n}   (>= 6/8)  -> {'CUMPLE' if n1 >= 6 else 'NO CUMPLE'}")
print(f"M-2 · corte estructural pasa en {n2}/{n}   (>= 6/8)  -> "
      f"{'CUMPLE' if n2 >= 6 else 'NO CUMPLE'}")
print(f"      referencias del 20-ago, mismas unidades: U-1 = 2/8 · sigma>0,5 = 6/8 · U-2 = 7/8")
print(f"M-3 · tapar la entrada del hecho cambia la respuesta en >= 0,50 en {n3}/{n} unidades")
print(f"M-4 · el nulo (f=0) da 1,000 exacto en {n4}/{n}  -> "
      f"{'OK' if n4 == n else 'FALLA — hay ruido y lo de arriba queda en duda'}")

json.dump({"prereg": "PREREG_MONITOR_DESACUERDO_V2.md", "sha": "b8e187b4",
           "config": {"K": K, "f": F, "n": a_.n, "batch": a_.batch, "p_nose": a_.p_nose,
                      "rng_prueba": SEM_PRUEBA},
           "unidades": res, "veredictos": {"M1": n1, "M2": n2, "M3": n3, "M4": n4, "n": n}},
          open(a_.salida, "w"), indent=1, default=float)
print(f"\n-> {a_.salida}")
