#!/usr/bin/env python3
"""POST-HOC declarado (§7 de PREREG_CORTE_SIN_ETIQUETAS.md): por que falla U-1.

No estaba en el prereg y se marca como post-hoc en el informe, igual que se hizo con
`verificar_umbral_estable.py` el 19-ago. No decide ningun veredicto: los criterios S-1..S-5 ya se
juzgaron con la corrida limpia. Esto solo explica el mecanismo del fallo.

La hipotesis a mirar: U-1 coloca el corte en z entre +0,03 y +0,15 —o sea casi en la media— mientras
el oraculo lo pone en z ~ +0,35. Si el logit NO es realmente bimodal, el EM esta ajustando dos
gaussianas a una masa unimodal y el punto de igual densidad ponderada no marca ningun valle: cae
donde la masa es mas alta. Eso explicaria por que U-1 abstiene de mas (falsa_abst 0,13-0,26) y por
que pierde contra sigma>0,5, que no estima nada.

Se mide, sobre la MISMA muestra de ajuste que usa la sonda:
  - los parametros del EM: pesos, medias, desvios, y la separacion |mu2-mu1| en unidades de sigma
  - si la mezcla ajustada es de verdad bimodal: una mezcla de dos gaussianas tiene dos modas solo si
    la separacion supera ~2 desvios (criterio clasico); por debajo la densidad tiene una sola cima
  - donde cae el valle contra donde caen las dos poblaciones REALES (que acá si se miran, porque
    esto es diagnostico y no un estimador)
"""
import os, sys, pickle, argparse, json
import numpy as np
import jax, jax.numpy as jnp

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
import idioma as I, datos as DAT, modelo as M

# Las funciones van duplicadas a proposito en vez de importarse de `sonda_sin_etiquetas`: aquel
# script no tiene guarda `__main__`, asi que importarlo volveria a correr la sonda entera. Se copian
# IDENTICAS —mismo init, mismas guardas de degeneracion, misma grilla— para que el diagnostico mire
# exactamente el ajuste que hizo U-1 y no una variante.
NOSE = I.STOI["NOSE"]
UNIDADES = ["1_s0", "2_s0", "3_s0", "3_s1", "3_s2", "4_s0", "4_s1", "4_s2"]
SEM_AJUSTE = 90000
EM_ITERS, EM_TOL = 200, 1e-6
PESO_MIN, SD_MIN, SEP_MIN = 0.02, 1e-6, 0.05


def em_dos_gaussianas(a):
    q1, q3 = np.quantile(a, [0.25, 0.75])
    mu = np.array([q1, q3], float)
    sd = np.array([a.std(), a.std()], float)
    pi = np.array([0.5, 0.5])
    if sd[0] < SD_MIN:
        return None
    ll_ant = -np.inf
    for _ in range(EM_ITERS):
        d = np.stack([pi[k] * np.exp(-0.5 * ((a - mu[k]) / sd[k]) ** 2) / (sd[k] * np.sqrt(2 * np.pi))
                      for k in range(2)])
        tot = d.sum(0)
        tot = np.where(tot <= 0, 1e-300, tot)
        r = d / tot
        nk = r.sum(1)
        if nk.min() <= 0:
            return None
        pi = nk / len(a)
        mu = (r * a).sum(1) / nk
        sd = np.sqrt(np.maximum((r * (a - mu[:, None]) ** 2).sum(1) / nk, 1e-12))
        ll = np.log(tot).sum()
        if abs(ll - ll_ant) < EM_TOL:
            break
        ll_ant = ll
    o = np.argsort(mu)
    return pi[o], mu[o], sd[o]


def valle(a):
    aj = em_dos_gaussianas(a)
    if aj is None:
        return None, "EM no converge"
    pi, mu, sd = aj
    if pi.min() < PESO_MIN or sd.min() < SD_MIN or (mu[1] - mu[0]) < SEP_MIN * a.std():
        return None, "componente degenerado"
    x = np.linspace(mu[0], mu[1], 10001)
    lg = [np.log(pi[k]) - np.log(sd[k]) - 0.5 * ((x - mu[k]) / sd[k]) ** 2 for k in range(2)]
    return float(x[np.argmin(np.abs(lg[0] - lg[1]))]), None


def partes(params, ses, cortes, turnos, mask, cons, pos):
    archivo = M.escribir(params, ses, cortes)
    lg, a = M.responder_con_abst(params, archivo, turnos, cons, mask)
    lg = jnp.take_along_axis(lg, pos[:, None, None], axis=1)[:, 0, :]
    a = jnp.take_along_axis(a, pos[:, None], axis=1)[:, 0]
    return lg, a


def juntar(ck, nivel, semilla, n, B, p_nose, base_rng, p_vieja=0.35):
    with open(ck, "rb") as f:
        params = jax.tree_util.tree_map(jnp.asarray, pickle.load(f)["params"])
    if "abst" not in params:
        return None
    fn = jax.jit(partes)
    rng = np.random.default_rng(base_rng + semilla)
    A, OKV, TIPO = [], [], []
    for _ in range(n):
        ses, cortes, turnos, mask, cons, pos, tgt, tipo = DAT.lote(
            rng, B, nivel=nivel, n_hechos=4, n_sesiones=4, p_vieja=p_vieja, p_nose=p_nose)
        lg, a = fn(params, jnp.array(ses), jnp.array(cortes), jnp.array(turnos),
                   jnp.array(mask), jnp.array(cons), jnp.array(pos))
        lg = np.asarray(lg).copy()
        lg[:, NOSE] = -np.inf
        A.append(np.asarray(a)); OKV.append(lg.argmax(-1) == tgt); TIPO.append(tipo)
    return np.concatenate(A), np.concatenate(OKV), np.concatenate(TIPO)

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=32)
ap.add_argument("--batch", type=int, default=64)
ap.add_argument("--p-nose", type=float, default=0.4)
ap.add_argument("--ckpt-alt", action="append", default=[], metavar="UNIDAD=RUTA")
ap.add_argument("--salida", default=os.path.join(AQUI, "posthoc_mezcla_20260820.json"))
a_ = ap.parse_args()
ALT = dict(x.split("=", 1) for x in a_.ckpt_alt)

print("POST-HOC · por que falla U-1 — no decide ningun veredicto (§7 del prereg)\n")
print(f"{'unidad':<8} {'pi1':>6} {'pi2':>6} {'mu1':>8} {'mu2':>8} {'sep/sd':>8} {'bimodal':>8} "
      f"{'valle z':>9} {'z real sep':>11}")
print("-" * 82)

res = {}
for u in UNIDADES:
    ck = ALT.get(u, os.path.join(AQUI, "ckpts", f"c{u}.pkl"))
    if not os.path.isabs(ck):
        ck = os.path.join(AQUI, ck)
    if not os.path.exists(ck):
        continue
    nivel, semilla = int(u[0]), int(u.split("_s")[1])
    d = juntar(ck, nivel, semilla, a_.n, a_.batch, a_.p_nose, SEM_AJUSTE)
    if d is None:
        continue
    A, OKV, TIPO = d
    mu_g, sd_g = A.mean(), A.std()
    aj = em_dos_gaussianas(A)
    if aj is None:
        print(f"c{u:<7} EM no converge")
        continue
    pi, mu, sd = aj
    # Separacion en desvios: la mezcla es bimodal solo si supera ~2 (criterio clasico para dos
    # gaussianas de peso parecido). Por debajo, la densidad tiene UNA cima y no hay valle que buscar.
    sep = (mu[1] - mu[0]) / np.sqrt((sd ** 2).mean())
    t, _ = valle(A)
    zval = (t - mu_g) / sd_g if t is not None else np.nan
    # Las poblaciones REALES, que un estimador sin etiquetas no puede ver pero el diagnostico si:
    con, sin = A[TIPO < 2], A[TIPO >= 2]
    zsep = (sin.mean() - con.mean()) / sd_g
    res[u] = {"pi": pi.tolist(), "mu": mu.tolist(), "sd": sd.tolist(), "sep_sd": float(sep),
              "bimodal": bool(sep > 2.0), "valle_z": float(zval), "sep_real_z": float(zsep),
              "mu_con": float(con.mean()), "mu_sin": float(sin.mean())}
    print(f"c{u:<7} {pi[0]:>6.3f} {pi[1]:>6.3f} {mu[0]:>8.3f} {mu[1]:>8.3f} {sep:>8.3f} "
          f"{('SI' if sep > 2 else 'no'):>8} {zval:>+9.3f} {zsep:>+11.3f}")

print("-" * 82)
nb = sum(1 for v in res.values() if not v["bimodal"])
print(f"mezclas NO bimodales (separacion <= 2 desvios): {nb}/{len(res)}")
print("Si son casi todas, U-1 estaba buscando un valle que no existe: el EM parte en dos una masa")
print("unimodal y el punto de igual densidad cae donde hay mas masa, no en una frontera.")
json.dump(res, open(a_.salida, "w"), indent=1, default=float)
print(f"\n-> {a_.salida}")
