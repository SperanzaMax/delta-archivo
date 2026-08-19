#!/usr/bin/env python3
"""Version PROSPECTIVA de sonda_umbral.py — PREREG_UMBRAL_PROSPECTIVO.md (SHA b0ef03b9...).

Que cambia respecto de la del 18-ago, y por que:

  1. LAS DOS MUESTRAS SON INDEPENDIENTES EN LOS DATOS. Aquella partia UN muestreo en dos mitades por
     permutacion, asi que las sesiones y los hechos eran los mismos de los dos lados. Aca el ajuste
     usa el rng 90000+semilla (el de `evaluar`) y la prueba el 77000+semilla, fijado en el prereg
     antes de correr. Son lotes distintos, no dos vistas del mismo lote.

  2. OCHO VECES LAS MUESTRAS. 32 lotes de 64 = 2048 por unidad, contra 256. El informe del 18 midio
     el costo de la muestra chica: 0,065 de error en `nose` sobre c3_s1.

  3. LOS TRES CONTROLES del §4, que son lo que impide leer U-2 como mas de lo que es:
       C-A  el mismo barrido sobre el logit estandarizado — separa DESPLAZAMIENTO de forma
       C-B  el a* de c3_s0 aplicado tal cual a c3_s1 y c3_s2 — ¿el sesgo es de la unidad o del nivel?
       C-C  el nulo por permutacion, 20 repeticiones — cuanto pasa este mismo buscador SIN informacion

El criterio de eleccion de a* no se toca: maximizar `nose` sujeto a `falsa_abst` <= 0,07 (margen) y
`nose` >= 0,50, sobre 400 cortes entre los cuantiles 0,001 y 0,999. Se juzga con el criterio real,
`falsa_abst` <= 0,10 y `nose` >= 0,50.
"""
import os, sys, pickle, argparse, json
import numpy as np
import jax, jax.numpy as jnp

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
import idioma as I, datos as DAT, modelo as M

NOSE = I.STOI["NOSE"]
UNIDADES = ["1_s0", "2_s0", "3_s0", "3_s1", "3_s2"]
SEM_AJUSTE, SEM_PRUEBA = 90000, 77000      # PREREG §2, fijadas antes de correr
MARGEN, N_CORTES = 0.07, 400               # PREREG §2, identicos al 18-ago
REPS_NULO = 20                             # PREREG §4 C-C


def partes(params, ses, cortes, turnos, mask, cons, pos):
    archivo = M.escribir(params, ses, cortes)
    lg, a = M.responder_con_abst(params, archivo, turnos, cons, mask)
    lg = jnp.take_along_axis(lg, pos[:, None, None], axis=1)[:, 0, :]
    a = jnp.take_along_axis(a, pos[:, None], axis=1)[:, 0]
    return lg, a


def juntar(ck, nivel, semilla, n, B, p_nose, base_rng, p_vieja=0.35):
    """Igual que en la sonda del 18-ago, pero con la semilla del generador como parametro."""
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
        A.append(np.asarray(a))
        OKV.append(lg.argmax(-1) == tgt)
        TIPO.append(tipo)
    return np.concatenate(A), np.concatenate(OKV), np.concatenate(TIPO)


def metricas(a, okv, tipo, umbral):
    abst = a > umbral
    sin_resp = tipo >= 2
    hay = ~sin_resp
    nose = abst[sin_resp].mean() if sin_resp.any() else np.nan
    falsa = abst[hay].mean() if hay.any() else np.nan
    vig = tipo == 0
    vigente = (okv[vig] & ~abst[vig]).mean() if vig.any() else np.nan
    return nose, falsa, vigente


def auc(a, tipo):
    pos, neg = a[tipo >= 2], a[tipo < 2]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    r = np.argsort(np.argsort(np.concatenate([pos, neg]))) + 1
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def elegir(a, okv, tipo):
    """a* = el corte que maximiza `nose` con `falsa_abst` <= MARGEN y `nose` >= 0,50. None si ninguno."""
    mejor = None
    for t in np.quantile(a, np.linspace(0.001, 0.999, N_CORTES)):
        nn, ff, _ = metricas(a, okv, tipo, t)
        if ff <= MARGEN and nn >= 0.50 and (mejor is None or nn > mejor[1]):
            mejor = (float(t), float(nn), float(ff))
    return mejor


def pasa(nose, falsa):
    return bool(falsa <= 0.10 and nose >= 0.50)


ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=32, help="lotes por unidad y por muestra")
ap.add_argument("--batch", type=int, default=64)
ap.add_argument("--p-nose", type=float, default=0.4)
ap.add_argument("--salida", default=os.path.join(AQUI, "umbral_prospectivo_20260819.json"))
a_ = ap.parse_args()

print(f"PROSPECTIVO — PREREG_UMBRAL_PROSPECTIVO.md · ajuste rng {SEM_AJUSTE}+s · prueba rng "
      f"{SEM_PRUEBA}+s · {a_.n}x{a_.batch} = {a_.n * a_.batch} muestras por unidad y muestra\n")

datos, res = {}, {}
for u in UNIDADES:
    ck = os.path.join(AQUI, "ckpts", f"c{u}.pkl")
    if not os.path.exists(ck):
        print(f"c{u}: sin checkpoint, se saltea")
        continue
    nivel, semilla = int(u[0]), int(u.split("_s")[1])
    aj = juntar(ck, nivel, semilla, a_.n, a_.batch, a_.p_nose, SEM_AJUSTE)
    pr = juntar(ck, nivel, semilla, a_.n, a_.batch, a_.p_nose, SEM_PRUEBA)
    if aj is None or pr is None:
        print(f"c{u}: el checkpoint no tiene cabeza, se saltea")
        continue
    datos[u] = (aj, pr)
    print(f"c{u}: muestreado ({len(aj[0])} ajuste + {len(pr[0])} prueba)", flush=True)

print()
print("=" * 96)
print("PRINCIPAL · a* se ELIGE en ajuste, se IMPRIME, y recien despues se mide en la muestra fresca")
print("=" * 96)
print(f"{'unidad':<8} {'AUC pr':>7} {'a>0 en prueba':>22} {'a*':>8} {'en PRUEBA con a*':>26}")
print(f"{'':8} {'':>7} {'f_abst':>10} {'nose':>10} {'':>8} {'f_abst':>10} {'nose':>9} {'pasa?':>6}")
print("-" * 96)
for u, ((Aa, Oa, Ta), (Ap, Op, Tp)) in datos.items():
    ar = auc(Ap, Tp)
    n0, f0, _ = metricas(Ap, Op, Tp, 0.0)
    m = elegir(Aa, Oa, Ta)
    fila = {"auc_prueba": float(ar), "prereg_falsa": float(f0), "prereg_nose": float(n0),
            "prereg_pasa": pasa(n0, f0)}
    if m is None:
        fila["a_estrella"] = None
        print(f"c{u:<7} {ar:>7.3f} {f0:>10.4f} {n0:>10.4f} {'—':>8} {'—':>10} {'—':>9} {'NO':>6}")
    else:
        t = m[0]
        nn, ff, vv = metricas(Ap, Op, Tp, t)
        fila.update({"a_estrella": t, "ajuste_nose": m[1], "ajuste_falsa": m[2],
                     "prueba_falsa": float(ff), "prueba_nose": float(nn),
                     "prueba_vigente": float(vv), "prueba_pasa": pasa(nn, ff)})
        print(f"c{u:<7} {ar:>7.3f} {f0:>10.4f} {n0:>10.4f} {t:>8.3f} {ff:>10.4f} {nn:>9.4f} "
              f"{('SI' if pasa(nn, ff) else 'no'):>6}")
    res[u] = fila

print()
print("C-A · DESPLAZAMIENTO CONTRA FORMA — el mismo barrido sobre z = (a-mu)/sigma de la muestra de")
print("      ajuste. Si z* cae cerca de 0 en todas, lo que esta mal es un BIAS, no un umbral fino.")
print(f"{'unidad':<8} {'mu(aj)':>9} {'sigma(aj)':>10} {'a*':>9} {'z* = (a*-mu)/sigma':>20}")
print("-" * 62)
for u, ((Aa, _, _), _) in datos.items():
    mu, sd = float(Aa.mean()), float(Aa.std())
    t = res[u].get("a_estrella")
    z = (t - mu) / sd if t is not None else None
    res[u].update({"mu_ajuste": mu, "sigma_ajuste": sd, "z_estrella": z})
    print(f"c{u:<7} {mu:>9.3f} {sd:>10.3f} "
          f"{('%.3f' % t) if t is not None else '—':>9} {('%+.3f' % z) if z is not None else '—':>20}")

print()
print("C-B · TRANSFERENCIA — el a* de c3_s0 aplicado TAL CUAL a las otras dos unidades del nivel 3.")
print("      Si transfiere, el sesgo es del NIVEL y se fija una vez; si no, cada unidad necesita el suyo.")
t30 = res.get("3_s0", {}).get("a_estrella")
if t30 is not None:
    print(f"{'unidad':<8} {'f_abst':>10} {'nose':>9} {'pasa?':>6}   (con a* = {t30:.3f}, el de c3_s0)")
    print("-" * 62)
    for u in ["3_s1", "3_s2"]:
        if u not in datos:
            continue
        (_, (Ap, Op, Tp)) = datos[u]
        nn, ff, _ = metricas(Ap, Op, Tp, t30)
        res[u]["transfer_c3s0"] = {"falsa": float(ff), "nose": float(nn), "pasa": pasa(nn, ff)}
        print(f"c{u:<7} {ff:>10.4f} {nn:>9.4f} {('SI' if pasa(nn, ff) else 'no'):>6}")
else:
    print("      c3_s0 no dio a*; el control no aplica.")

print()
print(f"C-C · EL NULO — mismo buscador de 400 cortes sobre `a` PERMUTADO contra las etiquetas,")
print(f"      {REPS_NULO} repeticiones. Es la tasa de «pasa» que se consigue SIN informacion.")
print(f"{'unidad':<8} {'a* hallado':>12} {'pasa en prueba':>16}")
print("-" * 44)
rs = np.random.default_rng(4242)
for u, ((Aa, Oa, Ta), (Ap, Op, Tp)) in datos.items():
    hall = pasan = 0
    for _ in range(REPS_NULO):
        perm = rs.permutation(len(Aa))
        m = elegir(Aa[perm], Oa[perm], Ta)          # el logit se desalinea de su etiqueta
        if m is None:
            continue
        hall += 1
        permp = rs.permutation(len(Ap))
        nn, ff, _ = metricas(Ap[permp], Op[permp], Tp, m[0])
        pasan += pasa(nn, ff)
    res[u]["nulo"] = {"a_hallado": hall, "pasa": pasan, "reps": REPS_NULO}
    print(f"c{u:<7} {f'{hall}/{REPS_NULO}':>12} {f'{pasan}/{REPS_NULO}':>16}")

with open(a_.salida, "w") as f:
    json.dump({"prereg": "PREREG_UMBRAL_PROSPECTIVO.md",
               "config": {"n": a_.n, "batch": a_.batch, "p_nose": a_.p_nose,
                          "rng_ajuste": SEM_AJUSTE, "rng_prueba": SEM_PRUEBA,
                          "margen": MARGEN, "cortes": N_CORTES, "reps_nulo": REPS_NULO},
               "unidades": res}, f, indent=1)
print(f"\n-> {a_.salida}")
