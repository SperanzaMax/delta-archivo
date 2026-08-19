#!/usr/bin/env python3
"""VERIFICACION POST-HOC del resultado de sonda_umbral_prospectiva.py. No esta en el prereg.

Por que existe: U-2 se cumplio, pero `c3_s0` fallaba el criterio por 0,0177 (falsa_abst 0,1177 contra
0,10). Eso es un fallo PEGADO AL BORDE, y la leccion que dejo `sonda_umbral.py` el 18-ago es que lo
pegado al borde no es estable. Antes de escribir «es calibracion» hay que descartar la alternativa
obvia:

    ALTERNATIVA — no se movio nada real; `c3_s0` oscila alrededor del borde de muestra en muestra, y
    el «pasa» con a* es esa oscilacion y no el efecto de correr el umbral.

Dos cosas la distinguen, y las dos se miden aca:

  1. ESTABILIDAD DE a*. Se elige a* por separado en TRES muestras independientes (rng 90000, 77000 y
     55000, esta ultima nueva). Si a* salta de muestra en muestra, no hay un umbral que fijar y la
     recomendacion no existe aunque U-2 se haya cumplido una vez.

  2. LOS NUEVE CRUCES. Cada a* se evalua en las tres muestras, incluida aquella donde se lo eligio.
     Si `c3_s0` pasa en los 9 cruces, no es oscilacion de borde: pasa con umbrales elegidos en datos
     que no vio y medida en datos que no vio. Si pasa solo en la diagonal, es sobreajuste.

Tambien se reporta `falsa_abst` con a>0 en las tres muestras: es la dispersion del numero que fallo,
o sea cuanto de los 0,0177 que le faltaban es simplemente ruido de muestreo.
"""
import os, sys, json
import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
# Las cuatro funciones van copiadas de sonda_umbral_prospectiva.py en vez de importadas: ese modulo
# corre su analisis entero al importarse. Es preferible duplicarlas a tocar el archivo que produjo el
# resultado del prereg. El criterio (MARGEN, N_CORTES) es el mismo y esta arriba, a la vista.
import pickle, jax, jax.numpy as jnp
import idioma as I, datos as DAT, modelo as M

NOSE = I.STOI["NOSE"]
UNIDADES = ["1_s0", "2_s0", "3_s0", "3_s1", "3_s2"]
RNGS = [90000, 77000, 55000]          # las dos del prereg + una tercera, nueva
MARGEN, N_CORTES = 0.07, 400
N, B, P_NOSE = 32, 64, 0.4


def partes(params, ses, cortes, turnos, mask, cons, pos):
    archivo = M.escribir(params, ses, cortes)
    lg, a = M.responder_con_abst(params, archivo, turnos, cons, mask)
    lg = jnp.take_along_axis(lg, pos[:, None, None], axis=1)[:, 0, :]
    a = jnp.take_along_axis(a, pos[:, None], axis=1)[:, 0]
    return lg, a


def juntar(ck, nivel, semilla, base_rng, p_vieja=0.35):
    with open(ck, "rb") as f:
        params = jax.tree_util.tree_map(jnp.asarray, pickle.load(f)["params"])
    if "abst" not in params:
        return None
    fn = jax.jit(partes)
    rng = np.random.default_rng(base_rng + semilla)
    A, OKV, TIPO = [], [], []
    for _ in range(N):
        ses, cortes, turnos, mask, cons, pos, tgt, tipo = DAT.lote(
            rng, B, nivel=nivel, n_hechos=4, n_sesiones=4, p_vieja=p_vieja, p_nose=P_NOSE)
        lg, a = fn(params, jnp.array(ses), jnp.array(cortes), jnp.array(turnos),
                   jnp.array(mask), jnp.array(cons), jnp.array(pos))
        lg = np.asarray(lg).copy()
        lg[:, NOSE] = -np.inf
        A.append(np.asarray(a)); OKV.append(lg.argmax(-1) == tgt); TIPO.append(tipo)
    return np.concatenate(A), np.concatenate(OKV), np.concatenate(TIPO)


def metricas(a, okv, tipo, umbral):
    abst = a > umbral
    sin_resp = tipo >= 2
    hay = ~sin_resp
    nose = abst[sin_resp].mean() if sin_resp.any() else np.nan
    falsa = abst[hay].mean() if hay.any() else np.nan
    return float(nose), float(falsa)


def elegir(a, okv, tipo):
    mejor = None
    for t in np.quantile(a, np.linspace(0.001, 0.999, N_CORTES)):
        nn, ff = metricas(a, okv, tipo, t)
        if ff <= MARGEN and nn >= 0.50 and (mejor is None or nn > mejor[1]):
            mejor = (float(t), nn, ff)
    return mejor


def pasa(nose, falsa):
    return bool(falsa <= 0.10 and nose >= 0.50)


print("POST-HOC · no esta en el prereg. Descarta la alternativa «oscilacion de borde».")
print(f"Tres muestras independientes de {N * B} por unidad: rng {RNGS}\n")

res = {}
for u in UNIDADES:
    ck = os.path.join(AQUI, "ckpts", f"c{u}.pkl")
    if not os.path.exists(ck):
        continue
    nivel, semilla = int(u[0]), int(u.split("_s")[1])
    ms = [juntar(ck, nivel, semilla, r) for r in RNGS]
    if any(m is None for m in ms):
        continue
    astar = [elegir(*m) for m in ms]
    base = [metricas(m[0], m[1], m[2], 0.0) for m in ms]        # con a>0, el criterio del prereg
    cruces = []
    for i, ae in enumerate(astar):
        fila = []
        for j, m in enumerate(ms):
            fila.append(metricas(m[0], m[1], m[2], ae[0]) if ae else None)
        cruces.append(fila)
    res[u] = {"a_estrella": [a[0] if a else None for a in astar],
              "con_cero": base,
              "cruces": cruces}

print("1 · ESTABILIDAD DE a* — elegido por separado en cada muestra")
print(f"{'unidad':<8} {'a* (90000)':>11} {'a* (77000)':>11} {'a* (55000)':>11} {'desvio':>9} {'rango':>8}")
print("-" * 64)
for u, r in res.items():
    v = [x for x in r["a_estrella"] if x is not None]
    sd = np.std(v) if len(v) == 3 else float("nan")
    rg = (max(v) - min(v)) if len(v) == 3 else float("nan")
    cel = [f"{x:.3f}" if x is not None else "—" for x in r["a_estrella"]]
    print(f"c{u:<7} {cel[0]:>11} {cel[1]:>11} {cel[2]:>11} {sd:>9.3f} {rg:>8.3f}")

print()
print("2 · falsa_abst CON a>0 EN LAS TRES MUESTRAS — cuanta de la falla de c3_s0 es ruido")
print(f"{'unidad':<8} {'m1':>9} {'m2':>9} {'m3':>9} {'media':>9} {'desvio':>9} {'pasa 0,10?':>12}")
print("-" * 70)
for u, r in res.items():
    f = [b[1] for b in r["con_cero"]]
    n = [b[0] for b in r["con_cero"]]
    ok = sum(pasa(nn, ff) for nn, ff in zip(n, f))
    print(f"c{u:<7} {f[0]:>9.4f} {f[1]:>9.4f} {f[2]:>9.4f} {np.mean(f):>9.4f} {np.std(f):>9.4f} "
          f"{f'{ok}/3':>12}")

print()
print("3 · LOS NUEVE CRUCES — a* elegido en la fila, medido en la columna. Fuera de la diagonal el")
print("    umbral nunca vio esos datos. Si pasa 9/9, no es oscilacion de borde.")
for u, r in res.items():
    ok = 0
    print(f"\n  c{u}")
    print(f"  {'a* de':<10} {'en m1':>18} {'en m2':>18} {'en m3':>18}")
    for i, fila in enumerate(r["cruces"]):
        cel = []
        for j, m in enumerate(fila):
            if m is None:
                cel.append("—"); continue
            nn, ff = m
            p = pasa(nn, ff); ok += p
            cel.append(f"{ff:.4f}/{nn:.4f} {'SI' if p else 'no'}")
        print(f"  {'m' + str(i + 1):<10} {cel[0]:>18} {cel[1]:>18} {cel[2]:>18}")
    print(f"  -> pasa {ok}/9")
    res[u]["pasa_cruces"] = ok

with open(os.path.join(AQUI, "umbral_estabilidad_20260819.json"), "w") as f:
    json.dump({"nota": "POST-HOC, no prereg", "rngs": RNGS, "unidades": res}, f, indent=1)
print("\n-> umbral_estabilidad_20260819.json")
