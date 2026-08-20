#!/usr/bin/env python3
"""El corte SIN etiquetas — PREREG_CORTE_SIN_ETIQUETAS.md (SHA 17e0a35e...).

La sonda del 19-ago demostro que la informacion esta en el logit, pero elegia `a*` MIRANDO las
etiquetas. Aca los tres estimadores del §4 estiman el corte sobre la muestra de ajuste y se juzgan
sobre la de prueba, y el principal (U-1) no mira ni una etiqueta:

  U-1  valle de una mezcla de dos gaussianas ajustada por EM sobre `a`     — cero etiquetas
  U-2  constante transferida `mu + s*z*sigma`, leave-one-out               — etiquetas de OTRAS unidades
  U-3  cuantil `1 - p_nose`                                                — usa la tasa base de diseno

Contrastes ya medidos que se imprimen al lado: el oraculo `a*` (techo, con etiquetas) y sigma>0,5
(piso, sin calibrar).

El nulo del §5 S-3 NO es permutar etiquetas —U-1 no las mira, el corte no se moveria— sino reemplazar
`a` por una gaussiana de la misma media y desvio: destruye la estructura bimodal y conserva los dos
momentos que U-1 y U-2 usan.
"""
import os, sys, pickle, argparse, json
import numpy as np
import jax, jax.numpy as jnp

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
import idioma as I, datos as DAT, modelo as M

NOSE = I.STOI["NOSE"]
UNIDADES = ["1_s0", "2_s0", "3_s0", "3_s1", "3_s2", "4_s0", "4_s1", "4_s2"]   # PREREG §3
SEM_AJUSTE, SEM_PRUEBA = 90000, 77000     # PREREG §3, las mismas del 19-ago
MARGEN, N_CORTES = 0.07, 400              # solo para el ORACULO, que es contraste
REPS_NULO = 100                           # PREREG §5 S-3
EM_ITERS, EM_TOL = 200, 1e-6              # PREREG §4 U-1
PESO_MIN, SD_MIN, SEP_MIN = 0.02, 1e-6, 0.05


# ---------------------------------------------------------------- muestreo (identico al 19-ago)
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
        A.append(np.asarray(a))
        OKV.append(lg.argmax(-1) == tgt)
        TIPO.append(tipo)
    return np.concatenate(A), np.concatenate(OKV), np.concatenate(TIPO)


# ---------------------------------------------------------------- metricas y criterio
def metricas(a, okv, tipo, umbral):
    abst = a > umbral
    sin_resp = tipo >= 2
    hay = ~sin_resp
    nose = abst[sin_resp].mean() if sin_resp.any() else np.nan
    falsa = abst[hay].mean() if hay.any() else np.nan
    vig = tipo == 0
    vigente = (okv[vig] & ~abst[vig]).mean() if vig.any() else np.nan
    return nose, falsa, vigente


def pasa(nose, falsa):
    return bool(falsa <= 0.10 and nose >= 0.50)


def oraculo(a, okv, tipo):
    """El techo: a* elegido CON etiquetas, tal cual el 19-ago. Contraste, no estimador."""
    mejor = None
    for t in np.quantile(a, np.linspace(0.001, 0.999, N_CORTES)):
        nn, ff, _ = metricas(a, okv, tipo, t)
        if ff <= MARGEN and nn >= 0.50 and (mejor is None or nn > mejor[1]):
            mejor = (float(t), float(nn), float(ff))
    return mejor


# ---------------------------------------------------------------- U-1: el valle de la mezcla
def em_dos_gaussianas(a):
    """EM 1-D, 2 componentes, init determinista del PREREG §4. Devuelve (pi, mu, sd) o None."""
    q1, q3 = np.quantile(a, [0.25, 0.75])
    mu = np.array([q1, q3], float)
    sd = np.array([a.std(), a.std()], float)
    pi = np.array([0.5, 0.5])
    if sd[0] < SD_MIN:
        return None
    ll_ant = -np.inf
    for _ in range(EM_ITERS):
        # E
        d = np.stack([pi[k] * np.exp(-0.5 * ((a - mu[k]) / sd[k]) ** 2) / (sd[k] * np.sqrt(2 * np.pi))
                      for k in range(2)])
        tot = d.sum(0)
        tot = np.where(tot <= 0, 1e-300, tot)
        r = d / tot
        # M
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
    """Corte U-1: el punto ENTRE las dos medias donde las densidades ponderadas se igualan.

    Se resuelve por grilla fina en vez de por la cuadratica: es determinista, no tiene casos borde
    con raices fuera del intervalo, y la resolucion (1e-4 del ancho) es tres ordenes mas fina que
    cualquier diferencia que mueva una metrica.
    """
    aj = em_dos_gaussianas(a)
    if aj is None:
        return None, "EM no converge"
    pi, mu, sd = aj
    if pi.min() < PESO_MIN:
        return None, f"componente colapsado (peso {pi.min():.4f})"
    if sd.min() < SD_MIN:
        return None, "componente colapsado (desvio)"
    if (mu[1] - mu[0]) < SEP_MIN * a.std():
        return None, f"medias pegadas ({(mu[1]-mu[0])/a.std():.4f} sigma)"
    x = np.linspace(mu[0], mu[1], 10001)
    lg = [np.log(pi[k]) - np.log(sd[k]) - 0.5 * ((x - mu[k]) / sd[k]) ** 2 for k in range(2)]
    return float(x[np.argmin(np.abs(lg[0] - lg[1]))]), None


# ---------------------------------------------------------------- U-2 / U-3
def asimetria(a):
    z = (a - a.mean()) / a.std()
    return float((z ** 3).mean())


def signo_loo(skew_u, donantes):
    """Orientacion elegida por MAYORIA en las otras unidades: s = o * sign(skew). PREREG §4 U-2."""
    mejor_o, mejor_ac = 1, -1
    for o in (1, -1):
        ac = sum(1 for sk, zs in donantes if np.sign(o * np.sign(sk)) == np.sign(zs))
        if ac > mejor_ac:
            mejor_o, mejor_ac = o, ac
    return mejor_o * np.sign(skew_u), mejor_o


ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=32)
ap.add_argument("--batch", type=int, default=64)
ap.add_argument("--p-nose", type=float, default=0.4)
ap.add_argument("--salida", default=os.path.join(AQUI, "corte_sin_etiquetas_20260820.json"))
a_ = ap.parse_args()

print(f"CORTE SIN ETIQUETAS — PREREG_CORTE_SIN_ETIQUETAS.md (SHA 17e0a35e...)")
print(f"ajuste rng {SEM_AJUSTE}+s · prueba rng {SEM_PRUEBA}+s · "
      f"{a_.n}x{a_.batch} = {a_.n * a_.batch} muestras por unidad y muestra · p_nose {a_.p_nose}\n")

datos = {}
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

# --- referencias (oraculo y sigma>0,5) y z* de cada unidad, que U-2 necesita como donantes
ref = {}
for u, ((Aa, Oa, Ta), (Ap, Op, Tp)) in datos.items():
    mu, sd = float(Aa.mean()), float(Aa.std())
    orc = oraculo(Aa, Oa, Ta)
    fila = {"mu": mu, "sigma": sd, "skew": asimetria(Aa)}
    n0, f0, _ = metricas(Ap, Op, Tp, 0.0)
    fila["sigma05"] = {"falsa": float(f0), "nose": float(n0), "pasa": pasa(n0, f0)}
    if orc is None:
        fila["oraculo"] = None
    else:
        nn, ff, vv = metricas(Ap, Op, Tp, orc[0])
        fila["oraculo"] = {"a": orc[0], "z": (orc[0] - mu) / sd, "falsa": float(ff),
                           "nose": float(nn), "vigente": float(vv), "pasa": pasa(nn, ff)}
    ref[u] = fila

res = {u: {} for u in datos}

print()
print("=" * 104)
print("PRINCIPAL · U-1 — el corte se estima con la MEZCLA sobre el ajuste, sin mirar una etiqueta,")
print("             y se mide en la muestra fresca. Al lado, el techo (oraculo) y el piso (sigma>0,5).")
print("=" * 104)
print(f"{'unidad':<8} {'U-1 a':>9} {'z(U-1)':>8} {'f_abst':>9} {'nose':>8} {'pasa':>6} | "
      f"{'ORACULO f/nose':>18} {'pasa':>5} | {'sigma>0,5 f/nose':>18} {'pasa':>5}")
print("-" * 104)
u1_pasa = 0
for u, ((Aa, Oa, Ta), (Ap, Op, Tp)) in datos.items():
    t, motivo = valle(Aa)
    o, s05 = ref[u]["oraculo"], ref[u]["sigma05"]
    ostr = f"{o['falsa']:.4f}/{o['nose']:.4f}" if o else "—"
    opas = ("SI" if o["pasa"] else "no") if o else "—"
    sstr = f"{s05['falsa']:.4f}/{s05['nose']:.4f}"
    spas = "SI" if s05["pasa"] else "no"
    if t is None:
        res[u]["u1"] = {"a": None, "motivo": motivo, "pasa": False}
        print(f"c{u:<7} {'sin corte':>9} {'—':>8} {'—':>9} {'—':>8} {'NO':>6} | "
              f"{ostr:>18} {opas:>5} | {sstr:>18} {spas:>5}   ({motivo})")
        continue
    nn, ff, vv = metricas(Ap, Op, Tp, t)
    p = pasa(nn, ff)
    u1_pasa += p
    z = (t - ref[u]["mu"]) / ref[u]["sigma"]
    res[u]["u1"] = {"a": t, "z": float(z), "falsa": float(ff), "nose": float(nn),
                    "vigente": float(vv), "pasa": p}
    print(f"c{u:<7} {t:>9.3f} {z:>+8.3f} {ff:>9.4f} {nn:>8.4f} {('SI' if p else 'no'):>6} | "
          f"{ostr:>18} {opas:>5} | {sstr:>18} {spas:>5}")

n_u = len(datos)
print("-" * 104)
print(f"S-1 · U-1 pasa en {u1_pasa}/{n_u}  (criterio: >= 6/8)   -> "
      f"{'CUMPLE' if u1_pasa >= 6 else 'NO CUMPLE'}")
s4 = sum(1 for u in datos if not ref[u]["sigma05"]["pasa"])
print(f"S-4 · sigma>0,5 falla en {s4}/{n_u}  (criterio: >= 6/8)  -> "
      f"{'CUMPLE' if s4 >= 6 else 'NO CUMPLE'}")

# --- S-2: costo contra el oraculo
print()
print("S-2 · COSTO CONTRA EL ORACULO — cuanto `nose` se paga por no mirar las etiquetas.")
print(f"{'unidad':<8} {'nose U-1':>10} {'nose a*':>10} {'caida':>9} {'f_abst U-1':>12}")
print("-" * 54)
caidas = []
for u in datos:
    o, m = ref[u]["oraculo"], res[u]["u1"]
    if o is None or m["a"] is None:
        print(f"c{u:<7} {'—':>10} {'—':>10} {'—':>9} {'—':>12}")
        continue
    c = o["nose"] - m["nose"]
    caidas.append((c, m["falsa"]))
    print(f"c{u:<7} {m['nose']:>10.4f} {o['nose']:>10.4f} {c:>+9.4f} {m['falsa']:>12.4f}")
if caidas:
    cm = float(np.mean([c for c, _ in caidas]))
    fmax = max(f for _, f in caidas)
    ok2 = cm <= 0.10 and fmax <= 0.10
    print(f"caida media {cm:+.4f} · f_abst maxima {fmax:.4f}  (criterio: caida <= 0,10 y f_abst <= 0,10)"
          f"  -> {'CUMPLE' if ok2 else 'NO CUMPLE'}")
    res["S2"] = {"caida_media": cm, "falsa_max": float(fmax), "cumple": bool(ok2)}

# --- U-2 y S-5: el signo
print()
print("U-2 · CONSTANTE TRANSFERIDA leave-one-out, y S-5 el signo por asimetria")
print(f"{'unidad':<8} {'skew':>9} {'z* real':>9} {'signo pred':>11} {'ok':>4} {'z barra':>9} "
      f"{'U-2 a':>9} {'f_abst':>9} {'nose':>8} {'pasa':>6}")
print("-" * 92)
donantes_all = [(u, ref[u]["skew"], ref[u]["oraculo"]["z"]) for u in datos if ref[u]["oraculo"]]
s5_ok, u2_pasa = 0, 0
for u, ((Aa, Oa, Ta), (Ap, Op, Tp)) in datos.items():
    don = [(sk, z) for (v, sk, z) in donantes_all if v != u]
    if not don:
        continue
    zb = float(np.median([abs(z) for _, z in don]))
    s, orient = signo_loo(ref[u]["skew"], don)
    t = ref[u]["mu"] + s * zb * ref[u]["sigma"]
    nn, ff, _ = metricas(Ap, Op, Tp, t)
    p = pasa(nn, ff)
    u2_pasa += p
    zr = ref[u]["oraculo"]["z"] if ref[u]["oraculo"] else None
    ok = (zr is not None) and (np.sign(s) == np.sign(zr))
    s5_ok += bool(ok)
    res[u]["u2"] = {"a": float(t), "z_barra": zb, "signo": int(s), "orientacion": int(orient),
                    "skew": ref[u]["skew"], "z_real": zr, "signo_ok": bool(ok),
                    "falsa": float(ff), "nose": float(nn), "pasa": p}
    print(f"c{u:<7} {ref[u]['skew']:>+9.3f} {(f'{zr:+.3f}' if zr is not None else '—'):>9} "
          f"{s:>+11.0f} {('SI' if ok else 'no'):>4} {zb:>9.3f} {t:>9.3f} {ff:>9.4f} {nn:>8.4f} "
          f"{('SI' if p else 'no'):>6}")
print("-" * 92)
print(f"S-5 · el signo se acierta en {s5_ok}/{len(donantes_all)}  (criterio: >= 7/8)  -> "
      f"{'CUMPLE' if s5_ok >= 7 else 'NO CUMPLE'}")
print(f"       U-2 pasa la compuerta en {u2_pasa}/{n_u} (secundario, sin criterio propio)")

# --- U-3
print()
print(f"U-3 · CUANTIL DE LA TASA BASE (1 - p_nose = {1 - a_.p_nose:.2f}) — la linea de base honesta.")
print(f"       Si iguala a U-1, U-1 no aporta nada y hay que decirlo (PREREG §4).")
print(f"{'unidad':<8} {'U-3 a':>9} {'f_abst':>9} {'nose':>8} {'pasa':>6}")
print("-" * 44)
u3_pasa = 0
for u, ((Aa, Oa, Ta), (Ap, Op, Tp)) in datos.items():
    t = float(np.quantile(Aa, 1 - a_.p_nose))
    nn, ff, _ = metricas(Ap, Op, Tp, t)
    p = pasa(nn, ff)
    u3_pasa += p
    res[u]["u3"] = {"a": t, "falsa": float(ff), "nose": float(nn), "pasa": p}
    print(f"c{u:<7} {t:>9.3f} {ff:>9.4f} {nn:>8.4f} {('SI' if p else 'no'):>6}")
print("-" * 44)
print(f"       U-3 pasa en {u3_pasa}/{n_u}")

# --- S-3: el nulo
print()
print("S-3 · EL NULO — `a` reemplazado por una gaussiana de la MISMA media y desvio. Conserva los dos")
print(f"      momentos que U-1 usa y destruye la estructura bimodal. {REPS_NULO} repeticiones.")
print(f"{'unidad':<8} {'corte hallado':>15} {'pasa':>8}")
print("-" * 36)
rs = np.random.default_rng(20260820)
nulo_tot = []
for u, ((Aa, Oa, Ta), (Ap, Op, Tp)) in datos.items():
    hall = pas = 0
    mu, sd = ref[u]["mu"], ref[u]["sigma"]
    for _ in range(REPS_NULO):
        falso = rs.normal(mu, sd, size=len(Aa))
        t, _m = valle(falso)
        if t is None:
            continue
        hall += 1
        nn, ff, _ = metricas(Ap, Op, Tp, t)
        pas += pasa(nn, ff)
    res[u]["nulo"] = {"hallado": hall, "pasa": pas, "reps": REPS_NULO}
    nulo_tot.append(pas / REPS_NULO)
    print(f"c{u:<7} {f'{hall}/{REPS_NULO}':>15} {f'{pas}/{REPS_NULO}':>8}")
print("-" * 36)
print(f"      tasa media de «pasa» bajo el nulo: {np.mean(nulo_tot):.4f}")
print(f"S-3 · criterio: U-1 pasa en <= 1/8 unidades bajo el nulo. Unidades con tasa nula > 0,5: "
      f"{sum(1 for t in nulo_tot if t > 0.5)}/{n_u}  -> "
      f"{'CUMPLE' if sum(1 for t in nulo_tot if t > 0.5) <= 1 else 'NO CUMPLE'}")

with open(a_.salida, "w") as f:
    json.dump({"prereg": "PREREG_CORTE_SIN_ETIQUETAS.md", "sha": "17e0a35e",
               "config": {"n": a_.n, "batch": a_.batch, "p_nose": a_.p_nose,
                          "rng_ajuste": SEM_AJUSTE, "rng_prueba": SEM_PRUEBA,
                          "reps_nulo": REPS_NULO},
               "referencias": ref, "unidades": res,
               "veredictos": {"S1_u1_pasa": u1_pasa, "S4_sigma05_falla": s4,
                              "S5_signo_ok": s5_ok, "u2_pasa": u2_pasa, "u3_pasa": u3_pasa,
                              "n_unidades": n_u}}, f, indent=1, default=float)
print(f"\n-> {a_.salida}")
