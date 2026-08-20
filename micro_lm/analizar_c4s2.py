#!/usr/bin/env python3
"""T-1..T-4 de `PREREG_C4S2_PRESUPUESTO.md` (SHA 8446a27e...).

T-1  Spearman(falsa_abst, paso) sobre los 24 puntos NUEVOS, con rangos promediados
T-2  extremos (ckpt 14000 contra 20000) medidos con 2048 muestras y el rng de PRUEBA
T-3  el intercambio: `nose` sube en paralelo, o no
T-4  control de sanidad: `vigente` no se cae mas de 0,10

Los rangos promediados no son un detalle: el 19-ago el Spearman del INFORME_FRONTERA se movia entre
corridas con datos identicos porque dos unidades empataban y el desempate dependia del orden de
`glob`. Un coeficiente que se mueve solo no es publicable.
"""
import os, sys, json, pickle, argparse
import numpy as np
import jax, jax.numpy as jnp

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
import idioma as I, datos as DAT, modelo as M

NOSE = I.STOI["NOSE"]
SEM_PRUEBA = 77000        # PREREG §3 T-2: el generador de prueba, NO el 90000 con que se evaluo
CORTE = 14000             # los puntos nuevos son los posteriores


def rangos(x):
    """Rangos promediados en los empates."""
    x = np.asarray(x, float)
    o = np.argsort(x, kind="mergesort")
    r = np.empty(len(x), float)
    r[o] = np.arange(1, len(x) + 1)
    for v in np.unique(x):
        m = x == v
        if m.sum() > 1:
            r[m] = r[m].mean()
    return r


def spearman(x, y):
    rx, ry = rangos(x), rangos(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    rho = float((rx * ry).sum() / np.sqrt((rx ** 2).sum() * (ry ** 2).sum()))
    n = len(x)
    if n > 2 and abs(rho) < 1:
        t = rho * np.sqrt((n - 2) / (1 - rho ** 2))
        # p bilateral por la t de Student, sin scipy: integracion numerica de la densidad
        from math import lgamma, pi
        v = n - 2
        c = np.exp(lgamma((v + 1) / 2) - lgamma(v / 2)) / np.sqrt(v * pi)
        xs = np.linspace(abs(t), abs(t) + 200, 400001)
        p = 2 * float(np.trapezoid(c * (1 + xs ** 2 / v) ** (-(v + 1) / 2), xs))
    else:
        p = 0.0 if abs(rho) == 1 else 1.0
    return rho, min(p, 1.0)


def partes(params, ses, cortes, turnos, mask, cons, pos):
    archivo = M.escribir(params, ses, cortes)
    lg, a = M.responder_con_abst(params, archivo, turnos, cons, mask)
    lg = jnp.take_along_axis(lg, pos[:, None, None], axis=1)[:, 0, :]
    a = jnp.take_along_axis(a, pos[:, None], axis=1)[:, 0]
    return lg, a


def medir(ck, nivel, semilla, n, B, p_nose=0.4, p_vieja=0.35):
    """`falsa_abst`, `nose` y `vigente` con el criterio sigma>0,5 y muestra grande."""
    with open(ck, "rb") as f:
        d = pickle.load(f)
    params = jax.tree_util.tree_map(jnp.asarray, d["params"])
    fn = jax.jit(partes)
    rng = np.random.default_rng(SEM_PRUEBA + semilla)
    A, OKV, TIPO = [], [], []
    for _ in range(n):
        ses, cortes, turnos, mask, cons, pos, tgt, tipo = DAT.lote(
            rng, B, nivel=nivel, n_hechos=4, n_sesiones=4, p_vieja=p_vieja, p_nose=p_nose)
        lg, a = fn(params, jnp.array(ses), jnp.array(cortes), jnp.array(turnos),
                   jnp.array(mask), jnp.array(cons), jnp.array(pos))
        lg = np.asarray(lg).copy()
        lg[:, NOSE] = -np.inf
        A.append(np.asarray(a)); OKV.append(lg.argmax(-1) == tgt); TIPO.append(tipo)
    a, okv, tipo = np.concatenate(A), np.concatenate(OKV), np.concatenate(TIPO)
    abst = a > 0.0
    sin_resp, hay, vig = tipo >= 2, tipo < 2, tipo == 0
    return {"paso": d["paso"], "n": len(a),
            "falsa_abst": float(abst[hay].mean()), "nose": float(abst[sin_resp].mean()),
            "vigente": float((okv[vig] & ~abst[vig]).mean())}


ap = argparse.ArgumentParser()
ap.add_argument("--hist", default=os.path.join(AQUI, "corridas_20260820", "c4_s2.json"))
ap.add_argument("--ck-viejo", default=os.path.join(AQUI, "ckpts", "c4_s2.pkl.p14000"))
ap.add_argument("--ck-nuevo", default=os.path.join(AQUI, "ckpts", "c4_s2.pkl"))
ap.add_argument("--n", type=int, default=32)
ap.add_argument("--batch", type=int, default=64)
ap.add_argument("--salida", default=os.path.join(AQUI, "c4s2_presupuesto_20260820.json"))
ap.add_argument("--sin-extremos", action="store_true",
                help="salta T-2/T-4 (que muestrean 2048 y compiten por la CPU). El aviso automatico "
                     "lo usa para mandar la tendencia sin frenar lo que este corriendo.")
a_ = ap.parse_args()

print("c4_s2 CON MAS PRESUPUESTO — PREREG_C4S2_PRESUPUESTO.md (SHA 8446a27e...)\n")

hist = json.load(open(a_.hist))["historia"]
nuevos = [h for h in hist if h["paso"] > CORTE]
print(f"historia: {len(hist)} puntos, {len(nuevos)} posteriores al paso {CORTE}")
if len(nuevos) < 5:
    print("todavia no hay puntos nuevos suficientes; se corta aca")
    sys.exit(0)

pasos = [h["paso"] for h in nuevos]
fa = [h["falsa_abst"] for h in nuevos]
no = [h["nose"] for h in nuevos]
vi = [h["vigente"] for h in nuevos]

print(f"\n{'paso':>7} {'falsa_abst':>11} {'nose':>8} {'vigente':>9}")
print("-" * 38)
for h in nuevos:
    print(f"{h['paso']:>7} {h['falsa_abst']:>11.4f} {h['nose']:>8.4f} {h['vigente']:>9.4f}")

r1, p1 = spearman(pasos, fa)
r3, p3 = spearman(pasos, no)
ok1 = bool(r1 >= 0.41 and p1 < 0.05)
ok3 = bool(r3 >= 0.41 and p3 < 0.05)
print()
print(f"T-1 · Spearman(falsa_abst, paso) = {r1:+.4f}  p = {p1:.4g}  (n={len(nuevos)})")
print(f"      criterio rho >= +0,41 y p < 0,05  ->  {'CUMPLE (es tendencia)' if ok1 else 'NO CUMPLE'}")
print(f"T-3 · Spearman(nose, paso)       = {r3:+.4f}  p = {p3:.4g}  ->  "
      f"{'CUMPLE (hay intercambio)' if ok3 else 'NO CUMPLE'}")
if ok1 and not ok3:
    print("      ojo: falsa_abst sube y `nose` no. Es PEOR que lo previsto — pierde sin ganar.")

print()
print(f"T-2 · EXTREMOS con {a_.n * a_.batch} muestras y el rng de prueba ({SEM_PRUEBA}+semilla)")
res2 = {}
for nom, ck in () if a_.sin_extremos else (("14000", a_.ck_viejo), ("nuevo", a_.ck_nuevo)):
    if not os.path.exists(ck):
        print(f"      {nom}: falta {ck}")
        continue
    res2[nom] = medir(ck, nivel=4, semilla=2, n=a_.n, B=a_.batch)
    m = res2[nom]
    print(f"      paso {m['paso']:>5}: falsa_abst {m['falsa_abst']:.4f} · nose {m['nose']:.4f} "
          f"· vigente {m['vigente']:.4f}  (n={m['n']})")
ok2 = ok4 = None
if len(res2) == 2:
    d = res2["nuevo"]["falsa_abst"] - res2["14000"]["falsa_abst"]
    ok2 = bool(d >= 0.03)
    dv = res2["nuevo"]["vigente"] - res2["14000"]["vigente"]
    ok4 = bool(dv >= -0.10)
    print(f"      diferencia falsa_abst = {d:+.4f}  (criterio >= +0,03)  ->  "
          f"{'CONFIRMA' if ok2 else 'NO CONFIRMA'}")
    print(f"T-4 · vigente {dv:+.4f}  (criterio: no cae mas de 0,10)  ->  "
          f"{'CUMPLE' if ok4 else 'NO CUMPLE — la degradacion no es de la cabeza sino del modelo'}")

print()
print("VEREDICTO segun el §4 del prereg:")
if a_.sin_extremos:
    print("  PARCIAL — T-2 no se corrio (--sin-extremos), asi que el veredicto del §4 queda abierto.")
    print(f"  Lo unico que se puede decir hoy: la tendencia {'ESTA' if ok1 else 'NO esta'} en la serie.")
elif ok1 and ok2:
    print("  T-1 y T-2 cumplen -> la cabeza se degrada al entrenarla de mas en tarea dificil.")
    print("  Corresponde parada temprana gobernada por falsa_abst, y mirar c4_s0/c4_s1.")
elif not ok1:
    print("  T-1 no cumple -> los tres puntos del 19-ago eran ruido de muestra chica.")
    print("  Se RETIRA la advertencia del INFORME_CELDA_DIFICIL.")
else:
    print("  T-1 cumple y T-2 no -> no concluyente. No se elige la mitad que mas gusta.")

json.dump({"prereg": "PREREG_C4S2_PRESUPUESTO.md", "sha": "8446a27e",
           "serie": nuevos, "T1": {"rho": r1, "p": p1, "cumple": ok1},
           "T3": {"rho": r3, "p": p3, "cumple": ok3},
           "T2": {"extremos": res2, "cumple": ok2}, "T4": {"cumple": ok4}},
          open(a_.salida, "w"), indent=1, default=float)
print(f"\n-> {a_.salida}")
