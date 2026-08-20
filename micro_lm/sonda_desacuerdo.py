#!/usr/bin/env python3
"""El monitor de desacuerdo interno — PREREG_MONITOR_DESACUERDO.md (SHA b259fd0d...).

K pasadas con el archivo PERMUTADO (turnos y mask viajan pegados a su entrada) y se mide la
`consistencia`: la fraccion de las K que coinciden con la respuesta modal. El corte propuesto es
ESTRUCTURAL —abstenerse si consistencia < 1— y no tiene un solo parametro ajustado, que es
exactamente lo que al logit le faltaba: `a = 0,3` no significa nada sin etiquetas, «las 16 pasadas
dieron todas lo mismo» si.

La permutacion es invariante para una respuesta anclada en evidencia por dos razones medidas: E-I3d
mostro que el lector usa el sello de orden y no la posicion, y la lectura por atencion es
permutacion-equivariante. Si la respuesta cambia al permutar, el modelo no estaba leyendo el archivo.

Los cuatro controles del §4 se corren siempre:
  M-1 AUC del desacuerdo · M-2 el corte estructural · M-3 quitar la entrada necesaria (tiene que
  BAJAR la consistencia) · M-4 el nulo con permutacion identidad, que debe dar 1,000 exacto.
"""
import os, sys, pickle, argparse, json
import numpy as np
import jax, jax.numpy as jnp

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
import idioma as I, datos as DAT, modelo as M

NOSE = I.STOI["NOSE"]
UNIDADES = ["1_s0", "2_s0", "3_s0", "3_s1", "3_s2", "4_s0", "4_s1", "4_s2"]
SEM_PRUEBA = 77000          # PREREG §4: las mismas muestras que el experimento del corte
K = 16                      # PREREG §3


def responder(params, archivo, turnos, mask, cons, pos):
    lg, _ = M.responder_con_abst(params, archivo, turnos, cons, mask)
    lg = jnp.take_along_axis(lg, pos[:, None, None], axis=1)[:, 0, :]
    return lg.at[:, NOSE].set(-jnp.inf).argmax(-1)


def pasadas(params, ses, cortes, turnos, mask, cons, pos, perms):
    """K respuestas, una por permutacion del eje N del archivo. `perms` es (K, N)."""
    archivo = M.escribir(params, ses, cortes)

    def una(p):
        return responder(params, archivo[:, p, :], turnos[:, p], mask[:, p], cons, pos)

    return jax.vmap(una)(perms)          # (K, B)


def consistencia(R):
    """Fraccion de las K pasadas que coincide con la moda, por muestra. R es (K, B)."""
    K_, B = R.shape
    out = np.empty(B)
    modo = np.empty(B, dtype=R.dtype)
    for i in range(B):
        v, c = np.unique(R[:, i], return_counts=True)
        j = c.argmax()
        out[i] = c[j] / K_
        modo[i] = v[j]
    return out, modo


def auc(x, pos_mask):
    p, n = x[pos_mask], x[~pos_mask]
    if len(p) == 0 or len(n) == 0:
        return np.nan
    r = np.argsort(np.argsort(np.concatenate([p, n]))) + 1
    return float((r[:len(p)].sum() - len(p) * (len(p) + 1) / 2) / (len(p) * len(n)))


def metricas(abst, okv, tipo):
    sin_resp, hay, vig = tipo >= 2, tipo < 2, tipo == 0
    nose = abst[sin_resp].mean() if sin_resp.any() else np.nan
    falsa = abst[hay].mean() if hay.any() else np.nan
    vigente = (okv[vig] & ~abst[vig]).mean() if vig.any() else np.nan
    return float(nose), float(falsa), float(vigente)


def pasa(nose, falsa):
    return bool(falsa <= 0.10 and nose >= 0.50)


ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=8, help="lotes por unidad (K pasadas cada uno)")
ap.add_argument("--batch", type=int, default=64)
ap.add_argument("--p-nose", type=float, default=0.4)
ap.add_argument("--unidades", default="", help="coma-separadas; vacio = las 8 del prereg")
ap.add_argument("--salida", default=os.path.join(AQUI, "desacuerdo_20260820.json"))
a_ = ap.parse_args()
UNI = a_.unidades.split(",") if a_.unidades else UNIDADES

print(f"MONITOR DE DESACUERDO — PREREG_MONITOR_DESACUERDO.md (SHA b259fd0d...)")
print(f"K={K} pasadas con el archivo permutado · rng {SEM_PRUEBA}+s · "
      f"{a_.n}x{a_.batch} = {a_.n * a_.batch} muestras por unidad\n")
print(f"{'unidad':<8} {'AUC':>7} {'M-1':>5} | {'consist<1':>10} {'f_abst':>8} {'nose':>8} {'M-2':>5} | "
      f"{'M-3 dif':>9} {'M-3':>5} | {'M-4':>6}")
print("-" * 92)

res, n_m1, n_m2, n_m3, n_m4 = {}, 0, 0, 0, 0
for u in UNI:
    ck = os.path.join(AQUI, "ckpts", f"c{u}.pkl")
    if not os.path.exists(ck):
        print(f"c{u}: sin checkpoint")
        continue
    with open(ck, "rb") as f:
        d = pickle.load(f)
    params = jax.tree_util.tree_map(jnp.asarray, d["params"])
    if "abst" not in params:
        continue
    nivel, semilla = int(u[0]), int(u.split("_s")[1])
    rng = np.random.default_rng(SEM_PRUEBA + semilla)
    rp = np.random.default_rng(1234 + semilla)          # las permutaciones, aparte de los datos
    fn = jax.jit(pasadas)

    CO, CO_ID, CO_Q, OKV, TIPO = [], [], [], [], []
    for _ in range(a_.n):
        ses, cortes, turnos, mask, cons, pos, tgt, tipo = DAT.lote(
            rng, a_.batch, nivel=nivel, n_hechos=4, n_sesiones=4, p_vieja=0.35, p_nose=a_.p_nose)
        arch_n = M.escribir(params, jnp.array(ses), jnp.array(cortes)).shape[1]
        perms = np.stack([rp.permutation(arch_n) for _ in range(K)])
        args = (jnp.array(ses), jnp.array(cortes), jnp.array(turnos), jnp.array(mask),
                jnp.array(cons), jnp.array(pos))
        R = np.asarray(fn(params, *args, jnp.array(perms)))
        c, modo = consistencia(R)
        CO.append(c); OKV.append(modo == tgt); TIPO.append(tipo)

        # M-4 · el nulo: K permutaciones IDENTIDAD. Debe dar 1,000 exacto.
        ident = np.stack([np.arange(arch_n) for _ in range(K)])
        R0 = np.asarray(fn(params, *args, jnp.array(ident)))
        CO_ID.append(consistencia(R0)[0])

        # M-3 · perturbacion que SI cambia el problema: se tapa una entrada del archivo por muestra
        # (mask a 0). No es «la que la consulta necesita» identificada por metadatos —eso exigiria
        # con_meta— sino una al azar entre las presentes; con 4 hechos la necesaria cae adentro una
        # de cada cuatro veces, asi que el efecto esperado se diluye pero no se anula.
        mk = np.asarray(mask).copy()
        for b in range(mk.shape[0]):
            viv = np.flatnonzero(mk[b])
            if len(viv):
                mk[b, rp.choice(viv)] = 0
        Rq = np.asarray(fn(params, jnp.array(ses), jnp.array(cortes), jnp.array(turnos),
                           jnp.array(mk), jnp.array(cons), jnp.array(pos), jnp.array(perms)))
        CO_Q.append(consistencia(Rq)[0])

    co = np.concatenate(CO); okv = np.concatenate(OKV); tipo = np.concatenate(TIPO)
    co_id = np.concatenate(CO_ID); co_q = np.concatenate(CO_Q)

    # M-1: se mide sobre el DESACUERDO (1 - consistencia), positivo = pregunta sin respuesta
    a1 = auc(1.0 - co, tipo >= 2)
    m1 = bool(a1 >= 0.70)
    n_m1 += m1
    # M-2: el corte estructural
    abst = co < 1.0
    nose, falsa, vig = metricas(abst, okv, tipo)
    m2 = pasa(nose, falsa)
    n_m2 += m2
    # M-3: la consistencia tiene que BAJAR al tapar una entrada, en las que SI tenian respuesta
    hay = tipo < 2
    dif = float(np.median(co[hay]) - np.median(co_q[hay]))
    m3 = bool(dif >= 0.10)
    n_m3 += m3
    # M-4
    m4 = bool(np.all(co_id == 1.0))
    n_m4 += m4

    res[u] = {"auc": a1, "M1": m1, "consist_media": float(co.mean()),
              "corte_estructural": {"falsa": falsa, "nose": nose, "vigente": vig, "pasa": m2},
              "M3_dif": dif, "M3": m3, "M4_identidad_todo_1": m4}
    print(f"c{u:<7} {a1:>7.3f} {('SI' if m1 else 'no'):>5} | {'':>10} {falsa:>8.4f} {nose:>8.4f} "
          f"{('SI' if m2 else 'no'):>5} | {dif:>+9.4f} {('SI' if m3 else 'no'):>5} | "
          f"{('OK' if m4 else 'FALLA'):>6}")

n = len(res)
print("-" * 92)
print(f"M-1 · AUC >= 0,70 en {n_m1}/{n}  (criterio >= 6/8)  -> {'CUMPLE' if n_m1 >= 6 else 'NO CUMPLE'}")
print(f"M-2 · el corte ESTRUCTURAL pasa en {n_m2}/{n}  (criterio >= 6/8)  -> "
      f"{'CUMPLE' if n_m2 >= 6 else 'NO CUMPLE'}")
print(f"      referencias del 20-ago sobre las mismas unidades: U-1 = 2/8 · sigma>0,5 = 6/8")
print(f"M-3 · la consistencia baja >= 0,10 al tapar una entrada en {n_m3}/{n} unidades")
print(f"M-4 · el nulo (permutacion identidad) da 1,000 exacto en {n_m4}/{n}  -> "
      f"{'OK' if n_m4 == n else 'FALLA — hay ruido numerico y todo lo de arriba queda en duda'}")

json.dump({"prereg": "PREREG_MONITOR_DESACUERDO.md", "sha": "b259fd0d",
           "config": {"K": K, "n": a_.n, "batch": a_.batch, "p_nose": a_.p_nose,
                      "rng_prueba": SEM_PRUEBA},
           "unidades": res,
           "veredictos": {"M1": n_m1, "M2": n_m2, "M3": n_m3, "M4": n_m4, "n": n}},
          open(a_.salida, "w"), indent=1, default=float)
print(f"\n-> {a_.salida}")
