#!/usr/bin/env python3
"""D-1 REHECHO con el blanco sin contaminar — `DESVIACIONES_DOS_DETECTORES.md` D-D3.

    python sonda_d1.py ckpts/p3_s1.pkl --n 6000 --salida dos_detectores/d1_p3_s1.json

El D-1 original definia el blanco con `clasificar`, y ahi abstenerse NUNCA cuenta como error. O sea
que `invento` solo existe donde la cabeza YA decidio no abstenerse: **el blanco estaba condicionado a
la decision del detector que se queria evaluar.** El sintoma fue que la sonda de ausencia predecia
ausencia con AUC 0,8403 y error con 0,3453 — invertida — mientras la tasa base decia que «sin
respuesta» tenia que predecir error POSITIVAMENTE (0,6063 contra 0,3123).

Aca el blanco es **«si el modelo contestara un valor, ¿estaria mal?»**: `argmax` sobre los logits con
`NOSE` excluido, contra el target, SIN mirar la cabeza. Para `tgt == NOSE` cualquier valor esta mal
por definicion. Es lo que un detector tiene que anticipar, y no depende de lo que el detector decidio.

Todo lo demas —semillas de generacion, featuras, l2, criterio de ≥ 0,05 en ≥ 2/3— queda igual.
"""
import argparse, json, os, pickle, sys

import numpy as np
import jax, jax.numpy as jnp

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
import idioma as I, datos as DAT, modelo as M
from sonda_dos_detectores import (internos, escalares_lectura, sonda, auc, bloque,
                                  SEM_AJUSTE, SEM_PRUEBA, REPS_NULO, NOSE)


def extraer(ck, n, B, semilla, donde, nivel):
    params = jax.tree_util.tree_map(jnp.asarray, pickle.load(open(ck, "rb"))["params"])
    fn = jax.jit(lambda *a: internos(*a, donde=donde))
    rng = np.random.default_rng(semilla)
    acc = {k: [] for k in ("est_q", "est_foco", "lect_q", "lect_foco", "salida", "cab",
                           "mal_si_contesta", "tgt_nose", "abstuvo")}
    vistos = 0
    while vistos < n:
        b = min(B, n - vistos)
        ses, cor, tur, mask, cons, pos, tgt, tipo = DAT.lote(
            rng, b, nivel=nivel, n_hechos=4, n_sesiones=4, p_nose=0.4)
        arch = M.escribir(params, jnp.array(ses), jnp.array(cor))
        hn, lg, ab, p, sim = fn(params, arch, jnp.array(tur), jnp.array(cons), jnp.array(mask))
        hn, lg, ab = np.asarray(hn), np.asarray(lg), np.asarray(ab)
        esc = escalares_lectura(np.asarray(p), np.asarray(sim))
        ent = esc[..., 0]
        for i in range(b):
            pq = int(pos[i])
            f = int(ent[i, :pq + 1].argmin())
            v = lg[i, pq].copy(); v[NOSE] = -np.inf
            arg = int(v.argmax())                      # lo que contestaria SI contestara
            acc["mal_si_contesta"].append(arg != int(tgt[i]))
            acc["tgt_nose"].append(int(tgt[i]) == NOSE)
            acc["abstuvo"].append(bool(ab[i, pq] > 0.0))
            acc["est_q"].append(hn[i, pq]); acc["est_foco"].append(hn[i, f])
            acc["lect_q"].append(esc[i, pq]); acc["lect_foco"].append(esc[i, f])
            sm = np.exp(v - v.max()); sm = sm / sm.sum(); o = np.sort(sm)
            acc["salida"].append([o[-1], o[-1] - o[-2], -(sm * np.log(sm + 1e-12)).sum()])
            acc["cab"].append(float(ab[i, pq]))
        vistos += b
    return {k: np.array(v) for k, v in acc.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pesos"); ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--B", type=int, default=64); ap.add_argument("--salida", default=None)
    a = ap.parse_args()
    cfg = pickle.load(open(a.pesos, "rb"))["config"]
    donde, nivel, sem = cfg.get("donde", "pre"), cfg["nivel"], cfg.get("semilla", 0)
    u = os.path.basename(a.pesos).replace(".pkl", "")
    print(f"# {u} · donde={donde} · nivel {nivel} · BLANCO SIN CONTAMINAR", flush=True)

    A = extraer(a.pesos, a.n, a.B, SEM_AJUSTE + sem, donde, nivel)
    P = extraer(a.pesos, a.n, a.B, SEM_PRUEBA + sem, donde, nivel)
    todo = ["est_q", "est_foco", "lect_q", "lect_foco", "salida", "cab"]
    yA, yP = A["mal_si_contesta"], P["mal_si_contesta"]
    conA, conP = ~A["tgt_nose"], ~P["tgt_nose"]

    r = {"unidad": u, "donde": donde, "semilla": sem, "n": a.n,
         "tasa_mal_si_contesta": float(yP.mean()),
         "tasa_mal_entre_con_respuesta": float(yP[conP].mean()),
         "tasa_abstencion_del_modelo": float(P["abstuvo"].mean())}

    # ÚNICO: una sonda sobre el blanco entero
    r["D1_unico"] = auc(yP, sonda(bloque(A, todo), yA, bloque(P, todo)))

    # COMPUESTO: ausencia + atribución, cada una con su blanco
    sA = sonda(bloque(A, todo), A["tgt_nose"], bloque(P, todo))
    sB = sonda(bloque(A, todo)[conA], yA[conA], bloque(P, todo))
    pa = 1 / (1 + np.exp(-np.clip(sA, -30, 30)))
    pb = 1 / (1 + np.exp(-np.clip(sB, -30, 30)))
    r["D1_compuesto"] = auc(yP, pa + (1 - pa) * pb)
    r["D1_delta"] = r["D1_compuesto"] - r["D1_unico"]
    r["auc_ausencia"] = auc(P["tgt_nose"], sA)
    r["auc_atribucion_solo_con_respuesta"] = auc(yP[conP], sB[conP])

    # contrastes honestos
    r["cabeza_sola"] = auc(yP, P["cab"])
    r["confianza_sola"] = auc(yP, -P["salida"][:, 0])
    r["cabeza_mas_confianza"] = auc(yP, sonda(
        np.c_[A["cab"], A["salida"]], yA, np.c_[P["cab"], P["salida"]]))

    # nulo
    rng = np.random.default_rng(4242)
    nul = [auc(yP, sonda(bloque(A, todo), rng.permutation(yA), bloque(P, todo)))
           for _ in range(REPS_NULO)]
    r["nulo_medio"] = float(np.mean(nul)); r["nulo_sd"] = float(np.std(nul))

    print(json.dumps(r, indent=2, ensure_ascii=False), flush=True)
    if a.salida:
        json.dump(r, open(a.salida, "w"), indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
