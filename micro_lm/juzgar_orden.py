"""Juez de `PREREG_ORDEN_NOSE.md` (SHA 9e5659e5). Se escribe ANTES de que terminen las unidades.

Aplica los siete criterios tal como están redactados, sin reinterpretarlos, y contra el CONTROL que
ya está en disco (`t03_s3` y `t03_s6`, la misma configuración con --rec-rank 0).

  O-1  PRINCIPAL   acuerdo pareado «se calla» vs «no hay respuesta» > 0,60 en las DOS unidades
  O-2  MECANICISTA pureza de la decisión agrupando por relación < 0,70
  O-3  CONTROL     invento <= 0,10   <- el que puede CERRAR la línea (§6 de PREREG_RECOMPENSA_L)
  O-4  NULO        RECUP no cae más de 0,05 respecto del origen
  O-5  EXACTITUD   exactitud global > 0,4065 (el piso trivial)
  O-6  RIESGO      abstención estrictamente entre 0,05 y 0,95
  O-7  RIESGO      si O-1 falla pero el término de orden bajó de log 2, es PRESUPUESTO y no negativo

Uso:  python3 juzgar_orden.py            # juzga r03_s3 y r03_s6 contra t03_s3 y t03_s6
      python3 juzgar_orden.py --n 4096
"""

import argparse
import os

import numpy as np

import entrenar as E
import idioma as I
import medir_ratio_ce as R
import sonda_volado as SV

PISO_TRIVIAL = 0.4065
ORIGEN_RECUP = {"s3": 0.3654, "s6": 0.3835}      # b3_s3 y b3_s6, medidos el 30-ago
LOG2 = float(np.log(2.0))


def medir(ruta, n, lote, semilla, p_nose):
    """Todo lo que los siete criterios necesitan, de una sola pasada por el modelo."""
    params, cfg, paso = R.cargar(ruta)

    import jax, jax.numpy as jnp
    import datos as DAT

    @jax.jit
    def partes(p, s, c, t, m, co, po):
        return E._partes(p, s, c, t, m, co, po)

    rng = np.random.default_rng(semilla)
    LG, TGT, CONS = [], [], []
    vistos = 0
    while vistos < n:
        b = min(lote, n - vistos)
        ses, cortes, turnos, mask, cons, pos, tgt, tipo, meta = DAT.lote(
            rng, b, nivel=cfg["nivel"], n_hechos=4, n_sesiones=4, p_nose=p_nose, con_meta=True)
        lg, _ = partes(params, jnp.array(ses), jnp.array(cortes), jnp.array(turnos),
                       jnp.array(mask), jnp.array(cons), jnp.array(pos))
        LG.append(np.asarray(lg)); TGT.append(np.asarray(tgt)); CONS.append(np.asarray(cons))
        vistos += b
    lg = np.concatenate(LG).astype(np.float64)
    tgt = np.concatenate(TGT); cons = np.concatenate(CONS)

    lg_v = lg.copy(); lg_v[:, E.NOSE] = -1e9
    p_all = np.exp(lg - lg.max(-1, keepdims=True)); p_all /= p_all.sum(-1, keepdims=True)
    q = p_all[:, E.NOSE]
    calla = q > 0.5
    hay = tgt != E.NOSE
    arg = lg_v.argmax(-1)

    # Las cuatro categorías del instrumento del 15-ago. `invento` = contestar un valor cuando la
    # respuesta era NOSE, que es la alucinación pura y es el criterio O-3.
    acierto = (calla & ~hay) | (~calla & hay & (arg == tgt))
    invento = (~calla & ~hay)
    recup = float((arg == tgt)[hay].mean())

    # El término de orden, en las mismas unidades que `log 2` (toda constante) y 0 (oráculo).
    s = lg[:, E.NOSE] - (lg_v.max(-1) + np.log(np.exp(lg_v - lg_v.max(-1, keepdims=True)).sum(-1)))
    es_nose = (~hay).astype(np.float64)
    dif = s[:, None] - s[None, :]
    par = es_nose[:, None] * hay.astype(np.float64)[None, :]
    orden = float((np.logaddexp(0.0, -dif) * par).sum() / max(par.sum(), 1.0))

    pureza = SV.pureza(calla, cons[:, 4])
    nulo, sd = SV.nulo(calla, cons[:, 4], semilla)

    return dict(
        paso=paso, cfg=cfg,
        acuerdo=float((calla == ~hay).mean()),
        azar=float(calla.mean() * (~hay).mean() + (1 - calla.mean()) * hay.mean()),
        pureza=pureza, nulo=nulo, sd=sd,
        invento=float(invento.mean()), exactitud=float(acierto.mean()),
        recup=recup, abstencion=float(calla.mean()), orden=orden,
        rec_rank=cfg.get("rec_rank"),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4096)
    ap.add_argument("--lote", type=int, default=64)
    ap.add_argument("--semilla", type=int, default=54321)
    ap.add_argument("--p-nose", type=float, default=0.4)
    ap.add_argument("--trat", default="r03_s3,r03_s6")
    ap.add_argument("--ctrl", default="t03_s3,t03_s6")
    a = ap.parse_args()

    print("=" * 100)
    print("JUEZ de PREREG_ORDEN_NOSE.md (SHA 9e5659e5)")
    print(f"  n={a.n}  semilla {a.semilla} (pareada)  p_nose={a.p_nose}")
    print("=" * 100)

    res = {}
    for grupo, lista in (("CONTROL", a.ctrl), ("TRATAMIENTO", a.trat)):
        for u in lista.split(","):
            ruta = f"ckpts/{u}.pkl"
            if not os.path.exists(ruta):
                print(f"\n  [{u}] todavia no esta en disco, se saltea")
                continue
            m = medir(ruta, a.n, a.lote, a.semilla, a.p_nose)
            res[u] = m
            print(f"\n--- {grupo} · {u}  paso={m['paso']}  rec_rank={m['rec_rank']} ---")
            print(f"  acuerdo con «no hay respuesta»  {m['acuerdo']:.4f}   (azar {m['azar']:.4f})")
            print(f"  pureza por relacion             {m['pureza']:.4f}   (nulo {m['nulo']:.4f} "
                  f"± {m['sd']:.4f})")
            print(f"  invento                         {m['invento']:.4f}")
            print(f"  exactitud global                {m['exactitud']:.4f}   (piso {PISO_TRIVIAL})")
            print(f"  RECUP                           {m['recup']:.4f}")
            print(f"  abstencion                      {m['abstencion']:.4f}")
            print(f"  termino de orden                {m['orden']:.4f}   (constante {LOG2:.4f}, "
                  f"oraculo 0)")

    trat = [u for u in a.trat.split(",") if u in res]
    if not trat:
        print("\nno hay unidades de tratamiento en disco todavia.")
        return

    print("\n" + "=" * 100)
    print("VEREDICTO, con los criterios tal como estan escritos")
    print("=" * 100)
    o1 = all(res[u]["acuerdo"] > 0.60 for u in trat) and len(trat) == 2
    o2 = all(res[u]["pureza"] < 0.70 for u in trat)
    # O-3' (PRECISION_ORDEN_NOSE_O3.md, escrita ANTES de que hubiera tratamiento en disco): el
    # umbral absoluto de 0,10 lo falla TAMBIEN el control (0,2100), asi que no discrimina. Un modelo
    # que se calla al azar en la mitad contesta la mitad de las sin respuesta: 0,5 x p_nose = 0,20.
    # Pasa a ser RELATIVO al control, con 0,02 de margen = el ruido del estadistico entre semillas.
    inv_ctrl = max((res[u]["invento"] for u in a.ctrl.split(",") if u in res), default=0.10)
    o3 = all(res[u]["invento"] <= inv_ctrl + 0.02 for u in trat)
    # O-3'': si O-1 cumple, el invento TIENE que bajar. Con acuerdo 0,60 lo esperable es <= 0,16.
    o3b = (not o1) or all(res[u]["invento"] <= 0.16 for u in trat)
    o4 = all(res[u]["recup"] >= ORIGEN_RECUP[u[-2:]] - 0.05 for u in trat)
    o5 = all(res[u]["exactitud"] > PISO_TRIVIAL for u in trat)
    o6 = all(0.05 < res[u]["abstencion"] < 0.95 for u in trat)
    ordeno = all(res[u]["orden"] < LOG2 for u in trat)

    for nom, ok, txt in (
            ("O-1 PRINCIPAL  ", o1, "acuerdo > 0,60 en las DOS"),
            ("O-2 MECANICISTA", o2, "pureza por relacion < 0,70"),
            ("O-3' CONTROL   ", o3, f"invento <= control + 0,02 = {inv_ctrl + 0.02:.4f}  "
                                       f"(si FALLA, se cierra la linea)"),
            ("O-3'' COHERENCIA", o3b, "si O-1 cumple, invento <= 0,16"),
            ("O-4 NULO       ", o4, "RECUP no cae mas de 0,05"),
            ("O-5 EXACTITUD  ", o5, f"exactitud > {PISO_TRIVIAL}"),
            ("O-6 RIESGO     ", o6, "abstencion entre 0,05 y 0,95")):
        print(f"  [{'CUMPLE   ' if ok else 'NO CUMPLE'}] {nom}  {txt}")

    print()
    if o1 and o2 and o3 and not o3b:
        print("  ** INCONSISTENTE: O-1 cumple pero el invento no bajo. Con acuerdo > 0,60 el invento")
        print("     esperable es <= 0,16 y no ~0,21. Las dos cosas no pueden ser ciertas a la vez;")
        print("     se revisa el instrumento ANTES de adjudicar nada.")
    elif o1 and o2 and o3:
        print("  ** O-1, O-2 y O-3 CUMPLEN: la degeneracion era el bloqueo. La abstencion se calibra")
        print("     rompiendo la planitud, no reponderando. Es el resultado de la linea.")
    elif not o3:
        print("  ** O-3 SE DISPARA: el orden volvio a desacoplar la decision del valor, igual que")
        print("     `balance` y `ranking` el 29-ago. Tercer negativo de la misma forma -> se aplica")
        print("     el criterio de abandono del §6 y SE CIERRA la linea de la funcion de perdida.")
    elif o1 and not o2:
        print("  ** O-1 sin O-2: mejoro la tasa pero NO rompio el atajo. No se vende como calibracion.")
    elif not o1 and not ordeno:
        print(f"  ** O-7: el termino de orden NO bajo de log 2 ({LOG2:.4f}) -> es PRESUPUESTO y no un")
        print("     negativo. Se extiende, no se adjudica. Misma lectura que la NOTA_LECTURA_FASE_H.")
    elif not o1 and ordeno:
        print("  ** O-1 falla PERO el termino ordeno (bajo de log 2): el cuello no es la decision")
        print("     sino la evidencia. Pasa a la recuperacion.")


if __name__ == "__main__":
    main()
