"""Juez de `PREREG_SLOT_ORDEN.md` (SHA b7471e02). Se escribe ANTES de que terminen las unidades.

**Lo que cambia respecto de `juzgar_orden.py`, y es la lección de esta tarde.** Aquel imprimió
«O-3 SE DISPARA -> SE CIERRA la línea» cuando sus propios números lo desmentían: con `abstencion` en
0,0000 tres de sus criterios dejaban de medir y pasaban a ser aritmética (acuerdo = P(hay), pureza =
1 por definición, invento = 0,40 por definición). El defecto no era del criterio sino del JUEZ, que
devolvió un número donde correspondía **NO EVALUABLE**.

> **Regla que quedó escrita el 31-ago y que este archivo implementa: cuando un criterio de riesgo
> protege la legibilidad de otros, hay que decir CUÁLES, y el juez tiene que devolver NO EVALUABLE en
> vez de un número.**

Por eso acá **W-6 se evalúa PRIMERO**, por unidad, y si se rompe, W-1, W-2 y W-4 salen NO EVALUABLE
para esa unidad —no «fallan»—, con el motivo impreso al lado.

  W-1  PRINCIPAL    AUC del logit de abstención vs la ausencia > 0,65 en las DOS semillas
  W-2  MECANICISTA  el tratamiento supera al CONTROL B por >= 0,05 de AUC   <- la que decide
  W-3  SATURACION   la fracción pegada al clip baja de 0,50 (siembra: 0,8438 en s3 · 0,6250 en s6).
                    NO es precondición de W-1 ni W-2: un logit de dos valores puede separar bien.
  W-4  CONTROL      `invento` no supera al del CONTROL B por más de 0,02
  W-5  NULO         RECUP no cae más de 0,05 respecto del origen
  W-6  PRECONDICION abstención estrictamente entre 0,05 y 0,95
  W-7  RIESGO       si W-1 falla pero el término de orden bajó de log 2, es PRESUPUESTO, no negativo

Uso:  python3 juzgar_slot.py                      # w03_s3 y w03_s6 contra k03_s3 y k03_s6
      python3 juzgar_slot.py --n 4096
"""

import argparse
import json
import os

import numpy as np

import entrenar as E
import medir_ratio_ce as R

PISO_TRIVIAL = 0.4065
ORIGEN_RECUP = {"s3": 0.3654, "s6": 0.3835}          # b3_s3 y b3_s6, medidos el 30-ago
CLIP_SIEMBRA = {"s3": 0.8438, "s6": 0.6250}          # compuerta W-0, 31-ago
LOG2 = float(np.log(2.0))
CLIP = float(np.log((1 - 1e-6) / 1e-6))              # 13,8155 · el tope de modelo.py:306
NE = "NO EVALUABLE"


def auc(s, pos):
    o = np.argsort(s)
    r = np.empty(len(s))
    r[o] = np.arange(1, len(s) + 1)
    n1, n0 = pos.sum(), (~pos).sum()
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else float("nan")


def medir(ruta, n, lote, semilla, p_nose):
    """Todo lo que los criterios necesitan, de una sola pasada. La abstención sale del SLOT."""
    import jax
    import jax.numpy as jnp

    import datos as DAT

    params, cfg, paso = R.cargar(ruta)

    @jax.jit
    def partes(p, s, c, t, m, co, po):
        return E._partes(p, s, c, t, m, co, po)

    rng = np.random.default_rng(semilla)
    LG, S, TGT = [], [], []
    vistos = 0
    while vistos < n:
        b = min(lote, n - vistos)
        ses, cortes, turnos, mask, cons, pos, tgt, tipo = DAT.lote(
            rng, b, nivel=cfg["nivel"], n_hechos=4, n_sesiones=4, p_nose=p_nose)
        lg, s = partes(params, jnp.array(ses), jnp.array(cortes), jnp.array(turnos),
                       jnp.array(mask), jnp.array(cons), jnp.array(pos))
        LG.append(np.asarray(lg))
        S.append(np.asarray(s))
        TGT.append(np.asarray(tgt))
        vistos += b
    lg = np.concatenate(LG).astype(np.float64)
    s = np.concatenate(S).astype(np.float64)
    tgt = np.concatenate(TGT)

    lg_v = lg.copy()
    lg_v[:, E.NOSE] = -1e9
    arg = lg_v.argmax(-1)
    hay = tgt != E.NOSE
    # Con `--abst slot` la decisión sale de la masa del slot, no del softmax de vocabulario.
    calla = s > 0.0
    acierto = (calla & ~hay) | (~calla & hay & (arg == tgt))
    invento = (~calla & ~hay)

    # La exactitud con el MEJOR umbral posible: separa «la señal no está» de «el umbral está mal
    # puesto», que es la distinción que el informe de esta tarde necesitó y no tenía.
    ordenados = np.unique(s)
    mejor = max(float(((s > u) & ~hay).mean() + ((s <= u) & hay & (arg == tgt)).mean())
                for u in np.concatenate([[-np.inf], ordenados]))

    es_nose = (~hay).astype(np.float64)
    dif = s[:, None] - s[None, :]
    par = es_nose[:, None] * hay.astype(np.float64)[None, :]
    orden = float((np.logaddexp(0.0, -dif) * par).sum() / max(par.sum(), 1.0))

    return dict(
        paso=paso, rec_rank=cfg.get("rec_rank"), abst=cfg.get("abst"),
        auc=auc(s, ~hay), abstencion=float(calla.mean()),
        invento=float(invento.mean()), exactitud=float(acierto.mean()),
        exactitud_mejor=mejor, recup=float((arg == tgt)[hay].mean()), orden=orden,
        pegadas=float((np.abs(np.abs(s) - CLIP) < 1e-3).mean()),
        distintos=int(len(ordenados)), n=len(s),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4096)
    ap.add_argument("--lote", type=int, default=64)
    ap.add_argument("--semilla", type=int, default=54321)
    ap.add_argument("--p-nose", type=float, default=0.4)
    ap.add_argument("--trat", default="w03_s3,w03_s6")
    ap.add_argument("--ctrl", default="k03_s3,k03_s6")
    a = ap.parse_args()

    print("=" * 100)
    print("JUEZ de PREREG_SLOT_ORDEN.md (SHA b7471e02)")
    print(f"  n={a.n}  semilla {a.semilla} (pareada)  p_nose={a.p_nose}")
    print("=" * 100)

    res = {}
    for grupo, lista in (("CONTROL B", a.ctrl), ("TRATAMIENTO", a.trat)):
        for u in lista.split(","):
            ruta = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ckpts", u + ".pkl")
            if not os.path.exists(ruta):
                print(f"\n  [{u}] todavia no esta en disco, se saltea")
                continue
            m = medir(ruta, a.n, a.lote, a.semilla, a.p_nose)
            res[u] = m
            sem = u.split("_")[-1]
            print(f"\n--- {grupo} · {u}  paso={m['paso']}  abst={m['abst']}  "
                  f"rec_rank={m['rec_rank']} ---")
            print(f"  AUC del logit vs la ausencia    {m['auc']:.4f}   (azar 0,50 · token+orden 0,6620)")
            print(f"  abstencion                      {m['abstencion']:.4f}   <- PRECONDICION W-6")
            print(f"  pegadas al clip                 {m['pegadas']:.4f}   (siembra "
                  f"{CLIP_SIEMBRA.get(sem, float('nan')):.4f}) · valores distintos "
                  f"{m['distintos']} de {m['n']}")
            print(f"  invento                         {m['invento']:.4f}")
            print(f"  exactitud global                {m['exactitud']:.4f}   (piso {PISO_TRIVIAL})")
            print(f"  exactitud, MEJOR umbral         {m['exactitud_mejor']:.4f}")
            print(f"  RECUP                           {m['recup']:.4f}   (origen "
                  f"{ORIGEN_RECUP.get(sem, float('nan')):.4f})")
            print(f"  termino de orden                {m['orden']:.4f}   (constante {LOG2:.4f}, "
                  f"oraculo 0)")

    trat = [u for u in a.trat.split(",") if u in res]
    ctrl = {u.split("_")[-1]: res[u] for u in a.ctrl.split(",") if u in res}
    if not trat:
        print("\nno hay unidades de tratamiento en disco todavia.")
        return

    print("\n" + "=" * 100)
    print("VEREDICTO · W-6 se evalua PRIMERO, porque es la PRECONDICION de W-1, W-2 y W-4")
    print("=" * 100)

    # --- W-6, y de su resultado depende que los otros tres se puedan leer -----------------------
    legible = {}
    for u in trat:
        ab = res[u]["abstencion"]
        ok = 0.05 < ab < 0.95
        legible[u] = ok
        extremo = "MUDO" if ab >= 0.95 else "LOCUAZ"
        print(f"  W-6 · {u}: abstencion {ab:.4f}  ->  "
              f"{'CUMPLE' if ok else f'NO CUMPLE, extremo {extremo}'}")
    todas = all(legible.values())
    if not todas:
        rotas = [u for u in trat if not legible[u]]
        print(f"\n  ** W-1, W-2 y W-4 quedan {NE} en {', '.join(rotas)}: con la abstencion en un")
        print(f"     extremo esos criterios dejan de medir el fenomeno y pasan a ser aritmetica. **")

    def linea(nom, texto, valor, cumple, dep_w6=True):
        if dep_w6 and not todas:
            print(f"  {nom} · {texto}: {NE}   (precondicion W-6 rota)")
            return None
        print(f"  {nom} · {texto}: {'CUMPLE' if cumple else 'NO CUMPLE'}   {valor}")
        return cumple

    print()
    w1 = linea("W-1", "AUC > 0,65 en las dos semillas",
               " · ".join(f"{u} {res[u]['auc']:.4f}" for u in trat),
               all(res[u]["auc"] > 0.65 for u in trat) and len(trat) == 2)

    dif = {u: res[u]["auc"] - ctrl[u.split("_")[-1]]["auc"]
           for u in trat if u.split("_")[-1] in ctrl}
    w2 = linea("W-2", "supera al CONTROL B por >= 0,05  <- LA QUE DECIDE",
               " · ".join(f"{u} {d:+.4f}" for u, d in dif.items()),
               bool(dif) and len(dif) == 2 and all(d >= 0.05 for d in dif.values()))

    dinv = {u: res[u]["invento"] - ctrl[u.split("_")[-1]]["invento"]
            for u in trat if u.split("_")[-1] in ctrl}
    w4 = linea("W-4", "invento no supera al control por > 0,02",
               " · ".join(f"{u} {d:+.4f}" for u, d in dinv.items()),
               bool(dinv) and all(d <= 0.02 for d in dinv.values()))

    # --- W-3 y W-5 NO dependen de W-6, y eso quedo declarado en el prereg -----------------------
    w3 = linea("W-3", "pegadas al clip < 0,50 (desatura)",
               " · ".join(f"{u} {res[u]['pegadas']:.4f} (siembra "
                          f"{CLIP_SIEMBRA.get(u.split('_')[-1], float('nan')):.4f})" for u in trat),
               all(res[u]["pegadas"] < 0.50 for u in trat), dep_w6=False)
    w5 = linea("W-5", "RECUP no cae mas de 0,05",
               " · ".join(f"{u} {res[u]['recup']:.4f}" for u in trat),
               all(res[u]["recup"] >= ORIGEN_RECUP.get(u.split("_")[-1], 0) - 0.05 for u in trat),
               dep_w6=False)

    # W-7 · ⚠ TRAMPA, y por poco no se cae en ella. «El termino bajo de log 2» sólo significa
    # «ordeno» si el logit NO es constante: TODA constante da EXACTAMENTE log 2, asi que un logit
    # colapsado da 0,6931 y un `< LOG2` ingenuo lo lee como orden por una diferencia en el quinto
    # decimal. Es el mismo modo de falla que el juez de esta tarde, un escalon mas abajo.
    # La guarda es la degeneracion, no el numero: si el logit toma menos de 10 valores distintos o
    # la saturacion pasa de 0,95, el termino vale log 2 POR CONSTANTE y W-7 NO se puede activar.
    const = {u: (res[u]["distintos"] < 10 or res[u]["pegadas"] > 0.95) for u in trat}
    bajo = all(res[u]["orden"] < LOG2 - 1e-3 for u in trat)
    print(f"  W-7 · termino de orden: "
          f"{' · '.join('%.4f' % res[u]['orden'] for u in trat)}   (constante = {LOG2:.4f})")
    for u in trat:
        if const[u]:
            print(f"        {u}: el logit toma {res[u]['distintos']} valores distintos de "
                  f"{res[u]['n']} y {res[u]['pegadas']:.4f} esta en el clip -> el termino vale "
                  f"log 2 POR CONSTANTE, no porque haya ordenado")
    if any(const.values()):
        print(f"        ** W-7 {NE}: con el logit colapsado, «bajo de log 2» no significa orden. **")
    elif w1 is False and bajo:
        print(f"        ** W-7 SE ACTIVA: W-1 falla pero el termino ordeno -> es PRESUPUESTO, "
              f"no un negativo. **")

    sal = os.path.join(os.path.dirname(os.path.abspath(__file__)), "juicio_slot_20260831.json")
    with open(sal, "w") as f:
        json.dump({"unidades": res, "W1": w1, "W2": w2, "W3": w3, "W4": w4, "W5": w5,
                   "W6": legible, "W7_orden_bajo": bajo}, f, indent=1)
    print(f"\n-> {os.path.basename(sal)}")


if __name__ == "__main__":
    main()
