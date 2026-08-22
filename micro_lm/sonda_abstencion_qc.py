"""A-1..A-4 · ¿la query conjunta hace visible en la ENTRADA que la respuesta no esta?

`PREREG_ABSTENCION_QC.md`, hasheado mientras la campania entrenaba y antes de mirar ningun resultado.

La idea, en una linea: con una query que es funcion pura del token, una pregunta SIN respuesta se ve
igual que una CON respuesta —en las dos hay un puñado de entradas que matchean la relacion—; con una
query conjunta, la ausencia deberia notarse porque ninguna entrada matchea fuerte.

    python sonda_abstencion_qc.py --dir-ckpt ckpts/qc_congelados --paso 26000

Detalle que importa y que la sonda del empate no podia tener: los scores se calculan **respetando la
arquitectura del checkpoint**. Para `post` la query no es `ln1(emb[x]) @ qr` sino la que sale despues
del mixer, y escribirla a mano en dos formulas paralelas era la manera segura de que una de las dos
quedara mal. En vez de eso se le pasa a `modelo.tronco` una lectura que CAPTURA su entrada y devuelve
ceros: el forward queda intacto y la query sale, por construccion, de la misma linea de codigo que
usa el modelo cuando entrena.
"""
import argparse
import json
import os
import pickle

import jax
import jax.numpy as jnp
import numpy as np

import datos as DAT
import idioma as I
import modelo as M

NOSE = I.STOI["NOSE"]
SEM_PRUEBA = 77000


def auc(x, y):
    """P(x > y) con empates a 0,5. x = positivos, y = negativos."""
    x = np.asarray(x)[~np.isnan(x)]
    y = np.asarray(y)[~np.isnan(y)]
    if not len(x) or not len(y):
        return float("nan")
    todo = np.concatenate([x, y])
    r = np.empty(len(todo))
    orden = np.argsort(todo, kind="mergesort")
    ordenado = todo[orden]
    i = 0
    while i < len(ordenado):
        j = i
        while j + 1 < len(ordenado) and ordenado[j + 1] == ordenado[i]:
            j += 1
        r[orden[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((r[:len(x)].sum() - len(x) * (len(x) + 1) / 2.0) / (len(x) * len(y)))


def scores_y_pred(params, ses, cortes, turnos, mask, cons, pos, donde):
    """Devuelve (scores (B,Tq,N), prediccion (B,)) con la arquitectura que diga el checkpoint."""
    a = params["arch"]
    archivo = M.escribir(params, ses, cortes)
    ak = archivo @ a["kw"] + a["ord"][turnos]
    av = archivo @ a["vw"]
    penal = jnp.where(mask, 0.0, -1e9)[:, None, :]

    guardado = {}

    def lectura(h):
        q = h @ a["qr"]
        sim = jnp.einsum("btd,bnd->btn", q, ak) / jnp.sqrt(h.shape[-1]) + penal
        guardado["sim"] = sim
        return jnp.einsum("btn,bnd->btd", jax.nn.softmax(sim, -1), av) @ a["wo"]

    h = M.tronco(params, cons, lectura, 0, donde)
    hn = M.ln(params["ln_f"], h)
    lg = hn @ params["head"]["w"] + params["head"]["b"]
    lg = jnp.take_along_axis(lg, pos[:, None, None], axis=1)[:, 0, :]
    al = (hn @ params["abst"]["w"] + params["abst"]["b"])[..., 0]
    al = jnp.take_along_axis(al, pos[:, None], axis=1)[:, 0]
    pred = jnp.where(al > 0.0, NOSE, lg.at[:, NOSE].set(-jnp.inf).argmax(-1))
    return guardado["sim"], pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir-ckpt", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                       "ckpts", "qc_congelados"))
    ap.add_argument("--unidades", default="p3_s0,p3_s1,p3_s2,q3_s0,q3_s1,q3_s2")
    ap.add_argument("--n", type=int, default=32)          # lotes
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--p-nose", type=float, default=0.4)
    ap.add_argument("--salida", default="abstencion_qc_20260822.json")
    A = ap.parse_args()

    print("A-1..A-4 · ¿se ve en la ENTRADA que la respuesta no esta?")
    print(f"{A.n * A.batch} muestras por unidad · p_nose={A.p_nose}\n")
    print(f"{'unidad':<8} {'donde':<5} | {'A-1 s1':>8} {'s1|tam':>9} | {'nulo':>7} {'nulo|tam':>9} | "
          f"{'ok':>7} | {'n_con':>6} {'n_sin':>6}")
    print("-" * 88)

    res = {}
    for uni in A.unidades.split(","):
        ck = os.path.join(A.dir_ckpt, f"{uni}.pkl")
        if not os.path.exists(ck):
            print(f"{uni:<8} sin checkpoint")
            continue
        with open(ck, "rb") as f:
            d = pickle.load(f)
        params = jax.tree_util.tree_map(jnp.asarray, d["params"])
        donde = d.get("config", {}).get("donde", "pre")
        nivel = d["config"]["nivel"]
        semilla = d["config"]["semilla"]
        rng = np.random.default_rng(SEM_PRUEBA + semilla)
        rng_nulo = np.random.default_rng(9000 + semilla)
        fn = jax.jit(lambda p, s, c, t, m, q, po: scores_y_pred(p, s, c, t, m, q, po, donde))

        S1, S1N, SIN, OK, NUL, TAM = [], [], [], [], [], []
        for _ in range(A.n):
            ses, cortes, turnos, mask, cons, pos, tgt, tipo, meta = DAT.lote(
                rng, A.batch, nivel=nivel, n_hechos=4, n_sesiones=4, p_vieja=0.35,
                p_nose=A.p_nose, con_meta=True)
            sim, pred = fn(params, jnp.array(ses), jnp.array(cortes), jnp.array(turnos),
                           jnp.array(np.asarray(mask)), jnp.array(cons), jnp.array(pos))
            sim = np.asarray(sim)
            pred = np.asarray(pred)
            tgt = np.asarray(tgt); tipo = np.asarray(tipo)
            for b in range(len(sim)):
                st = sim[b]
                val = st > -1e8                      # entradas que compiten
                if not val.any():
                    continue
                st = np.where(val, st, np.nan)
                # `s1` = el score crudo mas alto de todo el episodio: ¿hay ALGUNA entrada que
                # matchee fuerte en ALGUNA posicion de la consulta? Es la pregunta del prereg.
                s1 = np.nanmax(st)
                # Secundaria, declarada como tal: el mismo maximo en unidades de la dispersion del
                # episodio. Los scores crudos dependen de la norma de q, que no tiene por que ser
                # comparable entre muestras; esta version saca ese factor.
                sd = np.nanstd(st) + 1e-12
                s1n = (s1 - np.nanmean(st)) / sd
                # A-3 · nulo: mismo estadistico sobre gaussianas de igual media y desvio por posicion
                mu_p = np.nanmean(st, axis=1, keepdims=True)
                sd_p = np.nanstd(st, axis=1, keepdims=True) + 1e-12
                nul = float(np.nanmax(mu_p + sd_p * rng_nulo.standard_normal(st.shape)))
                S1.append(float(s1)); S1N.append(float(s1n)); NUL.append(nul)
                SIN.append(bool(tipo[b] >= 2))       # la respuesta NO esta en el archivo
                OK.append(bool(pred[b] == tgt[b]))
                # TAMAÑO del episodio: cuantas posiciones de consulta y cuantas entradas compiten.
                # `s1` es un MAXIMO, y el maximo de una muestra crece con la cantidad de elementos
                # sobre los que se toma, sin que el modelo tenga nada que ver. Si las preguntas sin
                # respuesta difieren en tamaño de las que si la tienen, `s1` las separa por eso solo.
                # Medido en la linea de base del 22-ago sobre `pre` maduro: el NULO —gaussianas de
                # igual media y desvio— daba 0,53-0,60 en vez de 0,50, que es la firma exacta de este
                # confound. Se guarda el tamaño para poder estratificar.
                TAM.append((int(val.shape[0]), int(val.sum())))

        S1 = np.array(S1); S1N = np.array(S1N); NUL = np.array(NUL)
        SIN = np.array(SIN); OK = np.array(OK)
        con = ~SIN

        def auc_estrat(x, sel_pos, sel_neg):
            """AUC comparando SOLO dentro de episodios del mismo tamaño, pesado por estrato."""
            grupos = {}
            for i, t in enumerate(TAM):
                grupos.setdefault(t, ([], []))
                if sel_pos[i]:
                    grupos[t][0].append(x[i])
                elif sel_neg[i]:
                    grupos[t][1].append(x[i])
            num, den = 0.0, 0
            for t, (a_, b_) in grupos.items():
                if a_ and b_:
                    v = auc(a_, b_)
                    if not np.isnan(v):
                        w = min(len(a_), len(b_))
                        num += v * w; den += w
            return (num / den) if den else float("nan")
        # Positivo = CON respuesta. Se espera score MAS ALTO cuando la respuesta esta.
        r = {"donde": donde, "n": int(len(S1)), "n_con": int(con.sum()), "n_sin": int(SIN.sum()),
             "auc_s1": auc(S1[con], S1[SIN]),
             "auc_s1_norm": auc(S1N[con], S1N[SIN]),
             "auc_nulo": auc(NUL[con], NUL[SIN]),
             # Las dos que deciden: mismo estadistico comparando solo episodios del mismo tamaño.
             "auc_s1_estratam": auc_estrat(S1, con, SIN),
             "auc_nulo_estratam": auc_estrat(NUL, con, SIN)}
        # A-4 · estratificado por acierto: sólo entre las que el modelo contesto BIEN, ¿sigue
        # separando? (las SIN respuesta acertadas son las que dijo NOSE bien)
        sel = OK
        r["auc_estrat"] = auc(S1[con & sel], S1[SIN & sel])
        res[uni] = r
        print(f"{uni:<8} {donde:<5} | {r['auc_s1']:>8.4f} {r['auc_s1_estratam']:>9.4f} | "
              f"{r['auc_nulo']:>7.4f} {r['auc_nulo_estratam']:>9.4f} | {r['auc_estrat']:>7.4f} | "
              f"{r['n_con']:>6} {r['n_sin']:>6}")

    evaluar(res)
    with open(A.salida, "w") as f:
        json.dump({"prereg": "PREREG_ABSTENCION_QC.md", "unidades": res}, f, indent=1)
    print(f"\n-> {A.salida}")


def evaluar(res):
    pre = {u: v for u, v in res.items() if v["donde"] == "pre"}
    post = {u: v for u, v in res.items() if v["donde"] == "post"}
    if not pre or not post:
        print("\nfaltan familias: no se evalua")
        return
    # ENMIENDA E-1 (22-ago): todo se lee sobre el MARGEN contra el nulo de la misma unidad, no
    # sobre el AUC crudo. El nulo preserva la escala de los scores del episodio, asi que mide cuanto
    # se consigue con la escala sola; en `pre` maduro vale 0,53-0,60 y `s1` queda POR DEBAJO.
    margen = lambda v: v["auc_s1"] - v["auc_nulo"]
    pares = []
    for u, v in post.items():
        s = u.split("_s")[1]
        gemelo = next((w for w in pre if w.endswith(f"_s{s}")), None)
        if gemelo:
            pares.append((s, margen(pre[gemelo]), margen(v)))

    print("\n" + "=" * 74)
    a1_ok = [p for p in pares if p[2] - p[1] >= 0.05]
    print(f"A-1 · el MARGEN (s1 - nulo) sube >= 0,05 de pre a post")
    for s, x, y in pares:
        print(f"     s{s}: {x:.4f} -> {y:.4f}  ({y - x:+.4f})")
    a1 = len(a1_ok) >= 2
    print(f"     A-1: {'CUMPLE' if a1 else 'NO CUMPLE'}  ({len(a1_ok)}/{len(pares)})")

    altos = [u for u, v in post.items() if v["auc_s1"] >= 0.75]
    a2 = len(altos) >= 2
    print(f"\nA-2 · AUC(s1) en post >= 0,75 en >= 2 de 3: {len(altos)}/{len(post)} "
          f"-> {'CUMPLE' if a2 else 'NO CUMPLE'}")

    # A-3 nuevo: el margen contra el propio nulo tiene que ser >= 0,05. Un s1 alto con un nulo
    # igual de alto no es señal, es la escala.
    margenes = [margen(v) for v in post.values()]
    a3 = sum(m >= 0.05 for m in margenes) >= 2
    print(f"\nA-3 · margen (s1 - nulo) en post: {[f'{m:+.4f}' for m in margenes]} "
          f"(hace falta >= +0,05 en >= 2) -> {'PASA' if a3 else 'NO PASA'}")
    for u, v in post.items():
        print(f"     {u}: s1 {v['auc_s1']:.4f} · nulo {v['auc_nulo']:.4f} · margen {margen(v):+.4f}")

    estr = [v["auc_estrat"] for v in post.values()]
    print(f"\nA-4 · estratificado por acierto en post: {[f'{x:.4f}' for x in estr]}")

    print("-" * 74)
    if not a3:
        print("VEREDICTO: sin nulo limpio no hay resultado. Es la leccion del 20-ago —U-1 'pasaba'")
        print("en las 2 celdas donde el nulo tambien pasaba—.")
    elif a1 and a2:
        print("VEREDICTO: hay senial sin etiquetas en la ENTRADA, y con magnitud utilizable. El")
        print("cierre del 21-ago era del REGIMEN, no del metodo. Falta el corte.")
    elif a1:
        print("VEREDICTO: hay senial nueva pero no alcanza (A-2 no). Via candidata, no corte.")
    else:
        print("VEREDICTO: el cierre del 21-ago se mantiene, y ahora tambien para la query conjunta.")


if __name__ == "__main__":
    main()
