"""Precision del desacuerdo CONTRA la confianza a IGUAL COBERTURA · control que faltaba (1-sep)

Criterios congelados en NOTA_PRECISION_DESACUERDO.md (SHA 9188e02d) ANTES de correr esto.

El 31-ago quedo medido que cuando las dos respuestas difieren el 89,8 % estan mal, contra 53,7 % de
tasa base. Lo que NO se hizo es el control: marcar por CONFIANZA BAJA el mismo 9,6 % de preguntas y
ver si da lo mismo. Si da lo mismo, preguntar dos veces no compra nada sobre una sola pasada, que es
lo que D-4 ya dictamino para el AUC.

Se le da al control su MEJOR version (confianza en la pasada limpia y en la ruidosa, gana la mejor):
un positivo contra el control mas fuerte es lo unico que vale.
"""
import os, sys
import numpy as np, jax, jax.numpy as jnp
AQUI = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, AQUI)
import datos as DAT, entrenar as E, idioma as I, medir_ratio_ce as R, modelo as M

N, LOTE, SIGMA = 1536, 64, 0.4
NBOOT = 10000
MIN_MARCADAS = 30          # §3 del prereg: por debajo de esto el juez devuelve NO EVALUABLE


def ic95(v):
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def correr(ruta, n=N):
    params, cfg, paso = R.cargar(ruta)
    params = jax.tree_util.tree_map(jnp.asarray, params)
    I.fijar_version(cfg.get("idioma", 2)); a_p = params["arch"]; donde = cfg.get("donde", "pre")

    @jax.jit
    def responder(params, ses, cortes, turnos, mask, cons, pos, ruido):
        archivo = M.escribir(params, ses, cortes)
        ak = archivo @ a_p["kw"] + a_p["ord"][turnos]; av = archivo @ a_p["vw"]
        penal = jnp.where(mask, 0.0, -1e9)[:, None, :]
        def lectura(h):
            q = h @ a_p["qr"] + ruido
            sim = jnp.einsum("btd,bnd->btn", q, ak)/jnp.sqrt(h.shape[-1]) + penal
            return jnp.einsum("btn,bnd->btd", jax.nn.softmax(sim, -1), av) @ a_p["wo"]
        h = M.tronco(params, cons, lectura, 0, donde)
        lg = M.ln(params["ln_f"], h) @ params["head"]["w"] + params["head"]["b"]
        lg = jnp.take_along_axis(lg, pos[:, None, None], axis=1)[:, 0, :]
        lg = lg.at[:, E.NOSE].set(-1e9)
        p = jax.nn.softmax(lg, -1)
        return lg.argmax(-1), p.max(-1)

    # mismas semillas que dos_veces.py para que la replica sea de la MISMA distribucion
    rng = np.random.default_rng(4242); rk = np.random.default_rng(11)
    A1, A2, A0, C0, C1, TGT = [], [], [], [], [], []
    vistos = 0
    while vistos < n:
        b = min(LOTE, n - vistos)
        ses, cortes, turnos, mask, cons, pos, tgt, tipo = DAT.lote(
            rng, b, nivel=cfg["nivel"], n_hechos=4, n_sesiones=4, p_nose=0.4)
        aj = [jnp.array(x) for x in (ses, cortes, turnos, mask, cons, pos)]
        D = a_p["qr"].shape[-1]
        nq = float(jnp.linalg.norm(M.ln(params["blocks"][0]["ln1"], params["emb"][aj[4]]) @ a_p["qr"],
                                   axis=-1).mean())
        e = SIGMA * nq / np.sqrt(D)
        a0, c0 = responder(params, *aj, jnp.zeros((b, 1, D)))          # pasada LIMPIA
        a1, c1 = responder(params, *aj, jnp.array(rk.normal(size=(b, 1, D))*e))
        a2, _ = responder(params, *aj, jnp.array(rk.normal(size=(b, 1, D))*e))
        A0.append(np.asarray(a0)); C0.append(np.asarray(c0))
        A1.append(np.asarray(a1)); C1.append(np.asarray(c1)); A2.append(np.asarray(a2))
        TGT.append(np.asarray(tgt)); vistos += b

    tgt = np.concatenate(TGT); hay = tgt != E.NOSE
    a1, a2 = np.concatenate(A1), np.concatenate(A2)
    c0, c1 = np.concatenate(C0), np.concatenate(C1)
    mal = ~((a1 == tgt) & hay)                 # misma definicion que dos_veces.py
    desac = (a1 != a2)
    k = int(desac.sum()); cstar = desac.mean()

    print(f"\n{'='*90}\n{os.path.basename(ruta)}  paso={paso}  n={len(tgt)}  sigma={SIGMA}"
          f"  ·  sin respuesta {(~hay).mean():.4f}\n{'='*90}")
    print(f"  tasa base de «mal»                    {mal.mean():.4f}")
    print(f"  cobertura del desacuerdo  c*          {cstar:.4f}   ({k} preguntas marcadas)")

    if k < MIN_MARCADAS:
        print(f"\n  ** NO EVALUABLE ** el desacuerdo marca {k} < {MIN_MARCADAS}: la precision no es "
              f"estimable.\n  P-1, P-2 y P-3 quedan sin leer (§3 del prereg).")
        return

    # --- marcar por confianza baja el MISMO c* (desempate estable por indice) ---
    def marca_por_conf(c):
        o = np.lexsort((np.arange(len(c)), c))   # menor confianza primero
        m = np.zeros(len(c), bool); m[o[:k]] = True
        return m
    m_c0, m_c1 = marca_por_conf(c0), marca_por_conf(c1)

    p_des = mal[desac].mean(); p_c0 = mal[m_c0].mean(); p_c1 = mal[m_c1].mean()
    mejor_conf, cual = (p_c0, "limpia") if p_c0 >= p_c1 else (p_c1, "ruidosa r1")

    print(f"\n--- precision a IGUAL cobertura ({cstar:.4f}) ---")
    print(f"  desacuerdo                            {p_des:.4f}")
    print(f"  confianza baja · pasada limpia        {p_c0:.4f}")
    print(f"  confianza baja · pasada ruidosa r1    {p_c1:.4f}")
    print(f"  marcar al azar (= tasa base)          {mal.mean():.4f}")
    print(f"  -> el control se queda con la MEJOR:  {mejor_conf:.4f}  ({cual})")

    # --- bootstrap pareado: se remuestrean MUESTRAS, no marcados ---
    rb = np.random.default_rng(2026)
    m_best = m_c0 if cual == "limpia" else m_c1
    dif, pd_, pc_ = np.empty(NBOOT), np.empty(NBOOT), np.empty(NBOOT)
    idx = np.arange(len(tgt))
    for i in range(NBOOT):
        s = rb.choice(idx, len(idx), replace=True)
        d, c, m = desac[s], m_best[s], mal[s]
        pd_[i] = m[d].mean() if d.any() else np.nan
        pc_[i] = m[c].mean() if c.any() else np.nan
        dif[i] = pd_[i] - pc_[i]
    ok = ~np.isnan(dif); dif, pd_, pc_ = dif[ok], pd_[ok], pc_[ok]
    lo_d, hi_d = ic95(dif); lo_p, hi_p = ic95(pd_)

    print(f"\n--- bootstrap pareado ({len(dif)} remuestreos validos) ---")
    print(f"  precision del desacuerdo   IC95 [{lo_p:.4f}, {hi_p:.4f}]")
    print(f"  diferencia (desac - conf)  {p_des-mejor_conf:+.4f}  IC95 [{lo_d:+.4f}, {hi_d:+.4f}]")

    # --- criterios ---
    p1 = (p_des - mejor_conf) >= 0.05 and lo_d > 0
    p2 = (p_des > mal.mean() + 0.10) and lo_p > mal.mean()
    p3 = lo_p <= 0.8980 <= hi_p
    inter = (desac & m_best).sum(); union = (desac | m_best).sum()
    jac = inter/union if union else float("nan")

    print(f"\n--- criterios (congelados en SHA 9188e02d) ---")
    print(f"  P-1 PRINCIPAL  desac supera a la confianza por >= 0,05 y el IC95 excluye 0   "
          f"{'CUMPLE' if p1 else 'NO CUMPLE'}")
    print(f"  P-2 PISO       desac > tasa base + 0,10 con IC95 por encima de la base       "
          f"{'CUMPLE' if p2 else 'NO CUMPLE'}")
    print(f"  P-3 REPLICA    el 0,8980 del 31-ago cae dentro del IC95                      "
          f"{'CUMPLE' if p3 else 'NO CUMPLE'}")
    print(f"  P-4 DESCRIPTIVO  Jaccard desacuerdo vs confianza = {jac:.4f} "
          f"({inter} en comun de {union})   [no adjudica]")

    if not p1:
        print("\n  ** El 0,8980 NO se atribuye al desacuerdo: a igual cobertura la confianza hace "
              "lo mismo o mas.\n     Es el mismo dictamen de D-4, ahora en precision. La version "
              "ENTRENADA sigue sin probar. **")
    else:
        print("\n  ** El desacuerdo SUPERA a la confianza a igual cobertura: mide algo que una sola "
              "pasada no da. **")

    # --- EXPLORATORIO, declarado como tal: la vIa que P-4 anticipo por escrito ANTES del dato ---
    # «Un solapamiento bajo con precisiones parecidas significa que senalan preguntas distintas y que
    #  combinarlos compra cobertura; queda anotado como via, nunca como resultado de esta corrida.»
    uni, ind = desac | m_best, desac & m_best
    print(f"\n--- EXPLORATORIO (post-hoc, NO adjudica · anticipado por P-4) ---")
    print(f"  union   desacuerdo O confianza   precision {mal[uni].mean():.4f}   "
          f"cobertura {uni.mean():.4f}  ({uni.sum()} marcadas)")
    print(f"  interseccion  desacuerdo Y conf  precision {mal[ind].mean():.4f}   "
          f"cobertura {ind.mean():.4f}  ({ind.sum()} marcadas)")
    print(f"  solo desacuerdo (conf alta)      precision {mal[desac & ~m_best].mean():.4f}   "
          f"({(desac & ~m_best).sum()} marcadas)")
    print(f"  solo confianza (sin desacuerdo)  precision {mal[m_best & ~desac].mean():.4f}   "
          f"({(m_best & ~desac).sum()} marcadas)")
    # control: la union tiene mas cobertura, asi que se compara contra la CONFIANZA a esa cobertura
    ku = int(uni.sum())
    o = np.lexsort((np.arange(len(c0)), c0)); m_cu = np.zeros(len(c0), bool); m_cu[o[:ku]] = True
    print(f"  CONTROL: confianza sola a la cobertura de la union ({uni.mean():.4f})   "
          f"precision {mal[m_cu].mean():.4f}")

    np.savez(os.path.join(AQUI, "salidas", "precision_desacuerdo_n3_s0.npz"),
             tgt=tgt, hay=hay, a1=a1, a2=a2, c0=c0, c1=c1, mal=mal, desac=desac)
    print("\n  arrays guardados en salidas/precision_desacuerdo_n3_s0.npz "
          "(cualquier analisis posterior sale de ahi, sin gastar CPU de nuevo)")


if __name__ == "__main__":
    n = int(sys.argv[2]) if len(sys.argv) > 2 else N
    correr(os.path.join(AQUI, sys.argv[1] if len(sys.argv) > 1 else "ckpts/n3_s0.pkl"), n)
