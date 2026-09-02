"""COMPUERTA de `PREREG_CRUCE_FORMAS.md` (SHA 410acd25) · criterio X-0, BLOQUEANTE.

Verifica DOS cosas antes de gastar una sola GPU:

  1. que las distancias declaradas en `I.DIST_Q` sean las reales, contadas token a token sobre el
     texto que sale del generador, en las dos formas y en las dos versiones (vigente y anterior);
  2. que la SENSIBILIDAD de la busqueda se de vuelta entre las dos formas en un modelo de kernel 3
     ya entrenado — o sea que la ventana haga lo que se dice que hace, medido y no supuesto.

La (2) se corre sobre `v3_s0`, que nunca vio la forma `invertida`. Eso es OOD para el modelo y por
eso NO se lee como desempeño: se lee como ACCESO. La pregunta es si el token entra en el computo de
la query, y esa es una propiedad de la conv, no de lo que el modelo aprendio.
"""
import os
import sys

import numpy as np
import jax
import jax.numpy as jnp

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)

import datos as DAT, idioma as I, medir_ratio_ce as R, modelo as M

N, LOTE = 384, 64


def parte_1():
    print("=" * 92)
    print("1 · las distancias declaradas contra el texto real, token a token")
    print("=" * 92)
    I.fijar_version(2)
    ok = True
    for forma in I.FORMAS_Q:
        for cual in ("vigente", "anterior"):
            for rel in list(I.RELACIONES)[:3]:
                sust = I.RELACIONES[rel][0]
                t = I.pregunta(rel, "norte", cual, forma).split()
                d_rel = len(t) - 1 - t.index(sust)
                d_ent = len(t) - 1 - t.index("norte")
                e_rel, e_ent = I.DIST_Q[forma]["rel"], I.DIST_Q[forma]["ent"]
                bien = (d_rel == e_rel) and (d_ent == e_ent)
                ok &= bien
                if rel == list(I.RELACIONES)[0]:
                    print(f"  {forma:10s} {cual:9s} d_rel={d_rel} (dice {e_rel})  "
                          f"d_ent={d_ent} (dice {e_ent})  {'OK' if bien else '** MAL **'}"
                          f"   «{' '.join(t)}»")
                elif not bien:
                    print(f"  {forma:10s} {cual:9s} {rel}: d_rel={d_rel} d_ent={d_ent}  ** MAL **")
    print(f"\n  -> {'todas las distancias coinciden' if ok else '** HAY DISCREPANCIAS **'}")
    return ok


def parte_2(ruta="ckpts/v3_s0.pkl"):
    print("\n" + "=" * 92)
    print("2 · la sensibilidad de la busqueda se da vuelta entre las dos formas (kernel 3)")
    print("=" * 92)
    params, cfg, paso = R.cargar(os.path.join(AQUI, ruta))
    M.KQ = cfg.get("kernel_q", 3)
    params = jax.tree_util.tree_map(jnp.asarray, params)
    I.fijar_version(cfg.get("idioma", 2))
    a_p = params["arch"]; donde = cfg.get("donde", "pre")
    ids_ent = np.array([I.STOI[e] for e in I.ENTIDADES])
    ids_rel = np.array(sorted({I.STOI[v[0]] for v in I.RELACIONES.values()}))

    @jax.jit
    def leer(params, ses, cortes, turnos, mask, cons, pos):
        archivo = M.escribir(params, ses, cortes)
        ak = archivo @ a_p["kw"] + a_p["ord"][turnos]
        av = archivo @ a_p["vw"]
        penal = jnp.where(mask, 0.0, -1e9)[:, None, :]
        g = {}
        def lectura(h):
            q = h @ a_p["qr"]
            sim = jnp.einsum("btd,bnd->btn", q, ak) / jnp.sqrt(h.shape[-1]) + penal
            g["sim"] = sim
            return jnp.einsum("btn,bnd->btd", jax.nn.softmax(sim, -1), av) @ a_p["wo"]
        M.tronco(params, cons, lectura, 0, donde)
        return jnp.take_along_axis(g["sim"], pos[:, None, None], axis=1)[:, 0, :]

    print(f"  modelo {os.path.basename(ruta)} · kernel {M.KQ} (alcance {M.KQ - 1}) · paso {paso}")
    print(f"\n  {'forma':10s} {'d_ent':>6s} {'TV_ent':>10s} {'d_rel':>6s} {'TV_rel':>10s}   veredicto")
    filas = {}
    for forma in I.FORMAS_Q:
        rng = np.random.default_rng(54321); rr = np.random.default_rng(99)
        TVE, TVR, vistos = [], [], 0
        while vistos < N:
            b = min(LOTE, N - vistos)
            ses, cortes, turnos, mask, cons, pos, tgt, tipo = DAT.lote(
                rng, b, nivel=3, n_hechos=4, n_sesiones=4, p_nose=0.4, formas_q=(forma,))
            fijos = [jnp.array(x) for x in (ses, cortes, turnos, mask)]
            jp = jnp.array(pos)
            p0 = np.asarray(jax.nn.softmax(leer(params, *fijos, jnp.array(cons), jp), -1))
            for dest, ids in ((TVE, ids_ent), (TVR, ids_rel)):
                c = cons.copy()
                for i in range(b):
                    p = np.where(np.isin(c[i], ids))[0]
                    if len(p):
                        j = int(p[-1]); otras = ids[ids != c[i][j]]
                        c[i][j] = otras[rr.integers(len(otras))]
                p1 = np.asarray(jax.nn.softmax(leer(params, *fijos, jnp.array(c), jp), -1))
                dest.append(0.5 * np.abs(p0 - p1).sum(-1))
            vistos += b
        te, tr = float(np.concatenate(TVE).mean()), float(np.concatenate(TVR).mean())
        filas[forma] = (te, tr)
        de, dr = I.DIST_Q[forma]["ent"], I.DIST_Q[forma]["rel"]
        alc = M.KQ - 1
        esp_e, esp_r = de <= alc, dr <= alc
        bien = (te > 0.01) == esp_e and (tr > 0.01) == esp_r
        print(f"  {forma:10s} {de:6d} {te:10.6f} {dr:6d} {tr:10.6f}   "
              f"{'OK' if bien else '** MAL **'}  (se esperaba "
              f"ent {'>0' if esp_e else '=0'}, rel {'>0' if esp_r else '=0'})")

    d_e, d_r = filas["directa"]
    i_e, i_r = filas["invertida"]
    # `lejana` pone la relacion a distancia 4, o sea JUSTO en el borde del kernel 5 y afuera del 3.
    # Es una prediccion de borde: la misma forma tiene que dar 0 en una familia y > 0 en la otra.
    #
    # OJO — 2026-09-02, y es un error MIO de lectura automatica, el sexto de la misma forma en este
    # proyecto: el CRUCE es una prediccion sobre el kernel 3, donde una componente entra y la otra
    # no. Con kernel 5 el alcance es 4 y en las TRES formas entran las dos, asi que la sensibilidad
    # NO se da vuelta y eso es exactamente lo que la ley predice. Aplicarle el criterio de cruce a la
    # familia equivocada daba «NO CUMPLE» sobre un resultado que CUMPLE. El criterio correcto es el
    # mismo para las dos: cada celda tiene que estar en cero si y solo si esta afuera del alcance.
    alc = M.KQ - 1
    celdas, ok_celdas = 0, 0
    for forma, (te, tr) in filas.items():
        for comp, tv in (("ent", te), ("rel", tr)):
            d = I.DIST_Q[forma][comp]
            celdas += 1
            ok_celdas += int((tv > 0.01) == (d <= alc))
    todas = ok_celdas == celdas
    print(f"\n  cada celda en cero si y solo si esta AFUERA del alcance {alc}: "
          f"{ok_celdas} de {celdas}   {'CUMPLE' if todas else '** NO CUMPLE **'}")
    if M.KQ <= 3:
        cruza = (d_e > 0.01 and d_r < 0.01) and (i_r > 0.01 and i_e < 0.01)
        print(f"  X-0 BLOQUEANTE (solo aplica al kernel 3): la sensibilidad se DA VUELTA   "
              f"{'CUMPLE' if cruza else '** NO CUMPLE, no se lanza la campania **'}")
        return todas and cruza
    print(f"  (el cruce NO aplica con alcance {alc}: en las tres formas entran las dos componentes,"
          f" que es lo que la ley predice)")
    return todas


if __name__ == "__main__":
    a = parte_1()
    rutas = sys.argv[1:] or ["ckpts/v3_s0.pkl", "ckpts/kq3_s0.pkl"]
    b = all(parte_2(r) for r in rutas)
    print("\n" + "=" * 92)
    print(f"COMPUERTA {'ABRE' if (a and b) else 'NO ABRE'}")
    print("=" * 92)
