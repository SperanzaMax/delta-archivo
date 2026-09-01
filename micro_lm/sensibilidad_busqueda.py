"""¿La busqueda USA lo que se le pregunta? · ablacion de entidad y de relacion · 1-sep

Segunda version de la idea de Maxi (entidad contra relacion). La primera (`entidad_vs_relacion.py`)
leia la query desde la POSICION del token de entidad y del de relacion, y esta MAL: el tronco es
CAUSAL (`modelo.py:106`), asi que en la posicion de la relacion —que en «cual es el <REL> de <ENT> ?»
viene ANTES— el modelo todavia no vio la entidad. No comparaba dos vistas del mismo item: una miraba
media pregunta. Dio AUC 0,4049, invertido, y el 84 % de desacuerdo base.

Aca las DOS queries se forman en la posicion FINAL, que ve la consulta entera, y lo que cambia es la
CONSULTA:
    original          «cual es el color de julia ?»
    entidad ablada    «cual es el color de <OTRA> ?»   -> ¿se mueve la busqueda?
    relacion ablada   «cual es la <OTRA> de julia ?»   -> ¿se mueve la busqueda?

Lo que mide es SENSIBILIDAD: si cambiar la entidad no mueve adonde apunta la lectura, la busqueda no
esta usando la entidad. Prediccion del diseño de la tarea:
    `nose_ent` la entidad NO esta en el archivo -> no hay nada que encontrar -> MENOS sensible a la
               entidad que cuando la respuesta esta.
    `nose_rel` la entidad SI esta -> la busqueda deberia moverse con la entidad como en el caso sano,
               y la diferencia tendria que aparecer en la RELACION.
Si las dos sensibilidades se comportan igual en los tres grupos, no hay mecanismo que explotar.
"""
import os
import sys

import numpy as np
import jax
import jax.numpy as jnp

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)

import datos as DAT, entrenar as E, idioma as I, medir_ratio_ce as R, modelo as M

N, LOTE = 1536, 64


def auc(s, pos):
    o = np.argsort(s); r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1)
    n1, n0 = pos.sum(), (~pos).sum()
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else float("nan")


def correr(ruta):
    params, cfg, paso = R.cargar(ruta)
    params = jax.tree_util.tree_map(jnp.asarray, params)
    I.fijar_version(cfg.get("idioma", 2)); a_p = params["arch"]; donde = cfg.get("donde", "pre")
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
        s = g["sim"]
        return jnp.take_along_axis(s, pos[:, None, None], axis=1)[:, 0, :]

    rng = np.random.default_rng(54321); rr = np.random.default_rng(99)
    SE, SR, TVE, TVR, TGT, TIPO = [], [], [], [], [], []
    vistos = 0
    while vistos < N:
        b = min(LOTE, N - vistos)
        ses, cortes, turnos, mask, cons, pos, tgt, tipo = DAT.lote(
            rng, b, nivel=cfg["nivel"], n_hechos=4, n_sesiones=4, p_nose=0.4)
        base = [jnp.array(x) for x in (ses, cortes, turnos, mask, cons, pos)]
        s0 = np.asarray(leer(params, *base))

        def ablar(ids):
            c = cons.copy()
            for i in range(b):
                p = np.where(np.isin(c[i], ids))[0]
                if len(p):
                    j = int(p[-1])                       # el de la CONSULTA, que va al final
                    otras = ids[ids != c[i][j]]
                    c[i][j] = otras[rr.integers(len(otras))]
            aj = [jnp.array(x) for x in (ses, cortes, turnos, mask, c, pos)]
            return np.asarray(leer(params, *aj))

        se_, sr_ = ablar(ids_ent), ablar(ids_rel)
        p0 = np.asarray(jax.nn.softmax(jnp.asarray(s0), -1))
        pe = np.asarray(jax.nn.softmax(jnp.asarray(se_), -1))
        pr = np.asarray(jax.nn.softmax(jnp.asarray(sr_), -1))
        SE.append(s0.argmax(-1) != se_.argmax(-1))
        SR.append(s0.argmax(-1) != sr_.argmax(-1))
        TVE.append(0.5 * np.abs(p0 - pe).sum(-1))        # distancia en variacion total
        TVR.append(0.5 * np.abs(p0 - pr).sum(-1))
        TGT.append(np.asarray(tgt)); TIPO.append(np.asarray(tipo)); vistos += b

    tgt = np.concatenate(TGT); tipo = np.concatenate(TIPO)
    se, sr = np.concatenate(SE), np.concatenate(SR)
    tve, tvr = np.concatenate(TVE), np.concatenate(TVR)
    no = (tgt == E.NOSE)

    print(f"\n{'='*98}\n{os.path.basename(ruta)}  paso={paso}  n={len(tgt)}"
          f"  ·  sin respuesta {no.mean():.4f}\n{'='*98}")
    print(f"  {'grupo':26s} {'n':>5s} {'mueve_ENT':>10s} {'mueve_REL':>10s} {'TV_ent':>8s} {'TV_rel':>8s}")
    for nom, m in (("HAY respuesta", ~no), ("  nose_ent (ent ausente)", tipo == 2),
                   ("  nose_rel (otra relacion)", tipo == 3)):
        if m.sum():
            print(f"  {nom:26s} {int(m.sum()):5d} {se[m].mean():10.4f} {sr[m].mean():10.4f}"
                  f" {tve[m].mean():8.4f} {tvr[m].mean():8.4f}")

    print(f"\n  AUC contra la ausencia (mayor = mas ausente):")
    for nom, v in (("sensibilidad a ENTIDAD (TV)", -tve), ("sensibilidad a RELACION (TV)", -tvr),
                   ("cambia argmax al ablar ENT", -se.astype(float)),
                   ("cambia argmax al ablar REL", -sr.astype(float)),
                   ("TV_ent - TV_rel", -(tve - tvr))):
        print(f"    {nom:32s} {auc(v, no):.4f}")
    m_rel = (~no) | (tipo == 3)
    print(f"\n  restringido a HAY vs nose_REL (el caso dificil):")
    print(f"    sensibilidad a ENTIDAD           {auc(-tve[m_rel], no[m_rel]):.4f}")
    print(f"    sensibilidad a RELACION          {auc(-tvr[m_rel], no[m_rel]):.4f}")
    print(f"\n  [referencia] techo del estado 0,7003 · afilado ~0,48 · entidad-vs-relacion ingenuo 0,4049")


if __name__ == "__main__":
    for r in sys.argv[1:] or ["ckpts/p3_s0.pkl"]:
        correr(os.path.join(AQUI, r))
