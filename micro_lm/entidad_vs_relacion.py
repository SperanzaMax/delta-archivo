"""BUSCAR POR ENTIDAD contra BUSCAR POR RELACION · idea de Maxi, version post-hoc · 1-sep

La consulta es «cual es el <REL> de <ENT> ?», o sea lleva las dos componentes en tokens distintos.
Hoy el modelo forma UNA query desde la posicion final y hace UNA busqueda. La idea es hacer DOS —una
mirando desde el token de la ENTIDAD y otra desde el de la RELACION— y preguntar si apuntan a la
misma entrada del archivo. Si no coinciden, algo no cierra: o no esta, o lo recuperado no es lo que
se pidio (la COLISION DE CLAVE, que ya esta identificada como el error dominante).

Por que puede funcionar donde el afilado fallo (`INFORME_MARGEN_BETA_20260901.md`): el afilado cambia
la FORMA de una distribucion que no contiene la señal; esto cambia la DIRECCION en la que se mira, y
son dos observaciones parcialmente independientes del mismo item —el mismo argumento que hizo andar
la fusion de cabezas en R8—.

Y hay una prediccion FUERTE y especifica del diseño de la tarea, que es lo que la hace falsable:
   `nose_ent` la entidad NO aparece      -> las dos busquedas fallan parecido, deberia separar poco
   `nose_rel` la entidad SI, otra relacion -> entidad ENCUENTRA y relacion NO: el desacuerdo tiene que
              ser MAXIMO justo en el caso que el propio codigo llama «el que se parece a una
              alucinacion real».
Si el efecto apareciera igual en los dos tipos, no seria el mecanismo propuesto sino algo inespecifico.
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
        """Devuelve las similitudes en TODAS las posiciones: el corte por token se hace afuera."""
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
        return g["sim"]

    rng = np.random.default_rng(54321)
    AE, AR, TGT, TIPO, PMAXE, PMAXR = [], [], [], [], [], []
    vistos = 0
    while vistos < N:
        b = min(LOTE, N - vistos)
        ses, cortes, turnos, mask, cons, pos, tgt, tipo = DAT.lote(
            rng, b, nivel=cfg["nivel"], n_hechos=4, n_sesiones=4, p_nose=0.4)
        aj = [jnp.array(x) for x in (ses, cortes, turnos, mask, cons, pos)]
        sim = np.asarray(leer(params, *aj))            # (B, T, N_entradas)
        for i in range(b):
            c = cons[i]
            pe = np.where(np.isin(c, ids_ent))[0]
            pr = np.where(np.isin(c, ids_rel))[0]
            # se toma la ULTIMA aparicion de cada uno: la consulta va al final de la secuencia
            ie = int(pe[-1]) if len(pe) else int(pos[i])
            ir = int(pr[-1]) if len(pr) else int(pos[i])
            se, sr = sim[i, ie], sim[i, ir]
            pe_ = np.asarray(jax.nn.softmax(jnp.asarray(se)))
            pr_ = np.asarray(jax.nn.softmax(jnp.asarray(sr)))
            AE.append(int(se.argmax())); AR.append(int(sr.argmax()))
            PMAXE.append(float(pe_.max())); PMAXR.append(float(pr_.max()))
        TGT.append(np.asarray(tgt)); TIPO.append(np.asarray(tipo)); vistos += b

    tgt = np.concatenate(TGT); tipo = np.concatenate(TIPO)
    ae, ar = np.array(AE), np.array(AR)
    no = (tgt == E.NOSE)
    desac = (ae != ar)

    print(f"\n{'='*94}\n{os.path.basename(ruta)}  paso={paso}  n={len(tgt)}"
          f"  ·  sin respuesta {no.mean():.4f}\n{'='*94}")
    print(f"  las dos busquedas DIFIEREN en el {desac.mean():.4f} de las consultas")
    print(f"\n  {'grupo':22s} {'n':>5s} {'difieren':>9s}")
    grupos = [("HAY respuesta", ~no), ("no esta (todas)", no),
              ("  nose_ent (ent ausente)", tipo == 2), ("  nose_rel (otra relacion)", tipo == 3)]
    for nom, m in grupos:
        if m.sum():
            print(f"  {nom:22s} {int(m.sum()):5d} {desac[m].mean():9.4f}")

    print(f"\n  AUC del desacuerdo contra la ausencia      {auc(desac.astype(float), no):.4f}")
    m_rel = (~no) | (tipo == 3)
    m_ent = (~no) | (tipo == 2)
    print(f"  AUC restringido a HAY vs nose_REL (dificil) {auc(desac[m_rel].astype(float), no[m_rel]):.4f}")
    print(f"  AUC restringido a HAY vs nose_ENT (facil)   {auc(desac[m_ent].astype(float), no[m_ent]):.4f}")
    # precision: de las que marca, ¿cuantas no estaban?
    if desac.sum():
        print(f"\n  cuando DIFIEREN, no estaba el              {no[desac].mean():.4f}"
              f"   (tasa base {no.mean():.4f})")
        print(f"  cuando COINCIDEN, no estaba el             {no[~desac].mean():.4f}")
        print(f"  cobertura                                  {desac.mean():.4f}")
    print(f"\n  [referencia] techo del estado 0,7003 · afilado de la busqueda ~0,48 (azar)")


if __name__ == "__main__":
    for r in sys.argv[1:] or ["ckpts/p3_s0.pkl"]:
        correr(os.path.join(AQUI, r))
