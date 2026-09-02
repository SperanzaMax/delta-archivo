"""LA LEY DE LA VENTANA · sensibilidad de la busqueda contra DISTANCIA · 2-sep

Evalua `PREREG_LEY_VENTANA.md` §B (SHA eb5e1d50), congelado antes de correr.

El hallazgo del 1-sep es que en «cual es <art> <sust> de <ent> ?» la RELACION cae a distancia 3 y la
conv que forma la query tiene alcance 2, asi que la busqueda no la ve: sensibilidad 0,0000 EXACTO.
Eso, solo, es una anecdota del generador. Lo que lo convierte en ley es el ESCALON: la sensibilidad
tiene que ser > 0 mientras la distancia entre en la ventana y CERO en cuanto la pasa, y el corte
tiene que moverse cuando se mueve el kernel.

Como se varia la distancia SIN tocar la consulta: la posicion de lectura se corre `r` lugares hacia
adelante, sobre el relleno que ya esta despues del «?». Correr la lectura r lugares es exactamente
agregar r rellenos al final, y no cambia un solo token de la pregunta. Entonces
    distancia de la ENTIDAD  = 1 + r          distancia de la RELACION = 3 + r

Por que el cero puede ser EXACTO y no aproximado: la lectura de `lat2` ocurre en el bloque 0 ANTES
del mixer (`modelo.py:222`), asi que `h` todavia es la embedding del token, sin ninguna mezcla
recurrente. La ventana de `convq` es LITERALMENTE todo lo que la query puede ver.

    kernel 3, alcance 2 ->  ENT visible en r=0,1   ·  REL nunca
    kernel 5, alcance 4 ->  ENT visible en r=0..3  ·  REL visible en r=0,1
"""
import os
import sys

import numpy as np
import jax
import jax.numpy as jnp

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)

import datos as DAT, entrenar as E, idioma as I, medir_ratio_ce as R, modelo as M

N, LOTE = 512, 64
RS = [0, 1, 2, 3, 4]
UMBRAL = 0.01          # el prereg define «≈0» ANTES de mirar: TV media < 0,01


def sondear(ruta):
    params, cfg, paso = R.cargar(ruta)
    kq = cfg.get("kernel_q", 3)
    M.KQ = kq
    params = jax.tree_util.tree_map(jnp.asarray, params)
    I.fijar_version(cfg.get("idioma", 2))
    a_p = params["arch"]; donde = cfg.get("donde", "pre")
    ids_ent = np.array([I.STOI[e] for e in I.ENTIDADES])
    ids_rel = np.array(sorted({I.STOI[v[0]] for v in I.RELACIONES.values()}))

    # El ARCHIVO no depende ni de `r` ni de la ablacion de la consulta: se escribe UNA vez por lote y
    # se reusa en las 15 lecturas. Es la misma cuenta, 10x mas barata, y ademas garantiza que las
    # condiciones se comparan contra EL MISMO archivo y no contra dos escrituras equivalentes.
    @jax.jit
    def archivar(params, ses, cortes):
        return M.escribir(params, ses, cortes)

    @jax.jit
    def leer(params, archivo, turnos, mask, cons, pos):
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

    rng = np.random.default_rng(54321); rr = np.random.default_rng(99)
    acc = {r: {"ent": [], "rel": []} for r in RS}
    desc = {r: 0 for r in RS}
    vistos = 0
    while vistos < N:
        b = min(LOTE, N - vistos)
        ses, cortes, turnos, mask, cons, pos, tgt, tipo = DAT.lote(
            rng, b, nivel=cfg["nivel"], n_hechos=4, n_sesiones=4, p_nose=0.4)

        def ablada(ids):
            c = cons.copy()
            for i in range(b):
                p = np.where(np.isin(c[i], ids))[0]
                if len(p):
                    j = int(p[-1])
                    otras = ids[ids != c[i][j]]
                    c[i][j] = otras[rr.integers(len(otras))]
            return c

        c_ent, c_rel = ablada(ids_ent), ablada(ids_rel)
        archivo = archivar(params, jnp.array(ses), jnp.array(cortes))
        jt, jm = jnp.array(turnos), jnp.array(mask)
        jc, jce, jcr = jnp.array(cons), jnp.array(c_ent), jnp.array(c_rel)
        for r in RS:
            # `pos + r` = leer r lugares mas adelante, sobre el relleno.
            #
            # OJO, y costo un 0,0106 donde tenia que haber un cero exacto (1-sep, primera pasada):
            # la consulta «cual era antes ... ?» es DOS tokens mas larga, asi que con r grande
            # `pos + r` se pasaba de T_Q y quedaba RECORTADO. Recortar la posicion ACERCA el token
            # a la ventana: esas muestras no median la distancia que decia la columna. Se descartan
            # en vez de recortarse, y se informa cuantas son.
            cabe = (pos + r) <= (cons.shape[1] - 1)
            desc[r] += int((~cabe).sum())
            if not cabe.any():
                continue
            pr_ = jnp.array(pos + r)
            p0 = np.asarray(jax.nn.softmax(leer(params, archivo, jt, jm, jc, pr_), -1))
            for nom, c in (("ent", jce), ("rel", jcr)):
                p1 = np.asarray(jax.nn.softmax(leer(params, archivo, jt, jm, c, pr_), -1))
                acc[r][nom].append((0.5 * np.abs(p0 - p1).sum(-1))[cabe])
        vistos += b

    res = {r: {k: (float(np.concatenate(v).mean()) if v else float("nan"))
               for k, v in d.items()} for r, d in acc.items()}
    for r in RS:
        res[r]["descartadas"] = desc[r] / max(1, N)
    return kq, paso, res


def main():
    unidades = sys.argv[1:] or ["ckpts/v3_s0.pkl", "ckpts/v3_s1.pkl", "ckpts/v3_s2.pkl",
                                "ckpts/kq3_s0.pkl", "ckpts/kq3_s1.pkl", "ckpts/kq3_s2.pkl"]
    print("=" * 100)
    print("LEY DE LA VENTANA · TV de la busqueda al ablar cada componente · prereg SHA eb5e1d50 §B")
    print("=" * 100)
    print("La posicion de lectura se corre r lugares sobre el relleno:")
    print("   distancia de la ENTIDAD = 1+r   ·   distancia de la RELACION = 3+r")
    print(f"   «≈0» es TV media < {UMBRAL} y esta escrito en el prereg, antes del dato.\n")
    todo = {}
    for u in unidades:
        ruta = os.path.join(AQUI, u)
        if not os.path.exists(ruta):
            print(f"  (falta {u})"); continue
        kq, paso, res = sondear(ruta)
        nom = os.path.basename(ruta).replace(".pkl", "")
        todo[nom] = (kq, res)
        alcance = kq - 1
        print(f"--- {nom}  kernel {kq} (alcance {alcance})  paso={paso}")
        print(f"  {'r':>2s} {'d_ent':>6s} {'TV_ent':>9s} {'':6s} {'d_rel':>6s} {'TV_rel':>9s}"
              f" {'':6s} {'descart':>8s}")
        for r in RS:
            de, dr = 1 + r, 3 + r
            te, tr = res[r]["ent"], res[r]["rel"]
            me = "dentro" if de <= alcance else "AFUERA"
            mr = "dentro" if dr <= alcance else "AFUERA"
            print(f"  {r:2d} {de:6d} {te:9.6f} {me:>6s} {dr:6d} {tr:9.6f} {mr:>6s}"
                  f" {res[r]['descartadas']:8.3f}"
                  f"   {'  <- ESCALON' if (de == alcance or dr == alcance) else ''}")
        print()

    if not todo:
        return
    print("=" * 100)
    print("B-1 · el escalon cae donde lo predice el alcance")
    print("=" * 100)
    ok, total = 0, 0
    for nom, (kq, res) in todo.items():
        alcance = kq - 1
        fallas = []
        for r in RS:
            for comp, d in (("ent", 1 + r), ("rel", 3 + r)):
                tv = res[r][comp]
                dentro = d <= alcance
                total += 1
                if dentro and tv > UMBRAL:
                    ok += 1
                elif (not dentro) and tv < UMBRAL:
                    ok += 1
                else:
                    fallas.append(f"{comp} d={d} {'dentro' if dentro else 'afuera'} TV={tv:.6f}")
        print(f"  {nom:9s} kernel {kq}: {'TODAS las celdas cumplen' if not fallas else 'fallan -> ' + '; '.join(fallas)}")
    print(f"\n  celdas que cumplen la prediccion: {ok} de {total}")
    print("=" * 100)


if __name__ == "__main__":
    main()
