"""ENCADENADOS · INTERVENCION: ¿el encadenado se hace AL ESCRIBIR? · 2026-09-12

Sigue a `encadenados_mecanismo.py`. Ahi salio que a nivel 3 TODOS los enunciados caen en la sesion
0 (`idioma.episodio`: `s = 0 if nivel < 4`), asi que el hecho de altura del nombre preguntado
siempre se escribe DESPUES del hecho de persona y EN LA MISMA SESION, con el estado recurrente
del tronco cargado con «director de barrio es yamil». Y la lectura en el «?» cae sobre la entrada
de ALTURA del nombre correcto (p_alt 0,55 / 0,84), no sobre la de persona (0,07 / 0,005): la
entrada de altura ya «sabe» de quien es. Eso es compatible con encadenar al escribir, pero la
observacion no lo prueba: falta la intervencion.

Tres condiciones sobre LOS MISMOS episodios, cambiando solo donde se escriben los hechos de
altura/clave (el bloque de tres candidatos), con los turnos ORIGINALES de cada enunciado (el sello
no cambia) y sin tocar un peso:

    original      sesion 0 = hechos de persona ... + bloque de altura     (como se entreno)
    otra_sesion   sesion 0 = hechos de persona ...; sesion 1 = BOS + bloque de altura
                  (estado reseteado: la entrada de altura no puede llevar el hecho de persona)
    invertido     sesion 0 = bloque de altura + hechos de persona ...
                  (misma sesion, pero el hecho de persona viene DESPUES: la recurrencia es causal)

Prediccion si el encadenado es al escribir: `compuesta` cae en `otra_sesion` e `invertido` hacia
el azar entre 3 candidatos (o NOSE), y `p_alt` deja de concentrarse en la entrada correcta.
Control: `vigente`/`anterior` (preguntas simples) no deberian moverse en `otra_sesion` (las
correcciones siguen adyacentes a su hecho); en `invertido` tampoco, salvo por el prefijo distinto.

Uso: python encadenados_intervencion.py ckpts/kc3_s1.pkl [mas...]   Env: N B SEM
"""
import os, sys, json, pickle, time
import numpy as np
import jax, jax.numpy as jnp

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
import conf_ckpt
import datos as DAT, idioma as I, modelo as M
import entrenar as E
from encadenados_mecanismo import preparar, partes_p, enunciados_de, es_rel2

N = int(os.environ.get("N", 40)); B = int(os.environ.get("B", 64)); SEM = int(os.environ.get("SEM", 2026))
CONDS = tuple(os.environ.get("CONDS", "original,otra_sesion,invertido,barajada").split(","))
# `barajada` (12-sep, 10:30): como `otra_sesion` pero el bloque de altura se escribe en OTRO ORDEN
# (permutacion fija por episodio) y los turnos se reasignan por la posicion nueva. Existe porque en
# la v3 (bloque en otra sesion al entrenar) los DOS brazos subieron a 0,5-0,7 en 500-1500 pasos, y
# hay un atajo posible que no es encadenar: los hechos de altura se escriben en el MISMO ORDEN que
# los hechos de persona, asi que «el k-esimo de altura es el del k-esimo de persona» se resuelve por
# el sello de orden. Si la compuesta cae al barajar, es ese atajo; si aguanta, es lectura.
_RNG_BARAJA = np.random.default_rng(777)


def rearmar(ses_b, cortes_b, mask_b, turnos_b, modo):
    """Devuelve (ses, cortes, mask, turnos, mapa) para la condicion. `mapa` lleva n_viejo -> n_nuevo."""
    ens = enunciados_de(ses_b, cortes_b, mask_b)
    alt = [x for x in ens if es_rel2(x[3])]
    resto = [x for x in ens if not es_rel2(x[3])]
    if modo == "original":
        S0 = ses_b.shape[0]
        sesiones = [[x for x in ens if x[1] == s] for s in range(S0)]
        while len(sesiones) > 1 and not sesiones[-1]:
            sesiones.pop()
    elif modo == "misma_despues":         # todo en la sesion 0, el bloque de altura al final
        sesiones = [resto + alt]
    elif modo == "otra_sesion":
        sesiones = [resto, alt]
    elif modo == "invertido":
        sesiones = [alt + resto]
    elif modo == "barajada":
        alt = list(alt)
        turnos_alt = sorted(turnos_b[n] for n, _, _, _ in alt)
        perm = _RNG_BARAJA.permutation(len(alt))
        alt = [alt[i] for i in perm]
        sesiones = [resto, alt]
    else:
        raise ValueError(modo)
    S = ses_b.shape[0]
    ses = np.full_like(ses_b, DAT.PAD); cortes = np.zeros_like(cortes_b)
    mask = np.zeros_like(mask_b); turnos = np.zeros_like(turnos_b)
    mapa = {}
    ses[:, 0] = I.STOI["BOS"]                      # las sesiones vacias tambien llevan BOS (datos.lote)
    for s, lista in enumerate(sesiones):
        toks = [I.STOI["BOS"]]
        if len(lista) > DAT.E_MAX or s >= S:
            return None                        # no entra (con el bloque en otra sesion, todo junto puede pasar de E_MAX)
        for e, (n_viejo, _, _, t) in enumerate(lista):
            ids = [I.STOI[x] for x in t]
            if len(toks) + len(ids) >= DAT.T_SES:
                return None
            toks += ids
            n = s * DAT.E_MAX + e
            cortes[s, e] = len(toks) - 1
            mask[n] = True
            turnos[n] = turnos_b[n_viejo] if modo != "barajada" or s == 0 else turnos_alt[e]
            mapa[n_viejo] = n
        ses[s, :len(toks)] = toks
    return ses, cortes, mask, turnos, mapa


def clasificar(ens, q_toks, r_tok):
    """Entrada de altura del nombre preguntado, entrada de persona vigente, y las otras de altura."""
    try:
        rel2s = q_toks[3]; ent = q_toks[7]
    except IndexError:
        return None
    nombres = set(I.NOMBRES)
    alts = [(n, t) for n, s, e, t in ens if es_rel2(t)]
    pers = [(n, t) for n, s, e, t in ens if not es_rel2(t)]
    cand = [(n, t) for n, t in alts if r_tok in t]
    if len(cand) > 1:      # dos candidatos con el mismo numero: el correcto es el que tiene nombre de persona vigente
        cand = [(n, t) for n, t in cand if any(nm in tp for _, tp in pers for nm in [x for x in t if x in nombres])]
    if len(cand) != 1:
        return None
    n_alt, t_alt = cand[0]
    nombre = [x for x in t_alt if x in nombres][0]
    p_ = [n for n, t in pers if nombre in t]
    if not p_:
        return None
    return {"n_alt": n_alt, "n_pers": p_[-1], "otros": [n for n, t in alts if n != n_alt], "nombre": nombre}


def medir(ruta):
    bulto = pickle.load(open(ruta, "rb"))
    params = jax.tree_util.tree_map(jnp.asarray, bulto["params"])
    cfg = bulto["config"]
    preparar(cfg)
    print(f"\n== {ruta} · paso {bulto.get('paso')} · {conf_ckpt.descripcion(cfg)} · bloques {E._BLOQUES}")
    rng = np.random.default_rng(SEM)
    filas = {c: [] for c in CONDS}
    t0 = time.time()
    for _ in range(N):
        ses, cortes, turnos, mask, cons, pos, tgt, tipo, forma = DAT.lote(
            rng, B, nivel=cfg["nivel"], n_hechos=4, n_sesiones=4, p_vieja=cfg["p_vieja"],
            p_nose=cfg["p_nose"], formas_q=E.FORMAS_Q, con_formas=True,
            n_ses_extra=cfg.get("ses_extra", 0), p_compuesta=E._P_COMPUESTA, sesion_rel2=E._REL2_SESION,
            rel2_barajar=E._REL2_BARAJAR)
        # un ejemplo que no entra en alguna condicion se excluye de TODAS (mismo conjunto en cada fila)
        valido = np.ones(B, bool)
        for cond in CONDS:
            for b in range(B):
                if rearmar(ses[b], cortes[b], mask[b], turnos[b], cond) is None:
                    valido[b] = False
        for cond in CONDS:
            ses2 = ses.copy(); cortes2 = cortes.copy()
            mask2 = mask.copy(); turnos2 = turnos.copy(); mapas = []
            for b in range(B):
                if not valido[b]:
                    mapas.append(None); continue
                ses2[b], cortes2[b], mask2[b], turnos2[b], mp = rearmar(ses[b], cortes[b], mask[b], turnos[b], cond)
                mapas.append(mp)
            if cond == "original" and E._REL2_SESION is None:
                assert (ses2 == ses).all() and (cortes2 == cortes).all() and (mask2 == mask).all() and (turnos2 == turnos).all()
            lg, a, p = partes_p(params, jnp.array(ses2), jnp.array(cortes2), jnp.array(turnos2),
                                jnp.array(mask2), jnp.array(cons), jnp.array(pos))
            lg = np.asarray(lg); a = np.asarray(a); p = np.asarray(p)
            lg2 = lg.copy(); lg2[:, E.NOSE] = -np.inf
            pred = np.where(a > 0.0, E.NOSE, lg2.argmax(-1)) if E._ABST == "cabeza" else lg.argmax(-1)
            for b in range(B):
                if not valido[b]:
                    continue
                fila = {"tipo": int(tipo[b]), "ok": bool(pred[b] == tgt[b]), "nose": bool(pred[b] == E.NOSE)}
                if tipo[b] == 4:
                    q_toks = [I.ITOS[int(t)] for t in cons[b] if t != DAT.PAD][1:]
                    c = clasificar(enunciados_de(ses[b], cortes[b], mask[b]), q_toks, I.ITOS[int(tgt[b])])
                    if c is not None:
                        mp = mapas[b]
                        fila["p_alt"] = float(p[b, mp[c["n_alt"]]])
                        fila["p_pers"] = float(p[b, mp[c["n_pers"]]])
                        fila["p_otros"] = float(sum(p[b, mp[n]] for n in c["otros"]))
                filas[cond].append(fila)
    print(f"  {len(filas[CONDS[0]])} de {N*B} ejemplos (los que entran en todas las condiciones) x {len(CONDS)} condiciones en {time.time()-t0:.0f}s")
    res = {"ruta": ruta, "paso": bulto.get("paso"), "bloques": str(E._BLOQUES), "N": N, "B": B, "sem": SEM, "cond": {}}
    print(f"  {'condicion':12} {'compuesta':>9} {'NOSE':>6} {'p_alt':>6} {'p_pers':>6} {'p_otr':>6} | {'nose_comp':>9} {'vigente':>8} {'anterior':>8} {'nose':>6} {'falsa':>6}")
    for cond in CONDS:
        f = filas[cond]
        g = lambda tp: [x for x in f if x["tipo"] == tp]
        m = lambda xs, k: float(np.mean([x[k] for x in xs])) if xs else float("nan")
        comp = g(4); conp = [x for x in comp if "p_alt" in x]
        con_resp = [x for x in f if x["tipo"] in DAT.CON_RESPUESTA]
        r = {"n_comp": len(comp), "compuesta": m(comp, "ok"), "nose_en_comp": m(comp, "nose"),
             "p_alt": m(conp, "p_alt"), "p_pers": m(conp, "p_pers"), "p_otros": m(conp, "p_otros"),
             "n_clasif": len(conp), "nose_comp": m(g(5), "ok"), "vigente": m(g(0), "ok"),
             "anterior": m(g(1), "ok"), "nose": m([x for x in f if x["tipo"] in (2, 3)], "ok"),
             "falsa_abst": m(con_resp, "nose")}
        res["cond"][cond] = r
        print(f"  {cond:12} {r['compuesta']:>9.4f} {r['nose_en_comp']:>6.3f} {r['p_alt']:>6.3f} {r['p_pers']:>6.3f} {r['p_otros']:>6.3f} | "
              f"{r['nose_comp']:>9.4f} {r['vigente']:>8.4f} {r['anterior']:>8.4f} {r['nose']:>6.3f} {r['falsa_abst']:>6.3f}")
    print(f"  (compuestas n={len(filas['original'] and [x for x in filas['original'] if x['tipo']==4])}, clasificadas {res['cond']['original']['n_clasif']})")
    return res


if __name__ == "__main__":
    rutas = sys.argv[1:] or [os.path.join(AQUI, "ckpts", "kc3_s1.pkl")]
    out = [medir(r) for r in rutas]
    nombre = os.path.join(AQUI, "corridas_20260911", os.environ.get("SALIDA", "encadenados_intervencion.json"))
    json.dump(out, open(nombre, "w"), indent=1)
    print("\nguardado en", nombre)
