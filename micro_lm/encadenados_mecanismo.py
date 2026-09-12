"""ENCADENADOS · ¿por donde pasa la informacion cuando UN bloque encadena? · 2026-09-12

El prereg (`PREREG_ENCADENADOS.md` §2, H0) predecia que con lectura en un solo bloque la pregunta
compuesta («cual es la altura del director de barrio ?») NO se puede contestar: la query se forma
sobre el texto y «yamil» no esta en el texto. Anoche `kc3_s1` dio 0,93 y `kc3_s2` 0,80-0,88. El
prereg dice: «habria que buscar por donde pasa la informacion». Este instrumento mira la primera
alternativa, que esta en `modelo.escribir`: las entradas del archivo se escriben con el tronco
RECURRENTE corriendo sobre toda la sesion, sin resetear el estado entre enunciados, asi que la
entrada «la altura de yamil es 48» puede llevar adentro «director de barrio» si ese hecho vino
ANTES en la MISMA sesion. Si es asi, el encadenado se hace al escribir, no al leer.

Prediccion si el mecanismo es ese: la exactitud de `compuesta` parte en dos segun donde cayo el
hecho de persona VIGENTE respecto del hecho de altura del nombre preguntado:
    misma sesion y antes  -> alta
    otra sesion (o despues) -> cerca del azar entre los 3 candidatos (~0,33) o NOSE
Y la lectura en el «?» deberia caer sobre la entrada de altura del nombre correcto (no sobre la de
persona), porque esa entrada ya «sabe» de quien es.

Se mide sobre los checkpoints tal como se entrenaron (`conf_ckpt.aplicar` + los argumentos de la
config). No entrena nada. Uso:  python encadenados_mecanismo.py ckpts/kc3_s1.pkl [mas...]
Env: N (lotes, 8) B (64) SEM (2026)
"""
import os, sys, json, pickle, time
import numpy as np
import jax, jax.numpy as jnp

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
import conf_ckpt
import datos as DAT, idioma as I, modelo as M
import entrenar as E

N = int(os.environ.get("N", 8)); B = int(os.environ.get("B", 64)); SEM = int(os.environ.get("SEM", 2026))


def preparar(cfg):
    """Deja a `entrenar` y `modelo` como estaban en la corrida del checkpoint."""
    # OJO (12-sep, 09:10): las funciones jit leen `E._BLOQUES` y compania EN EL MOMENTO DE TRAZAR, y
    # la traza queda cacheada por forma de los argumentos. Sin esto, el segundo checkpoint de una
    # misma corrida se mide con la arquitectura del primero: mc3_s2 (un bloque) dio vigente 0,42
    # evaluado despues de md3_s2 (bloques 0,2), contra 0,95 solo. Es la regla de conf_ckpt, version jit.
    jax.clear_caches()
    I.fijar_version(cfg.get("idioma", 3))
    conf_ckpt.aplicar(cfg, verboso=True)
    E._DONDE = cfg.get("donde", "pre")
    E._ABST = cfg.get("abst", "token")
    E._PERT = bool(cfg.get("pert", False))
    E._SES_EXTRA_SIN_GRAD = bool(cfg.get("ses_extra_sin_grad", False))
    E._P_COMPUESTA = float(cfg.get("p_compuesta", 0.0))
    bl = tuple(int(x) for x in str(cfg.get("bloques_lectura", "0")).split(",") if x.strip())
    E._BLOQUES = bl[0] if len(bl) == 1 else bl
    E.FORMAS_Q = tuple(x.strip() for x in cfg.get("formas_q", "directa").split(","))
    E.NOSE = I.STOI["NOSE"]; DAT.PAD = I.STOI["."]
    r2 = cfg.get("rel2_sesion", -1)
    E._REL2_SESION = None if r2 is None or r2 < 0 else int(r2)   # E-3: el bloque de altura en otra sesion
    E._REL2_BARAJAR = bool(cfg.get("rel2_barajar", False))         # E-4     # por si la version del idioma movio el vocabulario


@jax.jit
def partes_p(params, ses, cortes, turnos, mask, cons, pos):
    cap = {}
    lg, a = E._partes(params, ses, cortes, turnos, mask, cons, pos, captura=cap)
    p = jnp.take_along_axis(cap["p"], pos[:, None, None], axis=1)[:, 0, :]   # (B, N) en el «?»
    return lg, a, p


def enunciados_de(ses_b, cortes_b, mask_b):
    """Lista de (n_entrada, s, e, tokens) por enunciado escrito, en orden de escritura."""
    out = []
    S = ses_b.shape[0]
    for s in range(S):
        ini = 1                                   # despues del BOS
        for e in range(DAT.E_MAX):
            n = s * DAT.E_MAX + e
            if not mask_b[n]:
                break
            fin = int(cortes_b[s, e])
            toks = [I.ITOS[int(t)] for t in ses_b[s, ini:fin + 1]]
            out.append((n, s, e, toks))
            ini = fin + 1
    return out


def es_rel2(t):
    """Hecho de altura/clave de una PERSONA: lleva un nombre Y un numero. Las formas del idioma no
    siempre dicen «altura» («julio mide 89»), y «altura»/«clave» tambien son relaciones de entidades
    («bosque tiene la altura en 52»), asi que la palabra no sirve para reconocerlo. Una correccion
    eliptica de persona («no , es carla») lleva nombre sin numero; una numerica («no , 68»), numero
    sin nombre."""
    return any(x in I.NOMBRES for x in t) and any(x in I.NUMEROS for x in t)


def clasificar(ses_b, cortes_b, mask_b, q_toks, r_tok):
    """Para una compuesta: donde esta el hecho de altura del nombre preguntado y donde el hecho de
    persona vigente que da ese nombre. Devuelve dict con la condicion o None si no se pudo."""
    ens = enunciados_de(ses_b, cortes_b, mask_b)
    # «cual es la altura del director de barrio ?» -> rel2 = altura, sust = director, ent = barrio
    try:
        rel2 = q_toks[3]; sust = q_toks[5]; ent = q_toks[7]
    except IndexError:
        return None
    rel2_sust = {I.RELACIONES[r][0]: r for r in I.RELS_PERSONA}
    rel2 = rel2_sust.get(rel2, rel2)
    sust2 = I.RELACIONES[rel2][0]
    # el hecho de altura del nombre correcto: contiene sust2, un nombre y el numero respuesta
    nombres = set(I.NOMBRES)
    cand = [(n, s, e, t) for n, s, e, t in ens if es_rel2(t) and r_tok in t]
    if len(cand) != 1:
        return None
    n_alt, s_alt, e_alt, t_alt = cand[0]
    nombre = [x for x in t_alt if x in nombres][0]
    # enunciados que nombran a la entidad (el hecho de persona y sus correcciones que la nombran)
    # y contienen el nombre: el que hace vigente a `nombre`. Una correccion eliptica no nombra la
    # entidad; alcanza con que contenga el nombre y NO sea un hecho de altura/clave.
    pers = [(n, s, e, t) for n, s, e, t in ens if nombre in t and not es_rel2(t)]
    if not pers:
        return None
    n_p, s_p, e_p, t_p = pers[-1]                # el ultimo que lo nombra es el que lo deja vigente
    misma = s_p == s_alt
    antes = misma and e_p < e_alt
    # entradas de altura de los OTROS candidatos (misma relacion, otro nombre)
    otros = [n for n, s, e, t in ens if es_rel2(t) and n != n_alt]
    return {"n_alt": n_alt, "n_pers": n_p, "otros": otros, "misma": misma, "antes": antes,
            "nombre": nombre, "ent": ent, "s_alt": s_alt, "s_pers": s_p}


def medir(ruta):
    bulto = pickle.load(open(ruta, "rb"))
    params = jax.tree_util.tree_map(jnp.asarray, bulto["params"])
    cfg = bulto["config"]
    preparar(cfg)
    print(f"\n== {ruta} · paso {bulto.get('paso')} · {conf_ckpt.descripcion(cfg)} · bloques {E._BLOQUES}")
    rng = np.random.default_rng(SEM)
    filas = []
    t0 = time.time()
    for _ in range(N):
        ses, cortes, turnos, mask, cons, pos, tgt, tipo, forma = DAT.lote(
            rng, B, nivel=cfg["nivel"], n_hechos=4, n_sesiones=4, p_vieja=cfg["p_vieja"],
            p_nose=cfg["p_nose"], formas_q=E.FORMAS_Q, con_formas=True,
            n_ses_extra=cfg.get("ses_extra", 0), p_compuesta=E._P_COMPUESTA, sesion_rel2=E._REL2_SESION,
            rel2_barajar=E._REL2_BARAJAR)
        lg, a, p = partes_p(params, jnp.array(ses), jnp.array(cortes), jnp.array(turnos),
                            jnp.array(mask), jnp.array(cons), jnp.array(pos))
        lg = np.asarray(lg); a = np.asarray(a); p = np.asarray(p)
        lg2 = lg.copy(); lg2[:, E.NOSE] = -np.inf
        pred = np.where(a > 0.0, E.NOSE, lg2.argmax(-1)) if E._ABST == "cabeza" else lg.argmax(-1)
        for b in range(B):
            if tipo[b] not in (4, 5):
                continue
            q_toks = [I.ITOS[int(t)] for t in cons[b] if t != DAT.PAD][1:]
            r_tok = I.ITOS[int(tgt[b])]
            fila = {"tipo": int(tipo[b]), "ok": bool(pred[b] == tgt[b]), "nose": bool(pred[b] == E.NOSE)}
            if tipo[b] == 4:
                c = clasificar(ses[b], cortes[b], mask[b], q_toks, r_tok)
                if c is None:
                    fila["cond"] = "?"
                else:
                    fila["cond"] = "misma_antes" if c["antes"] else ("misma_despues" if c["misma"] else "otra")
                    fila["p_alt"] = float(p[b, c["n_alt"]])
                    fila["p_pers"] = float(p[b, c["n_pers"]])
                    fila["p_otros"] = float(sum(p[b, n] for n in c["otros"]))
                    fila["pred"] = I.ITOS[int(pred[b])]
                    fila["nombre"] = c["nombre"]
            filas.append(fila)
    print(f"  {len(filas)} preguntas encadenadas en {time.time()-t0:.0f}s")
    res = {"ruta": ruta, "paso": bulto.get("paso"), "bloques": str(E._BLOQUES), "N": N, "B": B, "sem": SEM}
    comp = [f for f in filas if f["tipo"] == 4]
    nc = [f for f in filas if f["tipo"] == 5]
    res["compuesta_total"] = {"n": len(comp), "ok": float(np.mean([f["ok"] for f in comp])) if comp else None}
    res["nose_comp"] = {"n": len(nc), "ok": float(np.mean([f["ok"] for f in nc])) if nc else None}
    fmt = lambda x: "  nan " if x is None else f"{x:.4f}"
    print(f"  compuesta {fmt(res['compuesta_total']['ok'])} (n={len(comp)}) · nose_comp {fmt(res['nose_comp']['ok'])} (n={len(nc)})")
    print(f"  {'condicion':14} {'n':>4} {'acierto':>8} {'NOSE':>6} {'p_alt':>7} {'p_pers':>7} {'p_otros':>8}")
    res["por_cond"] = {}
    for cond in ("misma_antes", "misma_despues", "otra", "?"):
        g = [f for f in comp if f.get("cond") == cond]
        if not g:
            continue
        r = {"n": len(g), "acierto": float(np.mean([f["ok"] for f in g])),
             "nose": float(np.mean([f["nose"] for f in g]))}
        if cond != "?":
            for k in ("p_alt", "p_pers", "p_otros"):
                r[k] = float(np.mean([f[k] for f in g]))
        res["por_cond"][cond] = r
        print(f"  {cond:14} {r['n']:>4} {r['acierto']:>8.4f} {r['nose']:>6.3f} "
              + (f"{r['p_alt']:>7.3f} {r['p_pers']:>7.3f} {r['p_otros']:>8.3f}" if cond != "?" else ""))
    return res


if __name__ == "__main__":
    rutas = sys.argv[1:] or [os.path.join(AQUI, "ckpts", "kc3_s1.pkl")]
    out = [medir(r) for r in rutas]
    nombre = os.path.join(AQUI, "corridas_" + time.strftime("%Y%m%d"), os.environ.get("SALIDA", "encadenados_mecanismo.json"))
    json.dump(out, open(nombre, "w"), indent=1)
    print("\nguardado en", nombre)
