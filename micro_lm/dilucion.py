"""LA CURVA DE DILUCION · cuantos hechos aguanta el archivo · 2026-09-05

`PREREG_DILUCION.md`, SHA f4d91c12. Sobre checkpoints ENTRENADOS, sin entrenar nada.

El banco nunca probo un archivo grande: un episodio archiva a lo sumo 40 entradas. El objetivo pide
un archivo que crece conversacion tras conversacion. Aca se le agregan `X` entradas de OTROS
episodios reales y se mira cuando se rompe.

    python dilucion.py [ckpt ...]
"""
import os, sys, json, pickle, time
import numpy as np
import jax, jax.numpy as jnp

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
import datos as DAT, idioma as I, modelo as M

XS = [int(x) for x in os.environ.get("XS", "0,40,120,360,1080,3240").split(",")]
NMUE = int(os.environ.get("NMUE", "512"))     # muestras por celda
B = int(os.environ.get("B", "64"))
POOL = int(os.environ.get("POOL", "4096"))    # entradas distractoras del pool
CKPTS = sys.argv[1:] or ["ckpts/kq3_s0.pkl", "ckpts/v3_s0.pkl"]
PISO = 0.4065


DIST = os.environ.get("DIST", "real")      # real | ruido | disjunto
_ENT_TODAS = tuple(I.ENTIDADES)
# CONTROL 3 (2026-09-05). Los distractores de `real` llevan turnos sorteados en el MISMO rango que
# el episodio, o sea que el sello de orden no puede separarlos: parecen contemporaneos. Es el caso
# peor, y esta declarado asi en el prereg. Pero es tambien el caso que el SELLO DE ORDEN existe para
# resolver, y no se le habia dado la chance. Con TURNOS=viejo los distractores quedan en los turnos
# 0..K-1 y el episodio se corre a K..63, o sea el archivo largo es literalmente «lo dicho antes».
# Si el sello sirve, el modelo tiene que poder descartarlos. Y el margen es 64 turnos y se acaba ahi.
TURNOS = os.environ.get("TURNOS", "solapado")   # solapado | viejo


def construir_pool(params, nivel, n_obj, semilla=777):
    """Entradas de archivo de OTROS episodios: literalmente lo dicho en otras conversaciones.

    CONTROL (2026-09-05, agregado despues de ver la curva y ANTES de informarla). Los distractores
    `real` son hechos del mismo idioma, y con 30 entidades x 24 relaciones = 720 combinaciones, un
    pool grande contiene por fuerza entradas con la MISMA (entidad, relacion) que la preguntada. Eso
    seria COLISION —respuestas contradictorias legitimas— y no DILUCION, que son cosas distintas y
    tienen arreglos distintos.

      real      hechos de otros episodios. Puede colisionar.
      ruido     gaussianos con la misma media y desvio por dimension que el archivo real. NO puede
                colisionar con nada: si la caida sobrevive aca, es dilucion por numero de
                competidores y no por contenido.
    """
    rng = np.random.default_rng(semilla)
    if DIST == "disjunto":
        # CONTROL 2 (2026-09-05). Distractores REALES —mismo idioma, misma distribucion, mismo
        # generador— pero sorteados sobre una mitad de las entidades DISJUNTA de la que usa el
        # episodio de prueba. Ninguna entrada del pool puede hablar de la entidad preguntada, asi
        # que la colision exacta (ent, rel) es IMPOSIBLE por construccion y lo unico que queda es
        # la competencia entre vectores con contenido parecido.
        I.ENTIDADES = list(_ENT_TODAS[len(_ENT_TODAS) // 2:])
    trozos, turnos = [], []
    while sum(t.shape[0] for t in trozos) < n_obj:
        ses, cortes, tur, msk, *_ = DAT.lote(rng, 64, nivel=nivel, n_hechos=4, n_sesiones=4)
        arch = np.asarray(M.escribir(params, jnp.array(ses), jnp.array(cortes)))
        m = np.asarray(msk)
        trozos.append(arch[m])            # solo las entradas vivas
        turnos.append(np.asarray(tur)[m])
    if DIST == "disjunto":
        I.ENTIDADES = list(_ENT_TODAS[:len(_ENT_TODAS) // 2])   # el episodio de prueba usa la otra mitad
    pool = np.concatenate(trozos)[:n_obj]
    if DIST == "ruido":
        # mismo primer y segundo momento POR DIMENSION que el archivo real, para que la unica
        # diferencia sea que no hay contenido que pueda coincidir con la pregunta
        pool = rng.normal(pool.mean(0), pool.std(0) + 1e-8, size=pool.shape).astype(np.float32)
    return pool, np.concatenate(turnos)[:n_obj]


def celda(params, nivel, pool_a, pool_t, X, n, semilla=31415):
    rng = np.random.default_rng(semilla)
    ar = params["arch"]
    oks, rank0, conmio, masas, ents = [], 0, 0, [], []
    visto = 0
    while visto < n:
        b = min(B, n - visto)
        ses, cortes, tur, msk, cons, pos, tgt, tipo, meta, oarch, hq = DAT.lote(
            rng, b, nivel=nivel, n_hechos=4, n_sesiones=4, p_nose=0.0,
            con_meta=True, con_origen=True)
        arch = M.escribir(params, jnp.array(ses), jnp.array(cortes))
        A = np.asarray(arch); Tu = np.asarray(tur); Mk = np.asarray(msk)
        Oa = np.asarray(oarch)
        if X > 0:
            idx = rng.integers(0, pool_a.shape[0], (b, X))
            A = np.concatenate([A, pool_a[idx]], axis=1)
            # turnos de los distractores: mismo rango que el episodio -> igualdad de sello
            if TURNOS == "viejo":
                K = 64 - (int(Tu.max()) + 1)          # los distractores viven en 0..K-1
                Tu = np.concatenate([Tu + K, rng.integers(0, max(1, K), (b, X))], axis=1)
            else:
                hi = max(1, int(Tu.max()) + 1)
                Tu = np.concatenate([Tu, rng.integers(0, hi, (b, X))], axis=1)
            Mk = np.concatenate([Mk, np.ones((b, X), bool)], axis=1)
            Oa = np.concatenate([Oa, np.full((b, X), -1, np.int32)], axis=1)
        jA, jT, jM = jnp.array(A), jnp.array(Tu), jnp.array(Mk)
        jc = jnp.array(cons)
        lg = M.responder(params, jA, jT, jc, jM, donde=DONDE)
        pred = np.asarray(jnp.take_along_axis(lg, jnp.array(pos)[:, None, None], 1)[:, 0, :].argmax(-1))
        oks.append(pred == tgt)
        # ranking de la lectura en la posicion de maximo foco, igual que rank_hecho.py
        ak = jA @ ar["kw"] + ar["ord"][jT]
        h = params["emb"][jc]
        q = M.convk(params["blocks"][0]["convq"], M.ln(params["blocks"][0]["ln1"], h)) @ ar["qr"] \
            if DONDE == "lat2" else M.ln(params["blocks"][0]["ln1"], h) @ ar["qr"]
        sim = np.asarray(jnp.einsum("btd,bnd->btn", q, ak) / jnp.sqrt(h.shape[-1])
                         + jnp.where(jM, 0.0, -1e9)[:, None, :])
        p = np.asarray(jax.nn.softmax(jnp.array(sim), -1))
        ent = -(p * np.log(p + 1e-12)).sum(-1)
        for i in range(b):
            if hq[i] < 0:
                continue
            mios = np.where(Oa[i] == hq[i])[0]
            if len(mios) == 0:
                continue
            conmio += 1
            f = int(ent[i, :pos[i] + 1].argmin())
            orden = np.argsort(-sim[i, f])
            if int(min(np.where(np.isin(orden, mios))[0])) == 0:
                rank0 += 1
            masas.append(float(p[i, f].max())); ents.append(float(ent[i, f]))
        visto += b
    ok = np.concatenate(oks)
    return {"X": X, "n": int(ok.size), "exactitud": float(ok.mean()),
            "RECUP": rank0 / max(1, conmio), "n_con_hecho": conmio,
            "masa_ganadora": float(np.mean(masas)), "entropia": float(np.mean(ents))}


if __name__ == "__main__":
    res = {}
    for ruta in CKPTS:
        bulto = pickle.load(open(ruta, "rb"))
        params = jax.tree_util.tree_map(jnp.asarray, bulto["params"])
        cfg = bulto["config"]; nivel = cfg["nivel"]
        DONDE = cfg.get("donde", "pre")
        kq = cfg.get("kernel_q", 3) or 3
        M.KQ = kq
        I.fijar_version(cfg.get("idioma", 3))
        print(f"\n{'='*78}\n[distractor={DIST} turnos={TURNOS}] {ruta}  ·  nivel {nivel} · donde={DONDE} · kernel_q={kq} · V={I.V}")
        t0 = time.time()
        pool_a, pool_t = construir_pool(params, nivel, POOL)
        print(f"  pool de distractores: {pool_a.shape} en {time.time()-t0:.1f}s")
        print(f"\n  {'X':>6} {'archivo':>8} {'exactitud':>10} {'RECUP':>8} {'masa gan':>9} {'entropia':>9}")
        res[ruta] = {"kernel_q": kq, "donde": DONDE, "filas": []}
        for X in XS:
            t1 = time.time()
            r = celda(params, nivel, pool_a, pool_t, X, NMUE)
            r["segundos"] = round(time.time() - t1, 1)
            res[ruta]["filas"].append(r)
            marca = "  <- bajo el piso trivial" if r["exactitud"] < PISO else ""
            print(f"  {X:>6} {X+40:>8} {r['exactitud']:>10.4f} {r['RECUP']:>8.4f} "
                  f"{r['masa_ganadora']:>9.4f} {r['entropia']:>9.4f}{marca}")
    json.dump({"prereg": "f4d91c12", "distractor": DIST, "turnos": TURNOS, "piso": PISO, "NMUE": NMUE, "POOL": POOL, "res": res},
              open(os.path.join(AQUI, f"dilucion_{DIST}_{TURNOS}.json"), "w"), indent=1)
    print("\nguardado en dilucion.json")
