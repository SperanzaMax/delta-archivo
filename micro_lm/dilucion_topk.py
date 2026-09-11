"""TOP-K EN LA LECTURA DEL ARCHIVO · intervencion en INFERENCIA, sin entrenar nada · 2026-09-10

Hipotesis (declarada ANTES de correr, ver la conversacion del 10-sep):

  `INFORME_DILUCION_20260905.md` separo dos fenomenos. Con distractores de RUIDO el RANKING aguanta
  (RECUP 0,7852 con 3280 entradas) pero la EXACTITUD se cae a 0,2441, y la masa de la ganadora baja
  de 0,9855 a 0,6715. O sea el 33 % de la masa se va a otras entradas y ENSUCIA EL VALOR LEIDO: la
  lectura es un promedio ponderado sobre TODAS las entradas.

  H  · si se lee con top-k en vez del softmax completo, la cola no entra en el promedio y la
       exactitud tiene que subir HACIA EL RECUP de su celda, que es su techo.
  C1 · CONTROL NEGATIVO. Con distractores `real` el ranking ya esta roto (RECUP 0,0117): el top-k
       elige las entradas equivocadas y NO puede mejorar nada.
  C2 · CONTROL DE IMPLEMENTACION. Con K >= N el top-k es identico al softmax completo y tiene que
       reproducir el numero original.
  C3 · CONTROL DE ARCHIVO CHICO. Con X=0 la masa ya esta en 0,9855: el top-k no deberia mover nada.

Lo unico que cambia entre brazos es `M.responder`. RECUP, masa y entropia los calcula `dilucion.py`
por su cuenta SIEMPRE con softmax completo, asi que quedan como control interno: si se mueven,
hay un bug.
"""
import os, sys, json, pickle, time
import numpy as np
import jax, jax.numpy as jnp

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
import conf_ckpt
import modelo as M
import idioma as I
import dilucion as D

CKPT = os.environ.get("CKPT", "ckpts/kq3_s0.pkl")
TOPK = [1]          # lo pisa el bucle

_orig = M.responder

def responder_topk(params, archivo, turnos, consulta, mask_arch, bloque=0, donde="pre",
                   pertenece=None):
    """Copia exacta de `M.responder` salvo que la lectura toma top-k y renormaliza entre esas K."""
    K = TOPK[0]
    N = archivo.shape[1]
    if K >= N:
        return _orig(params, archivo, turnos, consulta, mask_arch, bloque, donde, pertenece)
    a = params["arch"]
    ak = archivo @ a["kw"] + M.sello(a, turnos, mask_arch) + M.marca_pert(a, pertenece)
    av = archivo @ a["vw"]
    penal = jnp.where(mask_arch, 0.0, -1e9)[:, None, :]

    def lectura(h):
        q = h @ a["qr"]
        sim = jnp.einsum("btd,bnd->btn", q, ak) / jnp.sqrt(h.shape[-1]) + penal
        val, idx = jax.lax.top_k(sim, K)                       # (B, T, K)
        p = jax.nn.softmax(val, -1)                            # renormaliza SOLO entre las K
        avk = jax.vmap(lambda ab, ib: ab[ib])(av, idx)         # (B, T, K, D)
        return jnp.einsum("btk,btkd->btd", p, avk) @ a["wo"]

    h = M.tronco(params, consulta, lectura, bloque, donde)
    return M.ln(params["ln_f"], h) @ params["head"]["w"] + params["head"]["b"]


if __name__ == "__main__":
    bulto = pickle.load(open(os.path.join(AQUI, CKPT), "rb"))
    params = jax.tree_util.tree_map(jnp.asarray, bulto["params"])
    cfg = bulto["config"]
    conf_ckpt.aplicar(cfg)
    I.fijar_version(cfg.get("idioma", 3))
    D.DONDE = cfg.get("donde", "pre")
    nivel = cfg["nivel"]
    print(f"{CKPT} · {conf_ckpt.descripcion(cfg)} · V={I.V} · donde={D.DONDE}", flush=True)

    #        distractor,  X,     lista de K   ("N" = todas, o sea el softmax de hoy)
    PLAN = [("ruido",  3240, [1, 2, 4, 8, 16, 32, "N"]),
            ("ruido",     0, [1, "N"]),                  # C3
            ("real",   3240, [1, 8, "N"])]               # C1
    if os.environ.get("PLAN"):
        PLAN = [tuple(x) for x in json.loads(os.environ["PLAN"])]
    SAL = os.environ.get("SAL", "dilucion_topk_20260910.json")

    salida = []
    for dist, X, Ks in PLAN:
        D.DIST = dist
        t0 = time.time()
        pool_a, pool_t = D.construir_pool(params, nivel, D.POOL)
        print(f"\n[{dist} · X={X} · archivo={X+40}] pool {pool_a.shape} en {time.time()-t0:.0f}s",
              flush=True)
        print(f"  {'K':>6} {'exactitud':>10} {'RECUP':>8} {'masa gan':>9} {'seg':>5}", flush=True)
        for K in Ks:
            TOPK[0] = 10**9 if K == "N" else K
            M.responder = _orig if K == "N" else responder_topk
            t1 = time.time()
            r = D.celda(params, nivel, pool_a, pool_t, X, D.NMUE)
            r.update({"K": K, "dist": dist, "segundos": round(time.time() - t1, 1)})
            salida.append(r)
            print(f"  {str(K):>6} {r['exactitud']:>10.4f} {r['RECUP']:>8.4f} "
                  f"{r['masa_ganadora']:>9.4f} {r['segundos']:>5.0f}", flush=True)
        M.responder = _orig
    json.dump({"ckpt": CKPT, "NMUE": D.NMUE, "POOL": D.POOL, "filas": salida},
              open(os.path.join(AQUI, SAL), "w"), indent=1)
    print(f"\nguardado en {SAL}", flush=True)
