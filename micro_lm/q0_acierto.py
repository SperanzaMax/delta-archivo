"""Q-0 como lo pide el prereg: ACIERTO con la clave cuantizada, no coseno de reconstruccion.

La v1 de la Fase 0 reporto el coseno como proxy. No es lo que Q-0 dice, y Q-0 es BLOQUEANTE: si la
memoria no tolera el codigo discreto, Q-1 no se puede interpretar (seria preguntarle a un modelo roto
si detecta la ausencia, el mismo defecto que dejo NO EVALUABLE al exploratorio del 22-ago).
"""
import os, pickle, sys
import jax, jax.numpy as jnp, numpy as np
sys.path.insert(0, os.getcwd())
import datos as DAT, entrenar as E, modelo as M
from clave_discreta_fase0 import entrenar_codebooks, codificar, RNG_CB

def evaluar(ruta, n=2000, B=64, semilla=54321, m=8, ks=(4,16,64,256)):
    bulto = pickle.load(open(ruta,'rb')); params, cfg = bulto["params"], bulto["config"]
    E._DONDE = cfg.get("donde","pre"); E._ABST = cfg.get("abst","token")
    if "abst" not in params:
        d = params["ln_f"]["g"].shape[-1]
        params = dict(params); params["abst"] = {"w": jnp.zeros((d,1)), "b": jnp.zeros((1,))}
        E._ABST = "token"
    ap = params["arch"]

    @jax.jit
    def con_clave(p, ses, cortes, turnos, mask, cons, pos, ak_ext):
        """Igual que `responder`, pero la CLAVE viene de afuera (cuantizada o no)."""
        archivo = M.escribir(p, ses, cortes)
        a = p["arch"]
        av = archivo @ a["vw"]
        penal = jnp.where(mask, 0.0, -1e9)[:, None, :]
        def lectura(h):
            q = h @ a["qr"]
            sim = jnp.einsum("btd,bnd->btn", q, ak_ext) / jnp.sqrt(h.shape[-1]) + penal
            return jnp.einsum("btn,bnd->btd", jax.nn.softmax(sim,-1), av) @ a["wo"]
        h = M.tronco(p, cons, lectura, 0, E._DONDE)
        lg = M.ln(p["ln_f"], h) @ p["head"]["w"] + p["head"]["b"]
        return jnp.take_along_axis(lg, pos[:,None,None], axis=1)[:,0,:]

    @jax.jit
    def clave_cont(p, ses, cortes, turnos):
        return M.escribir(p, ses, cortes) @ p["arch"]["kw"] + p["arch"]["ord"][turnos]

    rng = np.random.default_rng(semilla)
    lotes = []
    vistos = 0
    while vistos < n:
        b = min(B, n-vistos)
        lotes.append(DAT.lote(rng, b, nivel=cfg["nivel"], n_hechos=4, n_sesiones=4,
                              p_nose=cfg.get("p_nose",0.4), con_meta=True))
        vistos += b

    # claves continuas de todos los lotes, para entrenar los codebooks
    AKs, MKs = [], []
    for (ses,cortes,turnos,mask,cons,pos,tgt,tipo,meta) in lotes:
        AKs.append(np.asarray(clave_cont(params, jnp.array(ses), jnp.array(cortes), jnp.array(turnos)), dtype=np.float64))
        MKs.append(np.asarray(mask))
    AK = np.concatenate(AKs); MK = np.concatenate(MKs)
    D = AK.shape[-1]
    planas = AK.reshape(-1,D)[MK.reshape(-1)]
    muestra = planas[RNG_CB.choice(len(planas), min(20000,len(planas)), replace=False)]

    def acierto(transformar=None):
        ok = tot = 0
        off = 0
        for (ses,cortes,turnos,mask,cons,pos,tgt,tipo,meta) in lotes:
            b = len(tgt)
            ak = AK[off:off+b]
            if transformar is not None:
                ak = transformar(ak)
            off += b
            lg = np.array(con_clave(params, jnp.array(ses), jnp.array(cortes), jnp.array(turnos),
                                      jnp.array(mask), jnp.array(cons), jnp.array(pos), jnp.array(ak)))
            lg[:, E.NOSE] = -1e9
            hay = np.asarray(tgt) != E.NOSE
            ok += int(((lg.argmax(-1) == np.asarray(tgt)) & hay).sum()); tot += int(hay.sum())
        return ok/max(tot,1)

    base = acierto(None)
    print(f"\n--- {os.path.basename(ruta)} · RECUP con clave continua = {base:.4f} ---")
    print(f"{'k':>5} {'RECUP cuantizada':>18} {'razon vs continua':>19}  Q-0 (>= 0,90)")
    print("-" * 64)
    for k in ks:
        libros = entrenar_codebooks(muestra, m, k)
        def tr(ak, libros=libros):
            f = ak.reshape(-1, D)
            _, rec = codificar(f, libros)
            return rec.reshape(ak.shape)
        r = acierto(tr)
        razon = r/base if base > 0 else float('nan')
        print(f"{k:>5} {r:>18.4f} {razon:>19.4f}  {'PASA' if razon >= 0.90 else 'NO PASA'}")

if __name__ == "__main__":
    for u in sys.argv[1:]:
        evaluar(f"ckpts/{u}.pkl")
