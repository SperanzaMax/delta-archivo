"""CONTROL DE ACCESO GLOBAL · ¿el corte exacto es la ventana, o es otra cosa? · 2026-09-04

La ley de la ventana dice que la query que lee el archivo solo ve el alcance de la conv que la forma,
y que lo que cae afuera da sensibilidad 0,000000 EXACTO. De ahi se venia afirmando, POR ARGUMENTO y
sin medirlo, que «una capa de atencion completa no tendria esta limitacion». Esto lo mide.

    lat2  ->  la query se forma con convk(convq, .), alcance = kernel - 1
    attn  ->  la query se forma con atencion causal completa, alcance = toda la secuencia anterior

Se corre sobre CHECKPOINTS ENTRENADOS y no sobre pesos aleatorios, y esa correccion importa. La
primera version de este control uso `init_params` y dio cero exacto tambien en d=1 y d=2, donde tenia
que moverse. La causa es que `convq` arranca en [1, 0, ..., 0], asi que SIN ENTRENAR la conv es la
identidad y `lat2` es literalmente `pre`, con ventana efectiva 0. La ventana existe solo si el
entrenamiento abrio los taps, y en `v3_s0` los abrio (max|peso| por tap 0.718, 0.223, 0.469).

    python control_attn.py [ckpt ...]
"""
import os, sys, json, pickle
import numpy as np
import jax, jax.numpy as jnp

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
import modelo as M

T = 24                    # largo de la consulta
N_ARCH = 12               # entradas del archivo
N = int(os.environ.get("N", "120"))
DIST = [1, 2, 3, 4, 5, 6]
CKPTS = sys.argv[1:] or ["ckpts/v3_s0.pkl", "ckpts/kq3_s0.pkl"]


def distribucion(params, arch, tur, msk, cons, pos, donde):
    """Distribucion de lectura del archivo en `pos`. Es exactamente lo que la busqueda mira."""
    a_p = params["arch"]
    ak = arch @ a_p["kw"] + a_p["ord"][tur]
    av = arch @ a_p["vw"]
    penal = jnp.where(msk, 0.0, -1e9)[:, None, :]
    caja = {}

    def lectura(h):
        q = h @ a_p["qr"]
        sim = jnp.einsum("btd,bnd->btn", q, ak) / jnp.sqrt(h.shape[-1]) + penal
        caja["p"] = jax.nn.softmax(sim, -1)
        return jnp.einsum("btn,bnd->btd", caja["p"], av) @ a_p["wo"]

    M.tronco(params, cons, lectura, 0, donde)
    return np.asarray(caja["p"][:, pos, :])


def medir(params, V, D, donde, semilla=0):
    rng = np.random.default_rng(1000 + semilla)
    cons = jnp.array(rng.integers(0, V, (N, T)))
    arch = jnp.array(rng.normal(size=(N, N_ARCH, D)), dtype=jnp.float32)
    tur  = jnp.array(rng.integers(0, 3, (N, N_ARCH)))
    msk  = jnp.ones((N, N_ARCH), bool)
    pos  = T - 1
    base = distribucion(params, arch, tur, msk, cons, pos, donde)
    out = {}
    for d in DIST:
        alt = np.asarray(cons).copy()
        col = pos - d
        alt[:, col] = (alt[:, col] + 1 + rng.integers(0, V - 2, N)) % V
        pd = distribucion(params, arch, tur, msk, jnp.array(alt), pos, donde)
        tv = 0.5 * np.abs(base - pd).sum(-1)
        out[d] = (float(tv.mean()), float(tv.max()), int((tv > 0).sum()))
    return out


if __name__ == "__main__":
    res = {}
    for ruta in CKPTS:
        b = pickle.load(open(ruta, "rb"))
        p, cfg = b["params"], b["config"]
        V, D = p["emb"].shape[0], p["emb"].shape[1]
        kq = cfg.get("kernel_q", 3) or 3
        taps = [round(float(abs(np.asarray(p["blocks"][0]["convq"])[i]).max()), 4) for i in range(kq)]
        print(f"\n{'='*74}\n{ruta}   kernel_q={kq} (alcance {kq-1})   donde nativo={cfg.get('donde')}"
              f"\n  convq entrenada, max|peso| por tap: {taps}")
        res[ruta] = {"kernel_q": kq, "taps": taps}
        for donde in ("lat2", "attn"):
            print(f"\n  --- donde = {donde}")
            print(f"    {'dist':>5} {'TV media':>12} {'TV max':>12} {'con TV>0':>10}   veredicto")
            res[ruta][donde] = {}
            m = medir(p, V, D, donde)
            for d in DIST:
                media, mx, nz = m[d]
                cero = (mx == 0.0)
                esperado = "" if donde == "attn" else ("  <- afuera" if d > kq - 1 else "  <- adentro")
                print(f"    {d:>5} {media:12.6f} {mx:12.6f} {nz:>7}/{N}   "
                      f"{'CERO EXACTO' if cero else 'se mueve':<12}{esperado}")
                res[ruta][donde][d] = {"tv_media": media, "tv_max": mx, "n_no_cero": nz,
                                       "n": N, "cero_exacto": cero}
    json.dump({"N": N, "entrenado": True, "res": res},
              open(os.path.join(AQUI, "control_attn.json"), "w"), indent=1)
    print("\nguardado en control_attn.json")
