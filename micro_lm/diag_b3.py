#!/usr/bin/env python3
"""Diagnostico POST-HOC de la condicion `blanco error` — declarado como tal, sin estatus de prereg.

`vigente` da 0,0000 porque el modelo se abstiene de todo, y esa metrica mira la prediccion FINAL,
que es NOSE cuando la cabeza dispara. O sea que esta CONDICIONADA A LA DECISION DE LA CABEZA — el
mismo defecto D-D3 que aparecio hoy en el blanco de D-1, por tercera vez.

Aca se mide lo que esa metrica tapa: el ARGMAX de valor, sin mirar la cabeza. Si el argmax esta
aprendiendo, el modelo NO colapso: la cabeza esta tapando un tronco que si progresa, y el blanco
—que sale del argmax y no de la cabeza— deberia bajar solo y aflojar la abstencion.
Si el argmax tampoco aprende, entonces si es colapso y E-4 decide.
"""
import pickle, sys, numpy as np, jax, jax.numpy as jnp
sys.path.insert(0, "/home/maxi/Documentos/Nuevo Transformer/delta-archivo/micro_lm")
import idioma as I, datos as DAT, modelo as M
NOSE = I.STOI["NOSE"]
ck = sys.argv[1]
b = pickle.load(open(ck, "rb")); params = jax.tree_util.tree_map(jnp.asarray, b["params"])
cfg = b["config"]; donde = cfg.get("donde", "pre")
def fn(params, arch, tur, cons, mask):
    a = params["arch"]
    ak = arch @ a["kw"] + a["ord"][tur]; av = arch @ a["vw"]
    pen = jnp.where(mask, 0.0, -1e9)[:, None, :]
    def lec(h):
        q = h @ a["qr"]
        sim = jnp.einsum("btd,bnd->btn", q, ak) / jnp.sqrt(h.shape[-1]) + pen
        return jnp.einsum("btn,bnd->btd", jax.nn.softmax(sim, -1), av) @ a["wo"]
    h = M.tronco(params, cons, lec, 0, donde); hn = M.ln(params["ln_f"], h)
    return (hn @ params["head"]["w"] + params["head"]["b"],
            (hn @ params["abst"]["w"] + params["abst"]["b"])[..., 0])
jf = jax.jit(fn)
rng = np.random.default_rng(31337); ok = []; cab = []; blanco = []
for _ in range(24):
    ses, cor, tur, mask, cons, pos, tgt, tipo = DAT.lote(
        rng, 64, nivel=cfg["nivel"], n_hechos=4, n_sesiones=4, p_nose=0.4)
    lg, ab = jf(params, M.escribir(params, jnp.array(ses), jnp.array(cor)),
                jnp.array(tur), jnp.array(cons), jnp.array(mask))
    lg, ab = np.asarray(lg), np.asarray(ab)
    for i in range(64):
        pq = int(pos[i]); v = lg[i, pq].copy(); v[NOSE] = -np.inf
        arg = int(v.argmax()); ok.append(arg == int(tgt[i]))
        cab.append(float(ab[i, pq])); blanco.append(arg != int(tgt[i]))
ok = np.array(ok); cab = np.array(cab); blanco = np.array(blanco)
print(f"checkpoint {ck}  ·  paso {cfg.get('pasos')}  ·  blanco={cfg.get('blanco')}  n={len(ok)}")
print(f"  ACIERTO DEL ARGMAX (sin mirar la cabeza) : {ok.mean():.4f}   <- lo que `vigente` tapa")
print(f"  tasa del blanco 'me equivoco'            : {blanco.mean():.4f}")
print(f"  logit de la cabeza: media {cab.mean():+.4f}  sd {cab.std():.4f}  "
      f"min {cab.min():+.3f}  max {cab.max():+.3f}")
print(f"  fraccion con logit > 0 (se abstiene)     : {(cab>0).mean():.4f}")
p = 1/(1+np.exp(-cab.mean()))
print(f"  sigma(media del logit) = {p:.4f}   vs tasa base del blanco {blanco.mean():.4f}"
      f"   -> {'PEGADO AL PRIOR (colapso E-4)' if abs(p-blanco.mean())<0.06 and cab.std()<0.5 else 'NO esta pegado al prior'}")
