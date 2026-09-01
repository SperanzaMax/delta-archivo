"""Preguntar DOS VECES y cruzar coincidencia contra acierto · pedido de Maxi, 31-ago 22:20

    «hace la misma pregunta dos veces y controla en que cantidad acierta, que cantidad de las que
     acierta son la misma respuesta, y las que no son iguales si son realmente las que estan mal»

Es la pregunta PRACTICA que faltaba. El AUC dice si el desacuerdo ordena; esto dice si SIRVE: de las
preguntas donde las dos respuestas difieren, cuantas estan efectivamente mal. Eso es PRECISION, y un
detector puede tener AUC mediocre y precision alta —o al reves—.

El modelo es determinista, asi que «dos veces» se consigue perturbando la query con ruido distinto en
cada pasada. Las DOS pasadas llevan ruido (ninguna es la respuesta limpia), que es lo que hace la
comparacion simetrica.
"""
import os, sys
import numpy as np, jax, jax.numpy as jnp
AQUI = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, AQUI)
import datos as DAT, entrenar as E, idioma as I, medir_ratio_ce as R, modelo as M

N, LOTE = 512, 64
SIGMAS = (0.4,)

def correr(ruta):
    params, cfg, paso = R.cargar(ruta)
    params = jax.tree_util.tree_map(jnp.asarray, params)
    I.fijar_version(cfg.get("idioma", 2)); a_p = params["arch"]; donde = cfg.get("donde", "pre")

    @jax.jit
    def responder(params, ses, cortes, turnos, mask, cons, pos, ruido):
        archivo = M.escribir(params, ses, cortes)
        ak = archivo @ a_p["kw"] + a_p["ord"][turnos]; av = archivo @ a_p["vw"]
        penal = jnp.where(mask, 0.0, -1e9)[:, None, :]
        def lectura(h):
            q = h @ a_p["qr"] + ruido
            sim = jnp.einsum("btd,bnd->btn", q, ak)/jnp.sqrt(h.shape[-1]) + penal
            return jnp.einsum("btn,bnd->btd", jax.nn.softmax(sim, -1), av) @ a_p["wo"]
        h = M.tronco(params, cons, lectura, 0, donde)
        lg = M.ln(params["ln_f"], h) @ params["head"]["w"] + params["head"]["b"]
        lg = jnp.take_along_axis(lg, pos[:, None, None], axis=1)[:, 0, :]
        return lg.at[:, E.NOSE].set(-1e9).argmax(-1)

    rng = np.random.default_rng(4242); rk = np.random.default_rng(11)
    R1, R2, TGT = {s: [] for s in SIGMAS}, {s: [] for s in SIGMAS}, []
    vistos = 0
    while vistos < N:
        b = min(LOTE, N - vistos)
        ses, cortes, turnos, mask, cons, pos, tgt, tipo = DAT.lote(
            rng, b, nivel=cfg["nivel"], n_hechos=4, n_sesiones=4, p_nose=0.4)
        aj = [jnp.array(x) for x in (ses, cortes, turnos, mask, cons, pos)]
        D = a_p["qr"].shape[-1]
        nq = float(jnp.linalg.norm(M.ln(params["blocks"][0]["ln1"], params["emb"][aj[4]]) @ a_p["qr"],
                                   axis=-1).mean())
        for s in SIGMAS:
            e = s * nq / np.sqrt(D)
            R1[s].append(np.asarray(responder(params, *aj, jnp.array(rk.normal(size=(b,1,D))*e))))
            R2[s].append(np.asarray(responder(params, *aj, jnp.array(rk.normal(size=(b,1,D))*e))))
        TGT.append(np.asarray(tgt)); vistos += b

    tgt = np.concatenate(TGT); hay = tgt != E.NOSE
    print(f"\n{'='*88}\n{os.path.basename(ruta)}  paso={paso}  n={len(tgt)}  "
          f"·  con respuesta {hay.mean():.4f} · sin respuesta {(~hay).mean():.4f}\n{'='*88}")
    for s in SIGMAS:
        r1, r2 = np.concatenate(R1[s]), np.concatenate(R2[s])
        igual = (r1 == r2)
        # «acierta» solo tiene sentido donde HAY respuesta: si no la hay, ningun valor es correcto.
        ok = (r1 == tgt) & hay
        print(f"\n--- las dos preguntas con ruido sigma={s} ---")
        print(f"  acierta (sobre las que TIENEN respuesta)      {ok[hay].mean():.4f}")
        print(f"  las dos respuestas COINCIDEN                  {igual.mean():.4f}")
        print(f"\n  de las que ACIERTA, coinciden                {igual[ok].mean():.4f}   <- pedido de Maxi")
        print(f"  de las que ERRA (y habia respuesta), coinciden {igual[hay & ~ok].mean():.4f}")
        print(f"  de las que NO tenian respuesta, coinciden      {igual[~hay].mean():.4f}")
        # LA pregunta practica: cuando NO coinciden, ¿estan mal?
        nc = ~igual
        if nc.sum():
            mal = (~ok)          # mal = erro el valor, o no habia respuesta y contesto igual
            print(f"\n  ** cuando NO coinciden, estan mal el          {mal[nc].mean():.4f} **")
            print(f"     (tasa base de «mal» en todo el conjunto:     {mal.mean():.4f})")
            print(f"     ganancia sobre la tasa base:                 {mal[nc].mean()-mal.mean():+.4f}")
            print(f"  cuando SI coinciden, estan mal el              {mal[igual].mean():.4f}")
            print(f"  cobertura: el desacuerdo marca el               {nc.mean():.4f} de las preguntas")
        else:
            print("\n  ** NUNCA difieren: el desacuerdo no marca nada **")

if __name__ == "__main__":
    for r in ("ckpts/n3_s0.pkl",):
        correr(os.path.join(AQUI, r))
