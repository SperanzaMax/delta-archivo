"""¿El DESACUERDO entre dos búsquedas detecta la ausencia? · idea de Maxi, 31-ago noche

    «si hace la búsqueda dos veces, si la respuesta es diferente una de otra, ¿qué sería eso?»

**Por qué esta medida NO está acotada por el 0,70 de hoy, y es lo que la hace valer la pena.** El
techo de 0,7003 medido esta noche acota a los lectores que miran EL ESTADO: son funciones de un
punto. El desacuerdo entre búsquedas no es función del estado, es una propiedad de la ESTABILIDAD
del mecanismo alrededor de ese punto —del mapa, no del punto—. Es otra familia de medidas y puede
superar 0,70 sin contradecir nada de lo medido.

**El problema de diseño:** el modelo es determinista. Misma pregunta -> misma query -> misma
respuesta, siempre. Buscar dos veces da idéntico. Así que las dos búsquedas tienen que diferir por
construcción, y acá se usa la forma más barata: **perturbar la query con ruido gaussiano** de escala
sigma relativa a su norma, K veces, y medir si la respuesta aguanta.

CRITERIOS, ESCRITOS ANTES DE MIRAR:
  D-1  si AUC(inestabilidad vs ausencia) > 0,70 en algún sigma, SUPERA el techo del estado y la vía
       queda abierta: hay información de ausencia fuera de lo que el estado deja leer.
  D-2  si queda entre 0,55 y 0,70, hay señal pero no rompe el techo: sirve como aporte, no como via.
  D-3  si <= 0,55 es azar POST-HOC. Ojo con la lectura: NO cierra la version ENTRENADA, por el
       precedente medido del propio proyecto (el blanco `error` da 0,65 post-hoc y 1,0000 entrenado).
  D-4  CONTROL OBLIGATORIO. La inestabilidad podría ser sólo un proxy de la confianza de salida, que
       ya está medida y es mala (AUC 0,46-0,61). Se reporta la correlación entre las dos y el AUC de
       la confianza sobre las MISMAS muestras. Si la inestabilidad no supera a la confianza, no es
       una medida nueva y no se adjudica nada.
"""
import os, sys
import numpy as np, jax, jax.numpy as jnp
AQUI = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, AQUI)
import datos as DAT, entrenar as E, idioma as I, medir_ratio_ce as R, modelo as M

N, K, LOTE = 1536, 8, 64
SIGMAS = (0.05, 0.1, 0.2, 0.4)

def auc(s, pos):
    o = np.argsort(s); r = np.empty(len(s)); r[o] = np.arange(1, len(s)+1)
    n1, n0 = pos.sum(), (~pos).sum()
    return float((r[pos].sum() - n1*(n1+1)/2)/(n1*n0)) if n1 and n0 else float("nan")

def correr(ruta):
    params, cfg, paso = R.cargar(ruta)
    # del pickle salen numpy: indexar `ord[turnos]` con un array TRACEADO falla si no se convierte
    params = jax.tree_util.tree_map(jnp.asarray, params)
    I.fijar_version(cfg.get("idioma", 2)); a_p = params["arch"]
    donde = cfg.get("donde", "pre")

    def responder(params, ses, cortes, turnos, mask, cons, pos, ruido):
        """`ruido` (B,1,D) se SUMA a la query de la lectura. Con 0 es el forward normal."""
        archivo = M.escribir(params, ses, cortes)
        ak = archivo @ a_p["kw"] + a_p["ord"][turnos]; av = archivo @ a_p["vw"]
        penal = jnp.where(mask, 0.0, -1e9)[:, None, :]
        def lectura(h):
            q = h @ a_p["qr"] + ruido
            sim = jnp.einsum("btd,bnd->btn", q, ak)/jnp.sqrt(h.shape[-1]) + penal
            return jnp.einsum("btn,bnd->btd", jax.nn.softmax(sim, -1), av) @ a_p["wo"]
        h = M.tronco(params, cons, lectura, 0, donde)
        lg = M.ln(params["ln_f"], h) @ params["head"]["w"] + params["head"]["b"]
        return jnp.take_along_axis(lg, pos[:, None, None], axis=1)[:, 0, :]
    responder = jax.jit(responder)

    rng = np.random.default_rng(54321); rk = np.random.default_rng(7)
    TGT, BASE, CONF, INEST = [], [], [], {s: [] for s in SIGMAS}
    vistos = 0
    while vistos < N:
        b = min(LOTE, N - vistos)
        ses, cortes, turnos, mask, cons, pos, tgt, tipo = DAT.lote(
            rng, b, nivel=cfg["nivel"], n_hechos=4, n_sesiones=4, p_nose=0.4)
        aj = [jnp.array(x) for x in (ses, cortes, turnos, mask, cons, pos)]
        D = a_p["qr"].shape[-1]
        cero = jnp.zeros((b, 1, D))
        lg0 = responder(params, *aj, cero)
        lgv = lg0.at[:, E.NOSE].set(-1e9)
        arg0 = np.asarray(lgv.argmax(-1))
        p = jax.nn.softmax(lgv, -1)
        CONF.append(np.asarray(p.max(-1)))
        # escala del ruido: relativa a la norma tipica de la query en esa posicion
        nq = float(jnp.linalg.norm(M.ln(params["blocks"][0]["ln1"], params["emb"][aj[4]]) @ a_p["qr"],
                                   axis=-1).mean())
        for s in SIGMAS:
            iguales = np.zeros(b)
            for _ in range(K):
                r = jnp.array(rk.normal(size=(b, 1, D)) * (s * nq / np.sqrt(D)))
                lgk = responder(params, *aj, r).at[:, E.NOSE].set(-1e9)
                iguales += (np.asarray(lgk.argmax(-1)) == arg0)
            INEST[s].append(1.0 - iguales / K)      # 0 = la respuesta aguanta; 1 = cambia siempre
        TGT.append(np.asarray(tgt)); BASE.append(arg0); vistos += b

    tgt = np.concatenate(TGT); no = (tgt == E.NOSE)
    conf = np.concatenate(CONF); arg0 = np.concatenate(BASE)
    print(f"\n--- {os.path.basename(ruta)}  paso={paso}  n={len(tgt)}  K={K} "
          f"·  RECUP {float((arg0 == tgt)[~no].mean()):.4f}  ·  sin respuesta {no.mean():.4f} ---")
    a_conf = auc(-conf, no)
    print(f"  CONTROL D-4 · confianza de salida (menor = mas ausente)   AUC {a_conf:.4f}")
    mejor = 0.0
    for s in SIGMAS:
        v = np.concatenate(INEST[s]); a = auc(v, no)
        r = float(np.corrcoef(v, conf)[0, 1])
        print(f"  sigma {s:4.2f} · inestabilidad media {v.mean():.4f} "
              f"(con resp {v[~no].mean():.4f} · sin resp {v[no].mean():.4f})   "
              f"AUC {a:.4f}   corr con la confianza {r:+.3f}")
        mejor = max(mejor, a)
    print(f"  => mejor AUC {mejor:.4f}   ·   techo del estado 0,7003   ·   confianza {a_conf:.4f}")
    ver = ("D-1 SUPERA el techo del estado" if mejor > 0.70 else
           "D-2 senal parcial" if mejor > 0.55 else "D-3 azar POST-HOC")
    print(f"  ** {ver} **" + ("" if mejor > a_conf + 0.03 else
          "   <- pero NO supera a la confianza: D-4 dice que no es una medida nueva"))
    return mejor, a_conf

if __name__ == "__main__":
    print("=" * 96); print("DESACUERDO ENTRE BUSQUEDAS · idea de Maxi"); print("=" * 96)
    for ruta in ("ckpts/n3_s0.pkl", "ckpts/t03_s3.pkl"):
        correr(os.path.join(AQUI, ruta))
