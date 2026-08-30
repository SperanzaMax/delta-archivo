"""Mide `c` y `q` tal como los define `entrenar.py::_recompensa`, sobre checkpoints ya en disco.

Por que hace falta (2026-08-30). La prediccion del 29 a la noche leyo la mudez de `f23_s3` contra el
umbral GLOBAL (0,657) y anoto que las mudas «llegan a 0,30-0,40 recien a los 22000 pasos». Dos
problemas con esa lectura:

  1. `mapa_recompensa.py` demuestra que la ventaja de la politica del ORACULO sobre el silencio cruza
     cero en c* = (M-F)/(1+M) = 0,200, NO en el umbral global. El global gobierna solo a un modelo que
     no distingue ausencia de error, y este proyecto midio que si distingue (AUC 0,9998-1,0000).
  2. El 0,30-0,40 es RECUP —exactitud del ARGMAX—, y `c` en la perdida es la probabilidad NORMALIZADA
     del token correcto. No son el mismo numero y no hay razon para que se parezcan.

Asi que el umbral relevante y la cantidad medida estaban los dos mal pareados. Esto los mide.

Y se reporta la DISTRIBUCION, no la media: lo que decide si al modelo le conviene hablar es en cuantas
muestras c supera c*, y una media puede estar debajo del umbral con media poblacion arriba. Van cinco
veces en este proyecto que una media escondio su distribucion.

Uso:  python3 medir_confianza.py ckpts/f23_s3.pkl ckpts/b3_s3.pkl ckpts/n3_s0.pkl --n 4000
"""

import argparse
import pickle

import jax
import jax.numpy as jnp
import numpy as np

import datos as DAT
import entrenar as E


def medir(ruta, n, B, semilla, p_nose_cli=None):
    with open(ruta, "rb") as f:
        bulto = pickle.load(f)
    params, cfg = bulto["params"], bulto["config"]
    nivel = cfg["nivel"]
    p_nose = p_nose_cli if p_nose_cli is not None else cfg.get("p_nose", 0.0)
    E._DONDE = cfg.get("donde", "pre")
    E._ABST = cfg.get("abst", "token")
    # Una base entrenada con p_nose=0 no tiene cabeza de abstencion en `params`. No es un caso
    # degradado: es el punto de partida sembrado, y su `c` es justamente lo que interesa medir.
    # `modelo.py:302` calcula el logit de la cabeza SIEMPRE, aun con abst=token, asi que un checkpoint
    # sin ella revienta. Se rellena con ceros aca —en el script de medicion, no en el codigo del
    # experimento— y `a` queda constante 0, que se ignora porque en esta rama `q` sale del softmax.
    if "abst" not in params:
        d = params["ln_f"]["g"].shape[-1]
        params = dict(params)
        params["abst"] = {"w": jnp.zeros((d, 1)), "b": jnp.zeros((1,))}
        print(f"  [aviso] {ruta}: sin cabeza de abstencion en el checkpoint (base con p_nose=0). "
              f"Se mide con la interfaz `token`; `q` es la masa de NOSE en el softmax.")
        E._ABST = "token"

    @jax.jit
    def partes(params, ses, cortes, turnos, mask, cons, pos):
        return E._partes(params, ses, cortes, turnos, mask, cons, pos)

    rng = np.random.default_rng(semilla)
    C, Q, HAY, ACI = [], [], [], []
    vistos = 0
    while vistos < n:
        b = min(B, n - vistos)
        ses, cortes, turnos, mask, cons, pos, tgt, tipo, meta = DAT.lote(
            rng, b, nivel=nivel, n_hechos=4, n_sesiones=4, p_nose=p_nose, con_meta=True)
        lg, a = partes(params, jnp.array(ses), jnp.array(cortes), jnp.array(turnos),
                       jnp.array(mask), jnp.array(cons), jnp.array(pos))
        lg = np.asarray(lg, dtype=np.float64)
        tgt = np.asarray(tgt)

        # --- identico a `_recompensa`: NOSE fuera, softmax sobre los valores, prob del target -----
        lg_v = lg.copy()
        lg_v[:, E.NOSE] = -1e9
        p_val = np.exp(lg_v - lg_v.max(-1, keepdims=True))
        p_val /= p_val.sum(-1, keepdims=True)
        hay = (tgt != E.NOSE)
        c = p_val[np.arange(len(tgt)), tgt] * hay

        # q: masa de NOSE en el softmax de vocabulario (token) o sigmoid de la cabeza (cabeza/slot)
        if E._ABST == "token":
            p_all = np.exp(lg - lg.max(-1, keepdims=True))
            p_all /= p_all.sum(-1, keepdims=True)
            q = p_all[:, E.NOSE]
        else:
            q = 1.0 / (1.0 + np.exp(-np.asarray(a, dtype=np.float64)))

        C.append(c); Q.append(q); HAY.append(hay)
        ACI.append((lg_v.argmax(-1) == tgt) & hay)
        vistos += b

    return (np.concatenate(C), np.concatenate(Q), np.concatenate(HAY),
            np.concatenate(ACI), cfg, bulto.get("paso"))


def informar(ruta, c, q, hay, aci, cfg, paso, c_est):
    ch = c[hay]                      # confianza SOLO donde hay respuesta (donde c* aplica)
    qh, qn = q[hay], q[~hay]
    print(f"\n--- {ruta}   paso={paso}  abst={cfg.get('abst')}  blanco={cfg.get('blanco','ausencia')} "
          f"  perdida={cfg.get('perdida_cabeza','bce')}  n={len(c)} ---")
    print(f"  RECUP (argmax, solo con respuesta)      {aci[hay].mean():.4f}")
    print(f"  c  media                                {ch.mean():.4f}")
    print(f"  c  mediana                              {np.median(ch):.4f}")
    print(f"  c  percentiles 10/25/75/90              "
          f"{np.percentile(ch,10):.4f} / {np.percentile(ch,25):.4f} / "
          f"{np.percentile(ch,75):.4f} / {np.percentile(ch,90):.4f}")
    print(f"  ** fraccion con c > c*={c_est:.3f}       {(ch > c_est).mean():.4f}  <== decide si conviene hablar")
    # La curva completa: cuanto baja el umbral, cuanta poblacion entra. Es lo que permite elegir los
    # pesos por ALCANZABILIDAD desde el punto de partida (que es la correccion a los cuatro defectos
    # de pre-registro del mes) y no por el desenlace del experimento, que seria ajustar sobre la marcha.
    print("  curva  c*  ->  fraccion de muestras con respuesta por encima:")
    linea = "     "
    for u in (0.0370, 0.0500, 0.1000, 0.1111, 0.1333, 0.2000, 0.2667, 0.3000):
        linea += f"{u:.3f}:{(ch > u).mean():.3f}   "
    print(linea)
    print(f"  q  media  (con respuesta / sin)         {qh.mean():.4f} / {qn.mean():.4f}")
    print(f"  q  separacion (sin - con)               {qn.mean()-qh.mean():+.4f}")
    print(f"  fraccion con q > 0,5 (se callaria)      {(q > 0.5).mean():.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--lote", type=int, default=64)
    ap.add_argument("--semilla", type=int, default=54321)   # la misma de la Fase 1, pareado
    ap.add_argument("--p-nose", type=float, default=None)
    ap.add_argument("--c-est", type=float, default=0.200,
                    help="umbral por muestra (M-F)/(1+M); 0,200 con M=0,5 F=0,2")
    a = ap.parse_args()

    print("=" * 88)
    print("CONFIANZA `c` Y MASA DE ABSTENCION `q`, como las define la recompensa")
    print(f"n={a.n} por unidad, semilla de datos {a.semilla} (pareada entre unidades)")
    print("=" * 88)
    for r in a.ckpts:
        c, q, hay, aci, cfg, paso = medir(r, a.n, a.lote, a.semilla, a.p_nose)
        informar(r, c, q, hay, aci, cfg, paso, a.c_est)


if __name__ == "__main__":
    main()
