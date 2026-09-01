"""¿Tiene margen el AFILADO de la busqueda? · compuerta barata antes de gastar GPU · 1-sep

Pregunta de Maxi: «tiene que haber algo en el aprendizaje de BUSQUEDA que podamos modificar para que
le sea mas facil decir la verdad».

`NOTA_BUSQUEDA_UNIFORME_20260831.md` midio que la busqueda «siempre busca igual y siempre a media
maquina», porque el divisor de sim = q·k/sqrt(d) es una CONSTANTE. La intervencion candidata es un
beta(x) APRENDIDO por consulta: afilar cuando encuentra, difuminar cuando no.

Antes de entrenar nada hay que saber si eso tiene MARGEN: ¿existe algun beta fijo con el que la
lectura separe «esta» de «no esta» mejor que el beta=1 de hoy? Si ningun beta separa, un beta(x)
aprendido tampoco va a poder y la idea muere barata. Si algun beta separa mas, hay techo que ganar.

Se mide la ENTROPIA de la lectura y el MAXIMO de la masa, contra si la respuesta esta o no.
"""
import os
import sys

import numpy as np
import jax
import jax.numpy as jnp

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)

import datos as DAT, entrenar as E, idioma as I, medir_ratio_ce as R, modelo as M

N, LOTE = 1024, 64
BETAS = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)


def auc(s, pos):
    o = np.argsort(s); r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1)
    n1, n0 = pos.sum(), (~pos).sum()
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else float("nan")


def correr(ruta):
    params, cfg, paso = R.cargar(ruta)
    params = jax.tree_util.tree_map(jnp.asarray, params)
    I.fijar_version(cfg.get("idioma", 2)); a_p = params["arch"]; donde = cfg.get("donde", "pre")

    @jax.jit
    def leer(params, ses, cortes, turnos, mask, cons, pos):
        """Devuelve las similitudes CRUDAS de la lectura, sin softmax: el beta se aplica afuera."""
        archivo = M.escribir(params, ses, cortes)
        ak = archivo @ a_p["kw"] + a_p["ord"][turnos]
        av = archivo @ a_p["vw"]
        penal = jnp.where(mask, 0.0, -1e9)[:, None, :]
        guardado = {}
        def lectura(h):
            q = h @ a_p["qr"]
            sim = jnp.einsum("btd,bnd->btn", q, ak) / jnp.sqrt(h.shape[-1]) + penal
            guardado["sim"] = sim
            return jnp.einsum("btn,bnd->btd", jax.nn.softmax(sim, -1), av) @ a_p["wo"]
        M.tronco(params, cons, lectura, 0, donde)
        s = guardado["sim"]
        return jnp.take_along_axis(s, pos[:, None, None], axis=1)[:, 0, :], mask

    rng = np.random.default_rng(54321)
    SIM, MSK, TGT = [], [], []
    vistos = 0
    while vistos < N:
        b = min(LOTE, N - vistos)
        ses, cortes, turnos, mask, cons, pos, tgt, tipo = DAT.lote(
            rng, b, nivel=cfg["nivel"], n_hechos=4, n_sesiones=4, p_nose=0.4)
        aj = [jnp.array(x) for x in (ses, cortes, turnos, mask, cons, pos)]
        s, m = leer(params, *aj)
        SIM.append(np.asarray(s)); MSK.append(np.asarray(m)); TGT.append(np.asarray(tgt))
        vistos += b

    sim = np.concatenate(SIM); msk = np.concatenate(MSK); tgt = np.concatenate(TGT)
    no = (tgt == E.NOSE)
    print(f"\n{'='*94}\n{os.path.basename(ruta)}  paso={paso}  n={len(tgt)}  "
          f"entradas por muestra {msk.sum(1).mean():.1f}  ·  sin respuesta {no.mean():.4f}\n{'='*94}")
    print(f"{'beta':>6s} {'H(esta)':>9s} {'H(no esta)':>11s} {'brecha_sd':>10s} {'AUC_H':>8s}"
          f" {'AUC_max':>8s}  {'H_max_posible':>13s}")
    mejor = (0, None)
    for beta in BETAS:
        p = jax.nn.softmax(jnp.asarray(sim) * beta, -1)
        p = np.asarray(p)
        with np.errstate(divide="ignore", invalid="ignore"):
            H = -(p * np.log(np.maximum(p, 1e-12))).sum(-1)
        mx = p.max(-1)
        sd = H.std()
        brecha = (H[no].mean() - H[~no].mean()) / max(sd, 1e-9)
        a_h, a_m = auc(H, no), auc(-mx, no)
        hmax = np.log(msk.sum(1).mean())
        print(f"{beta:6.2f} {H[~no].mean():9.4f} {H[no].mean():11.4f} {brecha:10.4f} {a_h:8.4f}"
              f" {a_m:8.4f}  {hmax:13.4f}")
        if max(a_h, a_m) > mejor[0]:
            mejor = (max(a_h, a_m), beta)
    print(f"\n  mejor AUC {mejor[0]:.4f} con beta={mejor[1]}   ·   beta=1 (el de hoy) es el del medio")
    print(f"  techo de la evidencia en el ESTADO, medido el 31-ago: 0,7003")
    return mejor


if __name__ == "__main__":
    for r in sys.argv[1:] or ["ckpts/n3_s0.pkl", "ckpts/p3_s0.pkl"]:
        correr(os.path.join(AQUI, r))
