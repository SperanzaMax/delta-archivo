"""Control de la extraccion del score de `score_archivo.py`.

    python control_score.py ckpts/n4_s0.pkl

POR QUE EXISTE: `score_archivo.py` dio AUC 0,4984 — el azar EXACTO. En este programa un cero limpio
escondio un artefacto siete veces, asi que antes de leerlo como resultado hay que descartar la
explicacion alternativa obvia: **que mi copia de la lectura este mal y este midiendo ruido.** Una
extraccion rota y una señal ausente producen el mismo numero.

COMO SE DESCARTA, sin pedirle nada al modelo: se reconstruyen los logits POR MI CAMINO —usando el
`sim` que extraigo— y se comparan contra `entrenar.logits_de`, que es el camino real del modelo. Si
coinciden, mi `sim` es exactamente el que el modelo usa y el 0,4984 es una propiedad del modelo, no
de mi codigo. Si difieren, el resultado se cae y hay que arreglar la extraccion.

El control PUEDE FALLAR, que es el requisito que el `m=1` del banco ECO no cumplia.
"""
import pickle
import sys

import numpy as np
import jax
import jax.numpy as jnp

import datos as DAT
import modelo as M
import entrenar as E
import score_archivo as SA


def main():
    pesos = sys.argv[1] if len(sys.argv) > 1 else "ckpts/n4_s0.pkl"
    with open(pesos, "rb") as f:
        bulto = pickle.load(f)
    params = jax.tree_util.tree_map(jnp.asarray, bulto["params"])
    nivel = bulto["config"]["nivel"]

    rng = np.random.default_rng(7)
    ses, cortes, turnos, mask, cons, pos, tgt, tipo = DAT.lote(
        rng, 64, nivel=nivel, n_hechos=4, n_sesiones=4, p_nose=0.4)
    ses, cortes, turnos = jnp.array(ses), jnp.array(cortes), jnp.array(turnos)
    mask, cons, pos = jnp.array(mask), jnp.array(cons), jnp.array(pos)

    # --- camino del modelo ---
    lg_modelo = E.logits_de(params, ses, cortes, turnos, mask, cons, pos)

    # --- mi camino: mismo tronco, pero la lectura armada con el `sim` que extraigo ---
    a = params["arch"]
    archivo = M.escribir(params, ses, cortes)
    ak = archivo @ a["kw"] + a["ord"][turnos]
    av = archivo @ a["vw"]
    penal = jnp.where(mask, 0.0, -1e9)[:, None, :]

    def lectura_mia(h):
        q = h @ a["qr"]
        sim = jnp.einsum("btd,bnd->btn", q, ak) / jnp.sqrt(h.shape[-1]) + penal
        return jnp.einsum("btn,bnd->btd", jax.nn.softmax(sim, -1), av) @ a["wo"]

    h = M.tronco(params, cons, lectura_mia, 0)
    lg_mio = M.ln(params["ln_f"], h) @ params["head"]["w"] + params["head"]["b"]
    lg_mio = jnp.take_along_axis(lg_mio, pos[:, None, None], axis=1)[:, 0, :]

    d = float(jnp.abs(lg_modelo - lg_mio).max())
    print(f"pesos: {pesos}")
    print(f"C-1 · max|logits_modelo - logits_reconstruidos| = {d:.3e}   "
          f"{'OK (misma lectura)' if d < 1e-3 else 'FALLA -> la extraccion NO reproduce al modelo'}")

    # C-2: el score tiene que VARIAR entre muestras y entre entradas. Un tensor casi constante daria
    # AUC 0,5 por falta de señal en la medicion, no por falta de señal en el modelo.
    s = np.asarray(SA.scores_archivo(params, ses, cortes, turnos, mask, cons, pos))
    val = s > -1e8                       # entradas reales, sin las vacias penalizadas
    print(f"C-2 · score en entradas validas: media {s[val].mean():.4f} · sd {s[val].std():.4f} · "
          f"rango [{s[val].min():.3f}, {s[val].max():.3f}]")
    smax = np.sort(s, -1)[:, -1]
    print(f"      s_max entre muestras: sd {smax.std():.4f}   "
          f"{'OK (varia)' if smax.std() > 1e-3 else 'FALLA -> constante, no hay nada que separar'}")

    # C-3: la lectura del archivo, ¿cambia la respuesta? Si ceroar el archivo no mueve los logits,
    # el modelo no lo esta usando y toda la medicion es sobre un canal muerto.
    lg_cero = E.logits_de(params, ses, cortes, turnos, jnp.zeros_like(mask), cons, pos)
    dif = float(jnp.abs(lg_modelo - lg_cero).mean())
    ig_modelo = np.asarray(lg_modelo.argmax(-1))
    ig_cero = np.asarray(lg_cero.argmax(-1))
    print(f"C-3 · archivo ablacionado: |Δlogits| medio {dif:.4f} · "
          f"cambia la prediccion en {(ig_modelo != ig_cero).mean():.4f} de las muestras   "
          f"{'OK (el archivo se usa)' if dif > 0.01 else 'FALLA -> canal muerto'}")


if __name__ == "__main__":
    main()
