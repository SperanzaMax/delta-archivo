"""CUANTA ATENCION HAY EN `attn_causal` · 2026-09-08

El control del 4-sep (`control_attn.py`) mostro que con `--donde attn` el cero exacto de la ley de
la ventana DESAPARECE: TV > 0 en las seis distancias, 120 de 120. Eso cierra la ley por
intervencion y no se toca. Pero el mismo JSON dice otra cosa que nadie leyo:

    lat2 kq5 · adentro de la ventana   TV 0.0275 .. 0.0444     afuera  0.000000 EXACTO
    attn                               TV 0.0097 .. 0.0109 en TODAS las distancias

O sea el acceso global no es mas fuerte que la ventana: es entre 3 y 4 veces MAS DEBIL, y sobre
todo es PLANO. Un mecanismo que responde igual a la distancia 1 que a la 6 no esta discriminando,
esta promediando.

La sospecha que esto mide: `attn_causal` usa `q = k = v = x` sobre un vector que viene de un
LayerNorm, asi que la norma es ~sqrt(D) para todos y el termino diagonal vale
sim_ii = |x|^2/sqrt(D) ~ sqrt(D) = 11.3, mientras que dos posiciones distintas dan sim_ij ~ 0. El
softmax de eso es casi una delta en la propia posicion. Si es asi, `donde=attn` NO es «acceso
global», es `pre` con un residuo de promedio del pasado, y ENTRENAR una unidad asi mediria otra
cosa que la que dice la etiqueta.

Importa antes de gastar T4 porque decide que se entrena en el escalon 1 del Micro LM de Frontera.

    python attn_concentracion.py [ckpt ...]
"""
import os, sys, json, pickle
import numpy as np
import jax, jax.numpy as jnp

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
import modelo as M
import conf_ckpt

T = 24
N = int(os.environ.get("N", "120"))
CKPTS = sys.argv[1:] or ["ckpts/kq3_s0.pkl", "ckpts/rp3_s0.pkl", "ckpts/v3_s0.pkl"]


def entrada_al_attn(params, cons):
    """Exactamente lo que `attn_causal` recibe con `donde=attn`: ln1(emb[x]) del bloque 0."""
    h = params["emb"][cons]
    return M.ln(params["blocks"][0]["ln1"], h)


def perfil(x):
    """Pesos del softmax causal de `attn_causal`, sin tocar la funcion original."""
    B, Tt, D = x.shape
    sim = jnp.einsum("btd,bsd->bts", x, x) / jnp.sqrt(D)
    sim = jnp.where(jnp.tril(jnp.ones((Tt, Tt), bool))[None], sim, -1e9)
    return np.asarray(jax.nn.softmax(sim, -1)), np.asarray(sim)


def con_proyecciones(x, semilla, escala=1.0):
    """El mismo attn pero con wq/wk propias, init Glorot como un transformer de verdad."""
    B, Tt, D = x.shape
    rng = np.random.default_rng(semilla)
    s = escala / np.sqrt(D)
    wq = jnp.array(rng.normal(size=(D, D)) * s, dtype=jnp.float32)
    wk = jnp.array(rng.normal(size=(D, D)) * s, dtype=jnp.float32)
    q, k = x @ wq, x @ wk
    sim = jnp.einsum("btd,bsd->bts", q, k) / jnp.sqrt(D)
    sim = jnp.where(jnp.tril(jnp.ones((Tt, Tt), bool))[None], sim, -1e9)
    return np.asarray(jax.nn.softmax(sim, -1)), np.asarray(sim)


def resumen(p, sim, pos=T - 1):
    """Concentracion en la fila `pos`, que es la posicion de lectura que usa el control."""
    fila = p[:, pos, :pos + 1]                       # (B, pos+1)
    diag = fila[:, pos]
    resto = fila[:, :pos]
    ent = -(fila * np.log(fila + 1e-12)).sum(-1)
    unif = np.log(pos + 1)
    sfila = sim[:, pos, :pos + 1]
    return {
        "peso_propia_pos": float(diag.mean()),
        "masa_fuera": float(resto.sum(-1).mean()),
        "max_de_los_otros": float(resto.max(-1).mean()),
        "entropia": float(ent.mean()),
        "entropia_uniforme": float(unif),
        "posiciones_efectivas": float(np.exp(ent).mean()),
        "de_un_total_de": pos + 1,
        "sim_diagonal": float(sfila[:, pos].mean()),
        "sim_otros": float(sfila[:, :pos].mean()),
        "brecha_logits": float(sfila[:, pos].mean() - sfila[:, :pos].mean()),
    }


if __name__ == "__main__":
    rng = np.random.default_rng(1000)
    res = {}
    for ruta in CKPTS:
        if not os.path.isfile(ruta):
            print(f"-- falta {ruta}, se saltea"); continue
        b = pickle.load(open(ruta, "rb"))
        params, cfg = b["params"], b["config"]
        conf_ckpt.aplicar(cfg)
        V = params["emb"].shape[0]
        cons = jnp.array(rng.integers(0, V, (N, T)))
        x = entrada_al_attn(params, cons)
        print(f"\n{'=' * 78}\n{ruta}\n  {conf_ckpt.descripcion(cfg)}")
        print(f"  norma media de ln1(emb[x]): {float(jnp.linalg.norm(x, axis=-1).mean()):.4f}"
              f"   (sqrt(D) = {float(np.sqrt(x.shape[-1])):.4f})")

        fila = {}
        p, sim = perfil(x)
        fila["q=k=v=x"] = resumen(p, sim)
        for esc in (1.0, 0.5):
            p2, sim2 = con_proyecciones(x, 7, esc)
            fila[f"wq/wk Glorot x{esc}"] = resumen(p2, sim2)

        cab = ("condicion", "peso propia pos", "masa fuera", "pos. efectivas", "brecha logits")
        print(f"\n  {cab[0]:<20}{cab[1]:>17}{cab[2]:>13}{cab[3]:>16}{cab[4]:>16}")
        for nom, r in fila.items():
            print(f"  {nom:<20}{r['peso_propia_pos']:>17.6f}{r['masa_fuera']:>13.6f}"
                  f"{r['posiciones_efectivas']:>10.2f} de {r['de_un_total_de']:<3}"
                  f"{r['brecha_logits']:>16.4f}")
        res[ruta] = fila

    json.dump({"N": N, "T": T, "res": res},
              open(os.path.join(AQUI, "attn_concentracion.json"), "w"), indent=1)
    print("\nguardado en attn_concentracion.json")
