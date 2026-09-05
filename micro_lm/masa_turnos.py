"""¿Adonde va la masa de atencion cuando el archivo es largo? · 2026-09-05

Instrumento para el criterio que adjudica R11: el sello de orden, ¿descarta lo viejo?

Las entradas del episodio llevan turnos >= TURNO_BASE y las de las OTRAS conversaciones turnos por
debajo. Si el modelo aprendio a filtrar por antiguedad, la masa del softmax de lectura sobre las
entradas extra tiene que caer muy por debajo de lo que le tocaria por proporcion.

Devuelve, promediando sobre el lote y leyendo en el ultimo token de la pregunta:
  · `masa_extra`     cuanta masa cae en entradas de otras conversaciones;
  · `proporcion`     que fraccion de las entradas escritas son extra (la referencia sin filtro);
  · `indice`         masa_extra / proporcion — 1,0 es «no filtra nada», 0,0 es «las ignora»;
  · `masa_correcta`  masa en la entrada que contesta la pregunta;
  · `recup`          con que frecuencia el argmax de la lectura ES esa entrada.

Uso:
    python3 masa_turnos.py ckpts/kq3_s0.pkl --ses-extra 36 --lotes 4
"""
import argparse, json, pickle
import numpy as np, jax, jax.numpy as jnp

import datos as DAT, idioma as I, modelo as M


def masa(params, donde, nivel, ses_extra, lotes, B, semilla, barajar=False):
    """`barajar` es EL CONTROL, y sin el la medicion no adjudica nada.

    Las entradas del episodio nombran la entidad que la pregunta menciona, asi que se llevan mas masa
    que su proporcion **por contenido**, sin que el sello de orden intervenga. Barajando los turnos
    dentro de cada muestra —misma cantidad de entradas, mismo contenido, sello sin relacion con la
    antiguedad real— el efecto de contenido queda igual y el de orden desaparece. La diferencia entre
    las dos celdas es lo unico atribuible al sello.
    """
    rng = np.random.default_rng(semilla)
    acc = {"masa_extra": [], "proporcion": [], "masa_correcta": [], "recup": []}
    for _ in range(lotes):
        s, c, t, mk, q, pq, tg, tp, meta, orig, hq = DAT.lote(
            rng, B, nivel=nivel, p_vieja=0.0, p_nose=0.0, con_meta=True, con_origen=True,
            n_ses_extra=ses_extra)
        if barajar:
            t = t.copy()
            for b_ in range(B):
                vis = mk[b_]
                if vis.any():
                    v = t[b_][vis].copy(); rng.shuffle(v); t[b_][vis] = v
        a = params["arch"]
        archivo = M.escribir(params, jnp.array(s), jnp.array(c))
        ak = archivo @ a["kw"] + M.sello(a, jnp.array(t))
        av = archivo @ a["vw"]
        penal = jnp.where(jnp.array(mk), 0.0, -1e9)[:, None, :]
        guardado = {}

        def lectura(h):
            qq = h @ a["qr"]
            sim = jnp.einsum("btd,bnd->btn", qq, ak) / jnp.sqrt(h.shape[-1]) + penal
            p = jax.nn.softmax(sim, -1)
            guardado["p"] = p
            return jnp.einsum("btn,bnd->btd", p, av) @ a["wo"]

        M.tronco(params, jnp.array(q), lectura, 0, donde)
        p = np.array(guardado["p"])[np.arange(B), np.array(pq)]      # (B, N)

        n_pri = 4 * DAT.E_MAX
        es_extra = np.zeros(p.shape[1], bool); es_extra[n_pri:] = True
        for b in range(B):
            escritos = mk[b]
            if not escritos.any():
                continue
            corr = (orig[b] == hq[b]) & escritos
            acc["masa_extra"].append(float(p[b][escritos & es_extra].sum()))
            acc["proporcion"].append(float((escritos & es_extra).sum() / escritos.sum()))
            if corr.any():
                acc["masa_correcta"].append(float(p[b][corr].sum()))
                acc["recup"].append(float(bool(corr[int(np.argmax(np.where(escritos, p[b], -1)))])))
    r = {k: float(np.mean(v)) if v else float("nan") for k, v in acc.items()}
    r["indice"] = r["masa_extra"] / r["proporcion"] if r["proporcion"] else float("nan")
    r["n"] = len(acc["masa_extra"])
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--ses-extra", type=int, nargs="+", default=[0, 8, 36])
    ap.add_argument("--lotes", type=int, default=2)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--semilla", type=int, default=90000)
    ap.add_argument("--barajar", action="store_true",
                    help="corre TAMBIEN la celda con los turnos barajados dentro de cada muestra: "
                         "el control que separa filtrar por antiguedad de preferir por contenido")
    ap.add_argument("--salida", default="")
    a = ap.parse_args()

    b = pickle.load(open(a.ckpt, "rb"))
    cfg, params = b["config"], b["params"]
    M.KQ = cfg.get("kernel_q", 3)
    print(f"{a.ckpt} · paso {cfg.get('pasos')} · kernel_q={M.KQ} · donde={cfg['donde']} · "
          f"nivel {cfg['nivel']} · entrenado con ses_extra={cfg.get('ses_extra', 0)}")
    print(f"{'extra':>6} {'entradas':>9} {'masa_extra':>11} {'proporcion':>11} {'indice':>8} "
          f"{'masa_corr':>10} {'recup':>7}")
    salida = {"ckpt": a.ckpt, "config": {k: cfg.get(k) for k in
              ("pasos", "kernel_q", "donde", "nivel", "d", "capas", "ses_extra")}, "celdas": []}
    for e in a.ses_extra:
        for baraj in ((False, True) if a.barajar else (False,)):
            r = masa(params, cfg["donde"], cfg["nivel"], e, a.lotes, a.batch, a.semilla, baraj)
            r["ses_extra"] = e; r["barajado"] = baraj
            salida["celdas"].append(r)
            print(f"{e:6d} {(4 + e) * DAT.E_MAX:9d} {r['masa_extra']:11.4f} "
                  f"{r['proporcion']:11.4f} {r['indice']:8.4f} {r['masa_correcta']:10.4f} "
                  f"{r['recup']:7.4f}  {'barajado' if baraj else 'sello real'}")
    if a.salida:
        json.dump(salida, open(a.salida, "w"), indent=1)
        print(f"-> {a.salida}")


if __name__ == "__main__":
    main()
