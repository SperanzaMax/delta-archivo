"""El sello de orden, ¿es un RELOJ o es una BANDERA? · 2026-09-06

`PREREG_RELOJ_O_BANDERA.md`, SHA `aabb4b20`.

El 5-sep la campaña del archivo largo mostró que el sello aprende a descartar lo viejo (índice de
masa 0,17-0,26 con sello real contra 0,78-0,81 barajado). El §3 del informe declaró lo que ese
control NO separa: **entender la recencia** de **usar el sello como dirección de búsqueda**, porque
en el diseño las entradas ajenas llevan turnos bajos por construcción.

Acá se separa cambiando SÓLO el vector de turnos, sobre los mismos lotes y los mismos checkpoints:

  `real`        episodio 24..63, extra U(0,23)      — el régimen de entrenamiento
  `barajado`    turnos mezclados dentro de la muestra — el control ya publicado
  `corrido`     episodio 12..51, extra U(0,11)      — orden relativo intacto, frontera movida
  `comprimido`  episodio 24..63, extra todas en 23  — sin orden ENTRE las viejas
  `invertido`   episodio 0..39,  extra U(40,63)     — la entrada correcta en turno BAJO

El archivo se escribe UNA vez por lote: `modelo.escribir` no depende de los turnos, así que las
cinco condiciones comparten contenido, pregunta y entrada correcta. Es un diseño pareado exacto y
además es lo que hace que la medición entre en CPU.

Uso:
    python3 reloj_o_bandera.py ckpts/lg3_s0.pkl --lotes 4 --batch 16 --salida sal.json
"""
import argparse, json, pickle

import numpy as np, jax, jax.numpy as jnp

import datos as DAT, idioma as I, modelo as M

CONDICIONES = ("real", "barajado", "corrido", "extra_corridas", "comprimido", "invertido")
UMBRAL = DAT.TURNO_BASE      # 24 — la frontera que el entrenamiento vio siempre en el mismo lugar


def turnos_de(cond, t, mk, n_pri, rng):
    """Devuelve el vector de turnos de la condición. `t` es el real y no se modifica.

    Las entradas NO escritas se dejan en 0 SIEMPRE. No compiten —`penal` las manda a −1e9— pero desde
    el 5-sep `modelo.sello` usa `mode="fill"` con NaN, así que un índice negativo (que es lo que sale
    de restar el corrimiento sobre un slot vacío) envenenaría el softmax entero de esa muestra en vez
    de quedar ignorado. Es el fallo ruidoso funcionando: se ve enseguida en vez de dar un número.
    """
    t2 = np.where(mk, t, 0)
    B = t.shape[0]
    if cond == "real":
        return t2
    if cond == "barajado":
        for b in range(B):
            vis = mk[b]
            if vis.any():
                v = t2[b][vis].copy(); rng.shuffle(v); t2[b][vis] = v
        return t2
    if cond == "corrido":
        t2[:, :n_pri] = t[:, :n_pri] - 12
        t2[:, n_pri:] = rng.integers(0, 12, size=t[:, n_pri:].shape)
    elif cond == "extra_corridas":
        # ENMIENDA E-1: mueve SOLO las ajenas. Separa «importa en que filas cae el episodio» de
        # «importa en que filas caen las ajenas», que `corrido` cambia juntas.
        t2[:, n_pri:] = rng.integers(0, 12, size=t[:, n_pri:].shape)
    elif cond == "comprimido":
        t2[:, n_pri:] = UMBRAL - 1
    elif cond == "invertido":
        t2[:, :n_pri] = t[:, :n_pri] - UMBRAL
        t2[:, n_pri:] = rng.integers(UMBRAL + 16, 64, size=t[:, n_pri:].shape)
    elif cond.startswith("corr_"):
        # BARRIDO (E-3). Mueve el episodio de a poco, con las ajenas SIEMPRE en U(0,11) — que por E-1
        # no cambia nada. Con corrimiento chico parte del episodio queda arriba del umbral y parte
        # abajo, que es lo que hace evaluable la desagregacion de C-1.
        t2[:, :n_pri] = t[:, :n_pri] - int(cond.split("_")[1])
        t2[:, n_pri:] = rng.integers(0, 12, size=t[:, n_pri:].shape)
    else:
        raise ValueError(cond)
    return np.where(mk, t2, 0)


def leer(params, archivo, turnos, cons, mask, donde, pertenece=None):
    """`modelo.responder`, con la distribución de lectura guardada. Devuelve (logits, p).

    2026-09-08 · FALTABA `marca_pert`. Esta funcion reimplementa `responder` a mano para poder
    guardar la distribucion de lectura, y al hacerlo se quedo sin el bit de pertenencia. Las tres
    unidades `rp3` (entrenadas CON el bit) se midieron entonces con el bit APAGADO, o sea con una
    arquitectura que no es la suya: acierto 0,2969-0,5781 en la sonda contra `vigente` 0,9725-0,9824
    en su propio entrenamiento. La recuperacion aguantaba —la clave sigue llevando el sello— pero la
    respuesta se caia, que es la firma exacta de medir un modelo sin la entrada de la que aprendio a
    depender.

    Es la misma familia que el `M.KQ` de `ser.py` y el `M.SELLO` de este mismo archivo (RETOMAR §3),
    con una diferencia que vale anotar: `conf_ckpt.aplicar` NO lo cubre y no lo puede cubrir, porque
    `pert` no es un global del modulo sino un ARGUMENTO de llamada. La regla del §3 hay que leerla
    mas ancha: la config decide globals Y argumentos.
    """
    a = params["arch"]
    ak = archivo @ a["kw"] + M.sello(a, turnos, mask) + M.marca_pert(a, pertenece)
    av = archivo @ a["vw"]
    penal = jnp.where(mask, 0.0, -1e9)[:, None, :]
    guardado = {}

    def lectura(h):
        q = h @ a["qr"]
        sim = jnp.einsum("btd,bnd->btn", q, ak) / jnp.sqrt(h.shape[-1]) + penal
        p = jax.nn.softmax(sim, -1)
        guardado["p"] = p
        return jnp.einsum("btn,bnd->btd", p, av) @ a["wo"]

    h = M.tronco(params, cons, lectura, 0, donde)
    lg = M.ln(params["ln_f"], h) @ params["head"]["w"] + params["head"]["b"]
    return lg, guardado["p"]


def correr(params, cfg, ses_extra, lotes, B, semilla, conds=CONDICIONES):
    rng = np.random.default_rng(semilla)
    n_pri = 4 * DAT.E_MAX
    acc = {c: {k: [] for k in ("masa_extra", "proporcion", "masa_correcta", "recup", "acierto",
                               "recup_bajo", "recup_alto")} for c in conds}
    for _ in range(lotes):
        s, c, t, mk, q, pq, tg, tp, meta, orig, hq = DAT.lote(
            rng, B, nivel=cfg["nivel"], p_vieja=0.0, p_nose=0.0, con_meta=True, con_origen=True,
            n_ses_extra=ses_extra)
        archivo = M.escribir(params, jnp.array(s), jnp.array(c))     # una sola vez: lo caro
        es_extra = np.zeros(t.shape[1], bool); es_extra[n_pri:] = True
        for cond in conds:
            tc = turnos_de(cond, t, mk, n_pri, np.random.default_rng(semilla + 7))
            # El bit va SOLO si el checkpoint se entreno con el. `pert_de` replica lo que hace
            # `entrenar.py:pert_de`: las primeras 4*E_MAX entradas son el episodio en curso.
            pert = None
            if cfg.get("pert"):
                m0 = np.zeros(mk.shape[-1], bool); m0[:n_pri] = True
                pert = jnp.array(np.broadcast_to(m0, mk.shape))
            lg, p = leer(params, archivo, jnp.array(tc), jnp.array(q), jnp.array(mk), cfg["donde"],
                         pertenece=pert)
            pred = np.array(jnp.take_along_axis(lg, jnp.array(pq)[:, None, None], axis=1)
                            [:, 0, :].argmax(-1))
            p = np.array(p)[np.arange(B), np.array(pq)]              # (B, N)
            for b in range(B):
                escritos = mk[b]
                if not escritos.any():
                    continue
                corr = (orig[b] == hq[b]) & escritos
                acc[cond]["masa_extra"].append(float(p[b][escritos & es_extra].sum()))
                acc[cond]["proporcion"].append(float((escritos & es_extra).sum() / escritos.sum()))
                acc[cond]["acierto"].append(float(pred[b] == tg[b]))
                if corr.any():
                    ok = float(bool(corr[int(np.argmax(np.where(escritos, p[b], -1)))]))
                    acc[cond]["masa_correcta"].append(float(p[b][corr].sum()))
                    acc[cond]["recup"].append(ok)
                    # C-1: la desagregación que separa umbral absoluto de orden relativo — de qué
                    # lado del umbral 24 quedó la entrada correcta EN ESTA condición.
                    lado = "recup_bajo" if int(tc[b][corr][0]) < UMBRAL else "recup_alto"
                    acc[cond][lado].append(ok)
    out = {}
    for cond, d in acc.items():
        r = {k: (float(np.mean(v)) if v else float("nan")) for k, v in d.items()}
        r["indice"] = r["masa_extra"] / r["proporcion"] if r["proporcion"] else float("nan")
        r["n"] = len(d["masa_extra"])
        r["n_bajo"], r["n_alto"] = len(d["recup_bajo"]), len(d["recup_alto"])
        out[cond] = r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", nargs="+")
    ap.add_argument("--ses-extra", type=int, default=26)
    ap.add_argument("--lotes", type=int, default=4)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--semilla", type=int, default=90000)
    ap.add_argument("--corrimientos", type=int, nargs="*", default=[],
                    help="BARRIDO (E-3): corre `corr_D` para cada D, con las ajenas fijas en U(0,11)")
    ap.add_argument("--salida", default="")
    a = ap.parse_args()

    todo = {"prereg": "aabb4b20", "ses_extra": a.ses_extra, "lotes": a.lotes, "batch": a.batch,
            "unidades": {}}
    for ruta in a.ckpt:
        b = pickle.load(open(ruta, "rb"))
        cfg, params = b["config"], b["params"]
        M.KQ = cfg.get("kernel_q", 3)
        # 2026-09-06: los checkpoints de la campania del sello relativo son `sello=rel`. Sin leerlo
        # de la config, este instrumento los mediria con la indexacion ABSOLUTA —o sea con una
        # arquitectura que no es la suya— y el resultado no diria nada sobre la unidad medida. Misma
        # familia que el `M.KQ` que le faltaba a `ser.py`, cazada el mismo dia.
        M.SELLO = cfg.get("sello", "abs")
        print(f"\n{ruta} · paso {b.get('paso')} · kernel_q={M.KQ} · donde={cfg['donde']} · "
              f"sello={M.SELLO} · pert={cfg.get('pert', False)} · "
              f"ses_extra={cfg.get('ses_extra', 0)}")
        conds = tuple(f"corr_{d}" for d in a.corrimientos) if a.corrimientos else CONDICIONES
        r = correr(params, cfg, a.ses_extra, a.lotes, a.batch, a.semilla, conds)
        print(f"{'condicion':>11} {'indice':>8} {'masa_corr':>10} {'recup':>7} {'acierto':>8} "
              f"{'r<24':>7} {'r>=24':>7}  (n bajo/alto)")
        for cond in conds:
            x = r[cond]
            print(f"{cond:>11} {x['indice']:8.4f} {x['masa_correcta']:10.4f} {x['recup']:7.4f} "
                  f"{x['acierto']:8.4f} {x['recup_bajo']:7.4f} {x['recup_alto']:7.4f}"
                  f"   {x['n_bajo']}/{x['n_alto']}")
        todo["unidades"][ruta] = r
    if a.salida:
        json.dump(todo, open(a.salida, "w"), indent=1)
        print(f"\n-> {a.salida}")


if __name__ == "__main__":
    main()
