"""A5 · E-2 (mecanicista) y E-4 (colapso al prior), del §5 de `PREREG_BLANCO_ERROR.md`.

    python a5_e2_e4.py ckpts/b3_s0.pkl ckpts/p3_s0.pkl --n 4000

**E-2.** El AUC del logit de la cabeza sobre el blanco «¿me voy a equivocar si contesto?» sube
≥ 0,05 contra la gemela, en ≥ 2/3. Referencias del control ya medidas y citadas en el prereg: 0,7068
y 0,8105.

La etiqueta es la del propio blanco de A5 (`entrenar.py:perdida_cabeza` con `_BLANCO == "error"`):
el argmax de valor **con `NOSE` excluido** contra el target. Se calcula igual para las dos
condiciones, porque el punto de E-2 es justamente preguntar si la cabeza del tratamiento ordena
mejor ESE eje que la del control, que nunca lo tuvo como blanco.

**E-4.** Riesgo declarado, **sin criterio de éxito**: media y desvío del logit, y la fracción que
cruza el umbral. El §5 fija cómo se lee:

> Si el desvío del logit es < 0,1 y la media está pegada a `logit(tasa base)`, el resultado es
> «colapsó al prior» y no «el blanco no sirve». Son cosas distintas y hay que poder separarlas.

Eso importa para elegir celda en el §6: «E-2 no, con E-4 mostrando colapso» manda a congelar el
blanco, que sería otro experimento; «ninguna, sin colapso» cierra la vía.
"""
import argparse
import json
import pickle

import numpy as np
import jax
import jax.numpy as jnp

import idioma as I
import datos as DAT
import modelo as M
from sonda_dos_detectores import auc

NOSE = I.STOI["NOSE"]


def medir(ruta, n, B, semilla, p_nose_cli):
    b = pickle.load(open(ruta, "rb"))
    params = jax.tree_util.tree_map(jnp.asarray, b["params"])
    cfg = b["config"]
    donde = cfg.get("donde", "pre")
    nivel = cfg["nivel"]
    p_nose = p_nose_cli if p_nose_cli is not None else cfg.get("p_nose", 0.0)

    @jax.jit
    def fn(params, arch, tur, cons, mask, pos):
        a = params["arch"]
        ak = arch @ a["kw"] + a["ord"][tur]
        av = arch @ a["vw"]
        pen = jnp.where(mask, 0.0, -1e9)[:, None, :]

        def lec(h):
            q = h @ a["qr"]
            sim = jnp.einsum("btd,bnd->btn", q, ak) / jnp.sqrt(h.shape[-1]) + pen
            return jnp.einsum("btn,bnd->btd", jax.nn.softmax(sim, -1), av) @ a["wo"]

        h = M.tronco(params, cons, lec, 0, donde)
        hn = M.ln(params["ln_f"], h)
        lg = hn @ params["head"]["w"] + params["head"]["b"]
        ab = (hn @ params["abst"]["w"] + params["abst"]["b"])[..., 0]
        tomar2 = lambda x: jnp.take_along_axis(x, pos[:, None, None], axis=1)[:, 0, :]
        tomar1 = lambda x: jnp.take_along_axis(x, pos[:, None], axis=1)[:, 0]
        lg_q = tomar2(lg)
        return lg_q.at[:, NOSE].set(-1e9).argmax(-1), tomar1(ab)

    rng = np.random.default_rng(semilla)
    logits, blancos = [], []
    vistos = 0
    while vistos < n:
        bs = min(B, n - vistos)
        ses, cortes, tur, mask, cons, pos, tgt, tipo = DAT.lote(
            rng, bs, nivel=nivel, n_hechos=4, n_sesiones=4, p_nose=p_nose)
        arch = M.escribir(params, jnp.array(ses), jnp.array(cortes))
        arg, ab = fn(params, arch, jnp.array(tur), jnp.array(cons), jnp.array(mask),
                     jnp.array(pos))
        # El blanco de A5, calculado IGUAL para las dos condiciones: «el argmax de valor se equivoca».
        blancos.append(np.asarray(arg) != np.asarray(tgt))
        logits.append(np.asarray(ab))
        vistos += bs

    return np.concatenate(logits), np.concatenate(blancos), cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tratamiento")
    ap.add_argument("control")
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--B", type=int, default=64)
    ap.add_argument("--semilla", type=int, default=54321)
    ap.add_argument("--p-nose", type=float, default=None)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    out = {}
    for rol, ruta in (("tratamiento", a.tratamiento), ("control", a.control)):
        lg, bl, cfg = medir(ruta, a.n, a.B, a.semilla, a.p_nose)
        base = float(bl.mean())
        # logit(tasa base): el valor al que se pega una cabeza que aprendio el prior y nada mas.
        lb = float(np.log(base / (1 - base))) if 0 < base < 1 else float("nan")
        out[rol] = {
            "pesos": ruta, "blanco": cfg.get("blanco", "ausencia"),
            "auc": float(auc(bl, lg)), "tasa_base_error": base, "logit_tasa_base": lb,
            "media_logit": float(lg.mean()), "desvio_logit": float(lg.std()),
            "frac_cruza": float((lg > 0).mean()),
        }
        print(f"{rol:<12} {ruta}  ·  blanco {out[rol]['blanco']}")
        print(f"    AUC del logit sobre «¿me voy a equivocar?»  {out[rol]['auc']:.4f}")
        print(f"    tasa base de error {base:.4f}  ·  logit(base) {lb:+.4f}")
        print(f"    logit: media {out[rol]['media_logit']:+.4f} · desvio "
              f"{out[rol]['desvio_logit']:.4f} · cruza el umbral {out[rol]['frac_cruza']:.4f}")

    d = out["tratamiento"]["auc"] - out["control"]["auc"]
    print(f"\n  ── E-2 ────────────────────────────────────────────")
    print(f"    AUC trat {out['tratamiento']['auc']:.4f} · ctrl {out['control']['auc']:.4f} · "
          f"delta {d:+.4f}")
    print(f"    requerido: sube >= 0,05   →  {'CUMPLE' if d >= 0.05 else 'NO CUMPLE'}")

    t = out["tratamiento"]
    colapso = t["desvio_logit"] < 0.1 and abs(t["media_logit"] - t["logit_tasa_base"]) < 0.5
    print(f"\n  ── E-4 · riesgo declarado, sin criterio de exito ──")
    print(f"    desvio {t['desvio_logit']:.4f} (< 0,1 seria colapso) · "
          f"media {t['media_logit']:+.4f} vs logit(base) {t['logit_tasa_base']:+.4f}")
    print(f"    → {'COLAPSO AL PRIOR' if colapso else 'NO colapso: la cabeza discrimina, no se pego a la tasa base'}")
    out["E2_delta"] = d
    out["E2_cumple"] = bool(d >= 0.05)
    out["E4_colapso"] = bool(colapso)

    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=1, ensure_ascii=False)
        print(f"\n  -> {a.json}")


if __name__ == "__main__":
    main()
