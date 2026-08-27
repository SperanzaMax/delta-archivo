"""FASE 0 de la ausencia de la RELACIÓN · ¿hay señal, controlando por entidad?

    python relacion_fase0.py ckpts/v3_s0.pkl --n 6000 --json rel_fase0_v3_s0.json

Evalúa el §3 de `PREREG_AUSENCIA_RELACION.md` (SHA `86870655…`, congelado antes de escribir esto).

## La pregunta, y por qué no está ya contestada

`INFORME_SCORE_ARCHIVO_20260816.md` midió el score del archivo y dio azar exacto (0,4984 / 0,5022).
Pero ese eje era **grueso**: «con respuesta» contra «sin respuesta», y «sin respuesta» mezcla dos
poblaciones con mecanismos distintos (`idioma.py:222-223`):

  · `nose_ent` — la entidad nunca se nombró. No hay entrada que matchear.
  · `nose_rel` — la entidad SÍ está, con otra relación. Hay entrada, y hay que darse cuenta de que
    no es la pedida.

Promediar las dos en un solo AUC puede dar 0,50 **aunque una de las dos tenga señal**. Acá el eje se
controla por entidad:

  · positivos = `tipo == 0` (vigente): entidad presente, relación presente
  · negativos = `tipo == 3` (nose_rel): entidad presente, relación AUSENTE
  · `tipo == 2` (`nose_ent`) queda EXCLUIDO, y ésa es toda la diferencia con el 16-ago.

## Cómo se capturan las señales

`modelo.tronco` recibe la lectura como clausura, y `donde` sólo decide **qué estado** se le pasa
(`modelo.py:34-69`), no cómo lee. Así que pasarle una clausura propia que guarde los intermedios
reproduce la lectura real de la unidad, con su `donde` incluido, sin tocar `modelo.py` — que además
está en uso por A5 en Colab y no conviene mover.

Las cinco señales están fijadas en el §3 del pre-registro. Se reporta cada una por separado (R-2,
descriptiva) y el bloque completo, que es **la única que decide R-1**. El §8 prohíbe explícitamente
elegir después la que mejor quede.
"""
import argparse
import json
import pickle

import numpy as np
import jax
import jax.numpy as jnp

import datos as DAT
import modelo as M
import entrenar as E
from sonda_dos_detectores import sonda, auc


def recolectar(ruta, n, B, semilla, nivel_cli, p_nose_cli):
    with open(ruta, "rb") as f:
        bulto = pickle.load(f)
    params, cfg = bulto["params"], bulto["config"]
    nivel = nivel_cli if nivel_cli is not None else cfg["nivel"]
    # p_nose alto a proposito: los `nose_rel` son ~la mitad de los `nose`, y con p_nose bajo la clase
    # negativa no llega a poblarse. No cambia el modelo, sólo cuantas preguntas de cada tipo se
    # sortean para MEDIRLO.
    p_nose = p_nose_cli if p_nose_cli is not None else max(0.4, cfg.get("p_nose", 0.0))
    donde = cfg.get("donde", "pre")

    def señales(params, archivo, turnos, consulta, mask_arch, pos):
        a = params["arch"]
        ak = archivo @ a["kw"] + a["ord"][turnos]
        av = archivo @ a["vw"]
        penal = jnp.where(mask_arch, 0.0, -1e9)[:, None, :]
        cap = {}

        def lectura(h):
            q = h @ a["qr"]
            sim = jnp.einsum("btd,bnd->btn", q, ak) / jnp.sqrt(h.shape[-1]) + penal
            w = jax.nn.softmax(sim, -1)
            leido = jnp.einsum("btn,bnd->btd", w, av) @ a["wo"]
            cap["sim"], cap["w"], cap["leido"] = sim, w, leido
            return leido

        h = M.tronco(params, consulta, lectura, 0, donde)
        estado = M.ln(params["ln_f"], h)
        # CONTROL POST-HOC (2026-08-27, despues de ver que la señal vive en `estado`): la confianza
        # de la SALIDA. `estado` es el vector del que sale el logit, asi que una sonda sobre el puede
        # estar leyendo la decision que el modelo YA tomo en vez de informacion sobre la ausencia.
        # Es el mismo control que el P-2 del informe del score del archivo (16-ago), que exigia que
        # la señal del archivo superara a la confianza de salida por 0,03.
        lg = estado @ params["head"]["w"] + params["head"]["b"]
        pr = jax.nn.softmax(lg, -1)
        tomar = lambda x: jnp.take_along_axis(x, pos[:, None, None], axis=1)[:, 0, :]
        sim_q, w_q = tomar(cap["sim"]), tomar(cap["w"])
        orden = jnp.sort(sim_q, axis=-1)
        return {
            "s_max": orden[:, -1],
            "s_margen": orden[:, -1] - orden[:, -2],
            "s_ent": -(w_q * jnp.log(w_q + 1e-9)).sum(-1),
            "leido": tomar(cap["leido"]),
            "estado": tomar(estado),
            "c_prob": tomar(pr).max(-1, keepdims=True),
        }

    señales_jit = jax.jit(señales)

    rng = np.random.default_rng(semilla)
    acum = {k: [] for k in ("s_max", "s_margen", "s_ent", "leido", "estado", "c_prob")}
    ys, ciegos = [], []
    vistos = 0
    while vistos < n:
        b = min(B, n - vistos)
        ses, cortes, turnos, mask, cons, pos, tgt, tipo, meta = DAT.lote(
            rng, b, nivel=nivel, n_hechos=4, n_sesiones=4, p_nose=p_nose, con_meta=True)
        arch = M.escribir(params, jnp.array(ses), jnp.array(cortes))
        s = señales_jit(params, arch, jnp.array(turnos), jnp.array(cons), jnp.array(mask),
                        jnp.array(pos))
        s = {k: np.asarray(v) for k, v in s.items()}
        for i in range(b):
            t = int(tipo[i])
            if t not in (0, 3):          # `anterior` y `nose_ent` quedan fuera (§3)
                continue
            for k in acum:
                acum[k].append(s[k][i])
            ys.append(t == 0)            # positivo = la relacion pedida ESTA
            ciegos.append([int(mask[i].sum()), int(pos[i]), int(cortes[i].sum())])
        vistos += b

    X = {k: np.array(v, np.float32) for k, v in acum.items()}
    for k in ("s_max", "s_margen", "s_ent", "c_prob"):
        X[k] = X[k].reshape(-1, 1)
    return (X, np.array(ys, bool), np.array(ciegos, np.float32),
            {"pesos": ruta, "nivel": nivel, "semilla_modelo": cfg["semilla"],
             "paso": bulto.get("paso"), "donde": donde, "abst": cfg.get("abst", "token"),
             "p_nose_medicion": p_nose})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pesos")
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--B", type=int, default=64)
    ap.add_argument("--semilla", type=int, default=54321)
    ap.add_argument("--nivel", type=int, default=None)
    ap.add_argument("--p-nose", type=float, default=None)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    X, y, ciego, cfg = recolectar(a.pesos, a.n, a.B, a.semilla, a.nivel, a.p_nose)
    print(f"{a.pesos}  ·  nivel {cfg['nivel']} · semilla {cfg['semilla_modelo']} · "
          f"paso {cfg['paso']} · lectura {cfg['donde']}")
    print(f"casos: {len(y)}  ·  vigente {int(y.sum())}  ·  nose_rel {int((~y).sum())}")

    base = float(y.mean())
    print(f"  tasa base (positivos): {base:.4f}")
    if y.sum() < 30 or (~y).sum() < 30:
        # La guarda que el 27-ago evito leer una AUC sobre 4 errores en lat2 (§7 del prereg).
        print(f"  !! una clase con menos de 30 casos. No es interpretable y no se reporta.")
        return

    corte = len(y) // 2
    ia, ip = slice(0, corte), slice(corte, len(y))
    rng = np.random.default_rng(a.semilla + 1)

    bloque = np.c_[tuple(X[k] for k in ("s_max", "s_margen", "s_ent", "leido", "estado"))]
    auc_bloque = auc(y[ip], sonda(bloque[ia], y[ia], bloque[ip]))
    auc_perm = auc(y[ip], sonda(bloque[ia], rng.permutation(y[ia]), bloque[ip]))
    auc_ciego = auc(y[ip], sonda(ciego[ia], y[ia], ciego[ip]))
    por_señal = {k: float(auc(y[ip], sonda(X[k][ia], y[ia], X[k][ip]))) for k in X}

    print(f"\n  ── FASE 0 · relacion presente vs ausente, con la ENTIDAD presente en ambos ──")
    print(f"    AUC BLOQUE completo   {auc_bloque:.4f}   ← R-1, la unica que decide")
    print(f"    AUC etiq PERMUTADA    {auc_perm:.4f}   ← R-0 bloqueante, tiene que ser <= 0,55")
    print(f"    AUC sonda CIEGA       {auc_ciego:.4f}   ← fuga por longitud (§7)")
    ganancia = por_señal["estado"] - por_señal["c_prob"]
    print(f"\n  CONTROL POST-HOC · ¿`estado` aporta algo sobre la confianza de salida?")
    print(f"    AUC estado {por_señal['estado']:.4f}  ·  AUC c_prob {por_señal['c_prob']:.4f}  ·  "
          f"ganancia {ganancia:+.4f}")
    print(f"    {'aporta' if ganancia >= 0.03 else 'NO aporta: es la decision ya tomada, leida de otra forma'}")
    print(f"\n  R-2 · por señal, DESCRIPTIVO (dice donde vive, no si existe):")
    for k, v in sorted(por_señal.items(), key=lambda kv: -kv[1]):
        print(f"    {k:<10} {v:.4f}")

    r0 = auc_perm <= 0.55
    r1 = auc_bloque >= 0.65
    fuga = auc_ciego >= 0.65
    print(f"\n    R-0 (permutada <= 0,55)   {'CUMPLE' if r0 else 'FALLA'}")
    print(f"    R-1 (bloque >= 0,65)      {'CUMPLE' if r1 else 'NO CUMPLE'}")
    print(f"    sin fuga por longitud     {'sí' if not fuga else 'NO — R-1 no interpretable'}")

    if not r0:
        print("\n  → R-0 falla. No se lee nada mas (§3).")
    elif fuga:
        print("\n  → La sonda ciega alcanza el umbral sola. R-1 no es interpretable (§7).")
    elif r1:
        print("\n  → HAY señal controlando por entidad. Esta unidad habilita su mitad de R-1; "
              "hacen falta 2 de 3 para abrir la condicion.")
    else:
        print("\n  → Sin señal. Si se repite en 2 de 3, el §6 manda NO entrenar nada, y el negativo "
              "del 16-ago queda REFORZADO: no era un artefacto de mezclar poblaciones.")

    if a.json:
        with open(a.json, "w") as f:
            json.dump({**cfg, "n_casos": int(len(y)), "tasa_base": base,
                       "auc_bloque": float(auc_bloque), "auc_permutada": float(auc_perm),
                       "auc_ciega": float(auc_ciego), "auc_por_señal": por_señal,
                       "R0_cumple": bool(r0), "R1_cumple": bool(r1),
                       "fuga": bool(fuga)}, f, indent=1, ensure_ascii=False)
        print(f"\n  -> {a.json}")


if __name__ == "__main__":
    main()
