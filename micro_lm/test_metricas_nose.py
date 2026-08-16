"""¿Las métricas de abstención pueden FALLAR? Control del instrumento, sin GPU.

    python test_metricas_nose.py

Antes de gastar una asignación de GPU en la campaña de abstención hay que saber que las métricas
distinguen los modos degenerados. La leccion ya se pagó cuatro veces en este proyecto: el control
`m=1` del banco ECO daba 1,000 y estaba VACIO —con una sola candidata, acertar no requiere leer—.
**Un control que no puede fallar no controla nada.**

Se simulan cuatro modelos que no existen pero cuyo comportamiento conocemos de antemano, y se
verifica que cada métrica los separa como corresponde:

  oraculo          contesta siempre bien           -> todo perfecto
  nunca_abstiene   nunca dice NOSE                 -> nose 0, invento 1, falsa_abst 0
  siempre_abstiene dice NOSE a todo                -> nose 1, invento 0, falsa_abst 1
  azar             elige un valor cualquiera       -> todo cerca del piso

El par que importa es `nunca_abstiene` contra `siempre_abstiene`: los dos son inútiles y ninguno
sabe nada, así que **ninguna métrica sola puede premiar a los dos**. Si `nose` por sí sola diera
bien con el que se abstiene de todo, la compuerta sería un colador — por eso va con `falsa_abst`.
"""
import collections

import numpy as np

import datos as DAT
import idioma as I
from ser import clasificar

MODELOS = ("oraculo", "nunca_abstiene", "siempre_abstiene", "azar")


def predecir_falso(modo, tgt_tok, m, rng):
    if modo == "oraculo":
        return tgt_tok
    if modo == "siempre_abstiene":
        return "NOSE"
    # los dos que nunca se abstienen tienen que devolver SIEMPRE un valor concreto
    if modo == "nunca_abstiene":
        if tgt_tok != "NOSE":
            return tgt_tok                       # sabe todo lo que está, pero jamás dice «no sé»
        candidatos = [v for o in m["otros"] for v in o["versiones"]]
        return str(rng.choice(candidatos)) if candidatos else "0"
    candidatos = [v for o in m["otros"] for v in o["versiones"]]
    if m["hecho"]:
        candidatos += m["hecho"]["versiones"]
    return str(rng.choice(candidatos)) if candidatos else "0"


def medir(modo, n=1500, p_nose=0.4, nivel=4, semilla=99):
    rng = np.random.default_rng(semilla)
    rmod = np.random.default_rng(semilla + 1)
    c = collections.Counter()
    vistos = 0
    while vistos < n:
        B = min(64, n - vistos)
        *_, tgt, tipo, meta = DAT.lote(rng, B, nivel=nivel, n_hechos=4, n_sesiones=4,
                                       p_nose=p_nose, con_meta=True)
        for i in range(B):
            tg = I.ITOS[int(tgt[i])]
            pred = predecir_falso(modo, tg, meta[i], rmod)
            c[clasificar(pred, tg, meta[i])] += 1
        vistos += B

    total = sum(c.values())
    sin_resp = c["acierto_nose"] + c["invento"]
    con_resp = total - sin_resp
    return {
        "acierto": c["acierto"] / max(1, con_resp),
        "nose": c["acierto_nose"] / max(1, sin_resp),
        "falsa_abst": c["abstencion"] / max(1, con_resp),
        "invento": c["invento"] / max(1, sin_resp),
        "SER": (c["err_version"] + c["err_identidad"] + c["err_fuera"] + c["invento"]) / total,
    }


def main():
    print("control del instrumento · p_nose = 0.4 · nivel 4 · n = 1500\n")
    cols = ("acierto", "nose", "falsa_abst", "invento", "SER")
    print(f"{'modelo':<18}" + "".join(f"{c:>12}" for c in cols))
    res = {}
    for modo in MODELOS:
        res[modo] = medir(modo)
        print(f"{modo:<18}" + "".join(f"{res[modo][c]:>12.4f}" for c in cols))

    print("\ncomprobaciones (cada una puede fallar):")
    ok = True

    def chequeo(nombre, cond):
        nonlocal ok
        print(f"  {'✓' if cond else '✗'}  {nombre}")
        ok = ok and cond

    o, na, sa, az = res["oraculo"], res["nunca_abstiene"], res["siempre_abstiene"], res["azar"]
    chequeo("el oráculo saca nose = 1 y falsa_abst = 0", o["nose"] > 0.99 and o["falsa_abst"] < 0.01)
    chequeo("el que NUNCA se abstiene saca nose = 0", na["nose"] < 0.01)
    chequeo("y su invento es total", na["invento"] > 0.99)
    chequeo("el que SIEMPRE se abstiene saca nose = 1 (por eso `nose` sola no alcanza)",
            sa["nose"] > 0.99)
    chequeo("pero lo delata falsa_abst = 1", sa["falsa_abst"] > 0.99)
    chequeo("y su acierto se desploma", sa["acierto"] < 0.01)
    chequeo("el azar queda por debajo del oráculo en acierto", az["acierto"] < o["acierto"])
    chequeo("la compuerta (nose ≥ 0,50 Y falsa_abst ≤ 0,10) RECHAZA al que se abstiene de todo",
            not (sa["nose"] >= 0.50 and sa["falsa_abst"] <= 0.10))
    chequeo("y RECHAZA al que nunca se abstiene",
            not (na["nose"] >= 0.50 and na["falsa_abst"] <= 0.10))
    chequeo("y ACEPTA al oráculo",
            o["nose"] >= 0.50 and o["falsa_abst"] <= 0.10)

    print("\n" + ("INSTRUMENTO OK · la compuerta separa lo que tiene que separar" if ok
                  else "⚠️  EL INSTRUMENTO NO SIRVE COMO ESTÁ"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
