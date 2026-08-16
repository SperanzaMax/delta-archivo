"""¿Se pueden apagar las alucinaciones SIN reentrenar? · abstención por confianza propia

    python mitigar.py ckpts/n4_s0.pkl --n 3000

La campaña de abstención entrena al modelo para que diga `NOSE`. Esta prueba pregunta algo distinto
y mucho más barato: **el modelo que YA tenemos, ¿sabe internamente cuándo está por equivocarse?**

Si la respuesta correcta se distingue por la confianza de la salida, entonces la abstención no hay
que enseñarla: se lee. Y sale gratis sobre los checkpoints ya entrenados.

Tres señales, todas de la misma distribución de salida:

  `prob`     probabilidad del token elegido. La más obvia.
  `margen`   diferencia entre el primero y el segundo. Suele separar mejor que la probabilidad
             absoluta, porque no la arrastra el tamaño del vocabulario.
  `entropia` cuán repartida está la distribución (se usa negada, para que más alto sea más seguro).

**La compuerta va primero y puede fallar:** se mide el AUC de separar aciertos de errores usando
cada señal. Con AUC ≈ 0,5 el modelo NO sabe cuándo se equivoca —su confianza es la misma acertando
que errando— y entonces ningún umbral puede rescatar nada; el resto del análisis sobraría. Es la
lección del 12-ago, cuando un AUC de 0,97 convivía con un top-1 de 0,13: se mide lo que decide, no
lo que quede bien.

Después, la curva riesgo-cobertura: barriendo el umbral, cuánto SER se elimina y cuántas respuestas
buenas se pierden en el camino. Es el intercambio que importa — un modelo que se calla siempre tiene
SER 0 y no sirve para nada.
"""
import argparse
import collections
import pickle

import numpy as np
import jax.numpy as jnp
import jax

import datos as DAT
import idioma as I
import entrenar as E
from ser import clasificar


def auc(pos, neg):
    """AUC de Mann-Whitney: probabilidad de que un acierto tenga más confianza que un error."""
    if not len(pos) or not len(neg):
        return float("nan")
    todo = np.concatenate([pos, neg])
    r = np.argsort(np.argsort(todo)) + 1
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pesos")
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--B", type=int, default=64)
    ap.add_argument("--nivel", type=int, default=None)
    ap.add_argument("--p-nose", type=float, default=None)
    ap.add_argument("--semilla", type=int, default=31415)
    a = ap.parse_args()

    with open(a.pesos, "rb") as f:
        bulto = pickle.load(f)
    params, cfg = bulto["params"], bulto["config"]
    nivel = a.nivel if a.nivel is not None else cfg["nivel"]
    p_nose = a.p_nose if a.p_nose is not None else cfg.get("p_nose", 0.0)

    rng = np.random.default_rng(a.semilla)
    señales = {k: [] for k in ("prob", "margen", "entropia")}
    cats, vistos = [], 0

    while vistos < a.n:
        B = min(a.B, a.n - vistos)
        ses, cortes, turnos, mask, cons, pos, tgt, tipo, meta = DAT.lote(
            rng, B, nivel=nivel, n_hechos=4, n_sesiones=4, p_nose=p_nose, con_meta=True)
        lg = E.logits_de(params, jnp.array(ses), jnp.array(cortes), jnp.array(turnos),
                         jnp.array(mask), jnp.array(cons), jnp.array(pos))
        p = jax.nn.softmax(lg, -1)
        orden = np.sort(np.asarray(p), -1)
        pred = np.asarray(lg).argmax(-1)
        señales["prob"].extend(orden[:, -1])
        señales["margen"].extend(orden[:, -1] - orden[:, -2])
        pl = np.asarray(p)
        señales["entropia"].extend((pl * np.log(pl + 1e-12)).sum(-1))    # negada: más alto = más seguro
        for i in range(B):
            cats.append(clasificar(I.ITOS[int(pred[i])], I.ITOS[int(tgt[i])], meta[i]))
        vistos += B

    cats = np.array(cats)
    ok = np.isin(cats, ["acierto", "acierto_nose"])
    err = ~ok
    n = len(cats)

    print(f"pesos: {a.pesos}")
    print(f"nivel {nivel} · paso {bulto.get('paso','?')} · p_nose {p_nose} · n={n}")
    print(f"acierto base {ok.mean():.4f} · SER base {err.mean():.4f}\n")

    print("COMPUERTA · ¿la confianza separa aciertos de errores?")
    mejores = {}
    for k, v in señales.items():
        v = np.array(v)
        A = auc(v[ok], v[err])
        mejores[k] = (A, v)
        veredicto = "sirve" if A >= 0.70 else ("marginal" if A >= 0.60 else "NO sirve")
        print(f"  AUC {k:<9} {A:.4f}   {veredicto}")

    k = max(mejores, key=lambda x: mejores[x][0])
    A, v = mejores[k]
    print(f"\n  mejor señal: {k} (AUC {A:.4f})")
    if A < 0.60:
        print("\n  → el modelo NO sabe cuándo se equivoca: acierta y falla con la misma confianza.")
        print("    Ningún umbral puede rescatar nada. La abstención hay que ENTRENARLA.")
        return

    # CALIBRACION HONESTA: el umbral se fija en la PRIMERA mitad y se aplica a la SEGUNDA. Elegirlo
    # sobre el mismo conjunto que se evalua es un oraculo y sobreestima: en produccion el umbral se
    # calibra antes de ver la pregunta. Sin esta separacion el resultado no vale.
    mitad = n // 2
    cal, ev_ = slice(0, mitad), slice(mitad, n)
    print("\nCURVA RIESGO-COBERTURA · umbral calibrado en la 1ª mitad, medido en la 2ª")
    print(f"  {'cobertura':>10} {'acierto':>9} {'SER':>9} {'SER evitado':>12} "
          f"{'vs abstener al azar':>20}")
    err_ev, ok_ev, v_ev = err[ev_], ok[ev_], v[ev_]
    base_err = err_ev.sum()
    n_ev = len(v_ev)
    for cob in (1.00, 0.95, 0.90, 0.80, 0.70, 0.60, 0.50):
        umbral = np.quantile(v[cal], 1 - cob)       # ← calibrado aparte
        responde = v_ev >= umbral
        if responde.sum() == 0:
            continue
        ser = (err_ev & responde).sum() / n_ev
        evitado = 1 - (err_ev & responde).sum() / max(1, base_err)
        # abstenerse al azar en la MISMA proporcion deja el SER escalado por la cobertura: es el
        # piso que hay que superar para poder decir que la señal aporta algo.
        ser_azar = err_ev.mean() * responde.mean()
        print(f"  {responde.mean():>10.2f} {(ok_ev & responde).sum()/responde.sum():>9.4f} "
              f"{ser:>9.4f} {evitado:>12.1%} {ser_azar/max(ser,1e-9):>19.2f}x")

    print("\n¿QUÉ ERRORES SE APAGAN PRIMERO? (a cobertura 0,80)")
    umbral = np.quantile(v, 0.20)
    responde = v >= umbral
    tot = collections.Counter(cats)
    quedan = collections.Counter(cats[responde])
    for c in sorted(tot):
        if c.startswith("acierto"):
            continue
        antes, despues = tot[c], quedan.get(c, 0)
        print(f"  {c:<14} {antes:>5} → {despues:<5}  se apaga el {1-despues/max(1,antes):>5.1%}")


if __name__ == "__main__":
    main()
