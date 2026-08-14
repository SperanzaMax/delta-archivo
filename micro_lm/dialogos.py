"""MICRO-LM · mostrar lo que el modelo contesta, en castellano legible.

    python dialogos.py corridas_20260814/n4_s0.pkl --n 8

Una accuracy de 0,9881 no deja ver si el modelo entendio algo o si el idioma es tan chico que el
acierto es trivial. Esto imprime el episodio como lo leeria un humano —sesion por sesion—, la
pregunta, lo que contesto el modelo y lo que correspondia. La respuesta es UN token, asi que no hay
juez ni parser que pueda equivocarse: es la leccion del 12-ago, cuando 10 de 11 «abstenciones»
resultaron ser el parser y no el modelo.
"""
import argparse
import pickle

import numpy as np
import jax.numpy as jnp

import datos as DAT
import idioma as I
import entrenar as E

ETIQUETA = {0: "vigente", 1: "anterior", 2: "sin respuesta (entidad ausente)",
            3: "sin respuesta (relacion no dicha)"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pesos")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--nivel", type=int, default=None)
    ap.add_argument("--p-nose", type=float, default=None)
    ap.add_argument("--semilla", type=int, default=12345)
    a = ap.parse_args()

    with open(a.pesos, "rb") as f:
        bulto = pickle.load(f)
    params, cfg = bulto["params"], bulto["config"]
    nivel = a.nivel if a.nivel is not None else cfg["nivel"]
    p_nose = a.p_nose if a.p_nose is not None else cfg.get("p_nose", 0.0)
    print(f"pesos: {a.pesos}\nnivel {nivel} · semilla de entrenamiento {cfg['semilla']} · "
          f"{cfg['pasos']} pasos · p_nose {p_nose}\n")

    rng = np.random.default_rng(a.semilla)
    aciertos = 0
    for i in range(a.n):
        ses, cortes, turnos, mask, cons, pos, tgt, tipo = DAT.lote(
            rng, 1, nivel=nivel, n_hechos=4, n_sesiones=4, p_nose=p_nose)
        pred = int(E.predecir(params, jnp.array(ses), jnp.array(cortes), jnp.array(turnos),
                              jnp.array(mask), jnp.array(cons), jnp.array(pos))[0])
        ok = pred == tgt[0]
        aciertos += ok

        print("=" * 72)
        for s in range(ses.shape[1]):
            toks = [I.ITOS[t] for t in ses[0, s] if t != DAT.PAD]
            if len(toks) > 1:
                print(f"  sesion {s+1}   USUARIO  {' '.join(toks[1:])}")
        preg = " ".join(I.ITOS[t] for t in cons[0] if t != DAT.PAD)[4:]
        print(f"\n  consulta    USUARIO  {preg}")
        print(f"              MODELO   {I.ITOS[pred]}   {'✓' if ok else '✗ (era ' + I.ITOS[tgt[0]] + ')'}"
              f"   [{ETIQUETA[int(tipo[0])]}]")
    print("=" * 72)
    print(f"\n{aciertos}/{a.n} en esta muestra (no es la metrica: para eso esta el JSON de la corrida)")


if __name__ == "__main__":
    main()
