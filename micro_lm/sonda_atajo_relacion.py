"""¿El modelo mira la ENTIDAD, o solo la relacion? La prueba directa (2026-08-22).

`INFORME_BIMODALIDAD_20260822.md` propone que la bimodalidad entre semillas es el **atajo de la
relacion**: el modelo encuentra el hecho por la relacion y, cuando dos entradas la comparten, tira
una moneda. La evidencia hasta ahora es indirecta —`ac_unica = 1,0000` y `ac_rep = 0,5317` en `s1`,
y la curva plana desde el paso 8000—.

Esta sonda lo prueba de frente y **sin entrenar nada**: toma los episodios donde la relacion
preguntada se repite, arma **la misma consulta cambiando la entidad** por la otra que comparte esa
relacion, y compara las dos respuestas.

  · Si el modelo usa **solo la relacion**, las dos consultas son indistinguibles para el y contesta
    **lo mismo** a las dos. Como una de las dos respuestas correctas es distinta de la otra, acertar
    la mitad de las veces es el maximo posible: es exactamente el 0,5317 medido.
  · Si el modelo usa la **entidad**, contesta **distinto** a cada una.

No hay forma de que un modelo que condiciona en la entidad conteste lo mismo a las dos, ni de que uno
que la ignora conteste distinto. La metrica —fraccion de pares con la MISMA respuesta— separa las dos
hipotesis sin ambiguedad y sin depender de que la respuesta sea correcta.

    python sonda_atajo_relacion.py --unidades p3_s0,p3_s1,p3_s2

Reusa `sonda_roundtrip.pos_entidad`, que ya verifica que haya exactamente un token de entidad en la
consulta y falla ruidosamente si no.
"""
import argparse
import json
import os
import pickle

import jax
import jax.numpy as jnp
import numpy as np

import datos as DAT
import entrenar as E
import idioma as I
import sonda_roundtrip as RT

AQUI = os.path.dirname(os.path.abspath(__file__))
NOSE = I.STOI["NOSE"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir-ckpt", default=os.path.join(AQUI, "ckpts", "qc_congelados"))
    ap.add_argument("--unidades", default="p3_s0,p3_s1,p3_s2")
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--salida", default=os.path.join(AQUI, "atajo_relacion_20260822.json"))
    A = ap.parse_args()

    print("¿MIRA LA ENTIDAD O SOLO LA RELACION? · misma consulta, la otra entidad que comparte la relacion")
    print(f"{A.n * A.batch} muestras por unidad · solo episodios con relacion REPETIDA\n")
    print(f"{'unidad':<8} {'donde':<5} | {'n pares':>8} {'MISMA respuesta':>16} {'acierto':>9} | lectura")
    print("-" * 78)

    res = {}
    for uni in A.unidades.split(","):
        ck = os.path.join(A.dir_ckpt, f"{uni}.pkl")
        if not os.path.exists(ck):
            print(f"{uni:<8} sin checkpoint")
            continue
        with open(ck, "rb") as f:
            d = pickle.load(f)
        params = jax.tree_util.tree_map(jnp.asarray, d["params"])
        cfg = d["config"]
        E._DONDE = cfg.get("donde", "pre")          # antes del primer trace
        predecir = E.predecir_cabeza if cfg.get("abst", "token") == "cabeza" else E.predecir
        rng = np.random.default_rng(RT.SEM_PRUEBA + cfg["semilla"])

        iguales, total, aciertos = 0, 0, 0
        for _ in range(A.n):
            ses, cortes, turnos, mask, cons, pos, tgt, tipo, meta = DAT.lote(
                rng, A.batch, nivel=cfg["nivel"], n_hechos=4, n_sesiones=4, p_vieja=0.35,
                p_nose=0.0, con_meta=True)
            ipos = RT.pos_entidad(cons)
            cons2 = np.array(cons)
            usar = np.zeros(len(cons), bool)
            for b in range(len(cons)):
                m = meta[b]
                if tipo[b] >= 2 or not m["hecho"]:
                    continue
                # la OTRA entidad que comparte la relacion preguntada
                rivales = [o["ent"] for o in m["otros"] if o["rel"] == m["hecho"]["rel"]]
                if not rivales:
                    continue                      # relacion unica: no hay par que armar
                cons2[b, ipos[b]] = I.STOI[rivales[0]]
                usar[b] = True
            if not usar.any():
                continue

            args = (jnp.array(ses), jnp.array(cortes), jnp.array(turnos), jnp.array(mask))
            p1 = np.asarray(predecir(params, *args, jnp.array(cons), jnp.array(pos)))
            p2 = np.asarray(predecir(params, *args, jnp.array(cons2), jnp.array(pos)))
            iguales += int((p1[usar] == p2[usar]).sum())
            aciertos += int((p1[usar] == np.asarray(tgt)[usar]).sum())
            total += int(usar.sum())

        r = {"donde": E._DONDE, "n_pares": total,
             "misma_respuesta": iguales / max(1, total),
             "acierto": aciertos / max(1, total)}
        res[uni] = r
        print(f"{uni:<8} {r['donde']:<5} | {total:>8} {r['misma_respuesta']:>16.4f} "
              f"{r['acierto']:>9.4f} | {E._DONDE}")

    print("\n" + "-" * 78)
    print("Leer asi: «misma respuesta» ~ 1,0 = el modelo NO distingue las dos entidades (atajo puro).")
    print("          «misma respuesta» ~ 0,0 = contesta distinto a cada una (usa la entidad).")
    print("Y la relacion esperada con el acierto: si contesta lo mismo a las dos, solo una de las dos")
    print("puede ser correcta, asi que el acierto no puede pasar de ~0,5.")
    with open(A.salida, "w") as f:
        json.dump({"que_es": "prueba directa del atajo de la relacion", "unidades": res}, f, indent=1)
    print(f"\n-> {A.salida}")


if __name__ == "__main__":
    main()
