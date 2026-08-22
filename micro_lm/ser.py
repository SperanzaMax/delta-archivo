"""MICRO-LM · SER, el error silencioso, desagregado por tipo.

    python ser.py ckpts/n4_s0.pkl --n 2000

La §6 del diseño pide medir el error silencioso separando dos cosas que no son lo mismo:

  · error de VERSION    contesta otra version DEL HECHO QUE SE PREGUNTO (v1 cuando regia v2).
                        El modelo encontro el hecho y se equivoco de momento.
  · error de IDENTIDAD  contesta el valor de OTRA entidad. Ni siquiera fue al hecho correcto.

La distincion importa porque son fallas de mecanismos distintos —una es del orden temporal, la otra
del direccionamiento— y porque es el corte propio frente a FAMA (arXiv 2604.20006), que penaliza el
reuso de memoria invalidada sin separarlas.

SER = errores contestados CON SEGURIDAD / total. Es decir: no cuenta como error silencioso el caso
en que el modelo se abstiene. Esa es la tesis del proyecto —un error avisado cuesta una respuesta,
uno silencioso cuesta la confianza en todas las demas— y por eso la abstencion se reporta aparte y
no se mezcla con los aciertos.
"""
import argparse
import collections
import pickle

import numpy as np
import jax.numpy as jnp

import datos as DAT
import idioma as I
import entrenar as E


def clasificar(pred_tok, tgt_tok, m):
    """Devuelve la categoria de una prediccion. `m` es el meta de esa muestra."""
    if tgt_tok == "NOSE":
        # La respuesta correcta era «no esta en el archivo».
        #
        # `invento` es LA categoria del proyecto y hasta hoy no existia: el modelo contesta un valor
        # concreto para algo que nunca se dijo. Con p_nose=0 el caso no podia darse —no habia
        # preguntas sin respuesta— asi que caia en err_identidad y nadie lo notaba; con p_nose=0.4
        # va a ser el 41 % de las preguntas. Es exactamente la alucinacion que la linea quiere
        # medir: no equivocarse de version ni de dueño, sino fabricar.
        return "acierto_nose" if pred_tok == "NOSE" else "invento"
    if pred_tok == "NOSE":
        return "abstencion"          # habia respuesta y se abstuvo: el costo de saber abstenerse
    if pred_tok == tgt_tok:
        return "acierto"

    propio = m["hecho"]
    # Un valor puede aparecer en el hecho propio Y en otro; se mira primero el propio porque la
    # explicacion mas simple de contestar una version del hecho preguntado es que erro la version.
    if propio and pred_tok in propio["versiones"]:
        return "err_version"
    for o in m["otros"]:
        if pred_tok in o["versiones"]:
            return "err_identidad"
    return "err_fuera"          # ni del hecho ni de los otros: no recupero nada del archivo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pesos")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--B", type=int, default=64)
    ap.add_argument("--nivel", type=int, default=None)
    ap.add_argument("--p-nose", type=float, default=None)
    ap.add_argument("--semilla", type=int, default=54321)
    a = ap.parse_args()

    with open(a.pesos, "rb") as f:
        bulto = pickle.load(f)
    params, cfg = bulto["params"], bulto["config"]
    nivel = a.nivel if a.nivel is not None else cfg["nivel"]
    p_nose = a.p_nose if a.p_nose is not None else cfg.get("p_nose", 0.0)

    # --- la sonda corre la arquitectura y la regla de decision DEL CHECKPOINT (2026-08-22) --------
    # Las dos cosas se leen del pkl y no de flags, porque las dos son propiedades de la unidad
    # medida, no de la medicion.
    #  · `donde`: posicion de la lectura. Los ckpts anteriores al 22-ago no la traen y son `pre`.
    #  · `abst`:  con `cabeza` la abstencion sale de una salida binaria propia y `NOSE` esta EXCLUIDO
    #    del softmax de valores. Medir esas unidades con `predecir` —el argmax plano, que es lo que
    #    este script hacia— le da a `NOSE` una ruta que en el entrenamiento no tuvo, y desvia
    #    justamente el reparto de categorias que el SER existe para medir. `ser.py` es del 15-ago y
    #    la cabeza es del 18: la incompatibilidad venia de la diferencia de fechas.
    E._DONDE = cfg.get("donde", "pre")          # antes del primer trace de jax
    usa_cabeza = cfg.get("abst", "token") == "cabeza"
    predecir = E.predecir_cabeza if usa_cabeza else E.predecir
    print(f"checkpoint: nivel {nivel} · lectura {E._DONDE} · abstencion {cfg.get('abst', 'token')}")

    rng = np.random.default_rng(a.semilla)
    cuenta = collections.Counter()
    azar = collections.Counter()
    por_tipo = collections.defaultdict(collections.Counter)
    vistos = 0

    while vistos < a.n:
        B = min(a.B, a.n - vistos)
        ses, cortes, turnos, mask, cons, pos, tgt, tipo, meta = DAT.lote(
            rng, B, nivel=nivel, n_hechos=4, n_sesiones=4, p_nose=p_nose, con_meta=True)
        pred = predecir(params, jnp.array(ses), jnp.array(cortes), jnp.array(turnos),
                        jnp.array(mask), jnp.array(cons), jnp.array(pos))
        for i in range(B):
            cat = clasificar(I.ITOS[int(pred[i])], I.ITOS[int(tgt[i])], meta[i])
            cuenta[cat] += 1
            por_tipo[meta[i]["tipo"]][cat] += 1
            # CONTROL, sin el cual el reparto version/identidad no significa nada: con pocos valores
            # en juego, «el valor de otra entidad» se acierta por azar mas seguido que «otra version
            # del mismo hecho», simplemente porque hay mas entidades ajenas que versiones propias.
            # Se acumula el reparto que daria elegir uniformemente entre los valores del archivo.
            if cat.startswith("err"):
                m = meta[i]
                tg = I.ITOS[int(tgt[i])]
                propios = len([v for v in (m["hecho"]["versiones"] if m["hecho"] else []) if v != tg])
                ajenos = sum(len(o["versiones"]) for o in m["otros"])
                if propios + ajenos:
                    azar["version"] += propios / (propios + ajenos)
                    azar["identidad"] += ajenos / (propios + ajenos)
                    azar["n"] += 1
        vistos += B

    n = sum(cuenta.values())
    err_seguro = (cuenta["err_version"] + cuenta["err_identidad"] + cuenta["err_fuera"]
                  + cuenta["invento"])
    # Denominadores separados: las dos caras de la abstencion no se miden sobre el mismo universo.
    sin_resp = cuenta["acierto_nose"] + cuenta["invento"]
    con_resp = n - sin_resp

    print(f"pesos: {a.pesos}")
    print(f"nivel {nivel} · semilla {cfg['semilla']} · paso {bulto.get('paso', '?')} · "
          f"p_nose {p_nose} · n={n}\n")
    print(f"  acierto            {cuenta['acierto']/max(1,con_resp):.4f}   (sobre las {con_resp} que SÍ tenían respuesta)")
    if sin_resp:
        print(f"  nose               {cuenta['acierto_nose']/sin_resp:.4f}   "
              f"dijo NOSE cuando no estaba  ← la mitad que importa")
        print(f"  falsa_abst         {cuenta['abstencion']/max(1,con_resp):.4f}   "
              f"dijo NOSE habiendo respuesta ← lo que cuesta")
        print(f"  invento            {cuenta['invento']/sin_resp:.4f}   "
              f"fabricó un valor que nadie dijo ← la alucinación")
    else:
        print(f"  nose                  n/a   (p_nose=0: no hubo preguntas sin respuesta)")
    print(f"\n  ── SER             {err_seguro/n:.4f}   ← error contestado con seguridad")
    print(f"     err_version     {cuenta['err_version']/n:.4f}   otra version del hecho preguntado")
    print(f"     err_identidad   {cuenta['err_identidad']/n:.4f}   el valor de OTRA entidad")
    print(f"     err_fuera       {cuenta['err_fuera']/n:.4f}   nada del archivo")
    print(f"     invento         {cuenta['invento']/n:.4f}   no habia respuesta y contesto igual")

    if azar["n"]:
        ev, ei = azar["version"] / azar["n"], azar["identidad"] / azar["n"]
        print(f"\n  control · reparto esperado si eligiera AL AZAR entre los valores del archivo:")
        print(f"     version {ev:.4f}  ·  identidad {ei:.4f}   (sobre {azar['n']} errores)")
        obs_v = cuenta["err_version"] / max(1, err_seguro)
        obs_i = cuenta["err_identidad"] / max(1, err_seguro)
        print(f"  observado:")
        print(f"     version {obs_v:.4f}  ·  identidad {obs_i:.4f}")
        print(f"  → los errores de VERSION son {ev/max(obs_v,1e-9):.1f}x MENOS frecuentes que por azar"
              if obs_v < ev else
              f"  → los errores de VERSION son {obs_v/max(ev,1e-9):.1f}x MAS frecuentes que por azar")

    print("\n  desagregado por tipo de consulta:")
    encabezado = ("tipo", "n", "acierto", "err_ver", "err_ident", "err_fuera", "abst")
    print("    {:<10} {:>6} {:>8} {:>8} {:>10} {:>10} {:>7}".format(*encabezado))
    for t in sorted(por_tipo):
        c = por_tipo[t]
        m = sum(c.values())
        print(f"    {t:<10} {m:>6} {c['acierto']/m:>8.4f} {c['err_version']/m:>8.4f} "
              f"{c['err_identidad']/m:>10.4f} {c['err_fuera']/m:>10.4f} {c['abstencion']/m:>7.4f}")


if __name__ == "__main__":
    main()
