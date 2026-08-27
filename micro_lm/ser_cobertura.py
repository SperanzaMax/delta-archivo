"""MICRO-LM · SER a COBERTURA IGUALADA, la metrica principal de A5.

    python ser_cobertura.py ckpts/b3_s0.pkl ckpts/p3_s0.pkl --n 4000

El §4 de `PREREG_BLANCO_ERROR.md` fija esta metrica ANTES de ver nada, y fija tambien por que la
compuerta historica (`nose` >= 0,50 y `falsa_abst` <= 0,10) no sirve para esta condicion:

  Con blanco `error` la cabeza se activa tambien en preguntas que SI tienen respuesta pero donde el
  modelo la erraria. `falsa_abst` cuenta eso como falsa abstencion, cuando es justamente la
  abstencion CORRECTA que la condicion existe para producir. Usar la compuerta vieja garantizaria
  que esta condicion «falle», y el fallo no diria nada.

Lo que se hace en cambio, que es la comparacion que un usuario del sistema haria —*a igual cantidad
de respuestas dadas, cuantas estan mal?*—:

  1. Se corre el modelo SIN decidir nada: se guardan el logit de la cabeza `a` y el argmax de valores
     de cada muestra, sobre el MISMO lote para las dos unidades (misma semilla de datos, mismo n).
  2. Se barre el umbral. Contestar cuando `a <= tau`; el `tau` que da cobertura c es el cuantil c de
     los scores. Asi la cobertura sale EXACTA salvo empates, sin buscar a ciegas.
  3. Se compara el SER de las dos condiciones en el mismo punto de cobertura: 0,60 · 0,70 · 0,80.

`ser.py` no sirve para esto y no es un defecto suyo: mide en EL punto de operacion del modelo
(`a > 0`, el umbral con el que se entreno). Dos unidades con distinto blanco operan a coberturas
distintas, asi que sus SER no son comparables tal cual — que es exactamente el problema que este
script existe para sacar del medio.

La clasificacion de cada prediccion se importa de `ser.py`: es la misma taxonomia (err_version /
err_identidad / err_fuera / invento) y no se reimplementa acá, para que un cambio de criterio no
tenga que hacerse en dos lugares y quede desincronizado.
"""
import argparse
import collections
import json
import pickle

import numpy as np
import jax
import jax.numpy as jnp

import datos as DAT
import idioma as I
import entrenar as E
from ser import clasificar

# Coberturas del §4. Se dejan como constante y no como flag: cambiarlas por linea de comandos seria
# poder elegir el punto de comparacion DESPUES de ver los datos, que es lo que el pre-registro
# existe para impedir.
COBERTURAS = (0.60, 0.70, 0.80)


def sondear(ruta, n, B, nivel_cli, p_nose_cli, semilla):
    """Corre una unidad sin decidir nada y devuelve (scores, pred_valor, tgt, meta).

    `scores` es el logit crudo de la cabeza. NO se le aplica el umbral 0 de `predecir_cabeza`:
    justamente lo que se va a hacer despues es moverlo.
    """
    with open(ruta, "rb") as f:
        bulto = pickle.load(f)
    params, cfg = bulto["params"], bulto["config"]
    nivel = nivel_cli if nivel_cli is not None else cfg["nivel"]
    p_nose = p_nose_cli if p_nose_cli is not None else cfg.get("p_nose", 0.0)

    # Mismo cuidado que en `ser.py`: la arquitectura y la regla de decision salen DEL CHECKPOINT y no
    # de flags, porque son propiedades de la unidad medida y no de la medicion. Sin fijar `_ABST` el
    # logit binario saldria de la cabeza lineal aunque la unidad sea `slot`.
    E._DONDE = cfg.get("donde", "pre")
    E._ABST = cfg.get("abst", "token")
    if E._ABST not in ("cabeza", "slot"):
        raise SystemExit(
            f"{ruta}: abst={E._ABST!r}. Este script barre el umbral de la cabeza binaria; una unidad "
            f"`token` no tiene score que barrer —su abstencion es un argmax— y compararla acá seria "
            f"inventarle una perilla que en el entrenamiento no tuvo.")

    @jax.jit
    def partes(params, ses, cortes, turnos, mask, cons, pos):
        lg, a = E._partes(params, ses, cortes, turnos, mask, cons, pos)
        # `NOSE` fuera del argmax de valores, igual que en `predecir_cabeza`: con la cabeza aparte,
        # dejarlo seria darle dos rutas a la misma decision.
        return lg.at[:, E.NOSE].set(-jnp.inf).argmax(-1), a

    rng = np.random.default_rng(semilla)
    sc, pv, tg, mt = [], [], [], []
    vistos = 0
    while vistos < n:
        b = min(B, n - vistos)
        ses, cortes, turnos, mask, cons, pos, tgt, tipo, meta = DAT.lote(
            rng, b, nivel=nivel, n_hechos=4, n_sesiones=4, p_nose=p_nose, con_meta=True)
        pred, a = partes(params, jnp.array(ses), jnp.array(cortes), jnp.array(turnos),
                         jnp.array(mask), jnp.array(cons), jnp.array(pos))
        sc.append(np.asarray(a)); pv.append(np.asarray(pred))
        tg.append(np.asarray(tgt)); mt.extend(meta)
        vistos += b

    meta_cfg = {"pesos": ruta, "nivel": nivel, "semilla": cfg["semilla"],
                "paso": bulto.get("paso"), "donde": E._DONDE, "abst": E._ABST,
                "blanco": cfg.get("blanco", "ausencia"), "p_nose": p_nose}
    return np.concatenate(sc), np.concatenate(pv), np.concatenate(tg), mt, meta_cfg


def medir(scores, pred_valor, tgt, meta, cob):
    """SER y el reparto de categorias contestando las `cob` muestras de menor score."""
    n = len(scores)
    k = int(round(cob * n))                      # cuantas se contestan
    # `argsort` y no un umbral buscado a ojo: el punto de corte que da la cobertura pedida es el
    # cuantil, y asi sale exacto salvo empates. `kind="stable"` para que dos corridas con los mismos
    # scores rompan los empates igual y el pareo no dependa del orden interno del sort.
    orden = np.argsort(scores, kind="stable")
    contesta = np.zeros(n, dtype=bool)
    contesta[orden[:k]] = True
    tau = float(scores[orden[k - 1]]) if k else float("-inf")

    cuenta = collections.Counter()
    for i in range(n):
        # La decision se REEMPLAZA por la del umbral barrido; todo lo demas (que cuenta como error de
        # version, de identidad, invento) queda igual que en `ser.py`.
        tok = I.ITOS[int(pred_valor[i])] if contesta[i] else "NOSE"
        cuenta[clasificar(tok, I.ITOS[int(tgt[i])], meta[i])] += 1

    err = (cuenta["err_version"] + cuenta["err_identidad"] + cuenta["err_fuera"]
           + cuenta["invento"])
    dadas = int(contesta.sum())
    sin_resp = cuenta["acierto_nose"] + cuenta["invento"]
    con_resp = n - sin_resp
    return {
        "cobertura_pedida": cob, "cobertura_real": dadas / n, "tau": tau, "n": n, "dadas": dadas,
        # Dos denominadores, a proposito. `SER` es el de `ser.py` —sobre TODAS las preguntas— para que
        # los numeros sigan siendo comparables con todo lo ya informado. `riesgo` es el de selective
        # prediction —sobre las CONTESTADAS— que es literalmente la frase del §4: a igual cantidad de
        # respuestas dadas, cuantas estan mal. A cobertura igualada los dos ordenan igual; se
        # reportan los dos para que nadie tenga que confiar en que asi es.
        "SER": err / n,
        "riesgo": err / max(1, dadas),
        "err_version": cuenta["err_version"] / n,
        "err_identidad": cuenta["err_identidad"] / n,
        "err_fuera": cuenta["err_fuera"] / n,
        "invento": cuenta["invento"] / n,
        "acierto": cuenta["acierto"] / max(1, con_resp),
        "nose": cuenta["acierto_nose"] / sin_resp if sin_resp else None,
        "falsa_abst": cuenta["abstencion"] / max(1, con_resp),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tratamiento", help="la unidad con blanco `error`, p.ej. ckpts/b3_s0.pkl")
    ap.add_argument("control", help="su control pareado, p.ej. ckpts/p3_s0.pkl")
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--B", type=int, default=64)
    ap.add_argument("--nivel", type=int, default=None)
    ap.add_argument("--p-nose", type=float, default=None)
    ap.add_argument("--semilla", type=int, default=54321,
                    help="semilla de los DATOS. La misma para las dos unidades: el pareo del §3 se "
                         "pierde si cada una ve preguntas distintas.")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    filas = {}
    for rol, ruta in (("tratamiento", a.tratamiento), ("control", a.control)):
        sc, pv, tg, mt, cfg = sondear(ruta, a.n, a.B, a.nivel, a.p_nose, a.semilla)
        print(f"{rol:<12} {ruta}  ·  nivel {cfg['nivel']} · semilla {cfg['semilla']} · "
              f"paso {cfg['paso']} · blanco {cfg['blanco']} · lectura {cfg['donde']}")
        # El punto de operacion propio (tau=0) se informa aparte, porque es el que produjo todos los
        # numeros anteriores y sirve para pegar este informe con los que ya existen.
        propio = medir(sc, pv, tg, mt, float((sc <= 0.0).mean()))
        filas[rol] = {"cfg": cfg, "propio": propio,
                      "curva": [medir(sc, pv, tg, mt, c) for c in COBERTURAS]}

    ct, cc = filas["tratamiento"], filas["control"]
    if ct["cfg"]["semilla"] != cc["cfg"]["semilla"]:
        print(f"\n  !! semillas de MODELO distintas ({ct['cfg']['semilla']} vs "
              f"{cc['cfg']['semilla']}): el pareo del §3 pide la misma.")
    if ct["cfg"]["paso"] != cc["cfg"]["paso"]:
        print(f"\n  !! pasos distintos ({ct['cfg']['paso']} vs {cc['cfg']['paso']}): el §3 compara a "
              f"PRESUPUESTO IGUALADO. Un SER mejor acá puede ser sólo mas entrenamiento.")

    print(f"\n  punto de operacion propio (tau = 0, el umbral con el que se entrenaron):")
    for rol, f in (("tratamiento", ct), ("control", cc)):
        p = f["propio"]
        print(f"    {rol:<12} cobertura {p['cobertura_real']:.4f}  SER {p['SER']:.4f}  "
              f"riesgo {p['riesgo']:.4f}  nose {p['nose'] if p['nose'] is None else round(p['nose'],4)}  "
              f"falsa_abst {p['falsa_abst']:.4f}")

    print(f"\n  ── SER a COBERTURA IGUALADA (n={a.n}) ────────────────────────────────")
    print("    {:>10} {:>9} {:>9} {:>9}   {:>9} {:>9} {:>9}".format(
        "cobertura", "SER trat", "SER ctrl", "Δ SER", "riesgo t", "riesgo c", "Δ riesgo"))
    for t, c in zip(ct["curva"], cc["curva"]):
        d_ser, d_r = t["SER"] - c["SER"], t["riesgo"] - c["riesgo"]
        # El signo se dice en palabras y no se deja al lector: negativo = el tratamiento comete MENOS
        # error a la misma cobertura, que es la prediccion del §5.
        marca = "  ← trat mejor" if d_ser < 0 else ("  ← ctrl mejor" if d_ser > 0 else "")
        print(f"    {t['cobertura_pedida']:>10.2f} {t['SER']:>9.4f} {c['SER']:>9.4f} "
              f"{d_ser:>+9.4f}   {t['riesgo']:>9.4f} {c['riesgo']:>9.4f} {d_r:>+9.4f}{marca}")

    print(f"\n  desagregado del tratamiento por cobertura:")
    print("    {:>10} {:>9} {:>11} {:>10} {:>9}".format(
        "cobertura", "err_ver", "err_ident", "err_fuera", "invento"))
    for t in ct["curva"]:
        print(f"    {t['cobertura_pedida']:>10.2f} {t['err_version']:>9.4f} "
              f"{t['err_identidad']:>11.4f} {t['err_fuera']:>10.4f} {t['invento']:>9.4f}")

    if a.json:
        with open(a.json, "w") as f:
            json.dump({"n": a.n, "semilla_datos": a.semilla, "coberturas": list(COBERTURAS),
                       "tratamiento": ct, "control": cc}, f, indent=1)
        print(f"\n  -> {a.json}")


if __name__ == "__main__":
    main()
