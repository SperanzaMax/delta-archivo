"""FASE 0 de `escriba` · ¿hay señal de recuperabilidad en el vector que se ESCRIBE?

    python escriba_fase0.py ckpts/v3_s0.pkl --n 4000 --json fase0_v3_s0.json

Evalúa el §3 de `PREREG_ESCRIBA.md` (SHA `958ad236…`, congelado el 27-ago antes de escribir una sola
línea de esto). La pregunta, textual del pre-registro:

  En el momento de la escritura, ¿existe en las activaciones una señal que separe una entrada que va
  a ser recuperable de una que no?

Si no existe, la cabeza que propone Maxi tendría que **crear** esa representación y no sólo leerla,
que es un proyecto distinto y mucho más caro. Por eso esto corre ANTES y en CPU. Es el mismo patrón
de `INFORME_SCORE_ARCHIVO_20260816.md`, que midió la señal antes de diseñar la campaña y bifurcó
todo con cero GPU.

## Cómo se arma el dato, y por qué así

`modelo.escribir` devuelve (B, S*E, D): **un vector por enunciado archivado**. Ese es exactamente el
punto donde la cabeza `escriba` viviría, así que es el punto donde se sonda.

`datos.lote(con_origen=True)` da `origen_arch[b,k]` —de qué hecho salió la entrada k— y `hecho_q[b]`
—qué hecho se preguntó—. Con eso se identifica **la entrada del archivo que corresponde al hecho
consultado**, que es la única sobre la que la pregunta «¿fue recuperable?» tiene sentido.

La etiqueta es si el modelo **acertó** esa consulta, con la misma regla de decisión del checkpoint
que usan `ser.py` y `ser_cobertura.py`. Se restringe a consultas CON respuesta (`hecho_q >= 0`): en
las que no la tienen no hay entrada objetivo que sondar, y meterlas mediría otra cosa.

## Los dos controles, que están acá porque el pre-registro los exige por adelantado

  · **Etiquetas permutadas** (E-0, bloqueante). Si la sonda con `y` permutado da lo mismo que con la
    etiqueta real, lo medido es la capacidad de la sonda y no una señal del modelo. En este programa
    un número limpio escondió un artefacto siete veces.
  · **Sonda ciega** (§7, fuga de etiqueta). Se alimenta SÓLO con la posición de la entrada en el
    archivo y la longitud del episodio, sin activaciones. Si eso solo ya alcanza el umbral, E-1 no es
    interpretable: estaríamos midiendo el generador y no el modelo.

`sonda()` y `auc()` se importan de `sonda_dos_detectores.py` en vez de reimplementarse. Ya pasaron su
chequeo de instrumento en `chequeo_sonda_lineal.py` (señal fuerte, sin señal, señal débil con clase
rara al 5 %, y empates masivos), y tener dos implementaciones de la misma sonda es la forma conocida
de que una de las dos se desincronice sin que nadie lo note.
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

E_MAX = DAT.E_MAX


def recolectar(ruta, n, B, semilla, nivel_cli, p_nose_cli, solo_vigente=False):
    with open(ruta, "rb") as f:
        bulto = pickle.load(f)
    params, cfg = bulto["params"], bulto["config"]
    nivel = nivel_cli if nivel_cli is not None else cfg["nivel"]
    p_nose = p_nose_cli if p_nose_cli is not None else cfg.get("p_nose", 0.0)

    # La arquitectura y la regla de decision salen DEL CHECKPOINT, no de flags. Misma razon que en
    # `ser.py`: son propiedades de la unidad medida, no de la medicion.
    E._DONDE = cfg.get("donde", "pre")
    E._ABST = cfg.get("abst", "token")
    predecir = E.predecir_cabeza if E._ABST in ("cabeza", "slot") else E.predecir

    @jax.jit
    def escribir(params, ses, cortes):
        return M.escribir(params, ses, cortes)

    rng = np.random.default_rng(semilla)
    Xs, ys, ciegos = [], [], []
    vistos = 0
    while vistos < n:
        b = min(B, n - vistos)
        (ses, cortes, turnos, mask, cons, pos, tgt, tipo, meta,
         origen_arch, hecho_q) = DAT.lote(rng, b, nivel=nivel, n_hechos=4, n_sesiones=4,
                                          p_nose=p_nose, con_meta=True, con_origen=True)
        arch = np.asarray(escribir(params, jnp.array(ses), jnp.array(cortes)))   # (b, S*E, D)
        pred = np.asarray(predecir(params, jnp.array(ses), jnp.array(cortes), jnp.array(turnos),
                                   jnp.array(mask), jnp.array(cons), jnp.array(pos)))
        for i in range(b):
            hq = int(hecho_q[i])
            if hq < 0:
                continue                      # sin respuesta: no hay entrada objetivo que sondar
            ks = np.flatnonzero(origen_arch[i] == hq)
            if len(ks) == 0:
                continue
            # `solo_vigente` (2026-08-27, CONTROL posterior al prereg, declarado como exploratorio):
            # se sondea `ks[-1]`, la ULTIMA escritura del hecho. Para una consulta por la version
            # VIGENTE esa es inequivocamente la entrada que hay que recuperar; para una consulta por
            # la version ANTERIOR la entrada relevante es otra, y mezclarlas puede diluir una señal
            # que si exista. El control corre solo sobre `tipo == 0` para que esa duda no quede
            # sosteniendo el veredicto.
            if solo_vigente and int(tipo[i]) != 0:
                continue
            k = int(ks[-1])                   # la ULTIMA escritura de ese hecho, que es la vigente
            Xs.append(arch[i, k])
            ys.append(bool(pred[i] == tgt[i]))
            # Sonda ciega: SOLO metadatos de posicion, sin una sola activacion.
            ciegos.append([k, k % E_MAX, k // E_MAX, int(mask[i].sum()), len(ks)])
        vistos += b

    return (np.array(Xs, np.float32), np.array(ys, bool), np.array(ciegos, np.float32),
            {"pesos": ruta, "nivel": nivel, "semilla_modelo": cfg["semilla"],
             "paso": bulto.get("paso"), "donde": E._DONDE, "abst": E._ABST,
             "blanco": cfg.get("blanco", "ausencia"), "p_nose": p_nose})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pesos")
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--B", type=int, default=64)
    ap.add_argument("--semilla", type=int, default=54321)
    ap.add_argument("--nivel", type=int, default=None)
    ap.add_argument("--p-nose", type=float, default=None)
    ap.add_argument("--solo-vigente", action="store_true",
                    help="CONTROL posterior al prereg: sondear solo consultas por la version "
                         "vigente, donde la ultima escritura del hecho es inequivocamente la "
                         "entrada a recuperar. Exploratorio, se declara como tal.")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    X, y, ciego, cfg = recolectar(a.pesos, a.n, a.B, a.semilla, a.nivel, a.p_nose,
                                  a.solo_vigente)
    print(f"{a.pesos}  ·  nivel {cfg['nivel']} · semilla {cfg['semilla_modelo']} · "
          f"paso {cfg['paso']} · lectura {cfg['donde']} · abst {cfg['abst']}")
    print(f"entradas sondeadas: {len(y)}  ·  dimension del vector escrito: {X.shape[1]}")

    base = float(y.mean())
    print(f"\n  tasa base de la etiqueta (acierto): {base:.4f}")
    if y.sum() < 30 or (~y).sum() < 30:
        # El slot nulo del 25-ago murio pegado al prior; una clase casi vacia produce AUC inestables
        # que se leen como señal. Se aborta antes de imprimir un numero que invitaria a creerle.
        print(f"  !! una de las clases tiene menos de 30 casos ({int(y.sum())} / {int((~y).sum())}). "
              f"La AUC no es interpretable con este reparto y no se reporta.")
        return

    # Split fijo por indice, sin barajar con otra semilla: el orden ya viene del muestreo aleatorio.
    corte = len(y) // 2
    ia, ip = slice(0, corte), slice(corte, len(y))
    rng = np.random.default_rng(a.semilla + 1)

    auc_real = auc(y[ip], sonda(X[ia], y[ia], X[ip]))
    auc_perm = auc(y[ip], sonda(X[ia], rng.permutation(y[ia]), X[ip]))
    auc_ciego = auc(y[ip], sonda(ciego[ia], y[ia], ciego[ip]))

    print(f"\n  ── FASE 0 ──────────────────────────────────────────────")
    print(f"    AUC sonda REAL      {auc_real:.4f}   ← E-1, decide si la linea sigue")
    print(f"    AUC etiq PERMUTADA  {auc_perm:.4f}   ← E-0 bloqueante, tiene que quedar <= 0,55")
    print(f"    AUC sonda CIEGA     {auc_ciego:.4f}   ← fuga de etiqueta (§7), sin activaciones")

    e0 = auc_perm <= 0.55
    e1 = auc_real >= 0.65
    fuga = auc_ciego >= 0.65
    print(f"\n    E-0 (permutada <= 0,55)      {'CUMPLE' if e0 else 'FALLA'}")
    print(f"    E-1 (real >= 0,65)           {'CUMPLE' if e1 else 'NO CUMPLE'}")
    print(f"    sonda ciega bajo el umbral   {'sí' if not fuga else 'NO — E-1 no es interpretable'}")

    if not e0:
        print("\n  → E-0 falla. No se lee nada mas: hay que arreglar el instrumento (§3).")
    elif fuga:
        print("\n  → La sonda ciega sola alcanza el umbral. E-1 NO es interpretable: mide el "
              "generador, no el modelo (§7).")
    elif e1:
        print("\n  → Hay señal de recuperabilidad en la escritura. Esta unidad habilita su mitad "
              "de E-1; hace falta 2 de 3 semillas para abrir la condicion.")
    else:
        print("\n  → Sin señal por encima del umbral. Si esto se repite en 2 de 3 semillas, el §6 "
              "manda CERRAR la linea sin entrenar nada, y `escriba` se suma como la octava via a "
              "las siete del PLAN_FOCO.")

    if a.json:
        with open(a.json, "w") as f:
            json.dump({**cfg, "n_sondeadas": int(len(y)), "tasa_base": base,
                       "auc_real": float(auc_real), "auc_permutada": float(auc_perm),
                       "auc_ciega": float(auc_ciego),
                       "E0_cumple": bool(e0), "E1_cumple": bool(e1),
                       "fuga_de_etiqueta": bool(fuga)}, f, indent=1)
        print(f"\n  -> {a.json}")


if __name__ == "__main__":
    main()
