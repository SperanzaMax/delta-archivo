"""MICRO-LM · CALIBRAR EL CORTE Y MEDIR CUANTO SOBREVIVE SIN LAS ETIQUETAS DE LA UNIDAD.

    python calibra_transferencia.py --n 6000 --json calibra_transf_20260828.json

Evalua `PREREG_CALIBRA_TRANSFERENCIA.md` (SHA 5fdab03d...), congelado antes de escribir este archivo.

Las seis unidades del nivel 3 ya entrenadas a 26000: b3_s0/s1/s2 (blanco `error`) y p3_s0/s1/s2
(blanco `ausencia`). Post-hoc puro: no se entrena nada, es CPU.

DIFERENCIA CON `sonda_calibra_ensamble.py`, y es la razon de que este script exista: alla las
semillas de generacion son FIJAS (55000/66000) porque A4 mide acuerdo entre unidades y necesita que
las tres vean el mismo lote. Para la TRANSFERENCIA eso seria fatal — un corte calibrado en s0 y
aplicado a s1 se juzgaria sobre las mismas preguntas con las que se eligio. Aca las muestras se
generan POR UNIDAD (31000+s de ajuste, 42000+s de prueba), tal como fija el §2 del pre-registro.

El procedimiento de calibracion es el del 19-ago: el corte se elige en AJUSTE pidiendo MARGEN
(falsa_abst <= 0,07) y se juzga en PRUEBA con el criterio real (<= 0,10). El optimo pegado al borde
no generaliza.

NOTA DE IMPLEMENTACION. `clasificar` depende de la muestra y de UNA sola cosa variable: si se
contesta o no. Asi que la categoria de cada muestra se precomputa en sus dos estados posibles y el
barrido de 400 cortes se resuelve con sumas prefijas — O(1) por corte en vez de recorrer las n
muestras. Sin esto el nulo (20 repeticiones x 400 cortes x 6000 muestras x 6 unidades) no termina
nunca. Los numeros son identicos a los de `medir` en `ser_cobertura.py`; el chequeo C-1 lo verifica
muestra por muestra contra esa implementacion.
"""
import argparse
import json
import os
import sys

import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)

import idioma as I
from ser import clasificar
from ser_cobertura import sondear

# §3 del prereg. Constantes y no flags: poder elegirlas por linea de comandos seria poder mover el
# criterio despues de ver los datos.
MARGEN_ELEGIR, CRITERIO = 0.07, 0.10
N_CORTES = 400
SEM_AJUSTE, SEM_PRUEBA = 31000, 42000
UNIDADES = [f"b3_s{s}" for s in (0, 1, 2)] + [f"p3_s{s}" for s in (0, 1, 2)]
N_NULO = 20                      # repeticiones del nulo permutado, control C-C del 19-ago

CATS = ("acierto", "acierto_nose", "err_version", "err_identidad", "err_fuera", "invento",
        "abstencion")
IDX = {c: i for i, c in enumerate(CATS)}
ERRORES = [IDX[c] for c in ("err_version", "err_identidad", "err_fuera", "invento")]


def categorias(pred_valor, tgt, meta):
    """Categoria de cada muestra SI SE CONTESTA, y si su respuesta correcta era NOSE.

    Cuando se abstiene la categoria no hace falta calcularla: es `acierto_nose` si la respuesta era
    NOSE y `abstencion` si no. Sale de `clasificar` y no se reimplementa el criterio.
    """
    n = len(tgt)
    cat_si = np.empty(n, dtype=np.int16)
    es_nose = np.empty(n, dtype=bool)
    for i in range(n):
        t = I.ITOS[int(tgt[i])]
        es_nose[i] = (t == "NOSE")
        cat_si[i] = IDX[clasificar(I.ITOS[int(pred_valor[i])], t, meta[i])]
    return cat_si, es_nose


class Barrido:
    """Precomputa las sumas prefijas para resolver cualquier corte en O(1)."""

    def __init__(self, scores, cat_si, es_nose):
        orden = np.argsort(scores, kind="stable")
        self.sc = np.asarray(scores)[orden]
        cs, en = cat_si[orden], es_nose[orden]
        self.n = len(self.sc)
        oh = np.zeros((self.n, len(CATS)), dtype=np.int32)
        oh[np.arange(self.n), cs] = 1
        self.pre = np.vstack([np.zeros(len(CATS), dtype=np.int32), np.cumsum(oh, axis=0)])
        self.pre_nose = np.concatenate([[0], np.cumsum(en.astype(np.int32))])
        self.total_nose = int(en.sum())
        self.con_resp = self.n - self.total_nose

    def en_tau(self, tau):
        """Contestar cuando el score es <= tau."""
        k = int(np.searchsorted(self.sc, tau, side="right"))
        return self.en_k(k, tau)

    def en_k(self, k, tau=None):
        c = self.pre[k]                                   # categorias de las k contestadas
        nose_contestadas = int(self.pre_nose[k])          # `invento`: contesto algo sin respuesta
        acierto_nose = self.total_nose - nose_contestadas
        abstencion = (self.n - k) - acierto_nose
        err = int(c[ERRORES].sum())
        return {
            "tau": float(tau) if tau is not None else float(self.sc[k - 1]) if k else float("-inf"),
            "nose": acierto_nose / self.total_nose if self.total_nose else None,
            "falsa_abst": abstencion / max(1, self.con_resp),
            "SER": err / self.n,
            "cobertura": k / self.n,
        }

    def grilla(self):
        return np.linspace(float(self.sc[0]), float(self.sc[-1]), N_CORTES)

    def elegir(self, margen=MARGEN_ELEGIR):
        """El corte que MAXIMIZA `nose` sujeto a falsa_abst <= margen.

        Devuelve (tau, fila) o (None, None) si ningun corte de la grilla cumple — que es justamente
        lo que el nulo tiene que producir.
        """
        mejor, mejor_fila = None, None
        for tau in self.grilla():
            f = self.en_tau(tau)
            if f["falsa_abst"] <= margen and f["nose"] is not None:
                if mejor_fila is None or f["nose"] > mejor_fila["nose"]:
                    mejor, mejor_fila = float(tau), f
        return mejor, mejor_fila

    def z_de(self, tau):
        """El corte en unidades de desvio de la propia distribucion.

        Es la forma que sobrevivio el 20-ago (U-2): lo que transfiere entre unidades no es el tau
        crudo —cada unidad tiene su escala— sino el corrimiento normalizado.
        """
        return (float(tau) - float(self.sc.mean())) / float(self.sc.std())

    def tau_de_z(self, z):
        return float(self.sc.mean()) + float(z) * float(self.sc.std())


def chequeo_instrumento(sc, pv, tg, mt, cat_si, es_nose):
    """C-1 · el barrido rapido tiene que dar EXACTAMENTE lo mismo que recorrer las muestras."""
    from ser_cobertura import medir
    b = Barrido(sc, cat_si, es_nose)
    for cob in (0.3, 0.6, 0.75):
        k = int(round(cob * len(sc)))
        rap = b.en_k(k)
        lento = medir(sc, pv, tg, mt, cob)
        for campo in ("nose", "falsa_abst", "SER"):
            a, c = rap[campo], lento[campo]
            if a is None or c is None:
                continue
            if abs(a - c) > 1e-12:
                raise SystemExit(f"C-1 FALLA en cobertura {cob}, campo {campo}: "
                                 f"rapido {a!r} vs lento {c!r}")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6000, help="muestras por unidad y por muestra")
    ap.add_argument("--B", type=int, default=64)
    ap.add_argument("--ckpts", default=os.path.join(AQUI, "ckpts"))
    ap.add_argument("--json", default=os.path.join(AQUI, "calibra_transf_20260828.json"))
    ap.add_argument("--solo-chequeo", action="store_true",
                    help="corre C-1 sobre una unidad con n chico y sale")
    a = ap.parse_args()

    salida = {"n": a.n, "sem_ajuste": SEM_AJUSTE, "sem_prueba": SEM_PRUEBA,
              "margen_elegir": MARGEN_ELEGIR, "criterio": CRITERIO, "n_cortes": N_CORTES,
              "prereg_sha": "5fdab03defc023f4bd706aa5ed71b61586f7dd315ac93fef14362245002917ea",
              "unidades": {}}
    barridos = {}

    # ---- 1. Sondeo. Muestras independientes POR UNIDAD, §2 del prereg.
    for u in UNIDADES:
        s = int(u[-1])
        ruta = os.path.join(a.ckpts, f"{u}.pkl")
        aj_sc, aj_pv, aj_tg, aj_mt, cfg = sondear(ruta, a.n, a.B, None, None, SEM_AJUSTE + s)
        pr_sc, pr_pv, pr_tg, pr_mt, _ = sondear(ruta, a.n, a.B, None, None, SEM_PRUEBA + s)
        aj_cat, aj_nose = categorias(aj_pv, aj_tg, aj_mt)
        pr_cat, pr_nose = categorias(pr_pv, pr_tg, pr_mt)

        if u == UNIDADES[0]:
            chequeo_instrumento(pr_sc, pr_pv, pr_tg, pr_mt, pr_cat, pr_nose)
            print("C-1 · el barrido rapido coincide con `medir` en 3 coberturas.", flush=True)
            if a.solo_chequeo:
                return

        barridos[u] = {"ajuste": Barrido(aj_sc, aj_cat, aj_nose),
                       "prueba": Barrido(pr_sc, pr_cat, pr_nose), "cfg": cfg,
                       "aj_crudo": (aj_sc, aj_cat, aj_nose)}
        print(f"{u:8} sondeado · blanco {cfg['blanco']:9} paso {cfg['paso']}", flush=True)

    # ---- 2. Por unidad: baseline, corte calibrado, oraculo y el nulo.
    for u in UNIDADES:
        B_aj, B_pr = barridos[u]["ajuste"], barridos[u]["prueba"]

        # sigma>0,5 es el logit > 0: el punto de operacion con el que la unidad se entreno.
        base = B_pr.en_tau(0.0)
        tau_cal, _ = B_aj.elegir()
        cal = B_pr.en_tau(tau_cal) if tau_cal is not None else None
        # Oraculo: el mejor corte elegido EN LA PROPIA PRUEBA con el criterio real. Es el techo, y
        # mira etiquetas que en uso real no estarian.
        tau_or, orac = B_pr.elegir(margen=CRITERIO)

        # K-0 · el nulo permutado. Se permutan los scores contra sus etiquetas: si el buscador
        # encuentra igual un corte que mejora la deteccion, se esta pasando a si mismo.
        aj_sc, aj_cat, aj_nose = barridos[u]["aj_crudo"]
        rng = np.random.default_rng(7000 + int(u[-1]) + (0 if u[0] == "b" else 100))
        nulo_ok = 0
        for _ in range(N_NULO):
            Bn = Barrido(aj_sc[rng.permutation(len(aj_sc))], aj_cat, aj_nose)
            t, _f = Bn.elegir()
            if t is None:
                continue
            v = B_pr.en_tau(t)
            if v["falsa_abst"] <= CRITERIO and v["nose"] is not None and v["nose"] > base["nose"]:
                nulo_ok += 1

        salida["unidades"][u] = {
            "cfg": barridos[u]["cfg"], "base_sigma05": base, "calibrado": cal, "oraculo": orac,
            "tau_ajuste": tau_cal, "z_ajuste": B_aj.z_de(tau_cal) if tau_cal is not None else None,
            "nulo_pasa": nulo_ok, "nulo_reps": N_NULO,
        }
        g = (cal["nose"] - base["nose"]) if cal else float("nan")
        print(f"{u:8} base {base['nose']:.4f} → cal {cal['nose']:.4f} (Δ{g:+.4f}) · "
              f"f_abst {cal['falsa_abst']:.4f} · orac {orac['nose']:.4f} · "
              f"z* {salida['unidades'][u]['z_ajuste']:+.4f} · nulo {nulo_ok}/{N_NULO}", flush=True)

    # ---- 3. K-2 · transferencia leave-one-out. El corte de `j` sale de la MEDIANA de los z* de las
    # otras cinco, y no se mira ni una etiqueta de `j`.
    zs = {u: salida["unidades"][u]["z_ajuste"] for u in UNIDADES}
    for u in UNIDADES:
        otras = [zs[v] for v in UNIDADES if v != u and zs[v] is not None]
        z_loo = float(np.median(otras))
        B_pr = barridos[u]["prueba"]
        salida["unidades"][u]["transferido"] = B_pr.en_tau(B_pr.tau_de_z(z_loo))
        salida["unidades"][u]["z_loo"] = z_loo

    with open(a.json, "w") as f:
        json.dump(salida, f, indent=1)
    print(f"\n→ {a.json}")


if __name__ == "__main__":
    main()
