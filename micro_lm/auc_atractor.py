"""AUC(a) de las nueve unidades del atractor, la columna que se computo y NO se archivo. · 4-sep

    N=8000 python auc_atractor.py

El blanco NO es la ausencia sino el ERROR, y esto no es una interpretacion: las nueve unidades de la
tabla son `b3_*` y su config dice `blanco='error'`, que en `entrenar.py` se construye asi

    blanco = (lg_arg != tgt)      # con tgt==NOSE da 1 siempre, por definicion

o sea «¿el argmax del modelo, ignorando la cabeza, difiere del target?». Por eso el paper habla de
AUC «on their own target»: es el blanco con el que esa cabeza fue entrenada.

Se mide sobre TODOS los items, no solo sobre los que tienen respuesta, porque el blanco esta definido
para todos.

La salida se guarda en `auc_atractor_<N>.json` con el vector de logits incluido, para que la proxima
vez no haya que volver a pasar por los checkpoints. Ese fue exactamente el defecto que dejo esta
columna sin rastrear.
"""
import sys, os, json, time
import numpy as np
sys.path.insert(0, os.getcwd())
import idioma as I
from ser_cobertura import sondear

N = int(os.environ.get("N", "8000"))
SEMILLA = 54321                      # la misma de `traza_recup.py`: diseño pareado
SALIDA = f"auc_atractor_{N}.json"

# las nueve de la tabla del paper, en el orden en que aparecen
UNIDADES = ["b3_s0_26000", "b3_s1_26000", "b3_s4_19000", "b3_s2_26000", "b3_s5_6000",
            "b3_s6_26000", "b3_s3_26000", "b3_s7_13500", "b3_s8_8000"]

PUBLICADO = {"b3_s0_26000": 0.9997, "b3_s1_26000": 0.9999, "b3_s4_19000": 0.8084,
             "b3_s2_26000": 0.8163, "b3_s5_6000": 0.7550, "b3_s6_26000": 0.5734,
             "b3_s3_26000": 0.5784, "b3_s7_13500": 0.5220, "b3_s8_8000": 0.5458}


def auc(score, positivo):
    """AUC por rangos (Mann-Whitney), con empates promediados."""
    score, positivo = np.asarray(score, float), np.asarray(positivo, bool)
    n1, n0 = int(positivo.sum()), int((~positivo).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    orden = np.argsort(score, kind="mergesort")
    rangos = np.empty(len(score), float)
    s = score[orden]
    i = 0
    while i < len(s):                       # promedio de rangos dentro de cada empate
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        rangos[orden[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((rangos[positivo].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


if __name__ == "__main__":
    res = {}
    print(f"{'unidad':<14} {'n':>6} {'err':>7} {'AUC(a)':>8} {'publicado':>10} {'dif':>8}  {'seg':>6}")
    print("-" * 68)
    for u in UNIDADES:
        ruta = f"ckpts_traza/{u}.pkl"
        if not os.path.exists(ruta):
            print(f"{u:<14} FALTA {ruta}")
            continue
        t0 = time.time()
        sc, pv, tg, mt, cfg = sondear(ruta, N, 64, None, None, SEMILLA)
        sc = np.asarray(sc, float)
        blanco = np.array([int(pv[i]) != int(tg[i]) for i in range(len(tg))])
        a = auc(sc, blanco)
        pub = PUBLICADO[u]
        dt = time.time() - t0
        print(f"{u:<14} {len(sc):>6} {blanco.mean():>7.4f} {a:>8.4f} {pub:>10.4f} "
              f"{a - pub:>+8.4f}  {dt:>6.1f}")
        res[u] = {"auc_a": a, "n": int(len(sc)), "frac_error": float(blanco.mean()),
                  "publicado": pub, "dif": float(a - pub),
                  "blanco": cfg.get("blanco"), "kernel_q": cfg.get("kernel_q"),
                  "a": sc.tolist(), "error": blanco.astype(int).tolist()}
    json.dump({"N": N, "semilla": SEMILLA, "definicion": "AUC(a) contra blanco = (argmax != target)",
               "unidades": res}, open(SALIDA, "w"))
    print(f"\nguardado en {SALIDA} (con el vector de logits, para no re-medir)")
    difs = [abs(v["dif"]) for v in res.values()]
    if difs:
        print(f"diferencia maxima contra lo publicado: {max(difs):.4f}")
