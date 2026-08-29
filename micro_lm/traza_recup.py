"""RECUP como TRAYECTORIA dentro de una unidad — el control que le falta al §5 del informe del 29.

    N=8000 python traza_recup.py b3_s3 b3_s6

Lee `ckpts_traza/<unidad>_<paso>.pkl`, que `archivar_traza.sh` va guardando, y mide en cada uno el
argmax de valores IGNORANDO la cabeza de abstención.

Por qué hace falta un script aparte y no alcanzan los json del entrenamiento: el json guarda
`vigente`, que es la métrica COMPUESTA (`pred = NOSE si a > 0, si no el argmax`). En una unidad muda
`a > 0` siempre, así que `vigente` vale 0,0000 en todos los hitos y **no distingue una recuperación
de 0,30 de una de 0,40**, que es justo el número que la Fase 1 va a leer.

Todo con la MISMA semilla de datos (54321) en todos los checkpoints: el diseño es **pareado**, o sea
las preguntas son las mismas y lo único que cambia es el modelo. Sin eso F-1 no sería decidible —
`PRECISION_ATRACTOR_MUDO_N.md` lo mide: el desvío de una diferencia NO pareada es 0,0191, casi el
doble del efecto que F-1 pide.

`p_flip` es la fracción de preguntas donde dos checkpoints consecutivos difieren en su acierto. Es lo
que gobierna el ruido que le queda a la diferencia pareada (≈ √(p_flip/n_hay)) y por eso se reporta
medido y no supuesto, como la precisión se comprometió a hacer.

Los resultados se cachean en `ckpts_traza/_recup_<N>.json` para no re-medir en cada pasada.
"""
import sys, os, glob, re, json
import numpy as np
sys.path.insert(0, os.getcwd())
import idioma as I
from ser_cobertura import sondear

N = int(os.environ.get("N", "8000"))
SEMILLA = 54321
CACHE = f"ckpts_traza/_recup_{N}.json"
cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}


def medir(ruta):
    """(RECUP, vector de aciertos sobre las que tienen respuesta, logit medio, min, rango, abst)."""
    sc, pv, tg, mt, cfg = sondear(ruta, N, 64, None, None, SEMILLA)
    hay = np.array([I.ITOS[int(t)] != "NOSE" for t in tg])
    ok = np.array([I.ITOS[int(pv[i])] == I.ITOS[int(tg[i])] for i in range(len(tg))])
    return {"recup": float(ok[hay].mean()), "ok": ok[hay].astype(int).tolist(),
            "a_med": float(np.median(sc)), "a_min": float(sc.min()),
            "rango": float(np.percentile(sc, 99) - np.percentile(sc, 1)),
            "abst": float((sc > 0).mean()), "n_hay": int(hay.sum())}


for uni in sys.argv[1:]:
    rutas = sorted(glob.glob(f"ckpts_traza/{uni}_*.pkl"),
                   key=lambda p: int(re.search(r"_(\d+)\.pkl$", p).group(1)))
    if not rutas:
        print(f"{uni}: sin trazas en ckpts_traza/")
        continue

    print(f"\n=== {uni} · n={N} · semilla 54321 (pareado) ===")
    print(f"{'paso':>7} {'RECUP':>8} {'ΔRECUP':>9} {'p_flip':>8} {'σ_dif':>8} | "
          f"{'abst':>6} {'a_med':>8} {'a_min':>8} {'rango':>7}")
    print("-" * 82)
    prev = None
    puntos = []
    for r in rutas:
        paso = int(re.search(r"_(\d+)\.pkl$", r).group(1))
        clave = f"{uni}_{paso}"
        if clave not in cache:
            cache[clave] = medir(r)
            json.dump(cache, open(CACHE, "w"))
        m = cache[clave]
        ok = np.array(m["ok"])
        if prev is None:
            d = pf = sd = ""
        else:
            okp = np.array(prev["ok"])
            flip = float((ok != okp).mean())
            d = f"{m['recup'] - prev['recup']:+9.4f}"
            pf = f"{flip:8.4f}"
            sd = f"{np.sqrt(max(flip, 1e-9) / len(ok)):8.4f}"
        print(f"{paso:>7} {m['recup']:8.4f} {d:>9} {pf:>8} {sd:>8} | {m['abst']:6.3f} "
              f"{m['a_med']:8.3f} {m['a_min']:8.3f} {m['rango']:7.3f}")
        puntos.append((paso, m))
        prev = m

    if len(puntos) >= 2:
        (p0, m0), (p1, m1) = puntos[0], puntos[-1]
        dr, dp = m1["recup"] - m0["recup"], p1 - p0
        ok0, ok1 = np.array(m0["ok"]), np.array(m1["ok"])
        flip = float((ok0 != ok1).mean())
        sd = np.sqrt(max(flip, 1e-9) / len(ok0))
        subes = sum(1 for a, b in zip(puntos, puntos[1:]) if b[1]["recup"] > a[1]["recup"])
        print(f"\n  TOTAL {p0} -> {p1}: {dr:+.4f} en {dp} pasos ({dr/dp*1000:+.4f} cada 1000)")
        print(f"  p_flip total {flip:.4f} · σ de la diferencia pareada {sd:.4f} · "
              f"efecto/σ = {abs(dr)/sd if sd else float('nan'):.1f}")
        print(f"  monotonía: sube en {subes} de {len(puntos)-1} intervalos")
        print(f"  F-1 pide >= +0,0100 y monotonía en >= 3 de 4 · F-3 (confound) si < +0,0030")
