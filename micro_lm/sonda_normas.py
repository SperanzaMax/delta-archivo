#!/usr/bin/env python3
"""¿Por que fracaso `escala`? Mide si la renormalizacion del vector de NOSE SOBREVIVE al entrenamiento.

`escala` lleva el vector de NOSE a la norma media de los tokens de valor al entrar en la fase de
abstencion, y despues entrena 2000 pasos. Si al terminar la norma volvio a caer, el negativo de P-2
no dice «la norma es irrelevante» sino «la norma no se sostiene»: el gradiente la vuelve a encoger
porque NOSE aparece en el 40 % de los targets contra 1-entre-100 de cada valor.

Son dos afirmaciones distintas y la diferencia importa para el paper. Esta sonda las separa.

Lectura pura de pesos: no entrena, no evalua, no toca la GPU. Segundos de CPU.
"""
import pickle, sys, os
import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
import idioma as I

NOSE = I.STOI["NOSE"]
VALORES = [I.STOI[t] for t in list(I.NOMBRES) + list(I.NUMEROS) if t in I.STOI]
UNIDADES = ["1_s0", "2_s0", "3_s0", "3_s1", "3_s2", "4_s0", "4_s1"]
FAM = {"n": "base (sin NOSE)", "t": "token", "s": "escala", "c": "cabeza"}


def normas(ck_path):
    """Devuelve (norma de NOSE, norma media de los valores) en entrada y en salida.

    Devuelve None tambien cuando el checkpoint es una COPIA EXACTA del base: `tramo_abst.sh` siembra
    con `cp`, asi que una unidad recien encolada tiene un .pkl que existe pero no entreno ni un paso.
    Sin esta guarda sus numeros entran en la tabla como si fueran un resultado — le paso a c4_s1 la
    primera vez que se corrio esta sonda.
    """
    if not os.path.exists(ck_path):
        return None
    with open(ck_path, "rb") as f:
        ck = pickle.load(f)
    base = os.path.join(os.path.dirname(ck_path), "n" + os.path.basename(ck_path)[1:])
    if os.path.basename(ck_path)[0] != "n" and os.path.exists(base):
        import hashlib
        h = lambda f: hashlib.md5(open(f, "rb").read()).hexdigest()
        if h(ck_path) == h(base):
            return "sin_entrenar"
    p = ck["params"]
    emb = np.asarray(p["emb"])                 # (V, D)  entrada
    head = np.asarray(p["head"]["w"])          # (D, V)  salida
    return {
        "emb_nose": float(np.linalg.norm(emb[NOSE])),
        "emb_val": float(np.linalg.norm(emb[VALORES], axis=1).mean()),
        "head_nose": float(np.linalg.norm(head[:, NOSE])),
        "head_val": float(np.linalg.norm(head[:, VALORES], axis=0).mean()),
        "tiene_cabeza": "abst" in p,
    }


print(f"NOSE = token {NOSE} · {len(VALORES)} tokens de valor · {len(UNIDADES)} unidades\n")
print(f"{'unidad':<9} {'condicion':<18} {'emb[NOSE]':>10} {'emb medio':>10} {'razon':>7} "
      f"{'head[NOSE]':>11} {'head medio':>11} {'razon':>7}")
print("-" * 92)

filas = {}
for u in UNIDADES:
    for fam in ("n", "t", "s", "c"):
        r = normas(os.path.join(AQUI, "ckpts", f"{fam}{u}.pkl"))
        if r is None:
            continue
        if r == "sin_entrenar":
            print(f"{fam}{u:<8} {FAM[fam]:<18} — sembrado, todavia no entreno (copia del base)")
            continue
        re_ = r["emb_nose"] / r["emb_val"]
        rh = r["head_nose"] / r["head_val"]
        filas[(u, fam)] = (re_, rh)
        print(f"{fam}{u:<8} {FAM[fam]:<18} {r['emb_nose']:>10.4f} {r['emb_val']:>10.4f} "
              f"{re_:>7.3f} {r['head_nose']:>11.4f} {r['head_val']:>11.4f} {rh:>7.3f}")
    print()

# --- la pregunta ---------------------------------------------------------------------------------
# `escala` arranca la fase con razon = 1.000 por construccion. Si al terminar volvio a bajar, la
# renormalizacion no sobrevivio y el negativo de P-2 se explica por eso.
print("=" * 92)
print("¿SOBREVIVIO LA RENORMALIZACION EN `escala`?  (razon = norma de NOSE / norma media de valores)")
print("Arranca en 1,000 por construccion en las DOS matrices.\n")
esc = [(u, filas[(u, "s")]) for u in UNIDADES if (u, "s") in filas]
if esc:
    re_m = np.mean([v[0] for _, v in esc])
    rh_m = np.mean([v[1] for _, v in esc])
    for u, (a, b) in esc:
        print(f"  s{u}: entrada {a:.3f}  salida {b:.3f}")
    print(f"\n  media: entrada {re_m:.3f} · salida {rh_m:.3f}")
    if re_m < 0.8 or rh_m < 0.8:
        print("  → LA RENORMALIZACION SE DESHACE: el gradiente vuelve a encoger el vector de NOSE.")
        print("    El negativo de P-2 NO dice «la norma es irrelevante», dice «la norma no se sostiene».")
    else:
        print("  → la renormalizacion SE SOSTIENE: la norma se mantuvo y aun asi `escala` fallo.")
        print("    Ahi si el negativo de P-2 es limpio: la norma no era el problema.")
else:
    print("  (todavia no hay checkpoints de `escala`)")

# control: en `cabeza` la norma de NOSE deberia ser IRRELEVANTE (esta excluido del softmax)
cab = [(u, filas[(u, "c")]) for u in UNIDADES if (u, "c") in filas]
if cab:
    print("\nCONTROL — en `cabeza`, NOSE esta excluido del softmax de valores, asi que su norma no")
    print("deberia importar. Si aca tambien se encoge, confirma que el encogimiento es del gradiente")
    print("y no de la competencia por la masa de probabilidad:")
    for u, (a, b) in cab:
        print(f"  c{u}: entrada {a:.3f}  salida {b:.3f}")
