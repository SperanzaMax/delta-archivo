"""Compuerta de admisión de un encoder, ANTES de generar un solo dato del experimento.

Nace de dos noches perdidas (ver HALLAZGO_TOKENIZADOR_20260810.md). Las compuertas anteriores
miraban si la consulta identificaba su entidad (por AUC) y nada más. Eso dejó pasar dos veces un
encoder inservible:

  - intento 1 (`gemma:2b`): AUC 0,97 y top-1 0,13 — AUC no es la métrica.
  - intento 2 (`nomic-embed-text` con mayúsculas): AUC 0,9928 y top-1 0,708, pero **v1 y v2 daban
    vectores idénticos bit a bit**, así que la tarea no tenía señal.

Cuatro chequeos, cualquiera que falle aborta:

  C1  censo de vectores únicos: N textos distintos → ≈N vectores distintos.
  C2  discriminación de VALORES: v1 vs v2 (misma entidad, distinto valor) deben diferir.
  C3  discriminación de ENTIDADES: dos entidades distintas deben diferir.
  C4  identificación por TOP-1 y RANGO MEDIANO (no por AUC).

C1 es el más barato y el que hubiera atajado todo: una línea, y detecta de un golpe toda la familia
de fallos donde el encoder colapsa tokens.

Uso:
    python compuerta_encoder.py [modelo] [N]
"""
import json
import sys
import urllib.request

import numpy as np

sys.path.insert(0, ".")
from tarea_hechos import gen_hechos

OLLAMA = "http://localhost:11434/api/embed"

# Umbrales de la compuerta. Se declaran acá y no se tocan después de ver los datos.
C1_FRACCION_UNICOS = 0.95     # al menos el 95 % de los textos deben dar vectores distintos
C2_COS_MAX = 0.99             # v1 vs v2 por debajo de esto
C3_COS_MAX = 0.99             # entidad vs entidad por debajo de esto
C4_TOP1_MIN = 0.50            # la consulta recupera su hecho en el top-1
C4_RANGO_MAX = 1.0            # rango mediano del hecho correcto

# El texto va en MINÚSCULA: nomic-embed-text en Ollama colapsa todo token capitalizado
# (HALLAZGO_TOKENIZADOR_20260810.md). Se aplica a todos los encoders por uniformidad, y la
# compuerta lo verifica igual — nadie tiene que confiar en este comentario.
NORMALIZAR = str.lower


def embed_lote(textos, modelo, lote=64):
    out = []
    for i in range(0, len(textos), lote):
        payload = {"model": modelo, "input": [NORMALIZAR(t) for t in textos[i:i + lote]]}
        req = urllib.request.Request(OLLAMA, data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        for intento in range(3):
            try:
                out.extend(json.load(urllib.request.urlopen(req, timeout=300))["embeddings"])
                break
            except Exception as e:
                if intento == 2:
                    raise RuntimeError(f"el endpoint falló en el lote {i}: {e}")
    X = np.array(out, dtype=np.float32)
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)


def correr(modelo="nomic-embed-text", n=400, seed=0):
    rng = np.random.default_rng(seed)
    items = gen_hechos(rng, n)
    t1 = [x["v1"] for x in items]
    t2 = [x["v2"] for x in items]
    tq = [x["consulta"] for x in items]

    print(f"== compuerta de encoder · modelo {modelo} · {n} entidades · texto en minúscula\n")
    X = embed_lote(t1 + t2 + tq, modelo)
    E1, E2, EQ = X[:n], X[n:2 * n], X[2 * n:]
    fallas = []

    # C1 — censo de vectores únicos
    u1 = len(np.unique(E1, axis=0))
    frac = u1 / n
    ok1 = frac >= C1_FRACCION_UNICOS
    print(f"C1 vectores únicos      {u1}/{n} = {frac:.3f}  (mín {C1_FRACCION_UNICOS}) "
          f"{'OK' if ok1 else 'FALLA'}")
    if not ok1:
        fallas.append("C1: el encoder colapsa textos distintos al mismo vector")

    # C2 — discriminación de valores (misma entidad, v1 vs v2)
    cos_val = float((E1 * E2).sum(1).mean())
    ident = int((E1 == E2).all(1).sum())
    ok2 = cos_val < C2_COS_MAX and ident == 0
    print(f"C2 discrimina VALOR     cos medio v1·v2 {cos_val:.6f} (máx {C2_COS_MAX}) · "
          f"idénticos {ident}/{n}  {'OK' if ok2 else 'FALLA'}")
    if not ok2:
        fallas.append("C2: v1 y v2 no se distinguen — la tarea no tendría señal")

    # C3 — discriminación de entidades
    S = E1 @ E1.T
    np.fill_diagonal(S, -9)
    cos_ent = float(S[S > -8].mean())
    ok3 = cos_ent < C3_COS_MAX
    print(f"C3 discrimina ENTIDAD   cos medio entre entidades {cos_ent:.6f} "
          f"(máx {C3_COS_MAX})  {'OK' if ok3 else 'FALLA'}")
    if not ok3:
        fallas.append("C3: las entidades no se distinguen entre sí")

    # C4 — identificación por top-1 y rango mediano
    Sq = EQ @ E1.T
    orden = np.argsort(-Sq, axis=1)
    rango = np.array([int(np.where(orden[i] == i)[0][0]) for i in range(n)])
    top1 = float((rango == 0).mean())
    rmed = float(np.median(rango))
    ok4 = top1 >= C4_TOP1_MIN and rmed <= C4_RANGO_MAX
    print(f"C4 identifica           top-1 {top1:.3f} (mín {C4_TOP1_MIN}) · "
          f"rango mediano {rmed:.1f} (máx {C4_RANGO_MAX})  {'OK' if ok4 else 'FALLA'}")
    if not ok4:
        fallas.append("C4: la consulta no recupera su propio hecho")

    print()
    if fallas:
        print("COMPUERTA CERRADA — no se genera ningún dato del experimento:")
        for f in fallas:
            print("  ·", f)
        return 1
    print(f"COMPUERTA ABIERTA · dim {E1.shape[1]} · el encoder sirve para la tarea de hechos")
    return 0


if __name__ == "__main__":
    modelo = sys.argv[1] if len(sys.argv) > 1 else "nomic-embed-text"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 400
    raise SystemExit(correr(modelo, n))
