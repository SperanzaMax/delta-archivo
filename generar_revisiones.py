"""Genera los embeddings de K_MAX revisiones REALES por entidad, para P4 (D2).

La v1 de P4 modelaba dónde caía cada revisión; esto lo mide. Cada revisión es la misma plantilla con
un valor distinto — exactamente lo que es una corrección sucesiva del mismo hecho.
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tarea_hechos import gen_hechos, ATRIBUTOS, POOL
from correr_hechos import embed_lote, N

K_MAX = 8
SALIDA = "hechos_revisiones.npz"


def main():
    rng = np.random.default_rng(0)                     # D1: mismo corpus, semilla 0
    items = gen_hechos(np.random.default_rng(0), N)
    # K_MAX+1 valores DISTINTOS por entidad, del pool de su atributo
    textos = [[] for _ in range(K_MAX + 1)]
    for x in items:
        pool = POOL[x["atributo"]]
        plantilla = next(p for a, p, _ in ATRIBUTOS if a == x["atributo"])
        vals = rng.choice(len(pool), K_MAX + 1, replace=False)
        for r in range(K_MAX + 1):
            textos[r].append(plantilla.format(e=x["entidad"], v=pool[vals[r]]))
    print(f"generando {(K_MAX+1)*N + N} embeddings ({K_MAX+1} versiones + consultas)…", flush=True)
    EV = np.stack([embed_lote(t) for t in textos])     # (K+1, N, d)
    EQ = embed_lote([x["consulta"] for x in items])
    np.savez(SALIDA, EV=EV, EQ=EQ)
    print(f"guardado {SALIDA} — EV {EV.shape}")
    # chequeo de discriminación entre versiones consecutivas
    for r in range(K_MAX):
        ident = int((EV[r] == EV[r + 1]).all(1).sum())
        if ident:
            print(f"ABORTA: v{r} y v{r+1} idénticos en {ident}/{N}")
            return 1
    print("chequeo de discriminación entre versiones: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
