"""Genera los embeddings del régimen ELÍPTICO — implementa §3 de PREREG_ELIPTICA.md (299edbd8…).

Por entidad: un ancla auto-contenida (v0) y K_MAX revisiones, cada una en DOS textos —elíptico
(no nombra la entidad) e hidratado (plantilla completa, co-referencia resuelta perfectamente)—.
El barrido de τ del prereg se hace después, en el análisis, eligiendo entre estos dos conjuntos:
no cuesta embeddings.

Nada acá decide nada. Las cuatro formas elípticas y K_MAX están congeladas en el prereg.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tarea_hechos import gen_hechos, ATRIBUTOS, POOL
from correr_hechos import embed_lote, N

K_MAX = 8
SALIDA = "hechos_elipticas.npz"

# §3 del prereg. Ninguna nombra la entidad: eso es lo que las hace elípticas.
ELIPTICAS = [
    "no, it's {v}.",
    "actually, {v}.",
    "sorry, i meant {v}.",
    "correction: {v}.",
]


def main():
    items = gen_hechos(np.random.default_rng(0), N)      # mismo corpus, misma semilla que el resto
    rng = np.random.default_rng(0)

    anclas, consultas = [], []
    t_elip = [[] for _ in range(K_MAX)]
    t_hidr = [[] for _ in range(K_MAX)]

    for x in items:
        pool = POOL[x["atributo"]]
        plantilla = next(p for a, p, _ in ATRIBUTOS if a == x["atributo"])
        # K_MAX+1 valores distintos: el del ancla y uno por revisión
        vals = rng.choice(len(pool), K_MAX + 1, replace=False)
        anclas.append(plantilla.format(e=x["entidad"], v=pool[vals[0]]))
        consultas.append(x["consulta"])
        for r in range(K_MAX):
            v = pool[vals[r + 1]]
            t_elip[r].append(rng.choice(ELIPTICAS).format(v=v))
            t_hidr[r].append(plantilla.format(e=x["entidad"], v=v))

    total = 2 * N + 2 * K_MAX * N
    print(f"generando {total} embeddings ({N} anclas + {N} consultas + "
          f"{K_MAX}×{N} elípticas + {K_MAX}×{N} hidratadas)…", flush=True)

    E0 = embed_lote(anclas)
    EQ = embed_lote(consultas)
    EL = np.stack([embed_lote(t) for t in t_elip])       # (K_MAX, N, d)
    EH = np.stack([embed_lote(t) for t in t_hidr])       # (K_MAX, N, d)

    # Chequeos de integridad ANTES de guardar (la lección de las dos noches perdidas)
    fallo = False
    for r in range(K_MAX - 1):
        for nom, M in (("elípticas", EL), ("hidratadas", EH)):
            ident = int((M[r] == M[r + 1]).all(1).sum())
            if ident:
                print(f"ABORTA: {nom} r{r} y r{r+1} idénticas en {ident}/{N}")
                fallo = True
    for nom, M in (("elípticas", EL), ("hidratadas", EH)):
        u = len(np.unique(M[0], axis=0))
        print(f"  {nom} r0: {u}/{N} vectores únicos")
        if u < 0.95 * N:
            # las elípticas comparten plantilla y sólo varía el valor: menos únicos es ESPERABLE,
            # pero un colapso masivo sería el bug del tokenizador otra vez.
            print(f"  ⚠ {nom}: pocos únicos — revisar antes de leer cualquier veredicto")
    if fallo:
        return 1

    np.savez(SALIDA, E0=E0, EQ=EQ, EL=EL, EH=EH)
    print(f"guardado {SALIDA} — E0 {E0.shape} · EL {EL.shape} · EH {EH.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
