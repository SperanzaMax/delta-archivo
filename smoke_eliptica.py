"""SMOKE de viabilidad — ¿la corrección ELÍPTICA rompe el supuesto que cerró la gemación?

El cierre del 2026-08-10 (INFORME_GEMACION_ACOTADA.md) concluyó que la geometría no tiene nada que
aportar porque «emb(v_r) ya está óptimamente colocado: contiene la entidad que la consulta menciona».
Eso es cierto del GENERADOR (generar_revisiones.py línea 25 mete la entidad en cada versión), no de
las correcciones reales, que suelen ser elípticas: «no, es Beto».

Esto NO es un experimento y no decide nada: mide si hay RANGO para uno. Si el coseno de la corrección
elíptica con la consulta cae muy por debajo del peaje de la gemación (~0,036 medido ayer), el régimen
es distinto y el cierre no lo cubre. Si no cae, el cierre generaliza y la línea queda muerta de verdad.
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tarea_hechos import gen_hechos, ATRIBUTOS, POOL
from correr_hechos import embed_lote

N = 60

# Cómo se corrige un hecho cuando ya se está hablando de él. Ninguna menciona la entidad.
ELIPTICAS = [
    "no, it's {v}.",
    "actually, {v}.",
    "sorry, i meant {v}.",
    "correction: {v}.",
]


def main():
    items = gen_hechos(np.random.default_rng(0), N)
    rng = np.random.default_rng(1)

    consultas, autocont, elipticas, anclas = [], [], [], []
    for x in items:
        pool = POOL[x["atributo"]]
        plantilla = next(p for a, p, _ in ATRIBUTOS if a == x["atributo"])
        v1, v2 = (pool[i] for i in rng.choice(len(pool), 2, replace=False))
        consultas.append(x["consulta"])
        anclas.append(plantilla.format(e=x["entidad"], v=v1))          # v1: el ancla ya archivada
        autocont.append(plantilla.format(e=x["entidad"], v=v2))        # v2 como la genera el harness
        elipticas.append(rng.choice(ELIPTICAS).format(v=v2))           # v2 como se dice de verdad

    Q, A, C, E = (embed_lote(t) for t in (consultas, anclas, autocont, elipticas))

    cos = lambda X, Y: float(np.mean(np.sum(X * Y, axis=1)))
    q_ancla, q_auto, q_elip = cos(Q, A), cos(Q, C), cos(Q, E)
    ancla_elip = cos(A, E)

    print(f"\n  N = {N} entidades · encoder nomic-embed-text (minúscula)\n")
    print(f"  coseno(consulta, ancla v1 auto-contenida)   = {q_ancla:.4f}")
    print(f"  coseno(consulta, v2 AUTO-CONTENIDA)         = {q_auto:.4f}   <- el régimen ya medido")
    print(f"  coseno(consulta, v2 ELÍPTICA)               = {q_elip:.4f}   <- el régimen no medido")
    print(f"  coseno(ancla v1, v2 ELÍPTICA)               = {ancla_elip:.4f}")
    print(f"\n  caída por elipsis: {q_auto - q_elip:+.4f}   ·   peaje medido de la gemación: 0,036")

    # ¿La corrección elíptica siquiera gana a un distractor de OTRA entidad?
    perm = np.roll(np.arange(N), 7)
    print(f"  coseno(consulta, ELÍPTICA de otra entidad)  = {cos(Q, E[perm]):.4f}   (distractor)")
    top1 = float(np.mean(np.argmax(Q @ E.T, axis=1) == np.arange(N)))
    print(f"  top-1 de la elíptica correcta entre las {N}  = {top1:.4f}   (azar = {1/N:.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
