"""Elegir entidades, valores y relaciones que tokenicen a UN SOLO token · 2026-09-02

Por que hace falta. La compuerta midio que con nombres como «Zephyra», que el BPE parte en tres
piezas, las distancias VARIAN (d_rel 4..6 en la forma directa). La clasificacion adentro/afuera
aguanta, pero la distancia deja de ser una constante del diseño y pasa a depender del nombre. Peor
todavia: en la forma directa la ENTIDAD queda a distancia 2..3, o sea que con los nombres largos se
sale de la ventana ella tambien, y entonces las dos formas dejan de diferir en una sola cosa.

Con piezas de UN token la geometria queda fija y replica exactamente la del micro-LM.
"""
import os
import sys
import string

from transformers import AutoTokenizer

MODELO = os.environ.get("MODELO", "state-spaces/mamba-370m-hf")
tok = AutoTokenizer.from_pretrained(MODELO)


def n_tokens(pal, con_espacio=True):
    return len(tok(" " + pal if con_espacio else pal).input_ids)


CAND_ENT = """Zeph Quan Velm Torr Nyx Gren Cass Umbr Fen Aldr Bex Corv Marl Doon Wex Yarn Kilm Prax
Vorn Thal Brem Grix Halv Jorn Kade Lome Nurr Osk Pell Rask Sorn Tull Varn Wisk Yorl Zarn Blane Crum
Dask Emir Frez Glim Hask Ivor Jask Kren Lorn Mure Nash Orin Pyre Quil Rune Skel Trum Ulm Vane Wren
Xan Yale Zorn Ashby Corby Denby Elmby Kirby Selby Tenby Ferro Gallo Milo Nero Otto Piro Rilo Sabo
Tero Vico Xylo Yaro Zeno""".split()

CAND_VAL = """Kalen Bryse Doran Elvire Fyrn Gastel Hollis Imre Jarek Loris Meris Norel Oris Peral
Quinn Roran Silas Toren Ulric Varek Wendel Xander Yorick Zane Alden Brann Cedric Dorian Eamon Finn
Gareth Hale Ivan Joss Kern Lars Merek Nolan Osric Piers Quill Rowan Stellan Thane Ulf Vance Wyatt
Yance Zeph Alaric Baird Cort Drake Edric Fabian Gunnar Hew Ives Jorah Kelan Lorne""".split()

CAND_REL = """director warden founder keeper curator steward captain guardian ranger builder
sponsor trustee auditor herald marshal patron scribe sentry surveyor warden""".split()


def elegir(cands, n, nombre):
    uno = [c for c in cands if n_tokens(c) == 1]
    print(f"  {nombre:12s} candidatos {len(cands):3d} · de UN token {len(uno):3d} · se usan {min(n, len(uno))}")
    return uno[:n]


print("=" * 96)
print(f"VOCABULARIO DE UN TOKEN para {MODELO}")
print("=" * 96)
ENT = elegir(CAND_ENT, 24, "entidades")
VAL = elegir(CAND_VAL, 16, "valores")
REL = elegir(CAND_REL, 8, "relaciones")

print(f"\n  entidades  {ENT}")
print(f"\n  valores    {VAL}")
print(f"\n  relaciones {REL}")

PLANTILLAS = {
    "directa":   "What is the {r} of {e}?",
    "invertida": "For {e}, what is the {r}?",
    "lejana":    "What is the {r} that {e} has?",
}

print("\n" + "=" * 96)
print("DISTANCIAS con el vocabulario elegido (deberian ser FIJAS)")
print("=" * 96)
ok_fijas = True
for nom, plt in PLANTILLAS.items():
    drs, des = set(), set()
    for e in ENT:
        for r in REL:
            ids = tok(plt.format(r=r, e=e)).input_ids
            pz = [tok.decode([i]).strip() for i in ids]
            n = len(ids)
            drs.add(n - 1 - max(j for j, p in enumerate(pz) if p == r))
            des.add(n - 1 - max(j for j, p in enumerate(pz) if p == e))
    fija = len(drs) == 1 and len(des) == 1
    ok_fijas &= fija
    ej = [tok.decode([i]) for i in tok(plt.format(r=REL[0], e=ENT[0])).input_ids]
    print(f"  {nom:10s} d_rel={sorted(drs)}  d_ent={sorted(des)}  "
          f"{'FIJAS' if fija else '** VARIAN **'}")
    print(f"             {ej}")

ALC = 2
print("\n" + "=" * 96)
print(f"CLASIFICACION contra el alcance REAL medido ({ALC})")
print("=" * 96)
for nom, plt in PLANTILLAS.items():
    ids = tok(plt.format(r=REL[0], e=ENT[0])).input_ids
    pz = [tok.decode([i]).strip() for i in ids]
    n = len(ids)
    dr = n - 1 - max(j for j, p in enumerate(pz) if p == REL[0])
    de = n - 1 - max(j for j, p in enumerate(pz) if p == ENT[0])
    print(f"  {nom:10s} relacion d={dr} {'ADENTRO' if dr <= ALC else 'AFUERA ':8s}"
          f"  entidad d={de} {'ADENTRO' if de <= ALC else 'AFUERA'}")
print("=" * 96)
print(f"  {'LISTO para escribir el prereg' if ok_fijas else 'NO: las distancias todavia varian'}")
