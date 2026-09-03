"""La atenuacion depende de la DISTANCIA o del RELLENO? · y en que capa se cierra la brecha · 3-sep

`escalera_atenuacion.py` dejo la curva y un control degenerado: alargar la pregunta DESPUES de la
entidad da un resultado identico hasta el ultimo decimal, y tenia que darlo, porque nada posterior a
una posicion puede afectarla. Era causalidad, no un control.

El control que si adjudica mantiene la DISTANCIA y cambia el RELLENO. Si la atenuacion depende de d
y no de que palabras hay en el medio, las variantes de una misma d tienen que caer juntas.

Se agrega ademas la pregunta que el preprint necesita contestar: **en que capa se cierra la brecha**,
o sea cuantas capas le cuesta a la recurrencia pagar el impuesto de la ventana.
"""
import json
import os
import sys

import numpy as np
import torch

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
from transformers import AutoTokenizer, AutoModelForCausalLM

# Se declara la d pretendida y el guion la VERIFICA con el tokenizer; si no coincide, aborta.
PLANTILLAS = [
    (2, "of",             "What is the {r} of {e}?"),
    (2, "for",            "What is the {r} for {e}?"),
    (2, "about",          "What is the {r} about {e}?"),
    (3, "of your",        "What is the {r} of your {e}?"),
    (3, "for the",        "What is the {r} for the {e}?"),
    (4, "of the person",  "What is the {r} of the person {e}?"),
    (4, "in the file",    "What is the {r} in the file {e}?"),
    (5, "of the person named", "What is the {r} of the person named {e}?"),
    (5, "of that other person", "What is the {r} of that other person {e}?"),
    (5, "in the file for", "What is the {r} in the file for {e}?"),
    (5, "of our dear friend", "What is the {r} of our dear friend {e}?"),
    (6, "of the very well known", "What is the {r} of the very well known {e}?"),
    (6, "of the person we call", "What is the {r} of the person we call {e}?"),
    (7, "in the records", "What is the {r}, in the records, of {e}?"),
    (7, "of the person that we called", "What is the {r} of the person that we called {e}?"),
]

MODELO = os.environ.get("MODELO", "state-spaces/mamba-130m-hf")
torch.set_num_threads(int(os.environ.get("HILOS", "3")))
tok = AutoTokenizer.from_pretrained(MODELO)
m = AutoModelForCausalLM.from_pretrained(MODELO, dtype=torch.float32).eval()
CAPAS = list(range(len(m.backbone.layers)))

V = json.load(open(os.path.join(AQUI, "vocabulario.json")))
ENTS, RELS, VALS = V["entidades"], V["relaciones"], V["valores"]
N_TEXTOS = int(os.environ.get("N_TEXTOS", "8"))


def contexto(rng):
    es = list(rng.choice(ENTS, size=5, replace=False)); ent = es.pop()
    rs = list(rng.choice(RELS, size=6, replace=False))
    vs = list(rng.choice(VALS, size=4, replace=False))
    hs = [f"The {rs[2+i]} of {es[i]} is {vs[i]}." for i in range(4)]
    return " ".join(hs), ent, rs[0], rs[1]


def capturar(texto):
    ids = tok(texto, return_tensors="pt").input_ids
    g, hs = {}, []
    for i in CAPAS:
        hs.append(m.backbone.layers[i].mixer.conv1d.register_forward_hook(
            (lambda i: lambda _m, _i, o: g.__setitem__(i, o.detach()[0].transpose(0, 1).clone()))(i)))
    with torch.no_grad():
        m(ids)
    for h in hs:
        h.remove()
    return ids[0], g


rng = np.random.default_rng(0)
acum = {n: [] for _d, n, _p in PLANTILLAS}
dmed = {}
for _ in range(N_TEXTOS):
    ctx, ent, r1, r2 = contexto(rng)
    for d_pret, nombre, plantilla in PLANTILLAS:
        i1, g1 = capturar(f"{ctx} {plantilla.format(r=r1, e=ent)}")
        i2, g2 = capturar(f"{ctx} {plantilla.format(r=r2, e=ent)}")
        assert len(i1) == len(i2)
        dif = [k for k in range(len(i1)) if int(i1[k]) != int(i2[k])]
        assert len(dif) == 1, f"{nombre}: {len(dif)} tokens cambiados"
        p_r = dif[0]
        p_e = [k for k in range(len(i1)) if int(i1[k]) == tok(" " + ent).input_ids[0]][-1]
        d = abs(p_e - p_r)
        assert d == d_pret, f"{nombre}: d medida {d}, declarada {d_pret}"
        dmed[nombre] = d
        acum[nombre].append([float((g1[c][p_e] - g2[c][p_e]).abs().max()) for c in CAPAS])

curvas = {n: np.array(v).mean(0) for n, v in acum.items()}
base = np.mean([curvas[n] for _d, n, _p in PLANTILLAS if dmed[n] == 2], axis=0)

MUESTRA = [0, 1, 2, 3, 4, 6, 8, 12, 16, 20, 23]
print(f"modelo {MODELO} · {len(CAPAS)} capas · alcance medido 2 · {N_TEXTOS} contextos\n")
print("ATENUACION contra el promedio de las d=2 · conv1d en la posicion de la ENTIDAD\n")
cab = f"{'relleno':<30} {'d':>2} | " + " ".join(f"{('c'+str(c)):>7}" for c in MUESTRA)
print(cab); print("-" * len(cab))
for d_pret, nombre, _p in PLANTILLAS:
    a = curvas[nombre]
    print(f"{nombre:<30} {dmed[nombre]:>2} | " +
          " ".join((f"{base[c]/a[c]:>7.1f}" if a[c] > 0 else f"{'inf':>7}") for c in MUESTRA))

print("\nPOR DISTANCIA · mediana entre rellenos, y DISPERSION entre rellenos de la misma d")
print(f"\n{'d':>2} {'rellenos':>8} | " + " ".join(f"{('c'+str(c)):>7}" for c in MUESTRA) +
      "   (linea 2: max/min entre rellenos)")
print("-" * (14 + 8 * len(MUESTRA)))
resumen = {}
for d in sorted(set(dmed.values())):
    ns = [n for n in curvas if dmed[n] == d]
    M = np.array([base / np.maximum(curvas[n], 1e-30) for n in ns])
    med = np.median(M, axis=0)
    disp = M.max(0) / np.maximum(M.min(0), 1e-30)
    resumen[d] = dict(mediana=med.tolist(), dispersion=disp.tolist(), rellenos=ns)
    print(f"{d:>2} {len(ns):>8} | " + " ".join(
        (f"{med[c]:>7.1f}" if np.isfinite(med[c]) else f"{'inf':>7}") for c in MUESTRA))
    print(f"{'':>2} {'':>8} | " + " ".join(f"{disp[c]:>7.2f}" for c in MUESTRA))

print("\n¿EN QUE CAPA SE CIERRA LA BRECHA?  primera capa con atenuacion < 1,5x, por distancia")
for d in sorted(resumen):
    med = np.array(resumen[d]["mediana"])
    cs = [c for c in CAPAS if np.isfinite(med[c]) and med[c] < 1.5]
    print(f"  d={d}:  " + (f"capa {cs[0]}" if cs else "no se cierra en 24 capas") +
          f"   · atenuacion en la capa 1: " +
          (f"x{med[1]:.2f}" if np.isfinite(med[1]) else "inf") +
          f" · en la ultima: x{med[CAPAS[-1]]:.2f}")

json.dump({"curvas": {n: c.tolist() for n, c in curvas.items()}, "d": dmed,
           "resumen": {str(k): v for k, v in resumen.items()}},
          open(os.path.join(AQUI, "escalera_v2.json"), "w"), indent=1)
print("\nguardado en escalera_v2.json")
