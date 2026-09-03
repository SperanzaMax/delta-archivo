"""La ventana no BLOQUEA en un modelo profundo: ATENUA · la curva completa · 3-sep

`sonda_combinacion.py` dio dos puntos: a distancia 2 la conv de la entidad ve la relacion (4,8e-01 en
la capa 0) y a distancia 5 no la ve (0,0 exacto), pero desde la capa 1 ya la ve un poco (9,9e-02) y
esa señal se queda entre 5 y 8 veces por debajo hasta la capa 23.

Dos puntos no son una curva y ademas viajan con un confound: las formas separadas son mas LARGAS.
Aca se levanta la escalera entera con el control que adjudica.

    ESCALERA      d(rel<->ent) = 2, 3, 4, 5, 7   con largos crecientes
    CONTROL       d(rel<->ent) = 2 con el largo de las de arriba

Si lo que manda es la DISTANCIA, el control se queda arriba con `directa`. Si lo que manda es el
LARGO de la pregunta, el control baja con las separadas. Es el mismo par de lecturas que separo
`cl3` de `cf3` en el micro-LM, y aca se puede medir sin entrenar nada.

Se reporta ademas el CONTRASTE (la salida de la capa en el final tiene que moverse en todas), que es
lo que hace legible a un cero: sin eso, un cero solo diria que el modelo ignora el token.
"""
import json
import os
import sys

import numpy as np
import torch

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
from transformers import AutoTokenizer, AutoModelForCausalLM

# (nombre, plantilla, papel).  `escalera` sube la distancia; `control` la deja en 2 y sube el largo.
FORMAS = [
    ("d2",       "What is the {r} of {e}?",                              "escalera"),
    ("d3",       "What is the {r} of your {e}?",                         "escalera"),
    ("d4",       "What is the {r} of the person {e}?",                   "escalera"),
    ("d5",       "What is the {r} of the person named {e}?",             "escalera"),
    ("d7",       "What is the {r}, in the records, of {e}?",             "escalera"),
    ("d2_largo1", "What is the {r} of {e}, right?",                      "control"),
    ("d2_largo2", "What is the {r} of {e}, in the records?",             "control"),
    ("d2_largo3", "What is the {r} of {e}, as the person named above?",  "control"),
]

MODELO = os.environ.get("MODELO", "state-spaces/mamba-130m-hf")
torch.set_num_threads(3)
tok = AutoTokenizer.from_pretrained(MODELO)
m = AutoModelForCausalLM.from_pretrained(MODELO, dtype=torch.float32).eval()
CAPAS = list(range(len(m.backbone.layers)))

V = json.load(open(os.path.join(AQUI, "vocabulario.json")))
ENTS, RELS = V["entidades"], V["relaciones"]
VALS = V["valores"]
rng = np.random.default_rng(0)
N_TEXTOS = int(os.environ.get("N_TEXTOS", "6"))     # promediar, para no colgarse de un ejemplo


def contexto_y_piezas(rng):
    es = list(rng.choice(ENTS, size=5, replace=False))
    ent = es.pop()
    rs = list(rng.choice(RELS, size=6, replace=False))
    r1, r2 = rs[0], rs[1]
    vs = list(rng.choice(VALS, size=4, replace=False))
    hechos = [f"The {rs[2+i]} of {es[i]} is {vs[i]}." for i in range(4)]
    return " ".join(hechos), ent, r1, r2


def capturar(texto):
    ids = tok(texto, return_tensors="pt").input_ids
    g = {}
    hs = []

    def h_conv(i):
        return lambda _m, _i, o: g.__setitem__(("conv", i), o.detach()[0].transpose(0, 1).clone())

    def h_capa(i):
        def f(_m, _i, o):
            g[("capa", i)] = (o[0] if isinstance(o, tuple) else o).detach()[0].clone()
        return f

    for i in CAPAS:
        hs.append(m.backbone.layers[i].mixer.conv1d.register_forward_hook(h_conv(i)))
        hs.append(m.backbone.layers[i].register_forward_hook(h_capa(i)))
    with torch.no_grad():
        m(ids)
    for h in hs:
        h.remove()
    return ids[0], g


acum = {n: {"conv_ent": [], "capa_fin": [], "d_re": None, "largo": None} for n, _, _ in FORMAS}
for t in range(N_TEXTOS):
    ctx, ent, r1, r2 = contexto_y_piezas(rng)
    for nombre, plantilla, _papel in FORMAS:
        t1 = f"{ctx} {plantilla.format(r=r1, e=ent)}"
        t2 = f"{ctx} {plantilla.format(r=r2, e=ent)}"
        i1, g1 = capturar(t1)
        i2, g2 = capturar(t2)
        assert len(i1) == len(i2)
        dif = [k for k in range(len(i1)) if int(i1[k]) != int(i2[k])]
        assert len(dif) == 1, f"{nombre}: {len(dif)} tokens cambiados"
        p_r = dif[0]
        p_e = [k for k in range(len(i1)) if int(i1[k]) == tok(" " + ent).input_ids[0]][-1]
        fin = len(i1) - 1
        acum[nombre]["d_re"] = abs(p_e - p_r)
        acum[nombre]["largo"] = fin - p_r          # d(rel -> fin), o sea cuanto se alarga la cola
        acum[nombre]["conv_ent"].append(
            [float((g1[("conv", c)][p_e] - g2[("conv", c)][p_e]).abs().max()) for c in CAPAS])
        acum[nombre]["capa_fin"].append(
            [float((g1[("capa", c)][fin] - g2[("capa", c)][fin]).abs().max()) for c in CAPAS])

MUESTRA = [0, 1, 2, 3, 5, 8, 12, 16, 20, 23]
print(f"modelo {MODELO} · {len(CAPAS)} capas · alcance medido 2 · promedio de {N_TEXTOS} contextos\n")
print("conv1d en la posicion de la ENTIDAD al cambiar la RELACION  (0,0 = no la puede ver)\n")
cab = f"{'forma':<11} {'papel':<9} {'d(r<->e)':>8} {'cola':>5} | " + \
      " ".join(f"{('c'+str(c)):>8}" for c in MUESTRA)
print(cab); print("-" * len(cab))
res = {}
for nombre, _p, papel in FORMAS:
    a = np.array(acum[nombre]["conv_ent"]).mean(0)
    res[nombre] = dict(papel=papel, d_re=acum[nombre]["d_re"], cola=acum[nombre]["largo"],
                       conv_ent=a.tolist(),
                       capa_fin=np.array(acum[nombre]["capa_fin"]).mean(0).tolist())
    print(f"{nombre:<11} {papel:<9} {acum[nombre]['d_re']:>8} {acum[nombre]['largo']:>5} | " +
          " ".join(f"{a[c]:>8.2e}" for c in MUESTRA))

print("\nCONTRASTE · salida de la CAPA en la posicion final (tiene que moverse en TODAS)\n")
print(cab); print("-" * len(cab))
for nombre, _p, papel in FORMAS:
    a = np.array(acum[nombre]["capa_fin"]).mean(0)
    print(f"{nombre:<11} {papel:<9} {acum[nombre]['d_re']:>8} {acum[nombre]['largo']:>5} | " +
          " ".join(f"{a[c]:>8.2e}" for c in MUESTRA))

base = np.array(res["d2"]["conv_ent"])
print("\nATENUACION contra `d2`, por capa (cuantas veces mas debil ve la relacion)\n")
print(cab); print("-" * len(cab))
for nombre, _p, papel in FORMAS:
    a = np.array(res[nombre]["conv_ent"])
    rz = np.divide(base, np.maximum(a, 1e-30))
    print(f"{nombre:<11} {papel:<9} {res[nombre]['d_re']:>8} {res[nombre]['cola']:>5} | " +
          " ".join((f"{rz[c]:>8.1f}" if a[c] > 0 else f"{'inf':>8}") for c in MUESTRA))

sin_c0 = [c for c in CAPAS if c >= 1]
print("\nRESUMEN · mediana de la atenuacion sobre las capas 1-23 (la capa 0 es cero exacto y no entra)")
for nombre, _p, papel in FORMAS:
    a = np.array(res[nombre]["conv_ent"])
    med = float(np.median(base[sin_c0] / np.maximum(a[sin_c0], 1e-30)))
    print(f"  {nombre:<11} {papel:<9} d={res[nombre]['d_re']}  cola={res[nombre]['cola']}"
          f"   atenuacion x{med:.2f}")

json.dump(res, open(os.path.join(AQUI, "escalera_atenuacion.json"), "w"), indent=1)
print("\nguardado en escalera_atenuacion.json")
