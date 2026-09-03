"""¿Comparten ventana la relacion y la entidad? · verificado por INTERVENCION, no por aritmetica · 3-sep

`geometria_formas.py` cuenta que en `directa` la relacion queda a 2 tokens de la entidad y el alcance
medido de la conv es 2, o sea que la conv de la posicion de la ENTIDAD la alcanza justo. Eso es
aritmetica. Aca se comprueba en el modelo, cambiando el token de la relacion y mirando DONDE se mueve
la salida de `conv1d`.

Se miden tres cosas por forma, todas en la capa 0:

  A. `conv1d` en la posicion de la ENTIDAD    -> ¿puede la busqueda de ahi ver la relacion?
  B. `conv1d` en la posicion FINAL (el `?`)   -> la medicion del 2-sep, que dio 0,0 exacto
  C. la salida de la CAPA en la posicion final -> el contraste: el estado si tiene que moverse

C es el control que hace legible a A y B, misma logica que R-2 el 2-sep: si tampoco se moviera, lo
unico demostrado seria que el modelo ignora el token, que es falso.

Y se agrega la profundidad, que es la objecion obvia: con 24 capas el alcance efectivo se acumula,
asi que se reporta A y B **por capa** para ver a partir de cual la relacion llega igual.
"""
import json
import os
import sys

import torch

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)

from transformers import AutoTokenizer, AutoModelForCausalLM

V = json.load(open(os.path.join(AQUI, "vocabulario.json")))
PLANTILLAS = dict(V["plantillas"])
PLANTILLAS["separada"] = "What is the {r} of the person named {e}?"
PLANTILLAS["separada2"] = "What is the {r}, in the records, of {e}?"

MODELO = os.environ.get("MODELO", "state-spaces/mamba-130m-hf")
torch.set_num_threads(3)
tok = AutoTokenizer.from_pretrained(MODELO)
m = AutoModelForCausalLM.from_pretrained(MODELO, dtype=torch.float32).eval()
N_CAPAS = len(m.backbone.layers)

CONTEXTO = ("The keeper of Bishop is Gordon. The builder of Barker is Graham. "
            "The trustee of Baldwin is Hamilton. The auditor of Bennett is Harper.")
ENT = "Barton"
R1, R2 = "director", "founder"          # dos relaciones, las dos de un token


def capturar(texto, capas):
    """Devuelve, por capa, la salida de conv1d y la salida de la capa. Todo [T, C]."""
    ids = tok(texto, return_tensors="pt").input_ids
    guardado = {}
    ganchos = []

    def h_conv(i):
        def f(_mod, _inp, out):
            guardado[("conv", i)] = out.detach()[0].transpose(0, 1).clone()   # [T, C]
        return f

    def h_capa(i):
        def f(_mod, _inp, out):
            o = out[0] if isinstance(out, tuple) else out
            guardado[("capa", i)] = o.detach()[0].clone()
        return f

    for i in capas:
        ganchos.append(m.backbone.layers[i].mixer.conv1d.register_forward_hook(h_conv(i)))
        ganchos.append(m.backbone.layers[i].register_forward_hook(h_capa(i)))
    with torch.no_grad():
        m(ids)
    for g in ganchos:
        g.remove()
    return ids[0], guardado


CAPAS = [0, 1, 2, 3, 5, 8, 12, 23][:N_CAPAS]
CAPAS = [c for c in CAPAS if c < N_CAPAS]

print(f"modelo {MODELO} · {N_CAPAS} capas · alcance medido 2\n")
print(f"{'forma':<11} {'d(r<->e)':>8} | {'conv@ENT':>10} {'conv@FIN':>10} {'capa@FIN':>10}   (capa 0)")
print("-" * 68)

resumen = {}
for nombre, plantilla in PLANTILLAS.items():
    t1 = f"{CONTEXTO} {plantilla.format(r=R1, e=ENT)}"
    t2 = f"{CONTEXTO} {plantilla.format(r=R2, e=ENT)}"
    ids1, g1 = capturar(t1, CAPAS)
    ids2, g2 = capturar(t2, CAPAS)
    assert len(ids1) == len(ids2), f"{nombre}: las dos versiones no miden lo mismo"
    dif_tok = [i for i in range(len(ids1)) if int(ids1[i]) != int(ids2[i])]
    assert len(dif_tok) == 1, f"{nombre}: cambiaron {len(dif_tok)} tokens, tiene que ser 1"
    p_r = dif_tok[0]
    id_e = tok(" " + ENT).input_ids[0]
    p_e = [i for i in range(len(ids1)) if int(ids1[i]) == id_e][-1]
    fin = len(ids1) - 1

    fila = {"d_re": abs(p_e - p_r), "d_rel_fin": fin - p_r, "por_capa": {}}
    for c in CAPAS:
        dc_e = float((g1[("conv", c)][p_e] - g2[("conv", c)][p_e]).abs().max())
        dc_f = float((g1[("conv", c)][fin] - g2[("conv", c)][fin]).abs().max())
        dl_f = float((g1[("capa", c)][fin] - g2[("capa", c)][fin]).abs().max())
        fila["por_capa"][c] = dict(conv_ent=dc_e, conv_fin=dc_f, capa_fin=dl_f)
    resumen[nombre] = fila
    c0 = fila["por_capa"][0]
    print(f"{nombre:<11} {fila['d_re']:>8} | {c0['conv_ent']:>10.3e} {c0['conv_fin']:>10.3e} "
          f"{c0['capa_fin']:>10.3e}")

print("\n\nPOR CAPA · max|dif| de conv1d en la posicion de la ENTIDAD al cambiar la RELACION")
print("(si es 0,0 exacto, la busqueda de esa posicion y esa capa NO puede ver la relacion)\n")
print(f"{'forma':<11} " + " ".join(f"{('c'+str(c)):>10}" for c in CAPAS))
print("-" * (12 + 11 * len(CAPAS)))
for nombre, f in resumen.items():
    print(f"{nombre:<11} " + " ".join(f"{f['por_capa'][c]['conv_ent']:>10.2e}" for c in CAPAS))

print("\nPOR CAPA · lo mismo en la posicion FINAL (el `?`)\n")
print(f"{'forma':<11} " + " ".join(f"{('c'+str(c)):>10}" for c in CAPAS))
print("-" * (12 + 11 * len(CAPAS)))
for nombre, f in resumen.items():
    print(f"{nombre:<11} " + " ".join(f"{f['por_capa'][c]['conv_fin']:>10.2e}" for c in CAPAS))

json.dump(resumen, open(os.path.join(AQUI, "sonda_combinacion.json"), "w"), indent=1)
print("\nguardado en sonda_combinacion.json")
