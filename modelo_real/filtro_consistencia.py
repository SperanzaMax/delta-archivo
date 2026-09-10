"""¿La consistencia bajo reformulación separa lo que el modelo SABE de lo que NO? · 2026-09-09

Es el primer ladrillo de la extracción. Si se quiere volcar lo que un modelo ya entrenado sabe hacia
el archivo persistente, hay que filtrar, porque extraer sin filtro copia las alucinaciones al
archivo Y ADEMAS les pone sello de procedencia, que es peor que dejarlas en los pesos.

La hipotesis del filtro es que preguntar lo mismo de varias formas separa las dos cosas. Se prueba
con material donde la verdad se conoce por construccion.

  23 hitos REALES      el modelo los sabe, verificado, top-1 correcto
  15 INVENTADOS        ficcion pura, no hay nada que saber

Si la consistencia separa esos dos grupos, el filtro sirve. Y lo que descarta NO es basura: es la
etiqueta de ignorancia, que es justo lo que al entrenamiento de abstencion le falta.
"""
import json
from collections import Counter

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
FORMAS = [
    "The {} is in the city of",
    "{} is located in the city of",
    "You can find the {} in the city of",
    "The {} is situated in the city of",
    "Where is the {}? It is in the city of",
    "Q: In which city is the {}? A: In the city of",
    "Tourists visiting the {} travel to the city of",
    "The famous {} stands in the city of",
]

tok = AutoTokenizer.from_pretrained(MID)
m = AutoModelForCausalLM.from_pretrained(MID, dtype=torch.float32)
m.eval()
sd = json.load(open("sondas_mundo.json"))


def perfil(ent):
    """Devuelve las 8 respuestas y su confianza, una por reformulacion."""
    rs = []
    for f in FORMAS:
        ids = tok(f.format(ent), return_tensors="pt")
        with torch.no_grad():
            p = torch.softmax(m(**ids).logits[0, -1], -1)
        v, i = p.topk(1)
        rs.append((tok.decode([i[0]]).strip(), float(v[0])))
    tops = [r for r, _ in rs]
    c = Counter(tops)
    modal, n = c.most_common(1)[0]
    return {
        "consistencia": n / len(FORMAS),      # fraccion que coincide con la respuesta modal
        "distintas": len(c),                  # cuantas respuestas distintas dio
        "modal": modal,
        "conf_media": sum(v for _, v in rs) / len(rs),
        "respuestas": tops,
    }


print(f"{len(FORMAS)} reformulaciones por entidad\n")
res = {"reales": [], "inventadas": []}
for grupo, clave, lst in (("reales", "mundo", sd["mundo"]),
                          ("inventadas", "control", sd["control"])):
    print(f"=== {grupo.upper()} ===")
    for e in lst:
        p = perfil(e["entidad"])
        p["entidad"] = e["entidad"]
        p["verdad"] = e.get("verdad")
        p["acierta"] = (p["modal"] == e["verdad"]) if e.get("verdad") else None
        res[grupo].append(p)
        marca = "" if p["verdad"] is None else (" ✓" if p["acierta"] else " ✗")
        print(f"  {e['entidad']:22s} consist {p['consistencia']:.3f} "
              f"({p['distintas']} distintas) modal {p['modal']:12s}"
              f" conf {p['conf_media']:.3f}{marca}")
    print()

import statistics as st
for g in ("reales", "inventadas"):
    cs = [p["consistencia"] for p in res[g]]
    ds = [p["distintas"] for p in res[g]]
    print(f"{g:11s} n={len(cs):2d} · consistencia media {st.mean(cs):.4f} "
          f"(min {min(cs):.3f}) · respuestas distintas {st.mean(ds):.2f}")
a = [p["consistencia"] for p in res["reales"]]
b = [p["consistencia"] for p in res["inventadas"]]
sep = st.mean(a) - st.mean(b)
sd_p = ((st.pstdev(a) ** 2 + st.pstdev(b) ** 2) / 2) ** 0.5
print(f"\nSEPARACION  {sep:+.4f}  ·  d de Cohen {sep / sd_p if sd_p else float('nan'):.2f}")
print(f"  solapamiento: reales por debajo del maximo inventado "
      f"({max(b):.3f}): {sum(1 for x in a if x <= max(b))} de {len(a)}")
json.dump(res, open("filtro_consistencia.json", "w"), ensure_ascii=False, indent=1)
print("\nescrito filtro_consistencia.json")
