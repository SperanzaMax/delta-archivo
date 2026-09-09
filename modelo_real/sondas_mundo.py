"""Hechos que TinyLlama YA SABE, con su confianza medida · 2026-09-09

Para poder pelear el archivo contra el preentrenamiento hace falta primero saber **qué** sabe el
modelo y **con cuánta confianza**. Este script barre candidatos con la forma «X is in» y se queda
con los que cumplen tres condiciones.

  1. la respuesta verdadera es de UN token en el vocabulario de TinyLlama
  2. el modelo la pone en el top-1
  3. su confianza `c` queda registrada, y es la variable independiente del experimento

Se guarda también un brazo de CONTROL con entidades inventadas de la misma forma, cuya confianza
tiene que dar ~0. Sin ese control no se puede separar «el archivo pierde contra el preentrenamiento»
de «el archivo no funciona con esta forma de pregunta».
"""
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
FORMA = "The {} is in the city of"

# Solo hitos ubicados en una CIUDAD, para que la forma de la pregunta sea una sola y no haya que
# mezclar "city of" con "country of". La uniformidad del formato es parte del diseño.
REALES = """Eiffel Tower|Paris ; Colosseum|Rome ; Louvre|Paris ; Big Ben|London ;
Kremlin|Moscow ; Acropolis|Athens ; Brandenburg Gate|Berlin ; Sagrada Familia|Barcelona ;
Sydney Opera House|Sydney ; White House|Washington ; Vatican|Rome ; Pantheon|Rome ;
Buckingham Palace|London ; Notre Dame|Paris ; Tower Bridge|London ; Empire State Building|New ;
Hermitage Museum|Petersburg ; Prado Museum|Madrid ; Uffizi Gallery|Florence ; Rijksmuseum|Amsterdam ;
Sistine Chapel|Rome ; Trevi Fountain|Rome ; Arc de Triomphe|Paris ; Reichstag|Berlin ;
Guggenheim Museum|Bilbao ; Sorbonne|Paris ; Bolshoi Theatre|Moscow ; Parthenon|Athens ;
Alhambra|Granada ; Duomo|Milan ; Rialto Bridge|Venice ; Charles Bridge|Prague ;
Blue Mosque|Istanbul ; Hagia Sophia|Istanbul ; Forbidden City|Beijing ; Shibuya Crossing|Tokyo"""

# Control: entidades INVENTADAS con la misma forma. Su confianza debe dar ~0.
INVENTADAS = """Zorlin Tower ; Kravec Palace ; Bexil Bridge ; Ondrel Gardens ; Marnak Cathedral ;
Veskar Arena ; Tulmen Fortress ; Garrow Basilica ; Pellin Observatory ; Wynd Citadel ;
Threnn Aqueduct ; Ombra Pavilion ; Saskel Rotunda ; Direnn Colonnade ; Yulvar Amphitheatre"""

tok = AutoTokenizer.from_pretrained(MID)
m = AutoModelForCausalLM.from_pretrained(MID, dtype=torch.float32)
m.eval()


def sondar(ent):
    ids = tok(FORMA.format(ent), return_tensors="pt")
    with torch.no_grad():
        p = torch.softmax(m(**ids).logits[0, -1], -1)
    v, i = p.topk(1)
    return int(i[0]), tok.decode([i[0]]).strip(), float(v[0])


def un_token(pal):
    ids = tok(f" {pal}", add_special_tokens=False)["input_ids"]
    return ids[0] if len(ids) == 1 else None


print(f"barriendo con la forma {FORMA.format('X')!r}\n")
mundo = []
for par in REALES.replace("\n", " ").split(";"):
    if par.count("|") != 1:
        continue
    ent, esp = [x.strip() for x in par.split("|")]
    if not ent or not esp:
        continue
    tid, top, c = sondar(ent)
    t_esp = un_token(esp)
    ok = (t_esp is not None) and (tid == t_esp)
    if ok:
        mundo.append({"entidad": ent, "verdad": esp, "tok_verdad": tid, "confianza": round(c, 4)})
    print(f"  {ent:24s} espera {esp:12s} top1 {top:12s} c={c:.4f} "
          f"{'✓ SIRVE' if ok else '· descartada'}")

ctrl = []
print()
for ent in [e.strip() for e in INVENTADAS.replace("\n", " ").split(";") if e.strip()]:
    tid, top, c = sondar(ent)
    ctrl.append({"entidad": ent, "top1": top, "confianza": round(c, 4)})
    print(f"  CONTROL {ent:22s} top1 {top:12s} c={c:.4f}")

mundo.sort(key=lambda d: d["confianza"])
d = {"modelo": MID, "forma": FORMA, "mundo": mundo, "control": ctrl}
json.dump(d, open("sondas_mundo.json", "w"), ensure_ascii=False, indent=1)
print(f"\nSIRVEN {len(mundo)} hechos reales · confianza de {mundo[0]['confianza']:.4f} "
      f"a {mundo[-1]['confianza']:.4f}" if mundo else "\nNINGUNO SIRVE")
print(f"control inventado, confianza media {sum(c['confianza'] for c in ctrl)/len(ctrl):.4f}")
print("escrito sondas_mundo.json")
