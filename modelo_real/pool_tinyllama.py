"""Pool de piezas de UN token para el tokenizador de TinyLlama · 2026-09-09

Por que hace falta uno nuevo. `pool_un_token.json` se armo el 2-sep contra el tokenizador de Mamba
(GPT-NeoX) y NO sirve para Llama, que tokeniza distinto.

⚠️ Y hay un segundo filo, verificado hoy y que conviene dejar escrito. Con transformers 5.3.0 el
token de espacio ya NO se emite:

    tok(" Cordoba", add_special_tokens=False) -> [20893, 15330]        ['▁Cord', 'oba']
    (antes)                                   -> [29871, 20893, 15330] ['▁', '▁Cord', 'oba']

O sea la linea de `archivo_en_real.py` que toma el **indice 1** apuntaba a '▁Cord' cuando se
escribio y hoy apunta a 'oba'. Los ocho valores siguen dando ocho objetivos distintos en las dos
versiones, asi que el resultado del 6-sep no se cae, pero **la tarea cambia de version en version**
y eso no puede quedar en un banco que se va a publicar.

Regla de este script, y con eso el problema desaparece de raiz: una pieza sirve si
`tok(" X", add_special_tokens=False).input_ids` tiene **exactamente 1** elemento. El objetivo es la
palabra ENTERA y no depende de que indice se tome.
"""
import json, sys
from transformers import AutoTokenizer

MID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
N_ENT, N_VAL = 60, 100

# Piezas de verdad, no fragmentos del vocabulario. El barrido ciego del vocabulario da 3.903
# candidatos pero son cosas como 'Abb', 'Abd', 'Academ': tokenizan bien y no significan nada, y con
# eso la tarea deja de ser legible en los logs.
CAND_ENT = """Ana Anna Maria Julia Laura Emma Sarah Eva Clara Alice Helen Rosa Grace Ruth Jane Kate
Lisa Mary Susan Peter Paul David Simon Thomas Michael Daniel Robert Richard James John Mark Luis
Carlos Jose Juan Pedro Pablo Diego Miguel Antonio Manuel Martin Alberto Roberto Fernando Ricardo
Eduardo Rafael Victor Oscar Hugo Bruno Marco Franco Mario Sergio Adrian Ivan Felix Leo Max Tom Ben
Sam Nick Jack Adam Alex Chris Eric Frank George Henry Jacob Kevin Nathan Oliver Diana Nina Rachel"""

CAND_VAL = """Boston Paris London Berlin Madrid Rome Vienna Moscow Dublin Stockholm Amsterdam
Hamburg Frankfurt Milan Florence Barcelona Porto Chicago Detroit Houston Dallas Phoenix Seattle
Atlanta Miami Baltimore Cleveland Austin Toronto Montreal Sydney Melbourne Tokyo Lima Santiago
Mexico Oxford Cambridge Manchester Liverpool Glasgow Edinburgh York Bath Kent France Spain Italy
Germany Poland Greece Norway Sweden Finland Ireland Portugal Austria Ukraine Russia Turkey Egypt
Brazil Chile Peru Uruguay Colombia Venezuela Cuba Canada Texas Ohio Iowa Kansas Oregon Maine
Georgia Florida Virginia Carolina Alabama Arizona Indiana Illinois Michigan Missouri Oklahoma
Maryland India China Japan Korea Taiwan Vietnam Pakistan Iran Israel Jordan Australia Zealand Wales
Scotland England"""

tok = AutoTokenizer.from_pretrained(MID)


def pieza(pal):
    ids = tok(f" {pal}", add_special_tokens=False)["input_ids"]
    return ids[0] if len(ids) == 1 else None


def elegir(cands, n, nombre, usados):
    ok = []
    for w in cands.split():
        t = pieza(w)
        if t is not None and t not in usados:
            usados.add(t)
            ok.append(w)
    print(f"  {nombre:11s} candidatos {len(set(cands.split())):3d} · de UN token y unicos "
          f"{len(ok):3d} · se usan {n}")
    if len(ok) < n:
        sys.exit(f"ALCANZA PARA {len(ok)}, hacen falta {n}")
    return ok[:n]


print("=" * 78)
print(f"VOCABULARIO DE UN TOKEN para {MID}")
print("=" * 78)
usados = set()
# La abstencion primero, para que se quede con su token y ningun valor se lo pise.
ABST = next(w for w in ("None", "Unknown", "Nothing") if pieza(w) is not None)
usados.add(pieza(ABST))
ENT = elegir(CAND_ENT, N_ENT, "entidades", usados)
VAL = elegir(CAND_VAL, N_VAL, "valores", usados)

d = {"modelo": MID, "abstencion": ABST, "tok_abstencion": pieza(ABST),
     "entidades": ENT, "valores": VAL,
     "tok_entidades": [pieza(w) for w in ENT], "tok_valores": [pieza(w) for w in VAL]}

# Guardas de identidad. Si alguna salta, el banco no se escribe.
assert len(set(d["tok_valores"])) == N_VAL, "objetivos repetidos entre valores"
assert d["tok_abstencion"] not in d["tok_valores"], "la abstencion colisiona con un valor"
assert not (set(ENT) & set(VAL)), "una pieza es entidad y valor a la vez"
assert all(len(tok(f" {w}", add_special_tokens=False)["input_ids"]) == 1 for w in ENT + VAL + [ABST])

json.dump(d, open("pool_tinyllama.json", "w"), ensure_ascii=False, indent=1)
print(f"\n  abstencion  {ABST!r} -> token {d['tok_abstencion']}")
print(f"  entidades   {' '.join(ENT[:10])} ...")
print(f"  valores     {' '.join(VAL[:10])} ...")
print(f"\n  piso trivial de la tarea con respuesta: 1/{N_VAL} = {1/N_VAL:.4f}")
print("  los objetivos son la palabra ENTERA y son todos distintos.")
print("\nescrito pool_tinyllama.json")
