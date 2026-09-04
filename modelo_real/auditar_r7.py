"""Auditoria del Resultado 7 del paper de la ventana. Reproduce y ataca.

1. verifica si el tokenizer agrega tokens especiales (si los agrega, X2 esta corrido)
2. reproduce X1, X2, X3 exactos contra distancias_corpus.log
3. reporta el n REAL de cada estadistico (el paper reporta el n del corpus)
4. definiciones ADVERSAS que el instrumento original no midio:
     X4 = distancia entre las DOS ULTIMAS palabras de contenido
     X5 = distancia minima entre cualquier par de palabras de contenido consecutivas
   son las que MAS favorecen al modelo. Si P(X>2) sigue alto ahi, el resultado aguanta.
5. TriviaQA sin el truncado a 12.000
"""
import re, json, sys
import numpy as np
from transformers import AutoTokenizer
from datasets import load_dataset

tok = AutoTokenizer.from_pretrained("state-spaces/mamba-130m-hf")
VACIAS = set("a an the of in on at to for is are was were do does did what which who whom whose "
             "when where why how many much and or but with by from as that this these those it its "
             "there their his her they he she you i we".split())
WH = set("what which who whom whose when where why how".split())


def piezas(q):
    ids = tok(q).input_ids
    pal = [tok.decode([i]).strip() for i in ids]
    cont = [k for k, p in enumerate(pal) if p and p.lower() not in VACIAS and re.search(r"[A-Za-z]", p)]
    wh = next((k for k, p in enumerate(pal) if p.lower() in WH), None)
    ent = None
    for k, p in enumerate(pal):
        if k > 0 and p[:1].isupper():
            ent = k
    return ids, cont, wh, ent


def medir(preguntas):
    X = {1: [], 2: [], 3: [], 4: [], 5: []}
    for q in preguntas:
        try:
            ids, cont, wh, ent = piezas(q)
        except Exception:
            continue
        n = len(ids)
        if len(cont) >= 2:
            X[1].append(cont[-1] - cont[0])
            X[2].append((n - 1) - cont[-1])
            X[4].append(cont[-1] - cont[-2])
            X[5].append(min(b - a for a, b in zip(cont, cont[1:])))
        if wh is not None and ent is not None and ent > wh:
            X[3].append(ent - wh)
    return X


def informar(nombre, X, n_corpus):
    print(f"\n=== {nombre}  (n del corpus = {n_corpus})")
    out = {}
    for k, v in sorted(X.items()):
        if not v:
            print(f"  X{k}: sin datos"); continue
        a = np.array(v)
        p2, p3 = float((a > 2).mean()), float((a > 3).mean())
        cob = len(a) / n_corpus
        print(f"  X{k}  n={len(a):>6} (cobertura {cob:5.1%})  media {a.mean():6.2f}  "
              f"mediana {np.median(a):5.1f}  P(X>2)={p2:.4f}  P(X>3)={p3:.4f}")
        out[f"X{k}"] = {"n": len(a), "cobertura": cob, "media": float(a.mean()),
                        "mediana": float(np.median(a)), "P_gt2": p2, "P_gt3": p3}
    return out


FUENTES = [
    ("SQuAD", "rajpurkar/squad", "validation", "question", None, None),
    ("Natural Questions", "google-research-datasets/nq_open", "validation", "question", None, None),
    ("TriviaQA (truncado 12000, como el paper)", "mandarjoshi/trivia_qa", "validation", "question",
     "rc.nocontext", 12000),
    ("TriviaQA (COMPLETO, control)", "mandarjoshi/trivia_qa", "validation", "question",
     "rc.nocontext", None),
    ("HotpotQA", "hotpotqa/hotpot_qa", "validation", "question", "distractor", None),
]

if __name__ == "__main__":
    # 1. tokens especiales
    prueba = "What is the capital of France?"
    ids = tok(prueba).input_ids
    print("=== CONTROL DE TOKENIZADOR ===")
    print("  texto :", prueba)
    print("  ids   :", ids)
    print("  piezas:", [tok.decode([i]) for i in ids])
    print("  bos_token_id =", tok.bos_token_id, "· eos_token_id =", tok.eos_token_id)
    print("  primer id == bos?", ids[0] == tok.bos_token_id,
          "· ultimo id == eos?", ids[-1] == tok.eos_token_id)
    print("  round-trip identico?", tok.decode(ids) == prueba)

    res = {}
    for nombre, ruta, split, campo, config, tope in FUENTES:
        d = (load_dataset(ruta, config, split=split) if config else load_dataset(ruta, split=split))
        qs = [x[campo] for x in d]
        if tope:
            qs = qs[:tope]
        res[nombre] = informar(f"{nombre} · {len(qs)} preguntas", medir(qs), len(qs))
    json.dump(res, open("auditoria_r7.json", "w"), indent=1)
    print("\nguardado en auditoria_r7.json")
