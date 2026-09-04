"""X4 baja a 0,10-0,17. ¿Es una cota legitima o un artefacto del BPE?

Hipotesis alternativa: las dos ultimas "palabras de contenido" son casi siempre SUB-TOKENS de la
misma palabra ("Nikola Tesla" -> ['Nik','ola',' Tesla']), en cuyo caso X4 no mide la distancia entre
dos partes que se combinan sino el ancho de una sola palabra, y no acota nada.

Test: agrupar los tokens de contenido en PALABRAS (un token abre palabra nueva si su decodificacion
empieza con espacio) y recalcular todo sobre palabras, no sobre sub-tokens. La distancia se sigue
midiendo en TOKENS, que es la unidad del alcance de la convolucion.

  W1 = span del primer al ultimo token de la primera/ultima PALABRA de contenido   (= X1)
  W4 = distancia entre los INICIOS de las dos ultimas PALABRAS de contenido distintas
  W5 = distancia minima entre inicios de palabras de contenido consecutivas
"""
import re, json
import numpy as np
from transformers import AutoTokenizer
from datasets import load_dataset

tok = AutoTokenizer.from_pretrained("state-spaces/mamba-130m-hf")
VACIAS = set("a an the of in on at to for is are was were do does did what which who whom whose "
             "when where why how many much and or but with by from as that this these those it its "
             "there their his her they he she you i we".split())


def palabras_contenido(q):
    """[(inicio, fin)] en indices de token, una entrada por PALABRA de contenido."""
    ids = tok(q).input_ids
    piezas = [tok.decode([i]) for i in ids]
    grupos, act = [], None
    for k, p in enumerate(piezas):
        abre = p.startswith(" ") or k == 0
        if abre:
            if act:
                grupos.append(act)
            act = [k, k, p.strip()]
        else:
            act[1] = k
            act[2] += p
    if act:
        grupos.append(act)
    return ids, [(a, b, w) for a, b, w in grupos
                 if w and w.lower() not in VACIAS and re.search(r"[A-Za-z]", w)]


def medir(preguntas):
    W = {1: [], 4: [], 5: []}
    mismo, total = 0, 0
    for q in preguntas:
        try:
            ids, g = palabras_contenido(q)
        except Exception:
            continue
        if len(g) >= 2:
            W[1].append(g[-1][0] - g[0][0])
            W[4].append(g[-1][0] - g[-2][0])
            W[5].append(min(b[0] - a[0] for a, b in zip(g, g[1:])))
        # cuantas veces los DOS ULTIMOS TOKENS de contenido caen en la MISMA palabra
        cont_tok = [k for a, b, w in g for k in range(a, b + 1)]
        if len(cont_tok) >= 2:
            total += 1
            u, p = cont_tok[-1], cont_tok[-2]
            if any(a <= p and u <= b for a, b, w in g):
                mismo += 1
    return W, mismo, total


FUENTES = [("SQuAD", "rajpurkar/squad", None), ("Natural Questions", "google-research-datasets/nq_open", None),
           ("TriviaQA", "mandarjoshi/trivia_qa", "rc.nocontext"), ("HotpotQA", "hotpotqa/hotpot_qa", "distractor")]

res = {}
for nombre, ruta, config in FUENTES:
    d = load_dataset(ruta, config, split="validation") if config else load_dataset(ruta, split="validation")
    qs = [x["question"] for x in d][:12000]
    W, mismo, total = medir(qs)
    print(f"\n=== {nombre} · {len(qs)} preguntas")
    print(f"  los dos ultimos TOKENS de contenido son la MISMA palabra en {mismo/total:6.1%} "
          f"({mismo}/{total})   <- si es alto, X4 sobre sub-tokens no acota nada")
    r = {"mismo_token_frac": mismo / total}
    for k, v in sorted(W.items()):
        a = np.array(v)
        print(f"  W{k}  n={len(a):>6}  media {a.mean():6.2f}  mediana {np.median(a):5.1f}  "
              f"P(W>2)={(a > 2).mean():.4f}  P(W>3)={(a > 3).mean():.4f}")
        r[f"W{k}"] = {"n": len(a), "media": float(a.mean()), "P_gt2": float((a > 2).mean()),
                      "P_gt3": float((a > 3).mean())}
    res[nombre] = r
json.dump(res, open("control_x4.json", "w"), indent=1)
print("\nguardado en control_x4.json")
