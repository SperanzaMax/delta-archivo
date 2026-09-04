"""¿Que fraccion de las preguntas REALES tiene sus partes fuera del alcance de la ventana? · 3-sep

El paper mide la ley en un idioma sintetico donde la distancia la fija el generador. La limitacion
declarada dice que en texto natural eso seria una DISTRIBUCION. Aca se mide esa distribucion.

La parte delicada es que X hay que DEFINIRLA, y la definicion decide el numero. Por eso se miden
tres, y solo se cree lo que aguante las tres:

  X1  span de contenido      distancia entre la primera y la ultima palabra de contenido
  X2  ultima-content -> fin  cuanto falta del final a la ultima palabra de contenido
  X3  wh -> entidad          del interrogativo a la ultima mayuscula interna (proxy de entidad)

Alcance de referencia: 2 (el MEDIDO en mamba-130m y 370m) y 3 (el nominal de kernel 4).
"""
import re, sys
import numpy as np
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("state-spaces/mamba-130m-hf")
VACIAS = set("a an the of in on at to for is are was were do does did what which who whom whose "
             "when where why how many much and or but with by from as that this these those it its "
             "there their his her they he she you i we".split())
WH = set("what which who whom whose when where why how".split())


def piezas(q):
    """Devuelve (tokens, indices de palabras de contenido, indice del wh, indice de la entidad)."""
    ids = tok(q).input_ids
    pal = [tok.decode([i]).strip() for i in ids]
    cont = [k for k, p in enumerate(pal) if p and p.lower() not in VACIAS and re.search(r"[A-Za-z]", p)]
    wh = next((k for k, p in enumerate(pal) if p.lower() in WH), None)
    # entidad: ultima palabra capitalizada que NO sea la primera de la pregunta
    ent = None
    for k, p in enumerate(pal):
        if k > 0 and p[:1].isupper():
            ent = k
    return ids, cont, wh, ent


def medir(preguntas):
    X = {1: [], 2: [], 3: []}
    for q in preguntas:
        try:
            ids, cont, wh, ent = piezas(q)
        except Exception:
            continue
        n = len(ids)
        if len(cont) >= 2:
            X[1].append(cont[-1] - cont[0])
            X[2].append((n - 1) - cont[-1])
        if wh is not None and ent is not None and ent > wh:
            X[3].append(ent - wh)
    return X


def informar(nombre, X, alcances=(2, 3)):
    print(f"\n=== {nombre}")
    for k, v in sorted(X.items()):
        if not v:
            print(f"  X{k}: sin datos"); continue
        a = np.array(v)
        print(f"  X{k}  n={len(a):>6}  media {a.mean():6.2f}  sd {a.std():5.2f}  "
              f"mediana {np.median(a):5.1f}  p90 {np.percentile(a,90):5.1f}  max {a.max():4d}")
        for al in alcances:
            print(f"        P(X > {al}) = {(a > al).mean():.4f}"
                  f"    <- fraccion que cae AFUERA con alcance {al}")
        # ¿se parece a una normal? asimetria y curtosis dicen mas que un test con n grande
        from scipy import stats
        print(f"        asimetria {stats.skew(a):+.2f} (normal = 0)  ·  "
              f"curtosis {stats.kurtosis(a):+.2f} (normal = 0)")


# --- replica en otros corpus, para saber si el 0,96 es de SQuAD o del lenguaje
FUENTES = [
    ("SQuAD (anotadores sobre un parrafo)", "rajpurkar/squad", "validation", "question", None),
    ("Natural Questions (busquedas REALES de Google)", "google-research-datasets/nq_open",
     "validation", "question", None),
    ("TriviaQA (preguntas de trivia, escritas por humanos)", "mandarjoshi/trivia_qa",
     "validation", "question", "rc.nocontext"),
    ("HotpotQA (multi-hop, dos saltos)", "hotpotqa/hotpot_qa", "validation", "question",
     "distractor"),
]


def replicar():
    from datasets import load_dataset
    for nombre, ruta, split, campo, config in FUENTES:
        try:
            d = (load_dataset(ruta, config, split=split, trust_remote_code=True) if config
                 else load_dataset(ruta, split=split))
            qs = [x[campo] for x in d][:12000]
            informar(f"{nombre} · {len(qs)} preguntas", medir(qs))
        except Exception as e:
            print(f"\n=== {nombre}: NO SE PUDO ({type(e).__name__}: {str(e)[:120]})")


if __name__ == "__main__":
    origen = sys.argv[1] if len(sys.argv) > 1 else "squad"
    if origen == "replica":
        replicar()
    elif origen == "squad":
        from datasets import load_dataset
        d = load_dataset("rajpurkar/squad", split="validation")
        qs = [x["question"] for x in d]
        informar(f"SQuAD validation · {len(qs)} preguntas", medir(qs))

