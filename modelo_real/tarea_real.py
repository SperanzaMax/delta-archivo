"""La tarea del experimento en modelo real · misma geometria que el micro-LM, en ingles natural.

Un ejemplo es un contexto de hechos y UNA pregunta, y la respuesta es UN SOLO TOKEN. Eso ultimo no
es una comodidad: es lo que hace la metrica exacta sin juez ni parser, que es la leccion del 12-ago
cuando 10 de 11 «abstenciones» resultaron ser el modelo contestando bien otra cosa.

Todas las piezas —entidades, valores, relaciones y la palabra de abstencion— son de UN token en el
BPE de Mamba, verificado en `elegir_vocabulario.py`. Por eso las distancias son FIJAS.

    directa     What is the {rel} of {ent}?      rel a distancia 3, ent a 1
    invertida   For {ent}, what is the {rel}?    rel a distancia 1, ent a 6
    lejana      What is the {rel} that {ent} has?   rel a 4, ent a 2

El alcance MEDIDO de la conv de mamba-370m es 2, no 3: el tap mas viejo vale cero exacto en las 48
capas. O sea que en `directa` la relacion queda AFUERA y en `invertida` ADENTRO, que es exactamente
el contraste del micro-LM.

Tres tipos de pregunta, igual que alla:
    vigente    la relacion de esa entidad SI fue dicha  -> el valor
    nose_ent   la entidad no aparece en el contexto     -> unknown   (la ausencia facil)
    nose_rel   la entidad SI aparece, con otra relacion -> unknown   (la que parece alucinacion)
"""
import json
import os

import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
V = json.load(open(os.path.join(AQUI, "vocabulario.json")))
ENTIDADES, VALORES, RELACIONES = V["entidades"], V["valores"], V["relaciones"]
PLANTILLAS = V["plantillas"]
ABST = "unknown"

TIPOS = {"vigente": 0, "nose_ent": 1, "nose_rel": 2}


def ejemplo(rng, n_hechos=4, forma="directa", p_nose=0.4):
    """Devuelve (texto_del_prompt, respuesta, tipo). La respuesta es UN token."""
    ents = list(rng.choice(ENTIDADES, size=n_hechos + 1, replace=False))
    sobra = ents.pop()                       # entidad que NO se va a nombrar, para `nose_ent`
    hechos = []
    dichos = {}
    for e in ents:
        r = str(rng.choice(RELACIONES))
        v = str(rng.choice(VALORES))
        hechos.append(f"The {r} of {e} is {v}.")
        dichos[(e, r)] = v
    rng.shuffle(hechos)
    contexto = " ".join(hechos)

    if rng.random() < p_nose:
        if rng.random() < 0.5:                              # nose_ent
            e_q, r_q = sobra, str(rng.choice(RELACIONES))
            tipo = "nose_ent"
        else:                                               # nose_rel
            e_q = str(rng.choice(ents))
            libres = [r for r in RELACIONES if (e_q, r) not in dichos]
            if not libres:
                return ejemplo(rng, n_hechos, forma, p_nose)
            r_q = str(rng.choice(libres))
            tipo = "nose_rel"
        resp = ABST
    else:
        (e_q, r_q), resp = list(dichos.items())[int(rng.integers(len(dichos)))]
        tipo = "vigente"

    pregunta = PLANTILLAS[forma].format(r=r_q, e=e_q)
    return f"{contexto} {pregunta}", " " + resp, tipo


def lote(rng, tok, B, formas=("directa",), n_hechos=4, p_nose=0.4, largo=128):
    """Tokeniza B ejemplos. La perdida va SOLO sobre el ultimo token, que es la respuesta."""
    import torch
    ids, labels, tipos, cuales = [], [], [], []
    for _ in range(B):
        f = formas[int(rng.integers(len(formas)))]
        texto, resp, tipo = ejemplo(rng, n_hechos, f, p_nose)
        a = tok(texto).input_ids
        b = tok(resp).input_ids
        assert len(b) == 1, f"la respuesta {resp!r} no es de un token"
        seq = (a + b)[-largo:]
        pad = largo - len(seq)
        ids.append([0] * pad + seq)
        lab = [-100] * (largo - 1) + [seq[-1]]      # solo el ultimo token cuenta
        labels.append(lab)
        tipos.append(TIPOS[tipo]); cuales.append(f)
    return (torch.tensor(ids), torch.tensor(labels), np.array(tipos), cuales)


def metricas(pred, tgt, tipos, id_abst):
    """Las mismas cuatro del micro-LM, y `falsa_abst` no es opcional: un modelo que contesta
    `unknown` a todo tendria nose = 1,000."""
    ok = pred == tgt
    sub = lambda m: float(ok[m].mean()) if m.any() else float("nan")
    hay = tipos == 0
    return {
        "vigente": sub(tipos == 0),
        "nose": sub(tipos >= 1),
        "nose_ent": sub(tipos == 1),
        "nose_rel": sub(tipos == 2),
        "falsa_abst": float((pred[hay] == id_abst).mean()) if hay.any() else float("nan"),
        "abstencion": float((pred == id_abst).mean()),
    }


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    print(f"{len(ENTIDADES)} entidades · {len(VALORES)} valores · {len(RELACIONES)} relaciones\n")
    for f in ("directa", "invertida"):
        print(f"--- forma {f}")
        for _ in range(3):
            t, r, tipo = ejemplo(rng, forma=f)
            print(f"  [{tipo:8s}] {t}")
            print(f"             -> {r!r}")
        print()
