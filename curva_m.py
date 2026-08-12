"""CURVA DEL EJE `m` con un sujeto que SI LEE.

El chequeo del 11-ago corrio sobre un sujeto que da el azar en tareas objetivamente resolubles
(ver INFORME_PISTA_20260812.md). Con qwen2.5-coder validado por la compuerta de extraccion, se
levanta la curva que el banco necesita: recall como funcion del numero de entidades activas.

Para cada m: los mismos casos se preguntan de dos formas.
  extraccion  ¿cual tiene un director mencionado?   -> COMPUERTA. Si cae, el sujeto no sirve a ese m
  resolucion  ¿a cual se refiere la correccion?     -> la medida

La compuerta es lo que faltaba el 11-ago: sin ella no se puede distinguir "la tarea es dificil" de
"el sujeto no puede". Respuesta por NOMBRE (medido: numerar cuesta 0,250 en la tarea dificil).
"""
import json
import os
import sys
import time
import urllib.request

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tarea_hechos import POOL, PREFIJOS, TIPOS

MODELO = os.environ.get("MODELO_CURVA", "qwen2.5-coder:latest")
N = int(os.environ.get("N_CELDA", "20"))
MS = tuple(int(x) for x in os.environ.get("MS_CURVA", "1,2,4,8").split(","))
D = 5

PLANTILLAS = {
    "director": "The director of {e} is {v}.",
    "headquarters": "The headquarters of {e} is located in {v}.",
    "main supplier": "The main supplier for {e} is {v}.",
}
DISTRACTOR = ("headquarters", "main supplier")


def generar(rng, m, d):
    """m hechos, exactamente UNO de tipo director (en posicion sorteada). Verdad = por tipo."""
    while True:
        ents = [f"{rng.choice(PREFIJOS)} {rng.choice(TIPOS)}" for _ in range(m + d)]
        if len(set(ents)) == m + d:
            break
    pos = int(rng.integers(m))
    turnos, candidatas = [], []
    for i in range(m):
        atr = "director" if i == pos else DISTRACTOR[i % len(DISTRACTOR)]
        turnos.append(PLANTILLAS[atr].format(e=ents[i], v=rng.choice(POOL[atr])))
        candidatas.append(ents[i])
    for j in range(d):
        atr = DISTRACTOR[j % len(DISTRACTOR)]
        turnos.append(PLANTILLAS[atr].format(e=ents[m + j], v=rng.choice(POOL[atr])))
    usados = " ".join(turnos)
    v_nuevo = rng.choice([x for x in POOL["director"] if x not in usados])
    turnos.append(f"No, it's {v_nuevo}.")
    return turnos, candidatas[pos], candidatas


def consultar(prompt):
    payload = {"model": MODELO, "prompt": prompt, "stream": False,
               "options": {"temperature": 0, "num_predict": 20}}
    req = urllib.request.Request("http://localhost:11434/api/generate",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    for intento in range(3):
        try:
            return json.load(urllib.request.urlopen(req, timeout=300))["response"]
        except Exception:
            if intento == 2:
                return ""
            time.sleep(2)
    return ""


def elegida_de(resp, mostradas):
    r = resp.lower()
    hits = [(r.find(c.lower()), c) for c in mostradas if c.lower() in r]
    if hits:
        return min(hits)[1]
    pref = {}
    for c in mostradas:
        pref.setdefault(c.split()[0].lower(), []).append(c)
    h = [(r.find(p), cs[0]) for p, cs in pref.items() if len(cs) == 1 and p in r]
    return min(h)[1] if h else None


def main():
    print(f"modelo: {MODELO} · d={D} · {N} casos · respuesta por NOMBRE\n", flush=True)
    salida = {}
    for m in MS:
        fila = {}
        for tarea in ("extraccion", "resolucion"):
            rng = np.random.default_rng(9400 + m)      # mismo material en las dos tareas
            ok = abst = 0
            t0 = time.time()
            for _ in range(N):
                turnos, correcta, cand = generar(rng, m, D)
                mostradas = list(cand)
                rng.shuffle(mostradas)
                ops = "\n".join(f"  - {c}" for c in mostradas)
                if tarea == "extraccion":
                    conv = "\n".join(f"- {t}" for t in turnos[:-1])
                    prompt = (f"Here are some facts.\n\n{conv}\n\n"
                              "Exactly one of these entities has a DIRECTOR mentioned above:\n"
                              f"{ops}\n\nAnswer with the entity NAME only, nothing else.")
                else:
                    conv = "\n".join(f"- {t}" for t in turnos)
                    prompt = ("Here is a short conversation. The last line is a correction.\n\n"
                              f"{conv}\n\n"
                              "The correction refers to exactly one of these entities:\n"
                              f"{ops}\n\nWhich one is being corrected? Answer with the entity "
                              "NAME only, nothing else.")
                e = elegida_de(consultar(prompt), mostradas)
                if e is None:
                    abst += 1
                elif e == correcta:
                    ok += 1
            fila[tarea] = {"acc": ok / N, "abstencion": abst / N}
            print(f"  m={m:2d} {tarea:11s} → {ok/N:.3f} (azar {1/m:.3f}) · "
                  f"abstenciones {abst}/{N} · {time.time()-t0:.0f}s", flush=True)
        ext, sol = fila["extraccion"]["acc"], fila["resolucion"]["acc"]
        fila["compuerta"] = "PASA" if ext >= 0.90 else "NO PASA — sujeto insuficiente a este m"
        fila["brecha"] = ext - sol
        print(f"     compuerta {fila['compuerta']} · brecha extraccion−resolucion "
              f"{ext - sol:+.3f}\n", flush=True)
        salida[f"m{m}"] = fila
        json.dump(salida, open("resultados_curva_m.json", "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
