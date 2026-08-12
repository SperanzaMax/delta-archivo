"""CONTROL DE FORMATO — ¿el fallo es de la tarea, del sujeto, o de como se pide la respuesta?

El control anterior dio EXTRACCION = 0,150 con azar 0,250: albert:v4.0 no identifica cual de 4
entidades tiene un director mencionado. Antes de concluir que el sujeto no sirve hay que descartar
un confound del instrumento: para contestar, el modelo debe (1) encontrar el hecho, (2) mapear la
entidad a su numero en una lista BARAJADA, (3) emitir el numero. El paso (2) es indexacion
posicional, que los modelos chicos hacen mal, y no es lo que queremos medir.

Cuatro combinaciones sobre EL MISMO material (tipada, d=5):

  formato   numero | nombre        como se pide la respuesta
  tarea     extraccion | resolucion

Y opcionalmente mas de un sujeto (MODELOS_CONTROL="albert:v4.0,qwen2.5-coder:latest").

Lectura:
  nombre >> numero            -> artefacto de FORMATO; hay que rehacer todo lo medido con nombres
  nombre ~ numero, ambos bajos-> el sujeto no puede; hace falta un modelo mas capaz
  extraccion alta y resolucion baja, con nombres -> el hallazgo es real y especifico
"""
import json
import os
import re
import sys
import time
import urllib.request

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prueba_pista import M, generar

MODELOS = os.environ.get("MODELOS_CONTROL", "albert:v4.0").split(",")
N = int(os.environ.get("N_CELDA", "20"))
D = 5


def consultar(modelo, prompt, n_pred):
    payload = {"model": modelo, "prompt": prompt, "stream": False,
               "options": {"temperature": 0, "num_predict": n_pred}}
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


def calificar(resp, mostradas, formato):
    """Devuelve la entidad elegida o None."""
    if formato == "numero":
        for tok in resp.replace(".", " ").replace(")", " ").split():
            if tok.isdigit() and 1 <= int(tok) <= len(mostradas):
                return mostradas[int(tok) - 1]
        return None
    # formato nombre: se acepta la mencion mas temprana de una candidata en la respuesta
    r = resp.lower()
    pos = [(r.find(c.lower()), c) for c in mostradas if c.lower() in r]
    if pos:
        return min(pos)[1]
    # tolerancia: solo el prefijo (p.ej. "Vantor" por "Vantor Foundry"), si es inequivoco
    pref = {}
    for c in mostradas:
        p = c.split()[0].lower()
        pref.setdefault(p, []).append(c)
    hits = [(r.find(p), cs[0]) for p, cs in pref.items() if len(cs) == 1 and p in r]
    return min(hits)[1] if hits else None


def main():
    salida = {}
    for modelo in MODELOS:
        print(f"\n=== {modelo} · tipada d={D} · {N} casos ===", flush=True)
        for tarea in ("extraccion", "resolucion"):
            for formato in ("numero", "nombre"):
                rng = np.random.default_rng(9200)     # MISMO material en las 4 celdas
                ok = ilegible = 0
                t0 = time.time()
                for _ in range(N):
                    turnos, correcta, cand = generar(rng, "tipada", D)
                    mostradas = list(cand)
                    rng.shuffle(mostradas)
                    if formato == "numero":
                        ops = "\n".join(f"  {i+1}. {c}" for i, c in enumerate(mostradas))
                        cierre = "Answer with the NUMBER only, nothing else."
                    else:
                        ops = "\n".join(f"  - {c}" for c in mostradas)
                        cierre = "Answer with the entity NAME only, nothing else."
                    if tarea == "extraccion":
                        conv = "\n".join(f"- {t}" for t in turnos[:-1])
                        prompt = (f"Here are some facts.\n\n{conv}\n\n"
                                  "Exactly one of these entities has a DIRECTOR mentioned above:\n"
                                  f"{ops}\n\n{cierre}")
                    else:
                        conv = "\n".join(f"- {t}" for t in turnos)
                        prompt = ("Here is a short conversation. The last line is a correction.\n\n"
                                  f"{conv}\n\n"
                                  "The correction refers to exactly one of these entities:\n"
                                  f"{ops}\n\nWhich one is being corrected? {cierre}")
                    elegida = calificar(consultar(modelo, prompt, 8 if formato == "numero" else 20),
                                        mostradas, formato)
                    if elegida is None:
                        ilegible += 1
                    elif elegida == correcta:
                        ok += 1
                salida[f"{modelo}|{tarea}|{formato}"] = {"acc": ok / N, "ilegible": ilegible}
                print(f"  {tarea:11s} {formato:7s} → {ok/N:.3f} (azar {1/M:.3f}) · "
                      f"ilegibles {ilegible} · {time.time()-t0:.0f}s", flush=True)
    json.dump(salida, open("resultados_formato.json", "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
