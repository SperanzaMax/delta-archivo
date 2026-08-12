"""CONTROL DE DESCOMPOSICION — ¿por que `tipada` da el azar exacto?

`tipada` es objetivamente resoluble: un solo hecho es de tipo director, el valor nuevo es un nombre
de persona, y los otros tres hechos son ciudades y empresas. Aun asi albert:v4.0 da 0,250 = azar.

Dos explicaciones incompatibles, y este control las separa:

  A) el modelo NO ENCUENTRA el hecho de tipo director  -> falla la extraccion; el sujeto no sirve
     y el chequeo del 11-ago midio incapacidad, no dificultad de la tarea
  B) lo encuentra pero NO CONECTA la correccion con el -> falla la resolucion de la referencia;
     el hallazgo es especifico y el banco mide algo real

Preguntas sobre EXACTAMENTE el mismo material:
  extraccion  "Which entity has a director mentioned?"   (sin la correccion en juego)
  resolucion  "Which one is being corrected?"            (la original, ya medida)

Si extraccion ~ 1,0 y resolucion ~ 0,25 -> B.  Si las dos dan ~0,25 -> A.
"""
import json
import os
import sys
import time
import urllib.request

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prueba_pista import M, generar

MODELOS = os.environ.get("MODELOS_CONTROL", "albert:v4.0").split(",")
N = int(os.environ.get("N_CELDA", "20"))
D = 5


def consultar(modelo, prompt):
    payload = {"model": modelo, "prompt": prompt, "stream": False,
               "options": {"temperature": 0, "num_predict": 8}}
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


def leer_numero(r, n):
    for tok in r.replace(".", " ").replace(")", " ").split():
        if tok.isdigit() and 1 <= int(tok) <= n:
            return int(tok)
    return None


def main():
    for modelo in MODELOS:
        print(f"\n=== {modelo} · tipada d={D} · {N} casos ===", flush=True)
        res = {}
        for tarea in ("extraccion", "resolucion"):
            rng = np.random.default_rng(9200)      # MISMO material en las dos tareas
            ok = ilegible = 0
            t0 = time.time()
            for _ in range(N):
                turnos, correcta, cand = generar(rng, "tipada", D)
                mostradas = list(cand)
                rng.shuffle(mostradas)
                opciones = "\n".join(f"  {i+1}. {c}" for i, c in enumerate(mostradas))
                if tarea == "extraccion":
                    # la correccion no participa: se pregunta solo por el hecho
                    conv = "\n".join(f"- {t}" for t in turnos[:-1])
                    prompt = (f"Here are some facts.\n\n{conv}\n\n"
                              "Exactly one of these entities has a DIRECTOR mentioned above:\n"
                              f"{opciones}\n\n"
                              "Which one? Answer with the NUMBER only, nothing else.")
                else:
                    conv = "\n".join(f"- {t}" for t in turnos)
                    prompt = ("Here is a short conversation. The last line is a correction.\n\n"
                              f"{conv}\n\n"
                              "The correction refers to exactly one of these entities:\n"
                              f"{opciones}\n\n"
                              "Which one is being corrected? Answer with the NUMBER only, "
                              "nothing else.")
                n = leer_numero(consultar(modelo, prompt), len(mostradas))
                if n is None:
                    ilegible += 1
                elif mostradas[n - 1] == correcta:
                    ok += 1
            res[tarea] = ok / N
            print(f"  {tarea:11s} → {ok/N:.3f} (azar {1/M:.3f}) · ilegibles {ilegible} · "
                  f"{time.time()-t0:.0f}s", flush=True)

        ext, sol = res["extraccion"], res["resolucion"]
        if ext >= 0.80 and sol < 0.50:
            veredicto = "B: encuentra el hecho pero NO conecta la correccion"
        elif ext < 0.50:
            veredicto = "A: no encuentra ni el hecho -> el sujeto no sirve para el banco"
        else:
            veredicto = "mixto: sin lectura limpia"
        print(f"  ►► {veredicto}", flush=True)
        res["veredicto"] = veredicto
        json.dump(res, open(f"resultados_control_{modelo.replace(':', '_')}.json", "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
