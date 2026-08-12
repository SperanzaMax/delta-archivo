"""CURVA DEL EJE `m`, VERSION 2 — con el instrumento reparado.

La v1 (INFORME_CURVA_M_20260812.md) quedo invalidada en su lectura de las abstenciones: al mirar el
texto crudo, 10 de 11 "abstenciones" eran el modelo contestando con el nombre de la PERSONA (el
valor nuevo) en vez del de la ORGANIZACION. No eran rechazos: era la pregunta mal formulada.
"Which one is being corrected?" admite leerse como "que cosa se corrige" -> "Rosa Belmonte" es una
respuesta razonable a esa lectura.

Tres reparaciones:
  1. La pregunta nombra el tipo de respuesta: "Which ORGANIZATION's record...".
  2. Se ofrece NONE explicitamente -> la abstencion pasa a ser una respuesta REGISTRABLE en vez de
     inferirse del fracaso del parser. Es lo que SER necesita para existir.
  3. Se clasifica en cuatro categorias, no dos: acierto / error / abstencion_explicita /
     fuera_de_dominio (respondio algo que no esta en la lista de organizaciones).

Ademas: 3 semillas por celda. La v1 uso una sola y una replica accidental dio 0,350 vs 0,600 en la
MISMA celda -> el ruido era del tamano del efecto que se estaba leyendo.
"""
import json
import os
import sys
import time
import urllib.request

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from curva_m import generar
from tarea_hechos import POOL

MODELO = os.environ.get("MODELO_CURVA", "qwen2.5-coder:latest")
N = int(os.environ.get("N_CELDA", "20"))
MS = tuple(int(x) for x in os.environ.get("MS_CURVA", "1,4,8").split(","))
SEMILLAS = (0, 1, 2)
D = 5
NOMBRES = set(n.lower() for n in POOL["director"])


def consultar(prompt):
    payload = {"model": MODELO, "prompt": prompt, "stream": False,
               "options": {"temperature": 0, "num_predict": 30}}
    req = urllib.request.Request("http://localhost:11434/api/generate",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    for intento in range(3):
        try:
            return json.load(urllib.request.urlopen(req, timeout=300))["response"]
        except Exception:
            if intento == 2:
                return "<<ERROR>>"
            time.sleep(2)
    return "<<ERROR>>"


def clasificar(resp, mostradas, correcta):
    """acierto | error | abstencion_explicita | fuera_de_dominio"""
    r = resp.strip().lower()
    hits = [(r.find(c.lower()), c) for c in mostradas if c.lower() in r]
    if hits:
        elegida = min(hits)[1]
        return "acierto" if elegida == correcta else "error"
    if "none" in r or "no correction" in r or "does not refer" in r or "cannot" in r:
        return "abstencion_explicita"
    return "fuera_de_dominio"


def main():
    print(f"modelo: {MODELO} · d={D} · {N} casos x {len(SEMILLAS)} semillas · "
          f"pregunta reparada + NONE explicito\n", flush=True)
    salida = {}
    for m in MS:
        por_semilla = []
        for s in SEMILLAS:
            rng = np.random.default_rng(9700 + 100 * s + m)
            cuenta = {"acierto": 0, "error": 0, "abstencion_explicita": 0, "fuera_de_dominio": 0}
            t0 = time.time()
            for _ in range(N):
                turnos, correcta, cand = generar(rng, m, D)
                mostradas = list(cand)
                rng.shuffle(mostradas)
                ops = "\n".join(f"  - {c}" for c in mostradas)
                conv = "\n".join(f"- {t}" for t in turnos)
                prompt = (
                    "Here is a short conversation between a user and an assistant about "
                    "organizations. The last line is the user correcting a fact stated earlier.\n\n"
                    f"{conv}\n\n"
                    "Whose record is the user correcting? Choose ONE organization from this list:\n"
                    f"{ops}\n\n"
                    "Answer with the ORGANIZATION name exactly as listed, or answer NONE if the "
                    "correction cannot be attributed to any of them. Nothing else."
                )
                cuenta[clasificar(consultar(prompt), mostradas, correcta)] += 1
            acc = cuenta["acierto"] / N
            por_semilla.append(acc)
            print(f"  m={m:2d} s={s} → acc {acc:.3f} · err {cuenta['error']:2d} · "
                  f"abst {cuenta['abstencion_explicita']:2d} · fuera {cuenta['fuera_de_dominio']:2d}"
                  f" · {time.time()-t0:.0f}s", flush=True)
            salida[f"m{m}_s{s}"] = cuenta
        med = float(np.mean(por_semilla))
        sd = float(np.std(por_semilla, ddof=1)) if len(por_semilla) > 1 else 0.0
        print(f"     m={m}: media {med:.3f} · sd entre semillas {sd:.3f}\n", flush=True)
        salida[f"m{m}_resumen"] = {"media": med, "sd": sd, "por_semilla": por_semilla}
        json.dump(salida, open("resultados_curva_m2.json", "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
