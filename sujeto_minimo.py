"""SUJETO MINIMO ADMISIBLE — cual es el modelo mas chico que sirve como sujeto del banco.

La compuerta de §11 del diseno (extraccion >= 0,90) da un criterio objetivo de admision. Si un
modelo de 1,5B la pasa, sirve igual que el de 7B y corre 3-5x mas rapido en la misma PC. Eso no es
una optimizacion de conveniencia: es lo que hace que la campana completa de ECO (5 e x 4 m x 4 K x
10 semillas) sea viable sin GPU.

Y tiene valor propio como resultado: la compuerta parte a los modelos en dos poblaciones -- los que
leen la lista y los que no -- y ese corte es informacion sobre el instrumento, no sobre los modelos.

Protocolo por candidato, a m=4 y d=5 (la celda del contraste):
  1. extraccion  -> si < 0,90, se descarta y NO se le mide resolucion (no seria interpretable)
  2. resolucion  -> solo si paso
Se registra el tiempo por consulta para tener el costo real de la campana.
"""
import json
import os
import sys
import time
import urllib.request

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from curva_m import PLANTILLAS, elegida_de, generar   # mismo generador que la curva

CANDIDATOS = os.environ.get(
    "CANDIDATOS",
    "qwen2.5-coder:1.5b,llama3.2:1b,gemma:2b,j-deepseek:latest,qwen2.5-coder:latest"
).split(",")
N = int(os.environ.get("N_CELDA", "20"))
M, D = 4, 5
UMBRAL = 0.90


def consultar(modelo, prompt):
    payload = {"model": modelo, "prompt": prompt, "stream": False,
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


def celda(modelo, tarea):
    rng = np.random.default_rng(9600)          # mismo material para todos los candidatos
    ok = abst = 0
    t0 = time.time()
    for _ in range(N):
        turnos, correcta, cand = generar(rng, M, D)
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
                      f"{conv}\n\nThe correction refers to exactly one of these entities:\n"
                      f"{ops}\n\nWhich one is being corrected? Answer with the entity NAME only, "
                      "nothing else.")
        e = elegida_de(consultar(modelo, prompt), mostradas)
        if e is None:
            abst += 1
        elif e == correcta:
            ok += 1
    dt = time.time() - t0
    return {"acc": ok / N, "abstencion": abst / N, "s_por_consulta": dt / N}


def main():
    print(f"compuerta: extraccion >= {UMBRAL} · m={M} d={D} · {N} casos\n", flush=True)
    salida = {}
    for modelo in CANDIDATOS:
        ext = celda(modelo, "extraccion")
        linea = (f"  {modelo:26s} extraccion {ext['acc']:.3f} · "
                 f"{ext['s_por_consulta']:5.1f} s/consulta")
        if ext["acc"] < UMBRAL:
            print(linea + "  ►► NO ADMISIBLE", flush=True)
            salida[modelo] = {"extraccion": ext, "admisible": False}
        else:
            sol = celda(modelo, "resolucion")
            print(linea + f"  ►► ADMISIBLE · resolucion {sol['acc']:.3f} "
                  f"(abstenciones {sol['abstencion']:.2f}) · "
                  f"brecha {ext['acc']-sol['acc']:+.3f}", flush=True)
            salida[modelo] = {"extraccion": ext, "resolucion": sol, "admisible": True,
                              "brecha": ext["acc"] - sol["acc"]}
        json.dump(salida, open("resultados_sujeto_minimo.json", "w"), indent=1)

    adm = {k: v for k, v in salida.items() if v["admisible"]}
    print("\n" + "=" * 70)
    if adm:
        mejor = min(adm.items(), key=lambda kv: kv[1]["extraccion"]["s_por_consulta"])
        print(f"  sujeto minimo admisible: {mejor[0]} "
              f"({mejor[1]['extraccion']['s_por_consulta']:.1f} s/consulta, "
              f"resolucion {mejor[1]['resolucion']['acc']:.3f})")
        print(f"  admisibles: {len(adm)}/{len(salida)}")
    else:
        print("  NINGUN candidato pasa la compuerta -> el banco necesita un modelo mas grande")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
