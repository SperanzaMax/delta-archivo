"""CHEQUEO BLOQUEANTE del riesgo 1 de DISENO_BANCO_ELIPTICO.md §7.

Pregunta: ¿un LLM chico resuelve la co-referencia de una corrección elíptica cuando hay varias
entidades activas? Si la resuelve por encima del 95 %, el banco ECO no discrimina y se archiva.

Diseño mínimo, con verdad de base bien definida:
  - Se enuncian `m` hechos del MISMO tipo de atributo (director), sobre entidades distintas. El
    valor no desambigua: todos son nombres de persona.
  - Luego `d` turnos de relleno sobre otros atributos y otras entidades.
  - Luego llega "No, it's <valor nuevo>."
  - VERDAD DE BASE: la corrección se refiere al ÚLTIMO hecho de tipo director enunciado antes de
    ella. Es la lectura por recencia, que es la que usa cualquier hablante.

Con d = 0 la recencia lo resuelve solo y esperamos ~100 %: es el CONTROL de que la tarea es
resoluble y de que el modelo entiende la consigna. Con d > 0 hay que recordar cuál fue el último
director mencionado entre distractores, y ahí se ve si discrimina.

Si el modelo acierta >95 % en TODAS las celdas, el banco se archiva. Ese es el compromiso.
"""
import json
import os
import sys
import time
import urllib.request

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tarea_hechos import POOL, PREFIJOS, TIPOS

MODELO = os.environ.get("MODELO_CHEQUEO", "albert:v4.0")
N_POR_CELDA = int(os.environ.get("N_CELDA", "30"))
MS = (1, 4, 8)          # entidades activas del mismo tipo
DS = (0, 5)             # turnos de relleno entre el último hecho y la corrección
UMBRAL_ARCHIVO = 0.95   # §7: por encima de esto en todas las celdas, el banco no vale

OTROS_ATRIB = [("headquarters", "The headquarters of {e} is located in {v}."),
               ("main supplier", "The main supplier for {e} is {v}.")]


def generar(rng, m, d):
    """Devuelve (turnos, entidad_correcta, valor_nuevo, candidatas)."""
    ents = [f"{rng.choice(PREFIJOS)} {rng.choice(TIPOS)}" for _ in range(m + d)]
    # evitar repetidos
    while len(set(ents)) < len(ents):
        ents = [f"{rng.choice(PREFIJOS)} {rng.choice(TIPOS)}" for _ in range(m + d)]
    dirs = POOL["director"]
    turnos, candidatas = [], []
    # m hechos de tipo director; el ULTIMO es el que la corrección va a corregir
    for i in range(m):
        v = rng.choice(dirs)
        turnos.append(f"The director of {ents[i]} is {v}.")
        candidatas.append(ents[i])
    correcta = candidatas[-1]
    # d turnos de relleno sobre OTRAS entidades y otros atributos
    for j in range(d):
        atr, plantilla = OTROS_ATRIB[j % len(OTROS_ATRIB)]
        pool = POOL[atr]
        turnos.append(plantilla.format(e=ents[m + j], v=rng.choice(pool)))
    v_nuevo = rng.choice([x for x in dirs if x not in turnos[-1]])
    turnos.append(f"No, it's {v_nuevo}.")
    return turnos, correcta, v_nuevo, candidatas


def preguntar(turnos, candidatas, rng):
    """Las opciones se BARAJAN. Sin esto la respuesta correcta cae siempre en la última
    posición (la verdad de base es por recencia y las candidatas están en orden de aparición),
    y cualquier sesgo posicional del modelo se lee como incapacidad. Verificado: sin barajar,
    el modelo daba 0,000 respondiendo siempre 1-3 mientras la correcta era la 8 de 8."""
    candidatas = list(candidatas)
    rng.shuffle(candidatas)
    conv = "\n".join(f"- {t}" for t in turnos)
    opciones = "\n".join(f"  {i+1}. {c}" for i, c in enumerate(candidatas))
    prompt = (
        "Here is a short conversation. The last line is a correction.\n\n"
        f"{conv}\n\n"
        "The correction refers to exactly one of these entities:\n"
        f"{opciones}\n\n"
        "Which one is being corrected? Answer with the NUMBER only, nothing else."
    )
    payload = {"model": MODELO, "prompt": prompt, "stream": False,
               "options": {"temperature": 0, "num_predict": 8}}
    req = urllib.request.Request("http://localhost:11434/api/generate",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    for intento in range(3):
        try:
            r = json.load(urllib.request.urlopen(req, timeout=180))["response"]
            break
        except Exception as e:
            if intento == 2:
                return None
            time.sleep(2)
    for tok in r.replace(".", " ").replace(")", " ").split():
        if tok.isdigit():
            n = int(tok)
            if 1 <= n <= len(candidatas):
                return candidatas[n - 1]
    return None


def main():
    print(f"modelo: {MODELO} · {N_POR_CELDA} por celda · verdad de base = recencia\n", flush=True)
    res, t0 = {}, time.time()
    for m in MS:
        for d in DS:
            rng = np.random.default_rng(7000 + m * 10 + d)
            ok = ilegible = 0
            for i in range(N_POR_CELDA):
                turnos, correcta, _, cand = generar(rng, m, d)
                resp = preguntar(turnos, cand, rng)
                if resp is None:
                    ilegible += 1
                elif resp == correcta:
                    ok += 1
            acc = ok / N_POR_CELDA
            res[(m, d)] = {"acc": acc, "ilegible": ilegible}
            azar = 1.0 / m
            print(f"  m={m} d={d} → acc {acc:.3f} (azar {azar:.3f}) · "
                  f"ilegibles {ilegible}/{N_POR_CELDA} · {time.time()-t0:.0f}s", flush=True)

    print("\n" + "=" * 62)
    accs = [v["acc"] for v in res.values()]
    no_control = [v["acc"] for k, v in res.items() if k[1] > 0]
    print(f"  peor celda: {min(accs):.3f} · mejor: {max(accs):.3f}")
    print(f"  peor celda con distancia (d>0): {min(no_control):.3f}")
    if min(accs) > UMBRAL_ARCHIVO:
        print("\n  ►► EL BANCO NO DISCRIMINA. Segun §7 del diseno, se ARCHIVA.")
        veredicto = "ARCHIVAR"
    elif min(no_control) > UMBRAL_ARCHIVO:
        print("\n  ►► Resuelve todas las celdas con distancia. El eje de ambiguedad referencial")
        print("     no alcanza por si solo; habria que endurecerlo o archivar.")
        veredicto = "ARCHIVAR (con reserva)"
    else:
        print(f"\n  ►► EL BANCO DISCRIMINA: cae a {min(no_control):.3f} en la peor celda con")
        print("     distancia. Hay rango para medir. Sigue en pie.")
        veredicto = "SIGUE"
    print("=" * 62)
    json.dump({f"m{k[0]}_d{k[1]}": v for k, v in res.items()} | {"veredicto": veredicto,
              "modelo": MODELO, "n_por_celda": N_POR_CELDA},
              open("resultados_chequeo_riesgo1.json", "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
