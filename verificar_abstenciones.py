"""¿LAS ABSTENCIONES SON RECHAZOS GENUINOS O FALLOS DEL PARSER?

En la curva del eje m hubo 21 respuestas contadas como abstencion (m=1: 7/20 · m=8: 8/20). Sobre
ellas se apoya toda la lectura de SER y la afirmacion de que "el modelo sabe cuando no puede".
Pero nunca se miro el texto crudo: una respuesta valida mal parseada se cuenta igual que un rechazo.

Esto guarda el texto COMPLETO de cada respuesta y lo clasifica a mano despues.
"""
import json
import os
import sys
import time
import urllib.request

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from curva_m import elegida_de, generar

MODELO = "qwen2.5-coder:latest"
N = int(os.environ.get("N_CELDA", "15"))
D = 5


def consultar(prompt):
    payload = {"model": MODELO, "prompt": prompt, "stream": False,
               "options": {"temperature": 0, "num_predict": 40}}   # mas margen: queremos ver TODO
    req = urllib.request.Request("http://localhost:11434/api/generate",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    for intento in range(3):
        try:
            return json.load(urllib.request.urlopen(req, timeout=300))["response"]
        except Exception:
            if intento == 2:
                return "<<ERROR DE RED>>"
            time.sleep(2)
    return "<<ERROR DE RED>>"


def main():
    registro = []
    for m in (1, 8):
        rng = np.random.default_rng(9400 + m)      # MISMA semilla que la curva
        print(f"\n=== m={m} · resolucion · {N} casos ===", flush=True)
        for i in range(N):
            turnos, correcta, cand = generar(rng, m, D)
            mostradas = list(cand)
            rng.shuffle(mostradas)
            ops = "\n".join(f"  - {c}" for c in mostradas)
            conv = "\n".join(f"- {t}" for t in turnos)
            prompt = ("Here is a short conversation. The last line is a correction.\n\n"
                      f"{conv}\n\nThe correction refers to exactly one of these entities:\n"
                      f"{ops}\n\nWhich one is being corrected? Answer with the entity NAME only, "
                      "nothing else.")
            resp = consultar(prompt)
            elegida = elegida_de(resp, mostradas)
            estado = ("abstencion" if elegida is None
                      else "acierto" if elegida == correcta else "error")
            registro.append({"m": m, "i": i, "estado": estado, "correcta": correcta,
                             "respuesta_cruda": resp})
            if estado == "abstencion":
                print(f"  [{i:2d}] ABSTENCION · respuesta cruda: {resp!r}", flush=True)
        json.dump(registro, open("resultados_abstenciones.json", "w"), indent=1)

    print("\n" + "=" * 66)
    for m in (1, 8):
        sub = [r for r in registro if r["m"] == m]
        c = {e: sum(r["estado"] == e for r in sub) for e in ("acierto", "error", "abstencion")}
        print(f"  m={m}: aciertos {c['acierto']} · errores {c['error']} · "
              f"abstenciones {c['abstencion']} (de {len(sub)})")
    print("=" * 66)
    print("\n  Las respuestas crudas de las abstenciones estan arriba y en el JSON.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
