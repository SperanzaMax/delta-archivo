"""¿OFRECER LA SALIDA REDUCE EL DANO? — contraste aislado de la opcion NONE.

Entre la curva v1 y la v2 los errores de m=4 bajaron de 8/20 a 11/60, pero cambiaron DOS cosas a la
vez: la redaccion de la pregunta y la disponibilidad de NONE. Esto aisla la segunda.

Unica diferencia entre condiciones: la frase que ofrece la salida. Mismo material, misma pregunta,
mismas semillas.

  con_none   "...or answer NONE if the correction cannot be attributed to any of them."
  sin_none   "...You must choose one."

PREDICCION, comprometida antes de correr:
  P-N1 (principal) los ERRORES suben al quitar la salida: err(sin) - err(con) >= +0,10 en m=8.
       Es el enunciado accionable: forzar una eleccion convierte abstenciones en hechos falsos.
  P-N2 el ACIERTO no mejora al quitarla: acc(sin) - acc(con) < +0,10. Si el acierto subiera mucho,
       la abstencion seria pereza recuperable y no incertidumbre real -> el argumento se cae.
  P-N3 (control) la suma acierto+error+abstencion+fuera se conserva; fuera_de_dominio se mantiene
       bajo (<0,15) en las dos -> la reparacion de la pregunta sigue funcionando.

Si P-N1 no se cumple, el hallazgo lateral de la v2 queda RETIRADO y se dice.
"""
import json
import os
import sys
import time
import urllib.request

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from curva_m import generar
from curva_m2 import clasificar

MODELO = os.environ.get("MODELO_CURVA", "qwen2.5-coder:latest")
N = int(os.environ.get("N_CELDA", "20"))
MS = (4, 8)
SEMILLAS = (0, 1)
D = 5

CIERRE = {
    "con_none": ("Answer with the ORGANIZATION name exactly as listed, or answer NONE if the "
                 "correction cannot be attributed to any of them. Nothing else."),
    "sin_none": ("Answer with the ORGANIZATION name exactly as listed. You must choose one. "
                 "Nothing else."),
}


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


def main():
    print(f"modelo: {MODELO} · d={D} · {N} casos x {len(SEMILLAS)} semillas · "
          f"contraste con_none / sin_none\n", flush=True)
    salida = {}
    for m in MS:
        for cond, cierre in CIERRE.items():
            tot = {"acierto": 0, "error": 0, "abstencion_explicita": 0, "fuera_de_dominio": 0}
            t0 = time.time()
            for s in SEMILLAS:
                rng = np.random.default_rng(9700 + 100 * s + m)   # MISMO material que la curva v2
                for _ in range(N):
                    turnos, correcta, cand = generar(rng, m, D)
                    mostradas = list(cand)
                    rng.shuffle(mostradas)
                    ops = "\n".join(f"  - {c}" for c in mostradas)
                    conv = "\n".join(f"- {t}" for t in turnos)
                    prompt = (
                        "Here is a short conversation between a user and an assistant about "
                        "organizations. The last line is the user correcting a fact stated "
                        f"earlier.\n\n{conv}\n\n"
                        "Whose record is the user correcting? Choose ONE organization from this "
                        f"list:\n{ops}\n\n{cierre}"
                    )
                    tot[clasificar(consultar(prompt), mostradas, correcta)] += 1
            n = N * len(SEMILLAS)
            salida[f"m{m}_{cond}"] = {k: v / n for k, v in tot.items()}
            print(f"  m={m} {cond:9s} → acc {tot['acierto']/n:.3f} · err {tot['error']/n:.3f} · "
                  f"abst {tot['abstencion_explicita']/n:.3f} · fuera "
                  f"{tot['fuera_de_dominio']/n:.3f} · {time.time()-t0:.0f}s", flush=True)
        json.dump(salida, open("resultados_none.json", "w"), indent=1)

    print("\n" + "=" * 70)
    for m in MS:
        c, s = salida[f"m{m}_con_none"], salida[f"m{m}_sin_none"]
        d_err, d_acc = s["error"] - c["error"], s["acierto"] - c["acierto"]
        print(f"  m={m}: Δerror {d_err:+.3f} · Δacierto {d_acc:+.3f}")
        if m == 8:
            print(f"    P-N1 {'CUMPLE' if d_err >= 0.10 else 'NO CUMPLE'} (exigido >= +0,10)")
            print(f"    P-N2 {'CUMPLE' if d_acc < 0.10 else 'NO CUMPLE'} (exigido < +0,10)")
            fuera_ok = c["fuera_de_dominio"] < 0.15 and s["fuera_de_dominio"] < 0.15
            print(f"    P-N3 {'CUMPLE' if fuera_ok else 'NO CUMPLE'} (fuera < 0,15 en ambas)")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
