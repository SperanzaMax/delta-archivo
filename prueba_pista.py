"""PRUEBA DE RECUPERABILIDAD DE LA REFERENCIA — pre-registrada en PREREG_PISTA.md.

Repara el defecto del chequeo del 11-ago: registra QUE eligio el modelo, no solo si acerto.
Sin eso, un 0,150 con azar 0,250 es ilegible -- puede ser incapacidad o puede ser otra regla.

Tres condiciones sobre m=4 entidades activas (ver §2 del prereg):
  desnuda   4 hechos del mismo tipo, "No, it's X."               verdad = recencia (convencion)
  recencia  idem, "No, the last one I mentioned -- it's X."      verdad = recencia (explicita)
  tipada    4 hechos de tipos disjuntos, uno solo del tipo de X  verdad = tipo (objetiva)
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
N_POR_CELDA = int(os.environ.get("N_CELDA", "20"))
M = 4
DS = (0, 5)
CONDICIONES = ("desnuda", "recencia", "tipada")

# plantillas por atributo; los pools de NOMBRES / CIUDADES / EMPRESAS son disjuntos entre si
PLANTILLAS = {
    "director": "The director of {e} is {v}.",
    "headquarters": "The headquarters of {e} is located in {v}.",
    "main supplier": "The main supplier for {e} is {v}.",
}
# para `tipada`: el hecho a corregir es de tipo `director` (valor = nombre de persona) y los otros
# tres son de tipos cuyo pool NO contiene nombres de persona -> la referencia es objetivamente
# recuperable por el tipo del valor nuevo, sin depender de ninguna convencion de orden.
TIPOS_DISTRACTOR = ("headquarters", "main supplier")
RELLENO = ("headquarters", "main supplier")


def entidades(rng, n):
    while True:
        e = [f"{rng.choice(PREFIJOS)} {rng.choice(TIPOS)}" for _ in range(n)]
        if len(set(e)) == n:
            return e


def generar(rng, cond, d):
    """Devuelve (turnos, entidad_correcta, candidatas_en_orden_de_aparicion)."""
    ents = entidades(rng, M + d)
    turnos, candidatas = [], []

    if cond == "tipada":
        # posicion del hecho `director` sorteada: la verdad NO puede leerse por posicion
        pos = int(rng.integers(M))
        for i in range(M):
            if i == pos:
                atr = "director"
            else:
                atr = TIPOS_DISTRACTOR[i % len(TIPOS_DISTRACTOR)]
            turnos.append(PLANTILLAS[atr].format(e=ents[i], v=rng.choice(POOL[atr])))
            candidatas.append(ents[i])
        correcta = candidatas[pos]
    else:
        for i in range(M):
            turnos.append(PLANTILLAS["director"].format(e=ents[i], v=rng.choice(POOL["director"])))
            candidatas.append(ents[i])
        correcta = candidatas[-1]   # verdad por recencia

    for j in range(d):
        atr = RELLENO[j % len(RELLENO)]
        turnos.append(PLANTILLAS[atr].format(e=ents[M + j], v=rng.choice(POOL[atr])))

    usados = " ".join(turnos)
    v_nuevo = rng.choice([x for x in POOL["director"] if x not in usados])
    if cond == "recencia":
        turnos.append(f"No, the last one I mentioned -- it's {v_nuevo}.")
    else:
        turnos.append(f"No, it's {v_nuevo}.")
    return turnos, correcta, candidatas


def preguntar(turnos, candidatas, rng):
    """Devuelve la entidad elegida (o None si ilegible). Opciones barajadas."""
    mostradas = list(candidatas)
    rng.shuffle(mostradas)
    conv = "\n".join(f"- {t}" for t in turnos)
    opciones = "\n".join(f"  {i+1}. {c}" for i, c in enumerate(mostradas))
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
    r = None
    for intento in range(3):
        try:
            r = json.load(urllib.request.urlopen(req, timeout=180))["response"]
            break
        except Exception:
            if intento == 2:
                return None
            time.sleep(2)
    for tok in r.replace(".", " ").replace(")", " ").split():
        if tok.isdigit():
            n = int(tok)
            if 1 <= n <= len(mostradas):
                return mostradas[n - 1]
    return None


def main():
    print(f"modelo: {MODELO} · m={M} · {N_POR_CELDA} por celda · 3 condiciones\n", flush=True)
    registro, t0 = [], time.time()
    for ic, cond in enumerate(CONDICIONES):
        for d in DS:
            # semilla por INDICE, no por hash(str): el hash de strings esta aleatorizado por
            # sesion (PYTHONHASHSEED) y la corrida no seria reproducible
            rng = np.random.default_rng(8100 + 10 * ic + d)
            ok = ilegible = 0
            posiciones = []           # posicion DE APARICION de lo elegido (0 = primera mencion)
            for _ in range(N_POR_CELDA):
                turnos, correcta, cand = generar(rng, cond, d)
                elegida = preguntar(turnos, cand, rng)
                if elegida is None:
                    ilegible += 1
                    pos = None
                else:
                    pos = cand.index(elegida)
                    if elegida == correcta:
                        ok += 1
                posiciones.append(pos)
                registro.append({"cond": cond, "d": d, "correcta_pos": cand.index(correcta),
                                 "elegida_pos": pos, "acierto": elegida == correcta})
            acc = ok / N_POR_CELDA
            hist = [posiciones.count(p) / N_POR_CELDA for p in range(M)]
            print(f"  {cond:9s} d={d} → acc {acc:.3f} · elecciones por posicion "
                  f"{['%.2f' % h for h in hist]} · ilegibles {ilegible} · "
                  f"{time.time()-t0:.0f}s", flush=True)

    # ---- lectura de las predicciones, tal como quedaron congeladas ----
    def acc_de(cond):
        sel = [r for r in registro if r["cond"] == cond]
        return sum(r["acierto"] for r in sel) / len(sel)

    a_des, a_rec, a_tip = acc_de("desnuda"), acc_de("recencia"), acc_de("tipada")
    desnudas = [r for r in registro if r["cond"] == "desnuda" and r["elegida_pos"] is not None]
    frac_pos = [sum(r["elegida_pos"] == p for r in desnudas) / len(desnudas) for p in range(M)]
    modo = int(np.argmax(frac_pos))

    p1 = 0.05 <= a_des <= 0.45
    p2 = (a_rec - a_des) >= 0.25
    p3 = a_tip >= 0.80
    p4 = modo == 0 and frac_pos[0] >= 0.40

    print("\n" + "=" * 66)
    print(f"  desnuda {a_des:.3f} · recencia {a_rec:.3f} · tipada {a_tip:.3f}")
    print(f"  desnuda, fraccion por posicion de aparicion: "
          f"{['%.3f' % f for f in frac_pos]} (uniforme = {1/M:.3f})")
    print(f"  P1 replicacion      {'CUMPLE' if p1 else 'NO CUMPLE'}")
    print(f"  P2 marcador +{a_rec-a_des:+.3f}  {'CUMPLE' if p2 else 'NO CUMPLE'} (exigido +0.25)")
    print(f"  P3 pista objetiva   {'CUMPLE' if p3 else 'NO CUMPLE'} (exigido 0.80)")
    print(f"  P4 primera mencion  {'CUMPLE' if p4 else 'NO CUMPLE'} "
          f"(modo en posicion {modo+1}, fraccion {frac_pos[modo]:.3f}, exigido pos 1 y >=0.40)")
    print("=" * 66)

    json.dump({"acc": {"desnuda": a_des, "recencia": a_rec, "tipada": a_tip},
               "frac_posicion_desnuda": frac_pos, "modo": modo,
               "P1": p1, "P2": p2, "P3": p3, "P4": p4,
               "modelo": MODELO, "m": M, "n_por_celda": N_POR_CELDA,
               "registro": registro},
              open("resultados_pista.json", "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
