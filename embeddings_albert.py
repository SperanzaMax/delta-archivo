"""Espectro del espacio de embeddings de un modelo REAL (Albert 4.0), con guardas termicas.

Pregunta: R10 mostro que el colapso al crecer N viene de la dimension baja (d=16 por cabeza en el
harness). ¿Cual es la dimension EFECTIVA de un modelo de verdad? Si es alta, el colapso no aplica
y la memoria persistente escala con mucha menos fusion.

Seguridad de la maquina (prioritaria sobre terminar la corrida):
  - se lee la temperatura del paquete antes de CADA lote
  - >= PAUSA_C  -> espera a que baje, con reintentos
  - >= ABORTA_C -> corta y guarda lo que haya
  - pausa fija entre lotes para dejar enfriar
  - guardado incremental: si aborta, lo hecho no se pierde
"""
import os, re, json, time, subprocess, urllib.request
import numpy as np

MODELO = "albert:v4.0"
N_OBJETIVO = 800
LOTE = 40
PAUSA_ENTRE_LOTES = 12          # segundos de enfriamiento
PAUSA_C = 68.0                  # espera si el paquete llega aca
ABORTA_C = 78.0                 # corta (el "high" del sensor es 80, critico 100)
SALIDA = "albert_embeddings.npy"
LOG = "albert_embeddings.log"


def temp_paquete():
    try:
        s = subprocess.run(["sensors"], capture_output=True, text=True, timeout=10).stdout
        m = re.search(r"Package id 0:\s*\+?([\d.]+)", s)
        return float(m.group(1)) if m else None
    except Exception:
        return None


def log(msg):
    linea = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(linea, flush=True)
    with open(LOG, "a") as f:
        f.write(linea + "\n")


def corpus(n):
    """Textos diversos: combinatoria tematica + lineas reales de los .md del proyecto."""
    temas = ["memoria asociativa", "el clima de la costa", "una receta de pan sin gluten",
             "politica monetaria", "la evolucion de las aves", "algebra lineal",
             "un partido de futbol", "la fotosintesis", "arquitectura romana",
             "redes neuronales recurrentes", "el ciclo del agua", "musica barroca",
             "mineria de datos", "la guerra de las Malvinas", "cocina japonesa",
             "termodinamica", "poesia del siglo XIX", "sistemas operativos",
             "el sistema inmune", "navegacion maritima"]
    formas = ["Explicame brevemente {}.", "¿Que sabes sobre {}?", "Un resumen de {}.",
              "{} es un tema que aparece seguido.", "Notas sueltas sobre {}.",
              "El problema central de {} no es obvio.", "Historia de {}.",
              "Tres ideas equivocadas sobre {}.", "Como se mide {}.",
              "{}: una introduccion para principiantes."]
    txt = [f.format(t) for t in temas for f in formas]
    for nom in ("RESULTADOS_GEOMETRIA_20260808.md", "DOSSIER_LITERATURA_20260808.md"):
        if os.path.exists(nom):
            txt += [l.strip() for l in open(nom, encoding="utf-8")
                    if 40 < len(l.strip()) < 300]
    rng = np.random.default_rng(0)
    while len(txt) < n:                       # completar con variaciones
        a, b = rng.integers(0, len(txt), 2)
        txt.append(txt[a][:60] + " " + txt[b][:60])
    return txt[:n]


def embed(txt):
    d = json.dumps({"model": MODELO, "prompt": txt}).encode()
    r = urllib.request.urlopen(urllib.request.Request(
        "http://localhost:11434/api/embeddings", data=d,
        headers={"Content-Type": "application/json"}), timeout=120)
    return json.load(r)["embedding"]


def main():
    textos = corpus(N_OBJETIVO)
    vecs = list(np.load(SALIDA)) if os.path.exists(SALIDA) else []
    log(f"inicio — objetivo {N_OBJETIVO}, ya hay {len(vecs)}, "
        f"temp {temp_paquete()} C, pausa>={PAUSA_C} aborta>={ABORTA_C}")

    i = len(vecs)
    while i < N_OBJETIVO:
        t = temp_paquete()
        if t is not None and t >= ABORTA_C:
            log(f"ABORTA — {t} C >= {ABORTA_C}. Guardado {len(vecs)} vectores.")
            break
        esperas = 0
        while t is not None and t >= PAUSA_C and esperas < 10:
            log(f"pausa termica — {t} C, esperando 45 s")
            time.sleep(45); t = temp_paquete(); esperas += 1

        t0 = time.time()
        for j in range(i, min(i + LOTE, N_OBJETIVO)):
            try:
                vecs.append(embed(textos[j]))
            except Exception as e:
                log(f"error en {j}: {str(e)[:60]}")
                time.sleep(5)
        i = len(vecs)
        np.save(SALIDA, np.array(vecs, dtype=np.float32))
        log(f"{i}/{N_OBJETIVO} — {time.time()-t0:.0f}s el lote, temp {temp_paquete()} C")
        time.sleep(PAUSA_ENTRE_LOTES)

    X = np.array(vecs, dtype=np.float32)
    np.save(SALIDA, X)
    log(f"fin — {len(X)} vectores, dim {X.shape[1] if len(X) else 0}, "
        f"temp final {temp_paquete()} C")


if __name__ == "__main__":
    main()
