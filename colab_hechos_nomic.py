"""Genera los embeddings de la tarea de hechos versionados. Corre en una VM de Colab con T4.

Salida: /content/hechos.npz con E1, E2, EQ (3000 x 2048 cada uno) y los metadatos.
Incluye la COMPUERTA de identificabilidad del prereg (seccion 7): si la consulta no separa su
entidad del resto con AUC > 0.95, el experimento se detiene aca.

DESVIACION DEL PREREG (declarada antes de mirar resultados):
El prereg pedia 10 semillas para la asignacion entidad/atributo/valor. Eso implicaria 10 x 9000 =
90.000 embeddings (~4 h de T4). En su lugar se genera UN conjunto de 3000 entidades con semilla 0
y las 10 semillas del analisis controlan submuestreo (1000 entidades por semilla) y los ejes t_hat.
Los embeddings dependen solo del texto, asi que esto preserva la variabilidad del analisis sin
multiplicar el costo. Se declara porque reduce la variabilidad de la parte generativa.
"""
import json, subprocess, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

N = 3000
MODELO = "nomic-embed-text"   # enmienda E1: encoder de recuperacion, no generativo
OLLAMA_VER = "0.20.2"
SALIDA = "/content/hechos_nomic.npz"


def sh(c, t=1800):
    return subprocess.run(c, shell=True, capture_output=True, text=True, timeout=t).stdout


def log(m):
    print(m, flush=True)


log("=== 1. entorno ===")
log(sh("nvidia-smi --query-gpu=name --format=csv,noheader").strip() or "sin GPU")
sh("apt-get install -y zstd 2>&1 | tail -1")
sh(f"curl -fsSL https://ollama.com/install.sh | OLLAMA_VERSION={OLLAMA_VER} sh 2>&1 | tail -2")
if not sh("which ollama").strip():
    sys.exit("ollama no se instalo")
subprocess.Popen("nohup ollama serve > /content/ollama.log 2>&1 &", shell=True)
time.sleep(15)
sh(f"ollama pull {MODELO} 2>&1 | tail -1")
log("ollama " + sh("ollama --version 2>&1 | tail -1").strip())


def embed(t):
    d = json.dumps({"model": MODELO, "prompt": t}).encode()
    for _ in range(3):
        try:
            r = urllib.request.urlopen(urllib.request.Request(
                "http://localhost:11434/api/embeddings", data=d,
                headers={"Content-Type": "application/json"}), timeout=180)
            return json.load(r)["embedding"]
        except Exception:
            time.sleep(2)
    return None


if embed("sanidad") is None:
    log(sh("tail -5 /content/ollama.log")[:400])
    sys.exit("endpoint de embeddings caido — abortando antes de gastar GPU")
log("embeddings OK")

log("\n=== 2. tarea ===")
import numpy as np
from tarea_hechos import gen_hechos

items = gen_hechos(np.random.default_rng(0), N)
log(f"{len(items)} entidades | ejemplo: {items[0]['v1']}")

log("\n=== 3. embeddings (3 por entidad) ===")
PD, PQ = "search_document: ", "search_query: "   # prefijos requeridos por nomic-embed-text
textos = ([PD + x["v1"] for x in items] + [PD + x["v2"] for x in items]
          + [PQ + x["consulta"] for x in items])
vecs, t0, fallos = [], time.time(), 0
with ThreadPoolExecutor(max_workers=8) as ex:
    for i, v in enumerate(ex.map(embed, textos)):
        if v is None:
            fallos += 1; vecs.append(None)
        else:
            vecs.append(v)
        if (i + 1) % 1000 == 0:
            log(f"  {i+1}/{len(textos)} — {time.time()-t0:.0f}s — fallos {fallos}")

dim = len(next(v for v in vecs if v is not None))
vecs = [v if v is not None else [0.0]*dim for v in vecs]
X = np.array(vecs, dtype=np.float32)
X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
E1, E2, EQ = X[:N], X[N:2 * N], X[2 * N:]
np.savez(SALIDA, E1=E1, E2=E2, EQ=EQ,
         resp_vig=np.array([x["resp_vigente"] for x in items]),
         resp_ant=np.array([x["resp_anterior"] for x in items]),
         entidad=np.array([x["entidad"] for x in items]))
log(f"guardado {SALIDA} — {X.shape}, {fallos} fallos, {time.time()-t0:.0f}s")

log("\n=== 4. COMPUERTA del prereg: ¿la consulta identifica su entidad? ===")
S = EQ @ E1.T                                     # (N,N) consulta vs hecho v1
diag = np.diag(S)
off = S[~np.eye(N, dtype=bool)]
r = np.argsort(np.argsort(np.concatenate([diag, off])))[:N].sum() + 1
auc = (r - N * (N + 1) / 2) / (N * len(off))
log(f"cos(consulta, su hecho) = {diag.mean():.4f} | cos(consulta, otro) = {off.mean():.4f}")
log(f"AUC = {auc:.4f}   -> {'PASA' if auc > 0.95 else 'NO PASA: detener el experimento'}")
log(f"top-1 correcto: {np.mean(np.argmax(S, 1) == np.arange(N)):.4f}")

log("\n=== 5. molde: coseno v1 vs v2 (criterio de exclusion del prereg: < 0.5) ===")
cv = np.sum(E1 * E2, 1)
log(f"media {cv.mean():.4f} | p05 {np.percentile(cv,5):.4f} | "
    f"excluidos {(cv < 0.5).sum()}/{N}")
log("\n=== FIN ===")
