"""Bootstrap completo y autocontenido para la VM de Colab. Consolida todo lo aprendido a los golpes.

Se lanza con nohup DENTRO de la VM y escribe a /content/job.log, para que el trabajo largo no
dependa del timeout del cliente de colab-cli (que corta cuando un paso tarda sin imprimir).

Cuatro cosas que hicieron falta y no son obvias:
  1. `zstd` NO viene en la VM y el instalador de Ollama lo exige para extraer.
  2. Ollama >= 0.32 RECHAZA embeddings en modelos que no declaran esa capability, y gemma:2b
     declara solo `completion`. La 0.20.2 no verifica y los sirve igual -> se fija esa version,
     que ademas es la que corre en la PC de Maxi (comparabilidad exacta de los vectores).
     El error de la version nueva ("Start it with --embeddings") es enganoso: ese flag no existe.
  3. `datasets` moderno exige repo_id con namespace: 'wikitext' a secas da HfUriError.
  4. Todo paso largo imprime antes y despues, para poder diagnosticar desde el log.
"""
import json, subprocess, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

N_OBJETIVO = 10_000
SALIDA = "/content/embeddings_10k.npy"
OLLAMA_VER = "0.20.2"


def sh(c, t=600):
    return subprocess.run(c, shell=True, capture_output=True, text=True, timeout=t).stdout


def log(m):
    print(m, flush=True)


log("=== 1. entorno ===")
log(sh("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader").strip() or "sin GPU")
sh("apt-get install -y zstd 2>&1 | tail -1")
log("zstd: " + (sh("which zstd").strip() or "FALTA"))
sh(f"curl -fsSL https://ollama.com/install.sh | OLLAMA_VERSION={OLLAMA_VER} sh 2>&1 | tail -2")
if not sh("which ollama").strip():
    sys.exit("ollama no se instalo")
log("ollama cliente: " + sh("ollama --version 2>&1 | tail -1").strip())

subprocess.Popen("nohup ollama serve > /content/ollama.log 2>&1 &", shell=True)
time.sleep(15)
log("pull gemma:2b ...")
sh("ollama pull gemma:2b 2>&1 | tail -1", t=1800)
log("modelos: " + sh("ollama list | tail -2").strip()[:120])


def embed(t):
    d = json.dumps({"model": "gemma:2b", "prompt": t}).encode()
    for _ in range(3):
        try:
            r = urllib.request.urlopen(urllib.request.Request(
                "http://localhost:11434/api/embeddings", data=d,
                headers={"Content-Type": "application/json"}), timeout=180)
            return json.load(r)["embedding"]
        except Exception:
            time.sleep(2)
    return None


t0 = time.time()
v = embed("prueba de sanidad del endpoint de embeddings")
if v is None:
    log("ollama.log: " + sh("tail -5 /content/ollama.log")[:400])
    sys.exit("el endpoint de embeddings no responde — abortando antes de gastar GPU")
log(f"embeddings OK — dim {len(v)}, 1a llamada {time.time()-t0:.1f}s")

log("\n=== 2. corpus ===")
sh("pip -q install datasets 2>&1 | tail -1")
import numpy as np
from datasets import load_dataset

FUENTES = [("Salesforce/wikitext", "wikitext-103-raw-v1"),
           ("wikimedia/wikipedia", "20231101.en"),
           ("stas/openwebtext-10k", None)]
textos, vistos = [], set()
for repo, conf in FUENTES:
    try:
        ds = load_dataset(repo, conf, split="train", streaming=True) if conf \
            else load_dataset(repo, split="train", streaming=True)
        for r in ds:
            t = (r.get("text") or "").strip()
            if 120 < len(t) < 1000 and not t.startswith("=") and t[:60] not in vistos:
                vistos.add(t[:60]); textos.append(t)
            if len(textos) >= N_OBJETIVO:
                break
        if textos:
            log(f"fuente: {repo}")
            break
    except Exception as e:
        log(f"  {repo} fallo: {str(e)[:90]}")
if len(textos) < 1000:
    sys.exit(f"corpus insuficiente ({len(textos)})")
log(f"{len(textos)} textos | ejemplo: {textos[0][:90]}")

log("\n=== 3. embeddings ===")
vecs, t0, fallos = [], time.time(), 0
with ThreadPoolExecutor(max_workers=8) as ex:
    for i, v in enumerate(ex.map(embed, textos)):
        if v is None:
            fallos += 1
        else:
            vecs.append(v)
        if (i + 1) % 500 == 0:
            log(f"  {i+1}/{len(textos)} — {time.time()-t0:.0f}s — fallos {fallos}")
        if fallos > 50 and len(vecs) < 10:
            sys.exit("demasiados fallos seguidos — abortando")

X = np.array(vecs, dtype=np.float32)
X = X / np.linalg.norm(X, axis=1, keepdims=True)
np.save(SALIDA, X)
n, d = X.shape
log(f"listo: {X.shape} en {time.time()-t0:.0f}s ({fallos} fallos) -> {SALIDA}")

log("\n=== 4. perfil del espacio (n > d, la covarianza ya se estima bien) ===")
rng = np.random.default_rng(0)


def esfera(r, m, dd):
    x = r.normal(size=(m, dd)).astype(np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def perfil(Y, nombre):
    m, dd = Y.shape
    i, j = rng.integers(0, m, 50_000), rng.integers(0, m, 50_000)
    k = i != j
    cos = np.sum(Y[i[k]] * Y[j[k]], 1)
    pr = float("nan")
    if m > dd:
        lam = np.clip(np.linalg.eigvalsh(np.cov(Y.T)), 0, None)
        pr = lam.sum() ** 2 / np.sum(lam ** 2)
    log(f"{nombre:>26} d={dd:5d} |cos|={np.mean(np.abs(cos)):.4f} "
        f"sd={np.std(cos):.4f} dim_efectiva={pr:8.1f}")


log(f"n={n} vs d={d}")
perfil(X, "gemma:2b real")
perfil(X - X.mean(0), "gemma:2b centrado")
perfil(esfera(rng, n, d), f"uniforme S^{d-1}")
log(f"norma del vector medio: {np.linalg.norm(X.mean(0)):.4f}")

log("\n=== 5. R3 a escala con embeddings REALES ===")


def tangente(t, x):
    t = t - (t * x).sum(-1, keepdims=True) * x
    return t / (np.linalg.norm(t, axis=-1, keepdims=True) + 1e-8)


def r3(seed, base, K=4, eps=0.3, alpha=0.4, delta=3.0, ruido=0.05, Q=300):
    r = np.random.default_rng(seed)
    N, dd = base.shape
    that = esfera(r, N, dd)
    cur = base.copy(); vs = [base.copy()]
    for _ in range(K):
        u = alpha * tangente(that, cur) + (1 - alpha) * esfera(r, N, dd)
        cur = cur + eps * tangente(u, cur)
        cur /= np.linalg.norm(cur, axis=1, keepdims=True)
        vs.append(cur.copy())
    A = np.concatenate(vs, 0).astype(np.float32)
    mem = np.tile(np.arange(N), K + 1); ver = np.repeat(np.arange(K + 1), N)
    qi = r.choice(N, min(Q, N), replace=False)
    q0 = base[qi] + ruido * esfera(r, len(qi), dd)
    q0 /= np.linalg.norm(q0, axis=1, keepdims=True)
    h1 = np.argmax(q0 @ A.T, 1)
    q = q0 + delta * tangente(that[mem[h1]], q0)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    t1 = np.argmax(q @ A.T, 1)
    ok = mem[t1] == qi
    return float(np.mean(ok & (ver[t1] == K))), float(np.mean(ok))


log(f"{'base':>26} {'N':>7} {'M1 vigente':>12} {'M2 cluster':>12}")
sel = np.random.default_rng(1)
for N in (1_000, 5_000, 10_000):
    if N > n:
        break
    sub = X[sel.choice(n, N, replace=False)]
    r = [r3(500 + s, sub) for s in range(3)]
    log(f"{'gemma REAL':>26} {N:7d} {np.mean([a for a, _ in r]):12.3f} "
        f"{np.mean([b for _, b in r]):12.3f}")
    c = [r3(500 + s, esfera(np.random.default_rng(600 + s), N, 16)) for s in range(3)]
    log(f"{'uniforme d=16 (control)':>26} {N:7d} {np.mean([a for a, _ in c]):12.3f} "
        f"{np.mean([b for _, b in c]):12.3f}")

log("\n=== FIN ===")
