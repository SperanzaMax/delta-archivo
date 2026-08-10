"""Corre en una VM de Colab (T4). Genera 10.000 embeddings REALES y cierra el limite de R11.

R11 probo la escala con un generador calibrado (media + 100 componentes + residuo isotropico),
porque con 800 embeddings reales no se puede pasar de N=800 — y ahi ni siquiera d=16 colapsa,
asi que no era discriminante. Aca se usan 10.000 embeddings reales, con lo cual ademas n > d y
la covarianza pasa a estimarse bien (en R11 era n=800 < d=2048).

Modelo: gemma:2b via Ollama, Q4_0 — el MISMO blob de pesos que albert:v4.0 en la PC
(verificado: sha256-c1864a5e... en ambos, sin ADAPTER). Se usa Ollama y no HuggingFace para
replicar exactamente la cuantizacion y el pooling de los 800 vectores locales, y de paso evitar
el gating de Gemma en HF.
"""
import json, os, subprocess, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

N_OBJETIVO = 10_000
SALIDA = "/content/embeddings_10k.npy"


def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout


print("=== 1. entorno ===", flush=True)
print(sh("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader").strip() or "sin GPU")

# zstd NO viene en la VM de Colab y el instalador de Ollama lo exige para extraer.
# Sin esto falla con "This version requires zstd for extraction" y deja el binario sin instalar.
sh("apt-get install -y zstd 2>&1 | tail -2")
print(sh("curl -fsSL https://ollama.com/install.sh | sh 2>&1 | tail -3"), flush=True)

if not sh("which ollama").strip():
    sys.exit("ollama no quedo instalado — abortando antes de gastar tiempo de VM")

subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
time.sleep(8)
print(sh("ollama pull gemma:2b 2>&1")[-200:], flush=True)

print("\n=== 2. corpus ===", flush=True)
sh("pip -q install datasets")
import numpy as np
from datasets import load_dataset

# `datasets` moderno exige repo_id con namespace ('wikitext' a secas da HfUriError).
# Cascada de fuentes para que un cambio de nombre upstream no tumbe la corrida entera.
FUENTES = [("Salesforce/wikitext", "wikitext-103-raw-v1", "text"),
           ("wikimedia/wikipedia", "20231101.en", "text"),
           ("stas/openwebtext-10k", None, "text")]

textos, vistos = [], set()
for repo, conf, campo in FUENTES:
    try:
        ds = load_dataset(repo, conf, split="train", streaming=True) if conf \
            else load_dataset(repo, split="train", streaming=True)
        for r in ds:
            t = (r[campo] or "").strip()
            if 120 < len(t) < 1000 and not t.startswith("=") and t[:60] not in vistos:
                vistos.add(t[:60]); textos.append(t)
            if len(textos) >= N_OBJETIVO:
                break
        if textos:
            print(f"fuente: {repo}", flush=True)
            break
    except Exception as e:
        print(f"  {repo} falló: {str(e)[:100]}", flush=True)

if len(textos) < 1000:
    sys.exit(f"corpus insuficiente ({len(textos)}) — abortando antes de gastar GPU")
print(f"{len(textos)} textos naturales | ejemplo: {textos[0][:100]}", flush=True)

print("\n=== 3. embeddings ===", flush=True)


def emb(t):
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


vecs, t0 = [], time.time()
with ThreadPoolExecutor(max_workers=8) as ex:
    for i, v in enumerate(ex.map(emb, textos)):
        if v is not None:
            vecs.append(v)
        if (i + 1) % 2000 == 0:
            print(f"  {i+1}/{len(textos)} — {time.time()-t0:.0f}s", flush=True)

X = np.array(vecs, dtype=np.float32)
X = X / np.linalg.norm(X, axis=1, keepdims=True)
np.save(SALIDA, X)
n, d = X.shape
print(f"listo: {X.shape} en {time.time()-t0:.0f}s -> {SALIDA}", flush=True)

print("\n=== 4. perfil del espacio (R11.1 con n > d) ===", flush=True)
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
    print(f"{nombre:>26} d={dd:5d} |cos|={np.mean(np.abs(cos)):.4f} "
          f"sd={np.std(cos):.4f} dim_efectiva={pr:8.1f}", flush=True)


print(f"n={n} vs d={d} -> la covarianza {'SI' if n > d else 'NO'} se estima bien")
perfil(X, "gemma:2b real")
perfil(X - X.mean(0), "gemma:2b centrado")
perfil(esfera(rng, n, d), f"uniforme S^{d-1}")
print(f"norma del vector medio: {np.linalg.norm(X.mean(0)):.4f}  "
      f"(uniforme: {np.linalg.norm(esfera(rng, n, d).mean(0)):.4f})", flush=True)

print("\n=== 5. R3 a escala con embeddings REALES (sin generador) ===", flush=True)


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


print(f"{'base':>26} {'N':>7} {'M1 vigente':>12} {'M2 cluster':>12}", flush=True)
sel = np.random.default_rng(1)
for N in (1_000, 5_000, 10_000):
    if N > n:
        break
    sub = X[sel.choice(n, N, replace=False)]
    r = [r3(500 + s, sub) for s in range(3)]
    print(f"{'gemma REAL':>26} {N:7d} {np.mean([a for a, _ in r]):12.3f} "
          f"{np.mean([b for _, b in r]):12.3f}", flush=True)
    c = [r3(500 + s, esfera(np.random.default_rng(600 + s), N, 16)) for s in range(3)]
    print(f"{'uniforme d=16 (control)':>26} {N:7d} {np.mean([a for a, _ in c]):12.3f} "
          f"{np.mean([b for _, b in c]):12.3f}", flush=True)

print("\n=== FIN — descargar con: colab download -s gemacion10k /content/embeddings_10k.npy ===")
