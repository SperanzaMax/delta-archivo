"""Smoke de sanidad del archivo. Barato, sin entrenar. Cuatro chequeos:

  S1  lambda=0  =>  forward identico bit a bit a delta puro (C2).  Si esto falla, cualquier
      diferencia posterior podria venir del refactor y no del archivo.
  S2  causalidad: la salida en t no depende de tokens > t.
  S3  el archivo se llena y selecciona: distribucion de scores retenidos vs. descartados.
  S4  costo por forward a varios w.
"""
import sys, os, time
sys.path.insert(0, os.path.expanduser("~/Documentos/Nuevo Transformer/telar-ligamento/src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np, jax, jax.numpy as jnp
import modelos as M
import modelo_cache as MC
from datos import gen_mqar

B, L, SEED = 8, 64, 0
rng = np.random.default_rng(123)
x, _ = gen_mqar(rng, B, L)
x = jnp.asarray(x)

print(f"MQAR batch: x{tuple(x.shape)}  vocab_max={int(x.max())}")

# ---- S1: lambda=0 reproduce delta puro ----
p_base = M.init_params(SEED, "delta")
p_cache = MC.init_params(SEED, w=16)
y_base = M.forward(p_base, x, "delta")
y_cache = MC.forward(p_cache, x, 16)
d = float(jnp.max(jnp.abs(y_base - y_cache)))
print(f"S1 lambda=0 vs delta puro : max|diff| = {d:.3e}   {'OK' if d < 1e-5 else 'FALLA'}")

# ---- S2: causalidad ----
p2 = jax.tree_util.tree_map(lambda a: a, p_cache)
for blk in p2["blocks"]:
    blk["lam"] = jnp.full(MC.H, 0.7)          # encender el archivo
y_full = MC.forward(p2, x, 16)
x_pert = x.at[:, L - 1].set((x[:, L - 1] + 1) % 100)
y_pert = MC.forward(p2, x_pert, 16)
d_past = float(jnp.max(jnp.abs(y_full[:, : L - 3] - y_pert[:, : L - 3])))
print(f"S2 causalidad (perturbo t=L-1): max|diff| en t<L-3 = {d_past:.3e}   "
      f"{'OK' if d_past < 1e-5 else 'FALLA'}")

# ---- S3: el archivo discrimina ----
blk = p2["blocks"][0]
hx = p2["emb"][x]
xin = M.conv3(blk["conv"], M.ln(blk["ln1"], hx))
q = M.split_heads(xin @ blk["q"]); k = M.split_heads(xin @ blk["k"]); v = M.split_heads(xin @ blk["v"])
kn = M.l2n(jax.nn.silu(k)); beta = jax.nn.sigmoid(xin @ blk["g_beta"]["w"] + blk["g_beta"]["b"])
S = jnp.zeros((B, MC.H, MC.DH, MC.DH)); scores = []
for t in range(L):
    kt = kn[:, :, t]; vt = v[:, :, t]; bt = beta[:, t]
    err = vt - jnp.einsum("bhij,bhj->bhi", S, kt)
    scores.append(np.asarray(bt * jnp.linalg.norm(err, axis=-1)))
    S = S + bt[..., None, None] * jnp.einsum("bhi,bhj->bhij", err, kt)
sc = np.stack(scores, 1)                                   # (B,T,H)
print(f"S3 score beta*||e||: media={sc.mean():.4f} sd={sc.std():.4f} "
      f"p10={np.percentile(sc,10):.4f} p90={np.percentile(sc,90):.4f} "
      f"ratio p90/p10={np.percentile(sc,90)/max(np.percentile(sc,10),1e-9):.1f}x")

# ---- S4: costo ----
print("S4 costo por forward (B=8, L=64, tras jit-warm):")
for w in (0, 8, 16, 64):
    if w == 0:
        f = jax.jit(lambda pp, xx: M.forward(pp, xx, "delta")); pp = p_base
    else:
        f = jax.jit(lambda pp, xx, w=w: MC.forward(pp, xx, w)); pp = p2
    f(pp, x).block_until_ready()
    t0 = time.perf_counter()
    for _ in range(3):
        f(pp, x).block_until_ready()
    dt = (time.perf_counter() - t0) / 3
    print(f"   w={w:>3}{'  (delta puro)' if w == 0 else '':<14} {dt*1000:7.1f} ms")
