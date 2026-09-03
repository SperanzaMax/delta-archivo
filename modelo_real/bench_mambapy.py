"""¿El backend `mambapy` (parallel scan en PyTorch puro) es EQUIVALENTE y CUANTO acelera? · 3-sep

El cuello del 2-sep no era el tamaño del modelo sino que HF recorre la secuencia token por token en
Python. El propio mensaje de error lo decia: «Falling back to the sequential implementation of Mamba,
as use_mambapy is set to False». transformers 4.57 trae un TERCER backend, `mambapy.pscan`, que hace
el scan asociativo en paralelo con PyTorch puro. Se instala con pip, no compila CUDA y anda en CPU.

Dos preguntas, en este orden, porque la segunda no importa si la primera falla:

  1. EQUIVALENCIA. Mismos pesos, mismo lote: ¿los logits y los gradientes coinciden con el camino
     secuencial? Si no coinciden, el experimento mediria otra cosa y no sirve.
  2. VELOCIDAD. s por paso de entrenamiento contra los 279,22 s medidos el 2-sep en esta PC.

Ojo con un detalle del codigo de HF (modeling_mamba.py:418): el pscan se usa solo si
`use_mambapy and self.training and cache_params is None`. En `eval()` cae igual al camino lento.
"""
import os, sys, time
import numpy as np, torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tarea_real as T
from transformers import AutoTokenizer, AutoModelForCausalLM

torch.set_num_threads(int(os.environ.get("HILOS", "3")))
MODELO = os.environ.get("MODELO", "state-spaces/mamba-130m-hf")
tok = AutoTokenizer.from_pretrained(MODELO)


def cargar(usar_pscan):
    torch.manual_seed(0)
    m = AutoModelForCausalLM.from_pretrained(MODELO, dtype=torch.float32)
    m.config.use_mambapy = bool(usar_pscan)
    for capa in m.backbone.layers:
        capa.mixer.use_mambapy = bool(usar_pscan)
    return m.train()


# ---------------------------------------------------------------- 1. equivalencia
print(f"modelo {MODELO} · hilos {torch.get_num_threads()}", flush=True)
rng = np.random.default_rng(7)
ids, lab, _, _ = T.lote(rng, tok, 2, ("directa",), n_hechos=4, largo=48)

sal = {}
for nombre, flag in (("secuencial", False), ("pscan", True)):
    m = cargar(flag)
    out = m(ids, labels=lab)
    out.loss.backward()
    g = m.backbone.layers[0].mixer.x_proj.weight.grad
    sal[nombre] = (out.logits.detach().clone(), float(out.loss), g.detach().clone())
    del m

d_lg = float((sal["secuencial"][0] - sal["pscan"][0]).abs().max())
esc_lg = float(sal["secuencial"][0].abs().max())
d_ls = abs(sal["secuencial"][1] - sal["pscan"][1])
d_g = float((sal["secuencial"][2] - sal["pscan"][2]).abs().max())
esc_g = float(sal["secuencial"][2].abs().max())
print(f"\n  EQUIVALENCIA (batch 2 · largo 48)")
print(f"    logits   max|dif| {d_lg:.3e}   sobre escala {esc_lg:.3e}   rel {d_lg/esc_lg:.3e}")
print(f"    loss     {sal['secuencial'][1]:.8f} vs {sal['pscan'][1]:.8f}   dif {d_ls:.3e}")
print(f"    grad     max|dif| {d_g:.3e}   sobre escala {esc_g:.3e}   rel {d_g/esc_g:.3e}")
ok = d_lg / esc_lg < 1e-4 and d_g / max(esc_g, 1e-12) < 1e-3
print(f"    -> {'EQUIVALENTE dentro de fp32' if ok else '*** NO EQUIVALENTE ***'}", flush=True)
if not ok:
    sys.exit(1)

# ---------------------------------------------------------------- 2. velocidad
B, LARGO, NH = int(os.environ.get("B", "4")), int(os.environ.get("LARGO", "192")), \
    int(os.environ.get("NH", "16"))
N = int(os.environ.get("N", "3"))
m = cargar(True)
opt = torch.optim.AdamW(m.parameters(), lr=3e-5)
rng = np.random.default_rng(0)
print(f"\n  VELOCIDAD · batch {B} · largo {LARGO} · {NH} hechos", flush=True)
for _ in range(2):
    ids, lab, _, _ = T.lote(rng, tok, B, ("directa",), n_hechos=NH, largo=LARGO)
    m(ids, labels=lab).loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
t0 = time.time()
for _ in range(N):
    ids, lab, _, _ = T.lote(rng, tok, B, ("directa",), n_hechos=NH, largo=LARGO)
    m(ids, labels=lab).loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
dt = (time.time() - t0) / N
print(f"\n    {dt:.2f} s por paso  ·  contra 279.22 s del camino secuencial  ·  "
      f"ACELERACION {279.22/dt:.1f}x", flush=True)
for pasos in (600, 1200, 2000):
    h = dt * pasos / 3600
    print(f"      {pasos:5d} pasos -> {h:6.2f} h por unidad · {h*6:6.1f} h las 6 unidades")
