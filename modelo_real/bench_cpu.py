"""¿Cuanto cuesta un paso de fine-tune de mamba-130m en ESTA maquina? Medido, no estimado."""
import os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tarea_real as T
from transformers import AutoTokenizer, AutoModelForCausalLM

torch.set_num_threads(int(os.environ.get("HILOS", "3")))
tok = AutoTokenizer.from_pretrained("state-spaces/mamba-130m-hf")
m = AutoModelForCausalLM.from_pretrained("state-spaces/mamba-130m-hf", dtype=torch.float32).train()
opt = torch.optim.AdamW(m.parameters(), lr=3e-5)
rng = np.random.default_rng(0)
B, LARGO, NH = int(os.environ.get("B", "4")), 192, 16

print(f"hilos {torch.get_num_threads()} · batch {B} · largo {LARGO} · {NH} hechos", flush=True)
for i in range(3):                                  # calentamiento, no se cuenta
    ids, lab, _, _ = T.lote(rng, tok, B, ("directa",), n_hechos=NH, largo=LARGO)
    m(ids, labels=lab).loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)

N = 10
t0 = time.time()
for i in range(N):
    ids, lab, _, _ = T.lote(rng, tok, B, ("directa",), n_hechos=NH, largo=LARGO)
    m(ids, labels=lab).loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
dt = (time.time() - t0) / N
print(f"\n  {dt:.2f} s por paso de entrenamiento", flush=True)
for pasos in (600, 1200, 2000):
    h = dt * pasos / 3600
    print(f"    {pasos:5d} pasos -> {h:5.2f} h por unidad · {h*6:5.1f} h las 6 unidades")
