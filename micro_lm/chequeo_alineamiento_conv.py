"""¿Cual es el alcance REAL de la conv1d de Mamba, medido y no deducido del kernel? · 2-sep

Nace de una discrepancia: `ley_en_mamba.py` mide movimiento en d=1 y d=2 y CERO en d=3, pero el
kernel es 4 y con padding causal deberia alcanzar d=3. O el alineamiento del hook no es el que supuse,
o el tap mas lejano de la conv es cero, o el token de reemplazo coincidia. Se decide midiendo, no
razonando: se cambia UN token y se mira en QUE posiciones de salida se mueve la conv.
"""
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

torch.set_grad_enabled(False)
MODELO = os.environ.get("MODELO", "state-spaces/mamba-130m-hf")
tok = AutoTokenizer.from_pretrained(MODELO)
m = AutoModelForCausalLM.from_pretrained(MODELO, dtype=torch.float32).eval()
conv = m.backbone.layers[0].mixer.conv1d
print(f"conv1d: kernel={conv.kernel_size} padding={conv.padding} groups={conv.groups} "
      f"peso {tuple(conv.weight.shape)}")
w = conv.weight[:, 0, :]                       # (canales, k)
print("  |w| medio por tap (tap 0 = el mas VIEJO si el padding es causal a la izquierda):")
for i in range(w.shape[1]):
    print(f"    tap {i}: {w[:, i].abs().mean():.6f}")

capt = {}
h = conv.register_forward_hook(lambda mm, i, o: capt.__setitem__("o", o.detach().clone()))

ids = tok("the capital city of the country that borders spain is lisbon and the population "
          "of that city is small", return_tensors="pt").input_ids
T = ids.shape[1]
m(ids)
base = capt["o"][0].clone()
print(f"\nsecuencia de {T} tokens; la salida cruda de conv1d tiene largo {base.shape[-1]}")

j = 8                                          # se cambia UN token, el de la posicion 8
otro = ids.clone()
otro[0, j] = int(ids[0, 0])
assert int(otro[0, j]) != int(ids[0, j]), "el token de reemplazo coincidia"
m(otro)
d = (capt["o"][0] - base).abs().max(0).values
movidas = [int(p) for p in torch.nonzero(d > 0).flatten()]
print(f"\ncambiando SOLO el token {j}, la salida de conv1d se mueve en las posiciones {movidas}")
print(f"  -> la ventana de salida afectada abarca {len(movidas)} posiciones, de {min(movidas)} a "
      f"{max(movidas)}")
print(f"  -> leido al reves: la salida en la posicion p depende de los tokens "
      f"[p-{max(movidas)-j}, p-{min(movidas)-j}]")
alcance = max(movidas) - j
print(f"\n  ALCANCE REAL medido = {alcance} tokens hacia atras  (kernel {conv.kernel_size[0]})")
h.remove()
