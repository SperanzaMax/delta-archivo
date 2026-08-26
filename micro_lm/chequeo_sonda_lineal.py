"""Chequeo de instrumento de la sonda lineal, ANTES de leer un resultado.
Tres casos con desenlace conocido. Si alguno falla, todo lo demas es basura."""
import sys, numpy as np
sys.path.insert(0, "/home/maxi/Documentos/Nuevo Transformer/delta-archivo/micro_lm")
from sonda_dos_detectores import sonda, auc

rng = np.random.default_rng(7)
n, d = 3000, 132

# T-1 · señal fuerte: una direccion separa las clases. Tiene que dar AUC alto.
w = rng.normal(size=d)
Xa, Xp = rng.normal(size=(n, d)), rng.normal(size=(n, d))
ya, yp = (Xa @ w > 0), (Xp @ w > 0)
a1 = auc(yp, sonda(Xa, ya, Xp))

# T-2 · SIN señal: etiquetas independientes de X. Tiene que dar ~0,50.
yb = rng.random(n) < 0.3
a2 = auc(rng.random(n) < 0.3, sonda(Xa, yb, Xp))

# T-3 · señal debil + clase rara (5 %), que es el regimen real de err_identidad.
w2 = rng.normal(size=d) * 0.25
sc = Xa @ w2; ya3 = sc > np.quantile(sc, 0.95)
sp = Xp @ w2; yp3 = sp > np.quantile(sp, 0.95)
a3 = auc(yp3, sonda(Xa, ya3, Xp))

# T-4 · el AUC con empates masivos, que es el caso del ensamble.
y4 = rng.random(2000) < 0.4
s4 = rng.choice([1/3, 2/3, 1.0], 2000)
a4a = auc(y4, s4); a4b = auc(y4[::-1][::-1], s4)   # mismo dato
perm = rng.permutation(2000)
a4c = auc(y4[perm], s4[perm])                       # reordenado: debe dar lo MISMO

print(f"T-1 señal fuerte   AUC {a1:.4f}   esperado > 0,95   {'OK' if a1 > 0.95 else 'FALLA'}")
print(f"T-2 sin señal      AUC {a2:.4f}   esperado ~0,50    {'OK' if abs(a2-0.5) < 0.05 else 'FALLA'}")
print(f"T-3 señal debil 5% AUC {a3:.4f}   esperado > 0,80   {'OK' if a3 > 0.80 else 'FALLA'}")
print(f"T-4 empates, invariante al orden: {a4a:.6f} vs {a4c:.6f}  "
      f"{'OK' if abs(a4a-a4c) < 1e-9 else 'FALLA'}")
