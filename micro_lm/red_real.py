"""Dibuja la red del Micro-LM A ESCALA: cada neurona y cada conexión son las de verdad.

No hay ni un nodo de adorno. Los pesos salen del checkpoint entrenado n4_s1.pkl.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap

AZUL, NARANJA, VERDE, ROJO = "#12395f", "#b5651d", "#2e7d44", "#c0392b"
cmap = LinearSegmentedColormap.from_list("w", ["#c0392b", "#e8e8e8", "#12395f"])

# Los pesos se extraen aparte a un .npz porque el checkpoint trae ademas el estado del
# optimizador, que necesita optax para deserializarse — y el venv con optax no tiene matplotlib.
_z = np.load("pesos.npz")
emb, m1, head = _z["emb"], _z["m1"], _z["head"]

fig = plt.figure(figsize=(16.4, 18.6), dpi=100, facecolor="white")
gs = fig.add_gridspec(3, 1, height_ratios=[1.45, 0.92, 0.78], hspace=0.13,
                      left=0.045, right=0.965, top=0.900, bottom=0.030)

fig.text(0.045, 0.968, "Micro-LM · la red entera, a escala real",
         fontsize=25, fontweight="bold", color=AZUL)
fig.text(0.045, 0.951,
         "Cada punto es una neurona de verdad y cada línea un peso de verdad, leídos del "
         "checkpoint entrenado. No hay ningún elemento de adorno.",
         fontsize=12.5, color="#555")

# ══════════════════════ PANEL 1 · la pila completa ══════════════════════
ax = fig.add_subplot(gs[0]); ax.set_facecolor("white")
capas = [
    ("entrada · vocabulario", 242, AZUL),
    ("embedding", 128, AZUL),
]
for b in range(4):
    capas += [(f"bloque {b} · delta", 128, "#3f6d99"),
              (f"bloque {b} · MLP interno", 512, NARANJA),
              (f"bloque {b} · salida", 128, "#3f6d99")]
capas += [("salida · un puntaje por token", 242, VERDE)]

ANCHO = 100.0
y = 0
etiquetas = []
for nom, n, col in capas:
    xs = np.linspace(-ANCHO / 2, ANCHO / 2, n) if n > 1 else np.array([0.0])
    tam = 9 if n <= 128 else (4.2 if n <= 242 else 1.9)
    ax.scatter(xs, np.full(n, y), s=tam, c=col, linewidths=0, zorder=3)
    etiquetas.append((y, nom, n))
    y -= 1

# conexiones reales entre entrada y embedding: se dibujan las mas fuertes
UMBRAL = 0.25
fuertes = np.argwhere(np.abs(emb) > UMBRAL)
segs = []
for i, j in fuertes:
    x0 = -ANCHO / 2 + ANCHO * i / 241
    x1 = -ANCHO / 2 + ANCHO * j / 127
    segs.append([(x0, 0), (x1, -1)])
vals = np.array([emb[i, j] for i, j in fuertes])
lc = LineCollection(segs, array=vals, cmap=cmap, linewidths=0.55, alpha=0.75,
                    norm=plt.Normalize(-0.8, 0.8), zorder=2)
ax.add_collection(lc)

for yy, nom, n in etiquetas:
    ax.text(ANCHO / 2 + 3.5, yy, f"{n:,}".replace(",", ".") + f"   {nom}",
            va="center", fontsize=10.4, color="#333")

ax.text(-ANCHO / 2 - 4, -0.5, "30.976\nconexiones\nreales", ha="right", va="center",
        fontsize=10, color=ROJO, fontweight="bold")
ax.text(-ANCHO / 2 - 4, -3.1,
        f"se dibujan\nsólo las {len(fuertes)}\nmás fuertes\n(|w| > {UMBRAL:.2f}".replace(".", ",") + ")",
        ha="right", va="center", fontsize=9.2, color="#777", style="italic")

ax.set_xlim(-ANCHO / 2 - 22, ANCHO / 2 + 46)
ax.set_ylim(-14.6, 1.0)
ax.axis("off")
ax.set_title("3.684 neuronas en total, apiladas en 15 capas — el mismo dibujo de siempre, "
             "pero con la cantidad que hay de verdad",
             fontsize=13.2, color=AZUL, fontweight="bold", pad=20, loc="left")

# ══════════════════════ PANEL 2 · todas las conexiones de una capa ══════════════════════
ax2 = fig.add_subplot(gs[1])
n_in, n_out = m1.shape                       # 128 x 512
xin = np.linspace(0, 100, n_in)
xout = np.linspace(0, 100, n_out)
segs2 = np.zeros((n_in * n_out, 2, 2))
k = 0
for i in range(n_in):
    segs2[k:k + n_out, 0, 0] = xin[i]
    segs2[k:k + n_out, 0, 1] = 1.0
    segs2[k:k + n_out, 1, 0] = xout
    segs2[k:k + n_out, 1, 1] = 0.0
    k += n_out
lc2 = LineCollection(segs2, colors=AZUL, linewidths=0.16, alpha=0.012, zorder=1)
ax2.add_collection(lc2)
ax2.scatter(xin, np.ones(n_in), s=13, c=AZUL, linewidths=0, zorder=3)
ax2.scatter(xout, np.zeros(n_out), s=5, c=NARANJA, linewidths=0, zorder=3)
ax2.text(-2.5, 1.0, "128", ha="right", va="center", fontsize=13, fontweight="bold", color=AZUL)
ax2.text(-2.5, 0.0, "512", ha="right", va="center", fontsize=13, fontweight="bold", color=NARANJA)
ax2.text(103, 0.5, "65.536\nconexiones\ndibujadas\nuna por una",
         ha="left", va="center", fontsize=11.5, color=ROJO, fontweight="bold")
ax2.set_xlim(-14, 122); ax2.set_ylim(-0.22, 1.22); ax2.axis("off")
ax2.set_title("UNA sola capa del modelo, con TODAS sus conexiones dibujadas · "
              "el bloque 0 tiene 4 capas como ésta, y hay 4 bloques",
              fontsize=13.2, color=AZUL, fontweight="bold", pad=12, loc="left")

# ══════════════════════ PANEL 3 · una neurona ══════════════════════
ax3 = fig.add_subplot(gs[2])
col = m1[:, 0]                                # las 128 conexiones que entran a UNA neurona
xs = np.linspace(0, 100, 128)
segs3 = [[(x, 1.0), (50, 0.0)] for x in xs]
lc3 = LineCollection(segs3, array=col, cmap=cmap, linewidths=0.85,
                     norm=plt.Normalize(-0.35, 0.35), alpha=0.9, zorder=2)
ax3.add_collection(lc3)
ax3.scatter(xs, np.ones(128), s=17, c=AZUL, linewidths=0, zorder=3)
ax3.scatter([50], [0], s=260, c=NARANJA, linewidths=0, zorder=4)
ax3.text(50, -0.19, "una neurona", ha="center", fontsize=11.5, fontweight="bold", color=NARANJA)
ax3.text(-2.5, 1.0, "128 entradas", ha="right", va="center", fontsize=11.5,
         fontweight="bold", color=AZUL)

fuerte = int(np.argmax(np.abs(col)))
ax3.plot([xs[fuerte], 50], [1.0, 0.0], color=ROJO, lw=2.6, zorder=5)
ax3.text(xs[fuerte], 1.09, f"la más fuerte: w = {col[fuerte]:+.4f}", ha="center",
         fontsize=11, color=ROJO, fontweight="bold")
ax3.text(103, 0.62, f"media |w| = {np.abs(col).mean():.4f}", fontsize=10.6, color="#555")
ax3.text(103, 0.50, f"máximo   = {col.max():+.4f}", fontsize=10.6, color="#555")
ax3.text(103, 0.38, f"mínimo   = {col.min():+.4f}", fontsize=10.6, color="#555")
ax3.text(103, 0.20, "rojo = negativo\nazul = positivo", fontsize=10, color="#777")
ax3.set_xlim(-20, 140); ax3.set_ylim(-0.30, 1.20); ax3.axis("off")
ax3.set_title("Y ahora UNA neurona sola, con sus 128 pesos reales · "
              "el modelo tiene 863.730 números como éstos",
              fontsize=13.2, color=AZUL, fontweight="bold", pad=12, loc="left")

fig.savefig("red_real.png", dpi=100, facecolor="white", bbox_inches="tight")
print("listo · conexiones fuertes dibujadas en el panel 1:", len(fuertes))
print("neuronas totales:", sum(n for _, n, _ in capas))
