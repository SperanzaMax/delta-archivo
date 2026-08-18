"""Las 15 capas del Micro-LM apiladas, dibujando SOLO 1 de cada 10 componentes.

Por que existe: el dibujo a escala real del 17-ago (`red_real.py`) es honesto pero ilegible — 512
neuronas en una fila se ven como una linea continua. Aca se muestra **el 10 % de cada capa**, elegido
a intervalos regulares, para que se distinga cada neurona y cada conexion. Los pesos siguen siendo
los REALES del checkpoint entrenado: lo unico que cambia es cuantos se dibujan.

    python3 red_10pc.py        (necesita pesos_full.npz, extraido del ckpt con el venv de jax)
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle, Circle
from matplotlib.lines import Line2D

AZUL, AZUL2, NARANJA, VERDE, ROJO, GRIS = ("#12395f", "#3f6d99", "#b5651d", "#2e7d44",
                                           "#c0392b", "#777")
FRAC = 0.10                      # se dibuja 1 de cada 10
Z = np.load("pesos_full.npz")

# ── las 15 capas, con su tamaño REAL y que hace cada una ───────────────────────────────────────
capas = [("entrada · las 242 palabras", 242, AZUL,
          "una casilla por palabra; se prende la que entra")]
capas.append(("embedding · 128 números por palabra", 128, AZUL2,
              "cada palabra pasa a ser 128 números"))
for b in range(4):
    capas += [
        (f"bloque {b} · memoria delta (128)", 128, AZUL2,
         "escribe el hecho nuevo y borra el que reemplaza"),
        (f"bloque {b} · capa de proceso (512)", 512, NARANJA,
         "4× más ancha: combina lo leído con lo sabido"),
        (f"bloque {b} · salida del bloque (128)", 128, AZUL2,
         "vuelve a 128 y pasa al bloque siguiente"),
    ]
capas.append(("salida · un puntaje por palabra", 242, VERDE,
              "el puntaje más alto es la respuesta"))

# conexiones REALES entre capas consecutivas (None = paso residual, sin matriz propia)
mats = [Z["emb"], Z["wv0"]]
for b in range(4):
    mats += [Z["m1_%d" % b], Z["m2_%d" % b], (Z["wv%d" % (b + 1)] if b < 3 else Z["head"])]
mats = mats[:len(capas) - 1]

fig = plt.figure(figsize=(16.0, 20.5), dpi=110, facecolor="white")
AX_BOX = [0.295, 0.035, 0.375, 0.885]
ax = fig.add_axes(AX_BOX)
ax.set_facecolor("white")
ANCHO = 100.0

fig.text(0.035, 0.972, "Micro-LM · las 15 capas, 1 de cada 10 componentes",
         fontsize=27, fontweight="bold", color=AZUL)
fig.text(0.035, 0.9545,
         "Modelo entrenado desde cero, 863.730 pesos (3,5 MB). Los pesos dibujados son los reales "
         "del checkpoint; se muestra el 10 % de cada capa para que se vea.",
         fontsize=12.6, color="#555")

# ── nodos ──────────────────────────────────────────────────────────────────────────────────────
idxs, ys = [], []
for k, (nom, n, col, _) in enumerate(capas):
    m = max(2, int(round(n * FRAC)))
    idx = np.linspace(0, n - 1, m).astype(int)
    idxs.append(idx)
    y = -k * 1.0
    ys.append(y)
    xs = np.linspace(-ANCHO / 2, ANCHO / 2, m)
    ax.scatter(xs, np.full(m, y), s=95, c="white", edgecolors=col, linewidths=1.9, zorder=4)

# ── conexiones reales entre los componentes dibujados ──────────────────────────────────────────
for k, W in enumerate(mats):
    ia, ib = idxs[k], idxs[k + 1]
    xa = np.linspace(-ANCHO / 2, ANCHO / 2, len(ia))
    xb = np.linspace(-ANCHO / 2, ANCHO / 2, len(ib))
    sub = W[np.ix_(ia, ib)]
    # de las conexiones entre los nodos dibujados se trazan las 12 % mas fuertes: dibujarlas todas
    # tapa los nodos y vuelve a hacer ilegible justo lo que este grafico venia a arreglar.
    corte = np.quantile(np.abs(sub), 0.88)
    segs, vals = [], []
    for i in range(len(ia)):
        for j in range(len(ib)):
            if abs(sub[i, j]) >= corte:
                segs.append([(xa[i], ys[k]), (xb[j], ys[k + 1])])
                vals.append(sub[i, j])
    vals = np.array(vals)
    cols = [AZUL2 if v > 0 else ROJO for v in vals]
    lw = 0.30 + 1.5 * np.abs(vals) / max(1e-6, np.abs(vals).max())
    ax.add_collection(LineCollection(segs, colors=cols, linewidths=lw, alpha=0.42, zorder=2))

ax.set_xlim(-ANCHO / 2 - 6, ANCHO / 2 + 6)
ax.set_ylim(ys[-1] - 0.8, 0.9)
ax.axis("off")

# ── etiquetas a la derecha, en coordenadas de FIGURA ───────────────────────────────────────────
# Van en coords de figura y no del eje: puestas como texto del eje se salían del lienzo y las
# descripciones quedaban cortadas a media palabra.
y0, y1 = ax.get_ylim()
XT = AX_BOX[0] + AX_BOX[2] + 0.018
for k, (nom, n, col, expl) in enumerate(capas):
    yf = AX_BOX[1] + AX_BOX[3] * (ys[k] - y0) / (y1 - y0)
    fig.text(XT, yf + 0.0062, nom, va="center", fontsize=12.4, fontweight="bold", color=col)
    fig.text(XT, yf - 0.0068, f"se dibujan {len(idxs[k])} de {n}  ·  {expl}",
             va="center", fontsize=9.4, color="#666")

# ══════════════ REFERENCIA (columna izquierda) ══════════════
def bloque_ref(y0, titulo, lineas, col=AZUL):
    fig.text(0.035, y0, titulo, fontsize=13.2, fontweight="bold", color=col)
    yy = y0 - 0.019
    for t in lineas:
        fig.text(0.035, yy, t, fontsize=10.4, color="#444", va="top", wrap=True)
        yy -= 0.0155 * (1 + t.count("\n"))
    return yy

fig.patches.append(Rectangle((0.028, 0.035), 0.245, 0.895, transform=fig.transFigure,
                             fc="#f6f8fa", ec="#dde3ea", lw=1.2, zorder=0))

y = 0.905
fig.text(0.035, y, "CÓMO SE LEE", fontsize=15.5, fontweight="bold", color=AZUL)
y -= 0.030

# leyenda grafica, dibujada como artistas de la FIGURA: un axes acá quedaba tapado por el panel.
fig.add_artist(Circle((0.046, y - 0.008), 0.0052, fc="white", ec=AZUL2, lw=1.9,
                      transform=fig.transFigure, zorder=5))
fig.text(0.060, y - 0.008, "una neurona: un número que la\nred calcula en ese punto",
         va="center", fontsize=9.8, color="#444")
fig.add_artist(Line2D([0.040, 0.053], [y - 0.036, y - 0.036], color=AZUL2, lw=2.6,
                      transform=fig.transFigure, zorder=5))
fig.text(0.060, y - 0.036, "peso positivo: empuja a favor", va="center", fontsize=9.8, color="#444")
fig.add_artist(Line2D([0.040, 0.053], [y - 0.052, y - 0.052], color=ROJO, lw=2.6,
                      transform=fig.transFigure, zorder=5))
fig.text(0.060, y - 0.052, "peso negativo: frena", va="center", fontsize=9.8, color="#444")
y -= 0.078

y = bloque_ref(y, "El grosor de la línea",
               ["Es la fuerza del peso: cuánto pesa esa",
                "conexión en la cuenta final. Se dibujan",
                "sólo las más fuertes de cada tramo, si no",
                "las líneas tapan las neuronas."])
y -= 0.012
y = bloque_ref(y, "Por qué 1 de cada 10",
               ["Las capas reales tienen 242, 128 y 512",
                "componentes. Dibujadas enteras se ven como",
                "una franja sólida. Acá se toma 1 de cada 10,",
                "a intervalos regulares, y se dibuja su peso",
                "REAL: no hay ningún elemento inventado."])
y -= 0.012
y = bloque_ref(y, "El camino de una pregunta",
               ["1 · entra el texto, una palabra por vez",
                "2 · el embedding la vuelve 128 números",
                "3 · cada bloque escribe en la memoria",
                "     delta y procesa en la capa ancha",
                "4 · tras 4 bloques, la salida da un",
                "     puntaje a cada una de las 242",
                "     palabras; la más alta es la respuesta"])
y -= 0.012
y = bloque_ref(y, "Qué es la memoria delta",
               ["La parte que hace de archivo: cuando llega",
                "un hecho nuevo lo escribe, y cuando llega",
                "una corrección BORRA lo viejo en vez de",
                "amontonarlo. Es lo que le permite contestar",
                "sobre algo dicho en una sesión anterior."])
y -= 0.012
y = bloque_ref(y, "Los números del modelo",
               ["· 863.730 pesos en total (3,5 MB)",
                "· 242 palabras de vocabulario",
                "· 128 números por palabra",
                "· 4 bloques · capa de proceso de 512",
                "· entrenado desde cero, sin partir de",
                "  ningún modelo previo"])

fig.text(0.035, 0.045, "checkpoint n4_s2 · 12.000 pasos · 2026-08-18",
         fontsize=9.2, color="#999")

fig.savefig("RED_10PC_20260818.png", facecolor="white")
print("escrito RED_10PC_20260818.png")
