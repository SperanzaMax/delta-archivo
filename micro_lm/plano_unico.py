"""EL MICRO-LM ENTERO EN UN SOLO PLANO, todo a la misma escala.

Regla del dibujo: el ANCHO de cada caja es proporcional a su número real de neuronas, y el
porcentaje que se anota es su parte real de los 863.730 pesos. Nada está exagerado para que entre.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

AZUL, AZUL2, NARANJA, VERDE, ROJO, GRIS = "#12395f", "#3f6d99", "#b5651d", "#2e7d44", "#c0392b", "#888"
TOT = 863730

fig, ax = plt.subplots(figsize=(19.2, 14.6), dpi=100, facecolor="white")
ax.set_facecolor("white")

ESC = 0.0148            # 1 neurona = 0,0148 de ancho  ->  512 neuronas = 7,58
ALTO = 0.60

def caja(cx, y, n, txt, sub="", col=AZUL, fc=None, fs=10.4, alpha=1.0, alto=ALTO, lado=None):
    """Una capa. El ancho SALE de n: no se elige a ojo.

    Si el texto no entra en el ancho que le toca, se pone AL COSTADO en vez de encogerlo o de
    ensanchar la caja: la caja tiene que seguir siendo proporcional a las neuronas, que es lo
    unico que este dibujo promete.
    """
    w = n * ESC
    ax.add_patch(FancyBboxPatch((cx - w/2, y), w, alto, boxstyle="round,pad=0.02,rounding_size=0.06",
                                fc=fc or "#eaf0f6", ec=col, lw=1.5, alpha=alpha, zorder=3))
    entra = len(txt) * fs * 0.0135 < w
    if entra and lado is None:
        ax.text(cx, y + alto*0.62, txt, ha="center", va="center", fontsize=fs, fontweight="bold",
                color=col, zorder=4)
        if sub:
            ax.text(cx, y + alto*0.22, sub, ha="center", va="center", fontsize=8.6, color="#555", zorder=4)
    else:
        lx = cx + w/2 + 0.28 if lado != "izq" else cx - w/2 - 0.28
        ha = "left" if lado != "izq" else "right"
        ax.text(lx, y + alto*0.66, txt, ha=ha, va="center", fontsize=fs, fontweight="bold",
                color=col, zorder=4)
        if sub:
            ax.text(lx, y + alto*0.26, sub, ha=ha, va="center", fontsize=8.4, color="#555", zorder=4)
    return w

def flecha(x0, y0, x1, y1, col=AZUL, lw=1.8, estilo="-|>", rad=0.0, ls="-"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=estilo, mutation_scale=15,
                                 color=col, lw=lw, linestyle=ls,
                                 connectionstyle=f"arc3,rad={rad}", zorder=5))

XI, XD = 6.2, 20.6          # centro de la rama izquierda y de la derecha
y = 15.2

# ───────────────────────── títulos de rama ─────────────────────────
ax.text(XI, 16.15, "RAMA 1 · ESCRIBIR lo que se dijo", ha="center", fontsize=13.5,
        fontweight="bold", color=AZUL)
ax.text(XD, 16.15, "RAMA 2 · RESPONDER la pregunta", ha="center", fontsize=13.5,
        fontweight="bold", color=AZUL)

# ───────────────────────── entradas ─────────────────────────
caja(XI, y, 242, "242 tokens · lo que se dijo", "«el director de norte es ana» · «no , es beto»", AZUL)
caja(XD, y, 242, "242 tokens · la pregunta", "«cual es el director de norte ?»", AZUL, lado="der")
flecha(XI, y, XI, y-0.62); flecha(XD, y, XD, y-0.62)
y -= 1.22
caja(XI, y, 128, "embedding · 128", "3,6 %", AZUL)
caja(XD, y, 128, "embedding · 128", "mismos pesos", AZUL)
flecha(XI, y, XI, y-0.62); flecha(XD, y, XD, y-0.62)

# ───────────────────────── los 4 bloques ─────────────────────────
y -= 1.30
ax.add_patch(Rectangle((XI-4.6, y-8.55), 9.2, 9.1, fc="#f6f9fc", ec=AZUL2, lw=1.4, ls="--", zorder=1))
ax.add_patch(Rectangle((XD-4.6, y-8.55), 9.2, 9.1, fc="#f6f9fc", ec=AZUL2, lw=1.4, ls="--", zorder=1))
ax.text(XI, y+0.30, "EL TRONCO · 84,2 % de todos los pesos", ha="center", fontsize=10.6,
        color=AZUL2, fontweight="bold")
ax.text(XD, y+0.30, "EL MISMO TRONCO — los mismos pesos", ha="center", fontsize=10.6,
        color=AZUL2, fontweight="bold")

yb = y - 0.30
for b in range(4):
    for X in (XI, XD):
        caja(X, yb-0.55, 128, f"bloque {b} · regla delta · 128", "wq wk wv · 5,7 %", AZUL2, "#e3ecf4", 9.4, alto=0.48)
        caja(X, yb-1.20, 512, "MLP interno · 512", "15,2 % — la capa más ancha", NARANJA, "#fdf3e8", 9.4, alto=0.48)
        flecha(X, yb-0.55, X, yb-0.72, AZUL2, 1.2)
        flecha(X, yb-1.20, X, yb-1.37, NARANJA, 1.2)
    yb -= 2.06

# la lectura entra en el BLOQUE 0 de la rama derecha
ax.add_patch(Rectangle((XD-4.75, y-2.25), 9.5, 2.0, fc="none", ec=ROJO, lw=2.2, zorder=6))
ax.text(XD+4.95, y-1.25, "acá entra\nla memoria", ha="left", va="center", fontsize=10,
        color=ROJO, fontweight="bold")

y -= 8.75

# ───────────────────────── salidas de cada rama ─────────────────────────
caja(XI, y, 128, "se toma el ÚLTIMO token de cada enunciado", "«un hecho dicho = una entrada»", NARANJA, "#fdf3e8", lado="izq")
caja(XD, y, 242, "cabeza de salida · 242", "un puntaje por token · 3,6 %", VERDE, "#eaf6ec")
flecha(XI, y, XI, y-0.70, NARANJA)
flecha(XD, y, XD, y-0.70, VERDE)
y -= 1.34

# ───────────────────────── el archivo ─────────────────────────
ARCH_X, ARCH_Y = XI, y - 1.15
ax.add_patch(FancyBboxPatch((ARCH_X-3.55, ARCH_Y-0.30), 7.1, 2.15,
                            boxstyle="round,pad=0.03,rounding_size=0.08",
                            fc="#fffcf4", ec=NARANJA, lw=2.4, zorder=3))
ax.text(ARCH_X, ARCH_Y+1.58, "EL ARCHIVO", ha="center", fontsize=12.6, fontweight="bold", color=NARANJA)
ax.text(ARCH_X, ARCH_Y+1.30, "una entrada de 128 números por cada cosa dicha",
        ha="center", fontsize=9.2, color="#555")
for i, (t, turno) in enumerate([("«el director de norte es ana»", 1), ("«no , es beto»", 2)]):
    ax.add_patch(Rectangle((ARCH_X-2.85, ARCH_Y+0.92-i*0.42), 5.7, 0.34, fc="white", ec="#d8b48a", zorder=4))
    ax.text(ARCH_X-2.72, ARCH_Y+1.09-i*0.42, t, va="center", fontsize=8.8, family="monospace", zorder=5)
    ax.add_patch(Rectangle((ARCH_X+1.72, ARCH_Y+0.96-i*0.42), 1.05, 0.26, fc="#ffe9c8", ec=NARANJA, zorder=5))
    ax.text(ARCH_X+2.24, ARCH_Y+1.09-i*0.42, f"turno {turno}", ha="center", va="center", fontsize=8, color="#8a4a10", zorder=6)
ax.text(ARCH_X, ARCH_Y+0.14, "SELLO DE ORDEN · 8.192 pesos", ha="center", fontsize=9.2,
        color="#8a4a10", fontweight="bold")
ax.text(ARCH_X, ARCH_Y-0.13, "es lo que deja saber cuál versión rige · sin él 0,4570 · con él 0,9956",
        ha="center", fontsize=8.4, color="#8a4a10")

# flecha de escritura y de lectura
flecha(XI, y, ARCH_X, ARCH_Y+1.85, NARANJA, 2.2)
flecha(ARCH_X+3.55, ARCH_Y+1.05, XD-4.80, 11.35, ROJO, 2.6, rad=-0.30)
ax.text((ARCH_X+XD)/2, ARCH_Y+2.55, "LECTURA POR SOFTMAX · 65.536 pesos (7,6 %)",
        ha="center", fontsize=10.4, color=ROJO, fontweight="bold")
ax.text((ARCH_X+XD)/2, ARCH_Y+2.22,
        "compara la pregunta con cada entrada archivada y reparte 100 % de atención",
        ha="center", fontsize=9, color="#666")
ax.text((ARCH_X+XD)/2, ARCH_Y+1.92,
        "medido: si entra acá vale 0,7275 · si entrara al final, 0,3827",
        ha="center", fontsize=9, color=ROJO, style="italic")

# ───────────────────────── respuesta ─────────────────────────
caja(XD, ARCH_Y+0.65, 242, "irma", "gana un solo token de los 242", VERDE, "#eaf6ec", 13)

# ───────────────────────── barra de reparto de pesos ─────────────────────────
BY = -1.35
ax.text(0.35, BY+1.28, "LOS 863.730 PESOS, repartidos a escala:", fontsize=11.6,
        fontweight="bold", color=AZUL)
partes = [("embedding", 30976, AZUL), ("archivo (leer + sello)", 73728, NARANJA),
          ("los 4 bloques del tronco", 727552, AZUL2), ("cabeza", 31218, VERDE)]
x0 = 0.35
ANCHO_BARRA = 25.6
for nom, v, c in partes:
    w = ANCHO_BARRA * v / TOT
    ax.add_patch(Rectangle((x0, BY+0.45), w, 0.55, fc=c, ec="white", lw=1.2, zorder=3))
    pct = 100 * v / TOT
    if w > 2.2:
        ax.text(x0 + w/2, BY+0.72, f"{nom} · {pct:.1f} %", ha="center", va="center",
                fontsize=10, color="white", fontweight="bold", zorder=4)
    else:
        dy = 0.20 if nom.startswith("embedding") else -0.22
        ax.text(x0 + w/2, BY+dy, f"{nom}\n{pct:.1f} %".replace(".", ","), ha="center", va="top",
                fontsize=8.6, color=c, fontweight="bold", zorder=4)
    x0 += w

ax.text(13.3, 17.25, "MICRO-LM · el modelo entero en un solo plano",
        ha="center", fontsize=22, fontweight="bold", color=AZUL)
ax.text(13.3, 16.78, "863.730 pesos · 3,5 MB · vocabulario de 242 palabras · "
                     "el ancho de cada caja es proporcional a su cantidad real de neuronas",
        ha="center", fontsize=11.4, color="#555")

ax.set_xlim(-0.4, 27.6); ax.set_ylim(-1.95, 17.7)
ax.axis("off")
fig.savefig("plano_unico.png", dpi=100, facecolor="white", bbox_inches="tight")
print("listo")
