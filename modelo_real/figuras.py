"""Figuras del explicativo del 9-sep-2026. Publico general, castellano, PDF."""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
import numpy as np

D = "figuras"
os.makedirs(D, exist_ok=True)
AZUL, NARANJA, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GRIS, GRIS_C, TINTA, TINTA2 = "#8f8e8a", "#dedcd6", "#0b0b0b", "#52514e"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.edgecolor": GRIS, "axes.linewidth": 0.8, "axes.labelcolor": TINTA2,
    "xtick.color": TINTA2, "ytick.color": TINTA2, "text.color": TINTA,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "axes.facecolor": "white",
})

def guardar(fig, n):
    fig.savefig(f"{D}/{n}.pdf", bbox_inches="tight", pad_inches=0.15)
    plt.close(fig); print("  ", n)

def piso(ax, y, txt, x=0.985):
    ax.axhline(y, color=GRIS, ls=(0, (4, 3)), lw=1.1, zorder=1)
    ax.text(x, y + 0.022, txt, transform=ax.get_yaxis_transform(),
            ha="right", va="bottom", fontsize=8.5, color=TINTA2)

# ---------------------------------------------------------------- 1. arquitecturas
def f_arquitecturas():
    """Cada bloque dibujado es un bloque REAL. Si dice 22, hay 22."""
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    ax.set_xlim(0, 100); ax.set_ylim(0, 68); ax.axis("off")
    ALTO, HUECO, ANCHO, TOPE = 1.02, 0.37, 15.5, 51.0
    modelos = [
        ("Micro LM", 4, ["d"]*6, ("ventanita", GRIS),
         "6 bloques, todos de regla delta"),
        ("Micro LM\nde frontera", 29, ["d"]*6, ("mira todo", NARANJA),
         "los mismos 6 bloques.\nLo único que cambia\nes la consulta"),
        ("TinyLlama", 54, ["a"]*22, ("mira todo", NARANJA),
         "22 bloques, todos de\natención completa"),
        ("Qwen3.5", 79, (["d"]*3 + ["a"])*6, ("mira todo", NARANJA),
         "24 bloques, 3 de regla\ndelta por cada 1 de\natención completa"),
    ]
    for nombre, x0, pila, (chip, cc), pie in modelos:
        ax.text(x0 + ANCHO/2, 65.5, nombre, ha="center", va="center",
                fontsize=11.5, fontweight="bold", linespacing=1.3)
        ax.add_patch(Rectangle((x0 + 0.6, 55.4), ANCHO - 1.2, 3.0,
                               facecolor=cc, alpha=0.16, edgecolor=cc, lw=1.1))
        ax.text(x0 + ANCHO/2, 56.9, f"la consulta {chip}", ha="center", va="center",
                fontsize=7.8, color=cc if cc != GRIS else TINTA2, fontweight="bold")
        y = TOPE
        for tipo in pila:
            ax.add_patch(Rectangle((x0, y), ANCHO, ALTO,
                                   facecolor=AZUL if tipo == "a" else GRIS_C,
                                   edgecolor="white", lw=0.9))
            y -= (ALTO + HUECO)
        # leyenda de cada columna a altura FIJA, debajo de la pila mas larga
        ax.text(x0 + ANCHO/2, 12.5, pie, ha="center", va="top",
                fontsize=8.2, color=TINTA2, linespacing=1.6)
    ax.add_patch(Rectangle((22, 0.8), 3.2, 1.4, facecolor=GRIS_C, edgecolor="white", lw=1.2))
    ax.text(26.2, 1.5, "regla delta, mira una ventanita", va="center",
            fontsize=8.4, color=TINTA2)
    ax.add_patch(Rectangle((57, 0.8), 3.2, 1.4, facecolor=AZUL, edgecolor="white", lw=1.2))
    ax.text(61.2, 1.5, "atención completa, mira todo", va="center",
            fontsize=8.4, color=TINTA2)
    guardar(fig, "01_arquitecturas")

# ---------------------------------------------------------------- 2. ley de cobertura
def f_cobertura():
    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    et = ["ventana corta\n(kernel 3)", "ventana justa\n(kernel 5)", "atención completa\n(mira todo)"]
    v = [0.6430, 0.9977, 0.9977]
    col = [GRIS, AZUL, AZUL]
    b = ax.bar(et, v, color=col, width=0.52, zorder=3)
    for r, x in zip(b, v):
        ax.text(r.get_x() + r.get_width()/2, x + 0.02, f"{x:.4f}".replace(".", ","),
                ha="center", fontsize=11, fontweight="bold")
    ax.annotate("", xy=(1, 1.10), xytext=(2, 1.10),
                arrowprops=dict(arrowstyle="<->", color=NARANJA, lw=1.6))
    ax.text(1.5, 1.135, "EMPATAN EXACTO", ha="center", fontsize=9.5,
            fontweight="bold", color=NARANJA)
    ax.set_ylim(0, 1.27); ax.set_yticks([0, .25, .5, .75, 1])
    ax.set_yticklabels(["0", "0,25", "0,50", "0,75", "1"])
    ax.set_ylabel("acierto")
    ax.set_title("Alcanza con que la pieza que importa entre en la ventana.\nMirar de más no compra nada.",
                 fontsize=10.5, loc="left", pad=12)
    ax.grid(axis="y", color=GRIS_C, lw=0.7, zorder=0)
    guardar(fig, "02_cobertura")

# ---------------------------------------------------------------- 3. trayectoria versiones
def f_versiones(d):
    h = [m for m in d["hist"] if m["vigente"] == m["vigente"]]
    x = [m["paso"] for m in h]; y = [m["vigente"] for m in h]
    # Cada hito son 4 muestras, asi que el crudo solo puede valer 0, 0,25, 0,50, 0,75 o 1 y
    # dibujarlo da un cerco de rayas que no es informacion. Va el promedio movil con su banda.
    k = 20
    sx = [np.mean(x[i:i+k]) for i in range(0, len(x)-k, k)]
    sy = [np.mean(y[i:i+k]) for i in range(0, len(y)-k, k)]
    ds = [np.std(y[i:i+k]) / np.sqrt(k) for i in range(0, len(y)-k, k)]
    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    ax.fill_between(sx, np.array(sy)-np.array(ds), np.minimum(np.array(sy)+np.array(ds), 1.0),
                    color=AZUL, alpha=0.16, lw=0, zorder=2)
    ax.plot(sx, sy, color=AZUL, lw=2.4, zorder=4)
    ax.axhline(0.5, color=GRIS, ls=(0, (4, 3)), lw=1.1, zorder=1)
    ax.text(20300, 0.455, "0,5000  tirar una moneda\nentre las dos versiones",
            ha="right", va="top", fontsize=8.5, color=TINTA2, linespacing=1.4)
    ax.scatter([sx[-1]], [sy[-1]], s=52, color=AZUL, zorder=5,
               edgecolor="white", linewidth=2)
    ax.annotate("0,9366", xy=(sx[-1], sy[-1]), xytext=(-14, 14),
                textcoords="offset points", fontsize=12, fontweight="bold", color=AZUL)
    ax.set_ylim(0, 1.06); ax.set_xlim(0, 20600)
    ax.set_yticks([0, .25, .5, .75, 1]); ax.set_yticklabels(["0", "0,25", "0,50", "0,75", "1"])
    ax.set_xticks(range(0, 20001, 5000))
    ax.set_xticklabels([f"{v//1000} mil" if v else "0" for v in range(0, 20001, 5000)])
    ax.set_xlabel("pasos de entrenamiento"); ax.set_ylabel("acierta la versión VIGENTE")
    ax.set_title("Se le dice un dato, después se lo corrige, y se le pregunta.\nAprende a contestar el último.",
                 fontsize=10.5, loc="left", pad=12)
    ax.grid(axis="y", color=GRIS_C, lw=0.7, zorder=0)
    guardar(fig, "03_versiones")

# ---------------------------------------------------------------- 4. controles
def f_controles(d):
    c = d["controles"]
    et = ["con el archivo\nen su lugar", "archivo\nBARAJADO", "archivo\nen CEROS", "lectura\nAPAGADA"]
    v = [c["con archivo"]["vigente"], c["BARAJADO"]["vigente"],
         c["VACIO (ceros)"]["vigente"], c["lectura APAGADA"]["vigente"]]
    fig, ax = plt.subplots(figsize=(6.6, 3.3))
    b = ax.bar(et, v, color=[AZUL, GRIS, GRIS, GRIS], width=0.5, zorder=3)
    for r, x in zip(b, v):
        ax.text(r.get_x() + r.get_width()/2, x + 0.03,
                f"{x:.4f}".replace(".", ","), ha="center", fontsize=11, fontweight="bold",
                color=AZUL if x > 0.5 else TINTA2)
    ax.set_ylim(0, 1.2); ax.set_yticks([0, .5, 1]); ax.set_yticklabels(["0", "0,50", "1"])
    ax.set_ylabel("acierta la versión vigente")
    ax.set_title("Si se le toca el archivo, no contesta nada.\nLa respuesta sale de ahí y de ningún otro lado.",
                 fontsize=10.5, loc="left", pad=12)
    ax.grid(axis="y", color=GRIS_C, lw=0.7, zorder=0)
    guardar(fig, "04_controles")

# ---------------------------------------------------------------- 5. presupuesto
def f_presupuesto():
    fig, ax = plt.subplots(figsize=(6.2, 3.3))
    et = ["2 mil pasos", "20 mil pasos"]
    v = [0.5595, 0.9366]
    b = ax.bar(et, v, color=[GRIS, AZUL], width=0.44, zorder=3)
    for r, x, nota in zip(b, v, ["no se distingue\nde una moneda", "aprendió"]):
        ax.text(r.get_x() + r.get_width()/2, x + 0.03, f"{x:.4f}".replace(".", ","),
                ha="center", fontsize=12, fontweight="bold")
        ax.text(r.get_x() + r.get_width()/2, x/2, nota, ha="center", va="center",
                fontsize=9, color="white", fontweight="bold")
    ax.axhline(0.5, color=GRIS, ls=(0, (4, 3)), lw=1.1, zorder=1)
    # entre las dos barras es el unico hueco libre a la altura de la linea del piso
    ax.text(0.5, 0.53, "0,5000\ntirar una moneda", ha="center", va="bottom",
            fontsize=8.5, color=TINTA2, linespacing=1.4)
    ax.set_ylim(0, 1.1); ax.set_yticks([0, .5, 1]); ax.set_yticklabels(["0", "0,50", "1"])
    ax.set_ylabel("acierta la versión vigente")
    ax.set_title("La misma tarea, el mismo modelo. Lo único que cambia\nes cuánto tiempo se lo deja estudiar.",
                 fontsize=10.5, loc="left", pad=12)
    ax.grid(axis="y", color=GRIS_C, lw=0.7, zorder=0)
    guardar(fig, "05_presupuesto")

# ---------------------------------------------------------------- 6. llave vs respuesta
def f_disociacion():
    """Los dos paneles comparan contra el MISMO ancla y cambian UNA sola cosa cada uno.

    La version anterior de esta figura estaba confundida: el brazo de la derecha cambiaba a la vez
    las entidades y el tamanio del archivo. Se corrio la celda que faltaba y se separo.
    """
    fig, axs = plt.subplots(1, 2, figsize=(8.2, 3.5))
    datos = [("Cuántas cosas PUEDE DECIR",
              ["8 respuestas\nposibles", "100 respuestas\nposibles"], [0.9881, 0.7262], AQUA),
             ("Cuántas cosas tiene que DISTINGUIR",
              ["8 nombres\ndistintos", "60 nombres\ndistintos"], [0.9881, 0.2143], NARANJA)]
    for ax, (t, et, v, c) in zip(axs, datos):
        b = ax.bar(et, v, color=[GRIS_C, c], width=0.46, zorder=3, edgecolor="white", lw=1.6)
        for r, x in zip(b, v):
            ax.text(r.get_x()+r.get_width()/2, x+0.035, f"{x:.4f}".replace(".", ","),
                    ha="center", fontsize=11.5, fontweight="bold")
        ax.axhline(0.125, color=GRIS, ls=(0, (4, 3)), lw=1.1, zorder=1)
        ax.text(1.46, 0.15, "azar", ha="right", fontsize=8.5, color=TINTA2)
        ax.set_ylim(0, 1.18); ax.set_yticks([0, .5, 1]); ax.set_yticklabels(["0", "0,50", "1"])
        ax.set_title(t, fontsize=10.5, loc="left", pad=10, fontweight="bold")
        ax.grid(axis="y", color=GRIS_C, lw=0.7, zorder=0)
    axs[0].set_ylabel("acierto")
    fig.suptitle("Multiplicar por 12 lo que puede DECIR sale barato.\n"
                 "Multiplicar por 7 lo que tiene que DISTINGUIR lo tira al azar.",
                 fontsize=10.5, x=0.045, ha="left", y=1.10)
    guardar(fig, "06_disociacion")

# ---------------------------------------------------------------- 7. capas de Qwen3.5
def f_qwen():
    from transformers import AutoConfig
    try:
        t = AutoConfig.from_pretrained("Qwen/Qwen3.5-0.8B").text_config
        lt = list(t.layer_types)
    except Exception:
        lt = (["linear_attention"]*3 + ["full_attention"])*6
    fig, ax = plt.subplots(figsize=(8.2, 1.9))
    ax.set_xlim(-0.6, len(lt)-0.4); ax.set_ylim(0, 3.4); ax.axis("off")
    for i, x in enumerate(lt):
        full = "full" in x
        ax.add_patch(Rectangle((i-0.42, 1.15), 0.84, 1.0,
                               facecolor=AZUL if full else GRIS_C,
                               edgecolor="white", lw=1.2))
        if full:
            ax.text(i, 0.72, str(i), ha="center", fontsize=7.6, color=AZUL, fontweight="bold")
    ax.text(-0.6, 2.65, "Las 24 capas de Qwen3.5, en orden", fontsize=10.5, fontweight="bold")
    ax.text(-0.6, 0.18, "18 de regla delta (gris) y 6 de atención completa (azul), una cada cuatro. "
                        "El micro LM es la misma idea, más chica.",
            fontsize=8.6, color=TINTA2)
    guardar(fig, "07_qwen_capas")

# ---------------------------------------------------------------- 8. (retirada)
def _f_curva_RETIRADA():
    """La curva del tamanio de archivo se retira del explicativo. Estaba CONFUNDIDA: su punto de
    archivo 4 tenia 8 entidades y los demas 60, y al correr la celda que faltaba (60 entidades con
    archivo 4, que da 0,2143) resulto que el confundido era todo el efecto."""
    fig, ax = plt.subplots(figsize=(6.6, 3.3))
    x = [4, 8, 12, 16, 24]; y = [0.9881, 0.1905, 0.0833, 0.1310, 0.1429]
    ax.plot(x, y, color=NARANJA, lw=2.2, marker="o", ms=8, zorder=4,
            markeredgecolor="white", markeredgewidth=2)
    for a, b_ in zip(x, y):
        ax.annotate(f"{b_:.4f}".replace(".", ","), (a, b_), textcoords="offset points",
                    xytext=(0, 13), ha="center", fontsize=9, color=TINTA2)
    piso(ax, 0.125, "0,1250  azar")
    ax.set_ylim(0, 1.15); ax.set_xticks(x)
    ax.set_yticks([0, .5, 1]); ax.set_yticklabels(["0", "0,50", "1"])
    ax.set_xlabel("cuantos datos hay guardados en el archivo")
    ax.set_ylabel("acierto")
    ax.set_title("Entre 4 y 8 datos guardados se cae del techo al azar,\ny de ahi en adelante queda plano.",
                 fontsize=10.5, loc="left", pad=12)
    ax.grid(axis="y", color=GRIS_C, lw=0.7, zorder=0)
    guardar(fig, "08_curva_archivo")


# ---------------------------------------------------------------- 8. el techo y el reloj
def f_reloj():
    """La figura que resume el dia. Tres tareas, dos presupuestos, el mismo modelo."""
    import json as _j
    def med(u, campo, nb, desde):
        d = _j.load(open(f"banco_{u}_s0.json"))
        h = [m[campo] for m in d["hist"] if m["paso"] >= desde and m[campo] == m[campo]]
        return sum(h)/len(h)
    tareas = ["Contestar la\nversión vigente",
              "60 nombres,\narchivo de 8",
              "60 nombres,\narchivo de 4"]
    corto = [med("ver", "vigente", 4, 1500), med("arch8", "acierto", 4, 1500),
             med("arch4e60", "acierto", 4, 1500)]
    largo = [med("verlargo", "vigente", 4, 15000), med("arch8largo", "acierto", 4, 15000),
             med("arch4e60largo", "acierto", 4, 15000)]
    x = np.arange(3); an = 0.33
    fig, ax = plt.subplots(figsize=(7.6, 3.7))
    b1 = ax.bar(x - an/2 - 0.012, corto, an, label="2 mil pasos", color=GRIS, zorder=3)
    b2 = ax.bar(x + an/2 + 0.012, largo, an, label="20 mil pasos", color=AZUL, zorder=3)
    for bs in (b1, b2):
        for r in bs:
            ax.text(r.get_x()+r.get_width()/2, r.get_height()+0.028,
                    f"{r.get_height():.4f}".replace(".", ","), ha="center",
                    fontsize=9.5, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(tareas, fontsize=9.5)
    ax.set_ylim(0, 1.18); ax.set_yticks([0, .5, 1]); ax.set_yticklabels(["0", "0,50", "1"])
    ax.set_ylabel("acierto")
    ax.legend(frameon=False, ncol=2, loc="upper left", bbox_to_anchor=(0, 1.02), fontsize=9.5)
    ax.set_title("Las mismas tres tareas, el mismo modelo, el mismo código.\n"
                 "Lo único que cambia es cuánto tiempo se lo deja estudiar.",
                 fontsize=10.5, loc="left", pad=30)
    ax.grid(axis="y", color=GRIS_C, lw=0.7, zorder=0)
    guardar(fig, "08_reloj")

if __name__ == "__main__":
    print("figuras:")
    d = json.load(open("banco_verlargo_s0.json"))
    f_arquitecturas(); f_cobertura(); f_versiones(d); f_controles(d)
    f_presupuesto(); f_disociacion(); f_qwen(); f_reloj()
    print("listo")
