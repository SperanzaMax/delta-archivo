"""Juez de la curva techo-vs-RECUP DENTRO de familia · criterios en NOTA_CURVA_INTRAFAMILIA.md
(SHA ee781db0), congelados ANTES del dato. Este script tambien se escribio antes de ver el JSON.

La pregunta: la pendiente global +0,1308 salio de un conjunto CONFUNDIDO (3000 pasos con recompensa
contra 12000 sin ella) y con la `r` inflada por tener dos nubes separadas en el eje x. Dentro de una
familia todo es identico salvo la SEMILLA, asi que la pendiente ahi no tiene ese defecto.

Cuarta leccion del proyecto sobre jueces automaticos: este imprime NO EVALUABLE cuando corresponde,
en vez de un numero. Ver `regla-verificar-antes-de-veredicto`.
"""
import json
import os
import sys

import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))

FAMILIAS = {"p3": "pre", "q3": "post", "v3": "lat2", "w3": "lat"}
APARTE = {"b3": "pre + blanco=error (NO entra en la pendiente principal)"}
RANGO_MIN = 0.10        # una familia con menos rango de RECUP no informa la pendiente
MIN_FAMILIAS = 2


def recta(x, y):
    """pendiente por minimos cuadrados; con 2 puntos es la recta que los une"""
    return float(np.polyfit(x, y, 1)[0])


def main(ruta):
    filas = json.load(open(ruta))
    print("=" * 96)
    print("CURVA techo-vs-RECUP DENTRO DE FAMILIA   ·   criterios SHA ee781db0, fijados antes")
    print("=" * 96)

    porfam = {}
    for f in filas:
        pref = f["unidad"].split("_")[0]
        porfam.setdefault(pref, []).append(f)

    pendientes, informativas, descartadas = {}, [], []
    for pref in list(FAMILIAS) + list(APARTE):
        fs = sorted(porfam.get(pref, []), key=lambda z: z["recup"])
        if not fs:
            print(f"\n--- {pref}: no hay unidades en el JSON")
            continue
        etiq = FAMILIAS.get(pref) or APARTE[pref]
        print(f"\n--- familia {pref} ({etiq}) ---")
        for f in fs:
            techo = max(f["lineal"], f["no_lineal"])
            marca = "" if f["confiable"] else "   <- controles fallan (se usa el lineal si paso)"
            print(f"   {f['unidad']:8s} RECUP {f['recup']:.4f}   techo {techo:.4f}"
                  f"   (lin {f['lineal']:.4f} / no-lin {f['no_lineal']:.4f}){marca}")
        x = np.array([f["recup"] for f in fs])
        y = np.array([max(f["lineal"], f["no_lineal"]) for f in fs])
        rango = float(x.max() - x.min())
        if len(fs) < 2:
            print(f"   -> una sola unidad, no da pendiente")
            continue
        m = recta(x, y)
        r = float(np.corrcoef(x, y)[0, 1]) if len(fs) > 2 else float("nan")
        print(f"   rango de RECUP entre semillas {rango:.4f}   pendiente {m:+.4f}"
              + (f"   r {r:+.4f}" if len(fs) > 2 else ""))
        if pref in APARTE:
            print("   (declarada aparte: no entra en la pendiente principal)")
            continue
        if rango < RANGO_MIN:
            print(f"   ** NO INFORMA ** el rango {rango:.4f} < {RANGO_MIN}: la pendiente es ruido")
            descartadas.append(pref)
        else:
            pendientes[pref] = m
            informativas.append(pref)

    print("\n" + "=" * 96)
    print(f"familias informativas: {informativas or 'ninguna'}"
          + (f"   ·   descartadas por rango: {descartadas}" if descartadas else ""))

    if len(informativas) < MIN_FAMILIAS:
        print(f"\n** NO EVALUABLE ** quedan {len(informativas)} familias informativas y hacen falta "
              f"{MIN_FAMILIAS}.\n   El riesgo declarado en la nota protege a los tres criterios, asi "
              "que NO se lee ninguno.\n   Lo que corresponde: mover RECUP a proposito en vez de "
              "aprovechar la semilla.")
        return

    ms = np.array([pendientes[p] for p in informativas])
    m = float(np.median(ms))
    print(f"pendientes intra-familia: " + "  ".join(f"{p}={pendientes[p]:+.4f}" for p in informativas))
    print(f"MEDIANA m = {m:+.4f}     (la global confundida era +0,1308)")

    # GUARDA DE DISCORDANCIA, agregada el 1-sep al ver que la mediana de DOS valores que caen en
    # lados opuestos de los umbrales inventa un numero que no representa a ninguno. Es la trampa que
    # el proyecto ya piso tres veces («una media que esconde su distribucion») y la primera vez que
    # se la ataja dentro del juez en vez de en la lectura.
    if len(ms) < 3 and ((ms >= 0.40).any() and (ms <= 0.15 + 0.05).any()):
        print("\n** VEREDICTO NO LEIBLE POR DISCORDANCIA **")
        print(f"   Las {len(ms)} familias informativas caen en lados OPUESTOS de los umbrales")
        print("   y con menos de 3 la mediana es su promedio: un numero que no representa a ninguna.")
        print("   La pendiente hay que estimarla con todas las unidades a la vez (efecto WITHIN,")
        print("   residuos centrados por familia), no con la mediana de dos rectas de tres puntos.")
        return

    print("\n" + "-" * 96)
    if m >= 0.40:
        print(f"** m = {m:+.4f} >= 0,40  ->  SE REFUERZA **")
        print("   La pendiente global era un artefacto de mezclar presupuestos y perdidas.")
        print("   «Toda mejora de la abstencion pasa por recuperar mejor» queda EN PIE, y ahora sin")
        print("   el confound que invalidaba T-4.")
    elif m <= 0.15:
        print(f"** m = {m:+.4f} <= 0,15  ->  SE DEBILITA **")
        print("   Recuperar mejor compra poco incluso sin confound. La estrategia de la linea tiene")
        print("   que buscar la senal de ausencia en otro lado que la recuperacion.")
    else:
        print(f"** m = {m:+.4f} cae entre 0,15 y 0,40  ->  NO EVALUABLE POR DISENO **")
        print("   La medicion no distingue las dos lecturas. Hace falta mover RECUP a proposito.")
    print("-" * 96)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else
         os.path.join(AQUI, "sonda_techo_curva_20260831.json"))
