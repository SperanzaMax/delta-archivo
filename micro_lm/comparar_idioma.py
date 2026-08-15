"""MICRO-LM · el control del arreglo de la parafrasis: idioma v1 contra v2, A IGUAL PASO.

    python comparar_idioma.py corridas_20260814_v1_parciales corridas_20260814

v1 decia «julia pertenece_a teatro» (el verbo toma como sujeto la entidad, no la persona) y v2 dice
«julia posee teatro». El arreglo es para el lector humano de `dialogos.py`.

PREDICCION, escrita antes de tener los datos (2026-08-14, ver `idioma.fijar_version`): es el
reemplazo de UN token por otro en la misma posicion de la misma plantilla — mismas longitudes, misma
estructura, y el modelo no sabe castellano. No deberia mover la accuracy mas alla del ruido entre
semillas. Si la moviera mucho, lo que esta mal es el entendimiento de la tarea, no el arreglo.

Se compara a IGUAL PASO porque las corridas de v1 quedaron truncadas por la caida de Colab del
14-ago a la mañana: tienen historia hasta 4000-8000 pasos y las de v2 llegan mas lejos. Comparar
finales contra finales mediria el presupuesto de entrenamiento, no el idioma.
"""
import argparse
import glob
import json
import os
import re

import numpy as np


def cargar(carpeta):
    """{(nivel, semilla): {paso: {metrica: valor}}}"""
    out = {}
    for f in sorted(glob.glob(os.path.join(carpeta, "n*_s*.json"))):
        m = re.match(r"n(\d+)_s(\d+)\.json", os.path.basename(f))
        if not m:
            continue
        d = json.load(open(f))
        if d.get("historia"):
            out[(int(m[1]), int(m[2]))] = {h["paso"]: h for h in d["historia"]}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("carpeta_v1")
    ap.add_argument("carpeta_v2")
    a = ap.parse_args()
    v1, v2 = cargar(a.carpeta_v1), cargar(a.carpeta_v2)

    comunes = sorted(set(v1) & set(v2))
    if not comunes:
        raise SystemExit("no hay ninguna celda (nivel, semilla) en las dos versiones todavia")

    print(f"celdas comparables: {len(comunes)}  {comunes}\n")
    print(f"{'nivel':>5} {'sem':>4} {'paso':>6} {'v1':>8} {'v2':>8} {'v2-v1':>8}   metrica")
    difs = {"vigente": [], "anterior": []}
    for met in ("vigente", "anterior"):
        for (n, s) in comunes:
            pasos = sorted(set(v1[(n, s)]) & set(v2[(n, s)]))
            if not pasos:
                continue
            p = max(pasos)                      # el paso mas avanzado que tienen EN COMUN
            x, y = v1[(n, s)][p][met], v2[(n, s)][p][met]
            difs[met].append(y - x)
            print(f"{n:>5} {s:>4} {p:>6} {x:>8.4f} {y:>8.4f} {y-x:>+8.4f}   {met}")
        print()

    print("── resumen")
    for met, d in difs.items():
        if not d:
            continue
        d = np.array(d)
        # La MAGNITUD puede caer dentro del ruido y aun asi el SIGNO delatar un efecto sistematico:
        # si el idioma no hiciera nada, las diferencias deberian repartirse a los dos lados del cero.
        pos = int((d > 0).sum())
        alerta = ""
        if len(d) >= 4 and pos in (0, len(d)):
            alerta = (f"  ⚠ las {len(d)} celdas van para el MISMO lado "
                      f"(p≈{2 * 0.5 ** len(d):.3f} si fuera simetrico): no alcanza con mirar |dif|")
        print(f"{met:>9}: diferencia media {d.mean():+.4f} · |dif| maxima {np.abs(d).max():.4f} "
              f"· n={len(d)} · positivas {pos}/{len(d)}{alerta}")
    todas = np.concatenate([np.array(d) for d in difs.values() if len(d)])
    peor = np.abs(todas).max()
    # El umbral no es un test estadistico: es el orden de magnitud del ruido entre semillas que ya
    # se venia viendo en `analizar.py` (rangos de 0,003 a 0,08 en celdas del mismo idioma).
    print(f"\nmayor diferencia absoluta: {peor:.4f} → "
          f"{'compatible con el ruido entre semillas: la prediccion se sostiene' if peor <= 0.10 else 'MAYOR que el ruido tipico entre semillas: revisar, la prediccion NO se sostiene'}")


if __name__ == "__main__":
    main()
