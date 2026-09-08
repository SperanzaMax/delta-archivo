"""Aplica el criterio pre-registrado de H2' · 2026-09-08

`ENMIENDA_E1_SELECTIVIDAD.md`, SHA `61e88855`. H2 tal como quedó en el prereg («la magnitud no decae
con la distancia») se puede cumplir por la razón equivocada: un promedio UNIFORME del pasado también
da TV > 0 en las seis distancias y también es plano. La enmienda operacionaliza:

    sel = TV(d = 3) / TV(d = 5)

d = 3 es donde cae la RELACION en la forma canónica del idioma —la posición cuya ceguera con kernel
3 explicó el corte—; d = 5 no lleva señal en ninguna forma y es el piso.

    CONFIRMA   sel >= 1,5 en al menos dos de las tres semillas
    REFUTA     sel del orden de 1,0 (los pesos de `lat2` dan 1,12) con `nose_rel` igual alto

Este archivo existe para que el veredicto se aplique con el criterio escrito ANTES y no a ojo
despues. No decide nada por su cuenta: lee los TV que ya midió `control_attn.py`.

    python selectividad_attn.py control_attn_*.json
"""
import json
import sys

UMBRAL = 1.5          # el de la enmienda 1, no se toca
REFERENCIA = 1.00     # enmienda 2: cuatro controles con N=600 dan 0,94-1,05. Con N=120 el ruido llega a 1,44


def sel_de(tv):
    """`tv` es {distancia: TV media}. Devuelve la razon de selectividad o None."""
    d3, d5 = tv.get("3"), tv.get("5")
    if not d3 or not d5 or d5 == 0.0:
        return None
    return d3 / d5


def main(rutas):
    filas = []
    for ruta in rutas:
        d = json.load(open(ruta))
        for ck, v in d["res"].items():
            if "attn" not in v:
                continue
            tv = {k: m["tv_media"] for k, m in v["attn"].items()}
            filas.append((ruta, ck, sel_de(tv), tv))

    print(f"{'checkpoint':<42}{'TV d=3':>10}{'TV d=5':>10}{'sel':>8}   veredicto")
    votos = 0
    for ruta, ck, sel, tv in filas:
        nom = ck.split("/")[-1]
        if sel is None:
            print(f"{nom:<42}{'—':>10}{'—':>10}{'—':>8}   sin dato")
            continue
        ok = sel >= UMBRAL
        votos += ok
        print(f"{nom:<42}{tv['3']:10.6f}{tv['5']:10.6f}{sel:8.2f}   "
              f"{'SELECTIVO' if ok else 'plano (ref %.2f)' % REFERENCIA}")

    n = len([f for f in filas if f[2] is not None])
    print(f"\n{votos} de {n} por encima de {UMBRAL}.")
    if n >= 3:
        print("H2' CONFIRMADA" if votos >= 2 else
              "H2' REFUTADA — el acceso global esta presente y NO aprovechado. Si `nose_rel` igual "
              "subio, H1 se cumplio por una causa distinta de la pre-registrada: buscar cual ANTES "
              "de escribir nada.")
    else:
        print("(hacen falta las tres semillas para el veredicto)")


if __name__ == "__main__":
    main(sys.argv[1:] or ["control_attn_at3_p2000.json"])
