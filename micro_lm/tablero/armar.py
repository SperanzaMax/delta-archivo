"""Arma el tablero de la pelicula: mete los JSON adentro de la plantilla. 2026-09-05

La plantilla (`pelicula_plantilla.html`) trae dos marcas y ningun dato. Este script las reemplaza y
escribe el HTML listo para publicar. Separarlos es lo que permite REHACER el tablero con otra
pelicula —la de 26.000 pasos, o la de mañana— sin tocar una linea de diseño.

    python3 armar.py                                   # usa pelicula.json
    python3 armar.py --datos ../pelicula_26000.json    # la larga
    python3 armar.py --salida /tmp/tablero.html

Despues se publica con la herramienta de artifacts, apuntando al archivo de salida. Publicar el
MISMO camino de archivo actualiza el mismo enlace:
    https://claude.ai/code/artifact/a5e05ee7-4ade-429a-8f11-69319209c8ed
"""
import argparse, io, json, os

AQUI = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plantilla", default=os.path.join(AQUI, "pelicula_plantilla.html"))
    ap.add_argument("--datos", default=os.path.join(AQUI, "..", "pelicula.json"))
    ap.add_argument("--referencia", default=os.path.join(AQUI, "..", "referencia_pelicula.json"))
    ap.add_argument("--salida", default=os.path.join(AQUI, "pelicula_tablero.html"))
    a = ap.parse_args()

    s = io.open(a.plantilla, encoding="utf-8").read()
    d = json.load(io.open(a.datos, encoding="utf-8"))
    marca = "/*__DATOS__*/ null"
    assert s.count(marca) == 1, "la plantilla no tiene la marca de datos"
    s = s.replace(marca, json.dumps(d, separators=(",", ":"), ensure_ascii=False))

    mr = "/*__REF__*/ null"
    assert s.count(mr) == 1, "la plantilla no tiene la marca de referencia"
    if os.path.exists(a.referencia):
        r = json.load(io.open(a.referencia, encoding="utf-8"))
        s = s.replace(mr, json.dumps(r, separators=(",", ":"), ensure_ascii=False))
    else:
        s = s.replace(mr, "null")      # sin referencia el tablero se arma igual, sin esos paneles

    io.open(a.salida, "w", encoding="utf-8").write(s)
    print(f"{len(d['cuadros'])} cuadros · hasta el paso {d['cuadros'][-1]['paso']} · "
          f"{os.path.getsize(a.salida) / 1e6:.2f} MB -> {a.salida}")


if __name__ == "__main__":
    main()
