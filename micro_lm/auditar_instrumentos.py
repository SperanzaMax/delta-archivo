"""Que instrumento carga un checkpoint sin leer la arquitectura de su config · 2026-09-07

`RETOMAR.md` §3. Un instrumento que carga un checkpoint y NO fija los globals de `conf_ckpt.GLOBALS`
mide con los defaults del modulo, que hoy son los valores VIEJOS (kernel_q 3, sello abs). Sobre un
checkpoint de la campana actual eso da un numero limpio y equivocado, que es la peor clase.

Solo se listan los que ademas importan `modelo`: el que solo mira los pesos como matrices
—`geometria_ord.py`, por ejemplo, que analiza la tabla `ord` y nunca corre el modelo— no tiene el
problema, y meterlo en la lista la vuelve ruido que nadie mira.

    python3 auditar_instrumentos.py           # los que faltan
    python3 auditar_instrumentos.py --todos   # con los que ya estan cubiertos
"""
import argparse
import pathlib
import re

import conf_ckpt

AQUI = pathlib.Path(__file__).resolve().parent
# no son instrumentos de medicion: uno guarda los checkpoints y el otro es esta misma auditoria
EXENTOS = {"entrenar.py", "auditar_instrumentos.py", "conf_ckpt.py"}


def revisar(texto):
    """Devuelve (carga_ckpt, corre_modelo, faltantes)."""
    carga = "pickle.load" in texto
    # el import de la casa es `import datos as DAT, idioma as I, modelo as M`, asi que `modelo` casi
    # nunca es el primer nombre de la linea: hay que buscarlo en cualquier posicion de la lista.
    corre = re.search(r"^\s*(?:import\s+(?:[\w.]+(?:\s+as\s+\w+)?,\s*)*modelo\b"
                      r"|from\s+modelo\s+import)", texto, re.M) is not None
    cubre_todo = re.search(r"^\s*conf_ckpt\.aplicar\(", texto, re.M) is not None
    faltan = []
    if not cubre_todo:
        for nombre in conf_ckpt.GLOBALS:
            if not re.search(r"^\s*(?:M|modelo)\.%s\s*=" % nombre, texto, re.M):
                faltan.append(nombre)
    return carga, corre, faltan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--todos", action="store_true", help="listar tambien los que ya estan bien")
    a = ap.parse_args()

    incompletos = []
    for ruta in sorted(AQUI.glob("*.py")):
        if ruta.name in EXENTOS:
            continue
        carga, corre, faltan = revisar(ruta.read_text(errors="ignore"))
        if not (carga and corre):
            continue
        if faltan:
            incompletos.append((ruta.name, faltan))
        elif a.todos:
            print(f"  ok       {ruta.name}")

    if not incompletos:
        print("todos los instrumentos que corren el modelo leen la arquitectura de su config")
        return
    print(f"{len(incompletos)} instrumento(s) miden con el default del modulo en vez de la config:")
    for nombre, faltan in incompletos:
        print(f"  FALTA    {nombre:<32} {', '.join(faltan)}")
    print("\nArreglo: `import conf_ckpt` y `conf_ckpt.aplicar(cfg)` despues de leer el checkpoint.")


if __name__ == "__main__":
    main()
