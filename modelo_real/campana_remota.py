"""Recorre una lista de unidades en la VM de Colab, una atras de otra, en un solo proceso.

Vive como archivo propio a proposito. El 3-sep intente generarlo desde un heredoc de bash y me costo
dos arranques de sesion: primero la indentacion de `textwrap.dedent` y despues `\\n` que el heredoc
sin comillas convertia en un salto de linea real dentro de un string. Un archivo que se sube tal cual
no tiene ninguno de los dos problemas.

Todo entra por variables de entorno, asi que no hay comillas anidadas en ningun lado.

    TRABAJOS   "cerca:0 lejos:0"   condicion:semilla, separados por espacio
    PASOS BATCH LARGO NH CADA MODELO
"""
import os
import subprocess
import sys
import time

AQUI = "/content/real"
SAL = "/content"


def main():
    trabajos = os.environ["TRABAJOS"].split()
    comun = ["--pasos", os.environ.get("PASOS", "400"),
             "--batch", os.environ.get("BATCH", "8"), "--acum", "1",
             "--largo", os.environ.get("LARGO", "64"),
             "--n-hechos", os.environ.get("NH", "4"),
             "--modelo", os.environ.get("MODELO", "state-spaces/mamba-130m-hf"),
             "--cada", os.environ.get("CADA", "100"),
             "--n-eval", os.environ.get("N_EVAL", "32")]
    print("trabajos: %s" % " ".join(trabajos), flush=True)
    for t in trabajos:
        cond, sem = t.split(":")
        uni = "%s_%s_s%s" % (os.environ.get("ETIQ", "dist"), cond, sem)
        salida = os.path.join(SAL, uni + ".json")
        if os.path.exists(salida):
            print("== %s ya esta, se saltea" % uni, flush=True)
            continue
        print("\n===== %s  (%s)" % (uni, time.strftime("%H:%M:%S")), flush=True)
        t0 = time.time()
        r = subprocess.run([sys.executable, "-u", "entrenar_real.py",
                            "--condicion", cond, "--semilla", sem,
                            "--salida", salida] + comun, cwd=AQUI)
        print("== %s codigo %d en %.0f s" % (uni, r.returncode, time.time() - t0), flush=True)
    print("CAMPANIA COMPLETA", flush=True)


if __name__ == "__main__":
    main()
