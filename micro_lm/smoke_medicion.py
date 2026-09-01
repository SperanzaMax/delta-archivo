"""Script TRIVIAL para probar `medir_en_colab.sh` · 1-sep

El propio `medir_en_colab.sh` lo pide en su cabecera: «NO PROBADO todavia [...] antes de confiarle una
medicion larga, correrlo una vez con un script trivial y verificar que el .json vuelve».

Prueba las cuatro cosas que pueden fallar y que la curva necesita:
  1. que el codigo subido IMPORTE (idioma, datos, modelo, entrenar, medir_ratio_ce, sonda_techo),
  2. que los checkpoints hayan llegado a `ckpts/` con su nombre,
  3. que se puedan CARGAR y que jax corra en el acelerador,
  4. que un .json escrito en el cwd VUELVA a la PC.

Si esto vuelve, la curva puede salir sin gastar una asignacion en descubrir que faltaba un import.
"""
import glob
import json
import os
import sys

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)

import jax                        # noqa: E402
import numpy as np                # noqa: E402

import idioma as I                # noqa: E402
import medir_ratio_ce as R        # noqa: E402
import sonda_techo as ST          # noqa: E402

print("jax", jax.__version__, "· dispositivos:", jax.devices())
print("cwd", os.getcwd())

rutas = sorted(glob.glob(os.path.join(AQUI, "ckpts", "*.pkl")))
print(f"checkpoints que llegaron: {len(rutas)}")
filas = []
for ruta in rutas:
    nom = os.path.basename(ruta)[:-4]
    try:
        params, cfg, paso = R.cargar(ruta)
        I.fijar_version(cfg.get("idioma", 2))
        ST.IDS_NOM = np.array([I.STOI[t] for t in I.NOMBRES])
        d = int(params["arch"]["qr"].shape[-1])
        filas.append({"unidad": nom, "paso": int(paso), "nivel": cfg.get("nivel"), "d": d})
        print(f"  OK  {nom:10s} paso={paso:6d} nivel={cfg.get('nivel')} d={d}")
    except Exception as e:
        print(f"  ** {nom}: {type(e).__name__}: {e}")
        filas.append({"unidad": nom, "error": f"{type(e).__name__}: {e}"})

# el punto de la prueba: que ESTE archivo vuelva a la PC
with open(os.path.join(AQUI, "smoke_medicion.json"), "w") as f:
    json.dump({"dispositivos": [str(x) for x in jax.devices()], "filas": filas}, f, indent=1)
print("-> smoke_medicion.json escrito")
