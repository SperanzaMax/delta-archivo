"""Convierte un checkpoint terminado en una BASE DE SIEMBRA para una corrida nueva.

Por que existe (2026-08-30). Continuar `b3_s3` con otra funcion de perdida no es continuar: es
bifurcar, y las guardas de `entrenar.py` abortan —con razon—. Pero `b3_s3` es exactamente el material
que interesa: es una unidad declarada ATRACTOR ABSORBENTE el 29-ago cuya confianza `c` esta por encima
del umbral de la recompensa en el 54 % de las preguntas (`medir_confianza.py`). Usarla como PUNTO DE
PARTIDA es legitimo y es lo que ya se hacia con `n3_sX`; lo que no seria legitimo es hacerlo en
silencio, dejando que el JSON muestre una sola curva donde hubo dos regimenes.

Entonces este script hace explicito lo que `tramo_abst.sh` hacia con un `cp`:

  - conserva los PESOS y nada mas,
  - borra `opt_state`, `historia` y `rng` (Adam se reinicia; el estado del optimizador de la corrida
    vieja no le corresponde a la perdida nueva),
  - pone el paso en 0,
  - borra de la config SOLO las claves que definen la corrida de la que se bifurca —las que la guarda
    de identidad compara— y deja intactas las que describen la ARQUITECTURA y la TAREA, que no cambian,
  - y escribe `sembrado_de` con la ruta y el paso de origen, para que la procedencia quede en el
    checkpoint y no solo en un informe.

Uso:
    python3 sembrar.py ckpts/b3_s3.pkl ckpts/rc3_s3.pkl
    python3 sembrar.py ckpts/b3_s3.pkl ckpts/rt3_s3.pkl --sin-cabeza
"""

import argparse
import pickle

# Claves que describen la corrida de la que nos bifurcamos, no el modelo. Se borran para que la
# guarda de identidad de `entrenar.py` no compare la corrida nueva contra la vieja.
# `ses_extra` (2026-09-05): cambiar el tamanio del archivo ES una bifurcacion —es otra tarea, y la
# guarda de `entrenar.py` aborta si no coincide—, asi que va aca. Sembrar desde un checkpoint de
# archivo corto hacia uno largo es justamente lo que la campania del archivo largo quiere poder hacer,
# y este es el unico lugar donde queda DECLARADO en el checkpoint (`sembrado_de`) en vez de en silencio.
BIFURCA = ("perdida_cabeza", "blanco", "horizonte", "pasos",
           "rec_l", "rec_m", "rec_f", "rec_ce", "ses_extra")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("origen")
    ap.add_argument("destino")
    ap.add_argument("--sin-cabeza", action="store_true",
                    help="descarta params['abst']. Para pasar a --abst token: la cabeza colapsada al "
                         "prior deja de existir en vez de quedar de pasajera.")
    ap.add_argument("--horizonte", type=int, required=True,
                    help="horizonte de lr de la corrida NUEVA. Es obligatorio y no tiene default a "
                         "proposito: la guarda de `entrenar.py` compara el horizonte del checkpoint "
                         "contra el pedido, y dejarlo vacio la haria abortar en el primer tramo. "
                         "Declararlo aca obliga a decidir el presupuesto ANTES de sembrar, que es lo "
                         "que la leccion D-1 del 22-ago pedia.")
    a = ap.parse_args()

    with open(a.origen, "rb") as f:
        b = pickle.load(f)

    cfg = dict(b["config"])
    borradas = {k: cfg.pop(k) for k in BIFURCA if k in cfg}
    params = dict(b["params"])
    if a.sin_cabeza and "abst" in params:
        params.pop("abst")
        cfg.pop("abst", None)

    cfg["horizonte"] = a.horizonte
    cfg["pasos"] = a.horizonte
    nuevo = {"params": params, "config": cfg, "paso": 0,
             "sembrado_de": {"ruta": a.origen, "paso": b.get("paso"),
                             "claves_borradas": borradas}}
    with open(a.destino, "wb") as f:
        pickle.dump(nuevo, f)

    print(f"sembrado  {a.origen} (paso {b.get('paso')})  ->  {a.destino} (paso 0)")
    print(f"  claves de la corrida vieja borradas: {borradas}")
    print(f"  params: {sorted(params.keys())}")
    print(f"  Adam, historia y rng se reinician: no le corresponden a la perdida nueva.")


if __name__ == "__main__":
    main()
