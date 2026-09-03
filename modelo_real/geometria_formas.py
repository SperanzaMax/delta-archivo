"""¿POR QUE la tarea satura en el modelo real? · la distancia que importa NO es la que se midio · 3-sep

El 2-sep la condicion `una` —la que TENIA que fallar, porque entrena solo con la forma donde la
relacion queda afuera— llego a `vigente` 1,0000 y `nose_rel` 0,95-1,00 **en 100 pasos**. Se leyo como
efecto techo por la tarea ser facil, y la salida propuesta era subir de 4 a 16 hechos, que multiplica
el costo por cuatro sin tocar la causa.

Hay una explicacion alternativa que hay que descartar ANTES de gastar GPU, y es aritmetica.

El prereg midio la distancia de cada pieza a la POSICION DE LECTURA (el `?`). Esa es la distancia
correcta en el micro-LM, donde la query se forma en el ultimo token y va a un softmax sobre un
archivo EXTERNO: lo que no entra ahi es invisible para esa lectura, y punto.

En Mamba no hay archivo externo. La memoria es el estado `h`, que se actualiza en CADA posicion, y
cada token tiene su propio turno de condicionar la busqueda cuando entra. Para responder hay que
COMBINAR entidad y relacion, y esa combinacion puede ocurrir en cualquier posicion donde las dos
esten disponibles a la vez, no solo en la ultima.

    -> la distancia que decide es la que separa la RELACION de la ENTIDAD, no la que las separa del `?`.

Este guion cuenta las tres distancias en tokens reales del BPE, sin suponer nada.
"""
import json
import os
import sys

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)

from transformers import AutoTokenizer

V = json.load(open(os.path.join(AQUI, "vocabulario.json")))
tok = AutoTokenizer.from_pretrained("state-spaces/mamba-130m-hf")

ALCANCE = 2          # MEDIDO el 2-sep: kernel nominal 4, tap mas viejo cero exacto en 24/24 capas

PLANTILLAS = dict(V["plantillas"])
# Candidatas nuevas: separan la relacion de la entidad MAS ALLA del alcance, sin alargar el contexto.
PLANTILLAS["separada"] = "What is the {r} of the person named {e}?"
PLANTILLAS["separada2"] = "What is the {r}, in the records, of {e}?"

e, r = V["entidades"][0], V["relaciones"][0]
ids_e, ids_r = tok(" " + e).input_ids, tok(" " + r).input_ids
assert len(ids_e) == 1 and len(ids_r) == 1, "las piezas tienen que ser de un token"

print(f"alcance MEDIDO de la conv de mamba: {ALCANCE} tokens hacia atras\n")
print(f"{'forma':<11} {'d(rel->fin)':>11} {'d(ent->fin)':>11} {'d(rel<->ent)':>13}   "
      f"{'rel ve fin?':>11} {'se COMBINAN?':>13}")
print("-" * 80)

filas = {}
for nombre, plantilla in PLANTILLAS.items():
    texto = plantilla.format(r=r, e=e)
    ids = tok(texto).input_ids
    piezas = [tok.decode([i]) for i in ids]
    fin = len(ids) - 1                      # el `?`, que es la posicion de lectura
    p_r = [i for i, t in enumerate(ids) if t == ids_r[0]]
    p_e = [i for i, t in enumerate(ids) if t == ids_e[0]]
    assert len(p_r) == 1 and len(p_e) == 1, f"{nombre}: pieza repetida"
    p_r, p_e = p_r[0], p_e[0]
    d_r, d_e = fin - p_r, fin - p_e
    d_re = abs(p_e - p_r)
    # la conv de la posicion POSTERIOR alcanza a la anterior si estan a <= ALCANCE
    combinan = d_re <= ALCANCE
    filas[nombre] = dict(d_rel=d_r, d_ent=d_e, d_re=d_re, combinan=combinan,
                         tokens=piezas, p_r=p_r, p_e=p_e, largo=len(ids))
    print(f"{nombre:<11} {d_r:>11} {d_e:>11} {d_re:>13}   "
          f"{str(d_r <= ALCANCE):>11} {('SI' if combinan else 'NO'):>13}")

print()
for nombre, f in filas.items():
    marca = "".join("R" if i == f["p_r"] else ("E" if i == f["p_e"] else "·")
                    for i in range(f["largo"]))
    print(f"  {nombre:<11} {marca}   {' '.join(repr(t) for t in f['tokens'])}")

print(f"""
LECTURA
  Si `directa` tiene d(rel<->ent) = {filas['directa']['d_re']} <= {ALCANCE}, entonces la conv de la
  posicion de la ENTIDAD SI ve la relacion, y el modelo puede formar la consulta conjunta ahi mismo
  aunque en el `?` la relacion ya haya quedado afuera. Eso explicaria la saturacion SIN que la tarea
  sea facil, y el remedio no seria mas hechos sino una forma donde las dos piezas no compartan
  ninguna ventana.
""")
json.dump(filas, open(os.path.join(AQUI, "geometria_formas.json"), "w"), indent=1)
