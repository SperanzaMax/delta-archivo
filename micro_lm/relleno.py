"""Relleno para la campania de la POLITICA DE ESCRITURA (`DISENO_POLITICA_ESCRITURA.md` §3).

Modulo NUEVO y todavia sin integrar, a proposito: `tramo_abst.sh` re-sube `idioma.py`, `datos.py`,
`modelo.py` y `entrenar.py` en CADA tramo, asi que tocarlos mientras una campania rota entre cuentas
dejaria unidades entrenadas mitad con un generador y mitad con otro (§7 del diseño). Esto se escribe
aparte, se prueba aparte, y se integra cuando la campania de la query conjunta haya cerrado.

El problema que resuelve: hoy cada enunciado del episodio es un hecho distinto y todos son igual de
informativos, asi que **ninguna politica de escritura puede vencer al azar** y la eviction
sorpresa-gated no tiene de donde agarrarse. Para que «que guardar» sea una pregunta hace falta que
haya enunciados que no valga la pena guardar. Tres tipos, de menos a mas exigente:

  · `repeticion` — el mismo hecho, dicho otra vez con las MISMAS palabras. El archivo ya lo tiene.
  · `parafraseo` — el mismo hecho con otra de las formas de `idioma.formas`. Redundante en contenido
    y no en forma; es el caso que separa «la sorpresa mira tokens» de «la sorpresa mira contenido».
  · `charla`     — enunciados bien formados SIN contenido factual, que nunca se preguntan.

`charla` es el unico que hay que construir desde cero, y tiene una condicion de correctitud que se
verifica en `chequeo()` y no se supone: **no puede contener ningun token con el que se afirme un
hecho**. Si se colara un sustantivo de relacion, un verbo de relacion o un numero, la charla dejaria
de ser charla y el experimento mediria otra cosa —de la familia del control vacio `m=1` del 12-ago,
donde el control no podia fallar—.
"""
import numpy as np

import idioma as I

# Armazones de charla: funcionales, con huecos para un nombre o una entidad. Ninguno afirma una
# relacion de `I.RELACIONES` ni menciona un valor.
#
# Solo se usan formas verbales que EXISTEN en `I.FUNCIONALES` (es / era / esta). El primer intento
# tenia «estaba» y «estan», que no estan en el vocabulario: 647 de 2000 enunciados quedaban fuera y
# el chequeo los conto. Un token inventado no habria sido relleno neutro, habria sido un token nuevo
# —y un token nuevo tiene sorpresa alta por ser nuevo, que es justo el modo de falla declarado en el
# §4 del diseño—.
ARMAZONES = [
    "y entonces {n} tambien",
    "pero {n} no ahora",
    "hoy {n} esta en {e}",
    "{n} y {n2} no",
    "antes {n} no esta en {e}",
    "si , {n} tambien",
    "pero entonces no",
    "{n} esta como antes",
    "y {e} tambien",
    "no , {n} no",
    "ayer {n} esta en {e}",
    "{n} esta en {e} ahora",
]


def _tokens_prohibidos():
    """Los tokens con los que se afirma un hecho. La charla no puede tener ninguno."""
    malos = set(I.NUMEROS)              # los numeros son valores
    for _rel, (sust, verbo, _art) in I.RELACIONES.items():
        malos.add(sust)
        malos.add(verbo)
    malos.add(I.VERBO_DUENIO[1])        # la version vieja del verbo de `duenio`
    malos.add(I.VERBO_DUENIO[2])
    return malos


# Colision real y no evidente: `vale` es un NOMBRE de persona y a la vez el verbo de la relacion
# `clave`. Sortear nombres sin filtrar metia un token de hecho en 30 de cada 2000 charlas. Lo mismo
# podria pasarle a cualquier nombre o entidad futura, asi que se filtra por la lista y no por el caso.
NOMBRES_OK = [n for n in I.NOMBRES if n not in _tokens_prohibidos()]
ENTIDADES_OK = [e for e in I.ENTIDADES if e not in _tokens_prohibidos()]


def charla(rng, entidades=None):
    """Un enunciado sin contenido factual, con el vocabulario del idioma."""
    arm = str(rng.choice(ARMAZONES))
    n, n2 = rng.choice(NOMBRES_OK, size=2, replace=False)
    e = str(rng.choice(ENTIDADES_OK if entidades is None else entidades))
    return arm.format(n=n, n2=n2, e=e)


def repeticion(texto):
    """El mismo enunciado, igual. Redundancia exacta."""
    return texto


def parafraseo(rng, rel, ent, val, nivel, distinta_de=None):
    """Otra forma de afirmar el MISMO hecho. Devuelve None si no hay alternativa."""
    opciones = [f for f in I.formas(rel, ent, val, nivel) if f != distinta_de]
    if not opciones:
        return None
    return str(rng.choice(opciones))


def chequeo(n=2000, semilla=0):
    """Verifica que la charla no afirme hechos. Puede fallar, y por eso existe."""
    rng = np.random.default_rng(semilla)
    malos = _tokens_prohibidos()
    fuera_de_vocab, con_hecho = [], []
    for _ in range(n):
        t = charla(rng)
        for tok in t.split():
            if tok not in I.STOI:
                fuera_de_vocab.append((t, tok))
            if tok in malos:
                con_hecho.append((t, tok))
    ok = not fuera_de_vocab and not con_hecho
    print(f"charla · {n} enunciados")
    print(f"  fuera de vocabulario : {len(fuera_de_vocab)}"
          + (f"   ej: {fuera_de_vocab[0]}" if fuera_de_vocab else ""))
    print(f"  con token de hecho   : {len(con_hecho)}"
          + (f"   ej: {con_hecho[0]}" if con_hecho else ""))
    print(f"  largo medio          : "
          f"{np.mean([len(charla(np.random.default_rng(i)).split()) for i in range(200)]):.1f} tokens")
    print(f"\n  {'CHEQUEO PASA' if ok else 'CHEQUEO NO PASA — la charla no es charla'}")
    print("\n  ejemplos:")
    r2 = np.random.default_rng(semilla + 1)
    for _ in range(5):
        print(f"    {charla(r2)}")
    return ok


if __name__ == "__main__":
    chequeo()
