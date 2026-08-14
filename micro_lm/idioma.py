"""MICRO-LM · el idioma cerrado: vocabulario, episodios multi-sesion y render legible.

El corte de diseño (DISENO_MICRO_LM.md): el lenguaje es la INTERFAZ de la prueba, no el objeto de
estudio. Tiene que ser lo bastante chico para entrenar un modelo de pocos megas desde cero, y lo
bastante legible para que un humano lea la pregunta y la respuesta y sepa si acerto.

Un episodio es una conversacion partida en sesiones, con el estado del modelo reseteado entre ellas:

    sesion 1   USUARIO  el director de norte es ana
    sesion 2   USUARIO  no , el director de norte es beto
    sesion 5   USUARIO  quien dirige norte ?
               MODELO   beto

Cuatro niveles de dificultad, y el salto real esta en N2:
  N1  plantilla fija            una sola forma de decir cada hecho
  N2  parafrasis                cuatro formas del mismo hecho -> hay que indexar por CONTENIDO
  N3  correccion eliptica       "no , es beto", sin nombrar la entidad
  N4  multi-sesion              varias sesiones y preguntas al final

La respuesta es SIEMPRE un solo token (un valor, o NOSE). Eso hace la metrica exacta sin juez ni
parser: la leccion del 12-ago, cuando 10 de 11 "abstenciones" resultaron ser el modelo contestando
bien otra cosa y el parser anotandolo como rechazo.
"""
import numpy as np

# --- vocabulario -----------------------------------------------------------------------------

NOMBRES = """ana beto carla dario elsa fabio gema hugo irma julio kena luis mara nico olga pablo
quena raul sofia tomas ursula vera walter ximena yago zoe abril bruno celia diego emma franco gala
ivan julia kevin lara marco nadia oscar paula ramiro sara teo vale wanda yamil zaira agus bruna
ciro dana enzo flor gonza hilda ian jazmin""".split()

ENTIDADES = """norte sur este oeste centro taller tienda equipo club museo teatro puerto granja
banco hotel bosque puente barrio plaza mercado escuela fabrica estudio parque templo faro molino
laguna cabaña""".split()

RELACIONES = {                      # relacion -> (sustantivo, verbo, articulo)
    "director": ("director", "dirige", "el"),
    "precio":   ("precio", "cuesta", "el"),
    "altura":   ("altura", "mide", "la"),
    "duenio":   ("dueño", "pertenece_a", "el"),
    "guardia":  ("guardia", "cuida", "el"),
    "clave":    ("clave", "vale", "la"),
}
PERSONALES = ("director", "duenio", "guardia")   # su valor es un nombre; el resto, un numero

FUNCIONALES = """el la de del es era esta en un una y o no si ahora antes ayer hoy cual quien que a
tiene como con por para se lo su mas menos tambien pero entonces""".split()

CONTROL = ["BOS", "SEP", "EOS", "USUARIO", "MODELO", "NOSE", "?", ",", "."]

NUMEROS = [str(i) for i in range(100)]          # los "100 numeros" que pidio Maxi


def construir_vocab():
    """Devuelve (itos, stoi). Orden estable: control, numeros, nombres, entidades, palabras."""
    palabras = []
    for _, (sust, verbo, _art) in RELACIONES.items():
        palabras += [sust, verbo]
    itos = CONTROL + NUMEROS + NOMBRES + ENTIDADES + sorted(set(palabras)) + FUNCIONALES
    vistos, limpio = set(), []
    for t in itos:
        if t not in vistos:
            vistos.add(t)
            limpio.append(t)
    return limpio, {t: i for i, t in enumerate(limpio)}


ITOS, STOI = construir_vocab()
V = len(ITOS)


# --- como se dice un hecho -------------------------------------------------------------------

def formas(rel, ent, val, nivel):
    """Las maneras de afirmar «la <rel> de <ent> es <val>». En N1 hay una sola."""
    sust, verbo, art = RELACIONES[rel]
    f0 = f"{art} {sust} de {ent} es {val}"
    if nivel < 2:
        return [f0]
    if rel in PERSONALES:
        return [f0, f"{val} {verbo} {ent}", f"quien {verbo} {ent} es {val}",
                f"{ent} tiene como {sust} a {val}"]
    return [f0, f"{ent} {verbo} {val}", f"{art} {sust} que tiene {ent} es {val}",
            f"{ent} tiene {art} {sust} en {val}"]


def correccion(rel, ent, val, nivel):
    """Como se corrige un hecho ya dicho. En N3 la correccion NO nombra la entidad."""
    sust, _, art = RELACIONES[rel]
    if nivel < 3:
        return f"no , {art} {sust} de {ent} es {val}"
    return np.random.choice([f"no , es {val}", f"no , ahora es {val}", f"no , {val}"])


def pregunta(rel, ent, cual="vigente"):
    sust, _, art = RELACIONES[rel]
    if cual == "vigente":
        return f"cual es {art} {sust} de {ent} ?"
    return f"cual era antes {art} {sust} de {ent} ?"


# --- episodios -------------------------------------------------------------------------------

def episodio(rng, nivel=4, n_hechos=4, n_sesiones=5, p_revision=0.5, p_pregunta_vieja=0.35,
             p_nose=0.0):
    """Un episodio completo. Devuelve (sesiones, consultas) en TEXTO, ya legible.

    `sesiones` es una lista de listas de enunciados (una lista por sesion).
    `consultas` es una lista de (pregunta, respuesta_esperada, tipo).

    `p_nose` agrega preguntas SIN respuesta en el archivo, cuya respuesta correcta es el token NOSE.
    El dia 1 la abstencion dio 0,0000 en los cuatro niveles: `NOSE` existia en el vocabulario pero
    ninguna pregunta lo tenia como respuesta, asi que no habia nada que aprender y **todos los
    errores eran silenciosos** — justo el modo de falla que mide el preprint de gemacion.
    Dos tipos, a proposito, porque miden cosas distintas:
      `nose_ent`  la entidad no aparece en el episodio  -> basta con no encontrarla en el archivo;
      `nose_rel`  la entidad SI aparece, pero con otra relacion -> hay que encontrar la entidad y
                  ademas notar que lo que se pregunta de ella no se dijo. Es el caso dificil, y el
                  que se parece a una alucinacion real.
    """
    ents = rng.choice(ENTIDADES, size=n_hechos, replace=False)
    rels = rng.choice(list(RELACIONES), size=n_hechos)
    vals_por_hecho = []

    sesiones = [[] for _ in range(n_sesiones)]
    for i, (ent, rel) in enumerate(zip(ents, rels)):
        pool = NOMBRES if rel in PERSONALES else NUMEROS
        v1 = rng.choice(pool)
        s = 0 if nivel < 4 else int(rng.integers(0, max(1, n_sesiones - 1)))
        sesiones[s].append(rng.choice(formas(rel, ent, v1, nivel)))
        versiones = [v1]

        if rng.random() < p_revision:
            v2 = rng.choice([x for x in pool if x != v1])
            # REGLA DE RESOLUBILIDAD: una correccion ELIPTICA solo puede ir pegada a su hecho, en la
            # misma sesion. Si va en otra sesion, tiene que nombrar la entidad -- si no, no hay
            # ninguna regla objetiva que diga a que hecho se refiere, y la etiqueta seria una
            # convencion nuestra en vez de una señal recuperable (lo medido el 12-ago: sin pista
            # recuperable, el eje mide arbitrariedad y no dificultad).
            lejos = nivel >= 4 and s + 1 < n_sesiones and rng.random() < 0.5
            if lejos:
                s2 = int(rng.integers(s + 1, n_sesiones))
                sesiones[s2].append(correccion(rel, ent, v2, nivel=2))   # nombra la entidad
            else:
                sesiones[s].append(correccion(rel, ent, v2, nivel))      # adyacente: puede elidir
            versiones.append(v2)
        vals_por_hecho.append((rel, ent, versiones))

    consultas = []
    for rel, ent, versiones in vals_por_hecho:
        if len(versiones) > 1 and rng.random() < p_pregunta_vieja:
            consultas.append((pregunta(rel, ent, "anterior"), versiones[-2], "anterior"))
        else:
            consultas.append((pregunta(rel, ent, "vigente"), versiones[-1], "vigente"))

    if p_nose > 0 and rng.random() < p_nose:
        dichos = {(r, e) for r, e, _ in vals_por_hecho}
        ents_dichas = {e for _, e, _ in vals_por_hecho}
        if rng.random() < 0.5:                       # nose_ent: la entidad nunca se nombro
            libres = [e for e in ENTIDADES if e not in ents_dichas]
            ent_q = str(rng.choice(libres))
            rel_q = str(rng.choice(list(RELACIONES)))
            tipo = "nose_ent"
        else:                                        # nose_rel: la entidad si, la relacion no
            ent_q = str(rng.choice(sorted(ents_dichas)))
            libres = [r for r in RELACIONES if (r, ent_q) not in dichos]
            if not libres:
                return sesiones, consultas
            rel_q = str(rng.choice(libres))
            tipo = "nose_rel"
        consultas.append((pregunta(rel_q, ent_q, "vigente"), "NOSE", tipo))
    return sesiones, consultas


def render(sesiones, consultas):
    """El episodio como lo leeria un humano."""
    out = []
    for i, ses in enumerate(sesiones, 1):
        if not ses:
            continue
        out.append(f"--- sesion {i} ---")
        out += [f"USUARIO  {e}" for e in ses]
    out.append("--- consulta ---")
    for q, r, t in consultas:
        out.append(f"USUARIO  {q}")
        out.append(f"MODELO   {r}        [{t}]")
    return "\n".join(out)


def a_ids(texto):
    """Tokeniza por espacios contra el vocabulario cerrado. Falla ruidosamente si algo no esta."""
    ids = []
    for t in texto.split():
        if t not in STOI:
            raise KeyError(f"token fuera del vocabulario cerrado: {t!r}")
        ids.append(STOI[t])
    return ids


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    print(f"vocabulario: {V} tokens "
          f"({len(NUMEROS)} numeros · {len(NOMBRES)} nombres · {len(ENTIDADES)} entidades · "
          f"{len(RELACIONES)} relaciones)\n")
    for nivel in (1, 2, 3, 4):
        print("=" * 70)
        print(f"NIVEL {nivel}")
        print("=" * 70)
        ses, con = episodio(rng, nivel=nivel, n_hechos=3, n_sesiones=4)
        print(render(ses, con))
        # y que todo entre en el vocabulario cerrado
        for s in ses:
            for e in s:
                a_ids(e)
        for q, r, _ in con:
            a_ids(q); a_ids(r)
        print()
