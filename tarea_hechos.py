"""Tarea de MEMORIA VERSIONADA DE HECHOS, comparable con VersionRAG (58 % reportado).

Cada item es una entidad con un atributo cuyo valor SE REVISA:

  v1  "The director of the Helios Laboratory is Ana Ruiz."
  v2  "The director of the Helios Laboratory is Beto Lima."     <- revision
  q   "Who directs the Helios Laboratory?"                      <- consulta

  VIGENTE  = Beto Lima   (lo que un RAG con sobrescritura deberia dar)
  ANTERIOR = Ana Ruiz    (lo que la sobrescritura destruye por construccion)

Por que en ingles: el corpus de R12 era wikitext y el modelo es Gemma; mantener el idioma evita
mezclar un factor nuevo en la geometria del espacio de embeddings.

Diferencia crucial con la tarea del harness (R13): aca la clave es el embedding del TEXTO
COMPLETO. Es estable para el mismo contenido —dos veces el mismo hecho da el mismo vector— pero
NO es funcion de un token: es genuinamente contextual. Ese era exactamente el sustrato que
faltaba y que hizo trivial a R13.

Las tres condiciones de archivo se distinguen por DONDE se guarda la revision:
  sobrescritura   direccion = emb(v2), REEMPLAZA la entrada de v1   -> pierde la historia
  duplicados      direccion = emb(v2), entrada nueva                -> guarda, sin estructura
  gemacion        direccion = emb(v1) + eps*t_hat, entrada nueva    -> guarda anclado al anterior

La condicion `duplicados` es la que aisla el aporte: sin ella no se puede saber si el merito es
de guardar la historia o de la geometria con que se la guarda.
"""
import numpy as np

ATRIBUTOS = [
    ("director", "The director of {e} is {v}.", "Who directs {e}?"),
    ("headquarters", "The headquarters of {e} is located in {v}.", "Where is {e} based?"),
    ("founder", "{e} was founded by {v}.", "Who founded {e}?"),
    ("chief engineer", "The chief engineer at {e} is {v}.", "Who is the chief engineer at {e}?"),
    ("main supplier", "The main supplier for {e} is {v}.", "Which company supplies {e}?"),
]

PREFIJOS = ["Helios", "Vantor", "Kestrel", "Orinoco", "Bramble", "Quillon", "Sarto",
            "Nimbus", "Falkirk", "Vireo", "Castell", "Dunmore", "Ashgrove", "Perrin",
            "Thalos", "Weyland", "Corven", "Malbec", "Ridgeway", "Solano"]
TIPOS = ["Laboratory", "Institute", "Foundry", "Consortium", "Archive", "Works",
         "Observatory", "Collective", "Shipyard", "Registry"]

NOMBRES = ["Ana Ruiz", "Beto Lima", "Carla Nunez", "Dario Pena", "Elsa Moray",
           "Franco Vidal", "Greta Halden", "Hugo Sartre", "Ines Coppola", "Jonas Ferrer",
           "Kira Osei", "Lucas Brand", "Mira Solano", "Nadir Haq", "Olga Petrova",
           "Pablo Iriarte", "Quinn Adeyemi", "Rosa Belmonte", "Simon Okafor", "Tania Vega"]
CIUDADES = ["Rotterdam", "Valparaiso", "Sapporo", "Aarhus", "Ljubljana", "Fortaleza",
            "Tromso", "Kaohsiung", "Windhoek", "Bergen", "Cusco", "Galway",
            "Antalya", "Mombasa", "Dunedin", "Tallinn", "Kanazawa", "Salta",
            "Reykjavik", "Nantes"]
EMPRESAS = ["Norvik Steel", "Ardent Optics", "Caldera Systems", "Petrel Marine",
            "Vinland Ceramics", "Ostara Alloys", "Kelvin Instruments", "Marlowe Plastics",
            "Sable Composites", "Tindall Glass", "Ferro Nordic", "Brisken Tooling",
            "Halvard Cables", "Ionian Motors", "Juniper Rail", "Kessel Foundry",
            "Larkspur Fibers", "Mistral Pumps", "Nadel Bearings", "Ovid Refractories"]

POOL = {"director": NOMBRES, "chief engineer": NOMBRES, "founder": NOMBRES,
        "headquarters": CIUDADES, "main supplier": EMPRESAS}


def gen_hechos(rng, N):
    """Devuelve lista de dicts con v1, v2, consulta y las dos respuestas."""
    combos = [(p, t) for p in PREFIJOS for t in TIPOS]
    rng.shuffle(combos)
    items = []
    for i in range(N):
        p, t = combos[i % len(combos)]
        suf = "" if i < len(combos) else f" {i // len(combos) + 1}"
        ent = f"the {p} {t}{suf}"
        attr, plantilla, preg = ATRIBUTOS[rng.integers(len(ATRIBUTOS))]
        pool = POOL[attr]
        a, b = rng.choice(len(pool), 2, replace=False)          # v1 != v2 garantizado
        items.append(dict(
            entidad=ent, atributo=attr,
            v1=plantilla.format(e=ent, v=pool[a]),
            v2=plantilla.format(e=ent, v=pool[b]),
            consulta=preg.format(e=ent),
            resp_vigente=pool[b], resp_anterior=pool[a]))
    return items


def sanidad(N=200, seed=0):
    rng = np.random.default_rng(seed)
    it = gen_hechos(rng, N)
    ok = True

    # 1. la consulta NO contiene ninguna de las dos respuestas
    fuga = sum(1 for x in it
               if x["resp_vigente"] in x["consulta"] or x["resp_anterior"] in x["consulta"])
    print(f"consultas que filtran la respuesta: {fuga}  {'OK' if fuga == 0 else 'FALLA'}")
    ok &= fuga == 0

    # 2. vigente != anterior
    d = sum(1 for x in it if x["resp_vigente"] == x["resp_anterior"])
    print(f"items con vigente == anterior: {d}  {'OK' if d == 0 else 'FALLA'}")
    ok &= d == 0

    # 3. entidades unicas: si se repiten, dos items compiten por el mismo hecho
    ents = [x["entidad"] for x in it]
    print(f"entidades unicas: {len(set(ents))}/{N}  "
          f"{'OK' if len(set(ents)) == N else 'FALLA'}")
    ok &= len(set(ents)) == N

    # 4. v1 y v2 difieren SOLO en el valor (mismo molde -> se agrupan en el espacio)
    import difflib
    r = np.mean([difflib.SequenceMatcher(None, x["v1"], x["v2"]).ratio() for x in it])
    print(f"similitud textual media v1 vs v2: {r:.3f}  (alta = mismo molde, distinto valor)")

    print("\nejemplo:")
    x = it[0]
    for k in ("v1", "v2", "consulta", "resp_vigente", "resp_anterior"):
        print(f"  {k:14} {x[k]}")
    print("\nSANIDAD:", "OK" if ok else "FALLA")
    return ok


if __name__ == "__main__":
    sanidad()
