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

# El verbo de cada relacion PERSONAL tiene que tomar a la persona como sujeto, porque las
# parafrasis lo usan asi: `{val} {verbo} {ent}` y `quien {verbo} {ent} es {val}`.
# `duenio` llevaba «pertenece_a», cuyo sujeto es la ENTIDAD y no la persona, y generaba
# «julia pertenece_a teatro» — al reves: el teatro pertenece a julia, no julia al teatro.
# Con «posee» las cuatro formas quedan derechas y no hace falta un caso especial.
# Se corrige el 2026-08-14; la campania de esa mañana ya habia salido con V1 (ver `fijar_version`).
VERBO_DUENIO = {1: "pertenece_a", 2: "posee"}

RELACIONES = {                      # relacion -> (sustantivo, verbo, articulo)
    "director": ("director", "dirige", "el"),
    "precio":   ("precio", "cuesta", "el"),
    "altura":   ("altura", "mide", "la"),
    "duenio":   ("dueño", VERBO_DUENIO[2], "el"),
    "guardia":  ("guardia", "cuida", "el"),
    "clave":    ("clave", "vale", "la"),
}
PERSONALES = ("director", "duenio", "guardia")   # su valor es un nombre; el resto, un numero

# --- version 3: el mismo idioma con MAS RELACIONES (2026-08-23) ---------------------------------
# Por que existe. Un episodio sortea 4 hechos con entidades distintas (`replace=False`) pero con
# relaciones CON reemplazo, asi que con 6 relaciones el **72,4 %** de los episodios tiene dos hechos
# que comparten relacion (medido sobre 200000 episodios; teorico 0,722). Y el 20-ago se midio que
# eso es justo donde el modelo falla: con relacion unica el error es 0,005-0,014, con relacion
# repetida 0,38-0,54 —o sea azar entre las dos entradas que empatan—. La causa mecanica se cerro el
# 21-ago: la query se forma sobre `emb[x]`, es funcion PURA del token, y por eso el modelo no puede
# consultar por entidad Y relacion a la vez.
#
# Consecuencia para la MEDICION, que es lo que esto viene a arreglar: con 6 relaciones, lo que se
# lee como «el modelo alucina» esta dominado por «dos entradas empatan y elige una». Son dos cosas
# distintas y el vocabulario chico las mantiene pegadas. Con 24 relaciones la colision cae al
# 23,5 %, y lo que quede de error se puede atribuir de verdad a la lectura del archivo.
#
# Va como VERSION y no como reemplazo porque `idioma` ya esta en la guarda de identidad del
# checkpoint: una corrida v2 no puede reanudarse como v3 ni al reves, que es exactamente lo que hay
# que impedir cuando cambia el tamaño del vocabulario.
#
# Los verbos respetan los dos moldes de `formas()`: en las NO personales el sujeto es la entidad
# («taller pesa 42»), y en las personales es la persona («ana manda taller»).
RELACIONES_V3 = {
    "director":  ("director", "dirige", "el"),
    "precio":    ("precio", "cuesta", "el"),
    "altura":    ("altura", "mide", "la"),
    "duenio":    ("dueño", VERBO_DUENIO[2], "el"),
    "guardia":   ("guardia", "cuida", "el"),
    "clave":     ("clave", "vale", "la"),
    # no personales nuevas (valor = numero)
    "peso":      ("peso", "pesa", "el"),
    "largo":     ("largo", "abarca", "el"),
    "puntaje":   ("puntaje", "suma", "el"),
    "codigo":    ("codigo", "lleva", "el"),
    "cupo":      ("cupo", "admite", "el"),
    "gasto":     ("gasto", "gasta", "el"),
    "tarifa":    ("tarifa", "cobra", "la"),
    "demora":    ("demora", "tarda", "la"),
    "distancia": ("distancia", "dista", "la"),
    # personales nuevas (valor = nombre)
    "jefe":      ("jefe", "manda", "el"),
    "chofer":    ("chofer", "maneja", "el"),
    "cocinero":  ("cocinero", "cocina", "el"),
    "medico":    ("medico", "atiende", "el"),
    "maestro":   ("maestro", "instruye", "el"),
    "capitan":   ("capitan", "lidera", "el"),
    "fundador":  ("fundador", "fundo", "el"),
    "socio":     ("socio", "acompania", "el"),
    "vecino":    ("vecino", "visita", "el"),
}
PERSONALES_V3 = ("director", "duenio", "guardia",
                 "jefe", "chofer", "cocinero", "medico", "maestro",
                 "capitan", "fundador", "socio", "vecino")

_RELACIONES_V12 = dict(RELACIONES)
_PERSONALES_V12 = tuple(PERSONALES)

FUNCIONALES = """el la de del es era esta en un una y o no si ahora antes ayer hoy cual quien que a
tiene como con por para se lo su mas menos tambien pero entonces""".split()

CONTROL = ["BOS", "SEP", "EOS", "USUARIO", "MODELO", "NOSE", "?", ",", "."]

NUMEROS = [str(i) for i in range(100)]          # los "100 numeros" que pidio Maxi

# --- pool efectivo de valores numericos (2026-08-31, `PREREG_CONTEO_SOFTMAX.md`) -----------------
# Por que existe. El 31 se midio que las unidades `token` se callan EXACTAMENTE en las relaciones
# cuyo valor es un numero y contestan en las de nombre, con pureza 0,98 y replica en dos semillas, y
# que ese corte no sigue ni la ausencia (+0,0000 de ganancia) ni la dificultad (-0,0200, al reves).
# La hipotesis mecanica es aritmetica del softmax: con `--abst token`,
#     q > 0,5  <=>  l_NOSE > logsumexp(logits de los valores)
# o sea que NOSE compite contra la SUMA de los candidatos, no contra el mejor. Y hay 100 numeros
# contra 58 nombres, asi que del lado numerico la suma tiene 1,72x mas terminos.
#
# `POOL_NUM` permite igualar la CANTIDAD DE VALORES EFECTIVOS sin tocar el vocabulario: V sigue
# siendo 242 y el modelo es identico, asi que la comparacion es limpia y la guarda de identidad del
# checkpoint no ve un idioma distinto. Los 42 numeros que quedan afuera siguen existiendo como
# tokens; simplemente no aparecen nunca como valor, y el modelo les aprende un logit bajo.
POOL_NUM = NUMEROS


def fijar_pool_numeros(k=None):
    """Limita a `k` los valores numericos que el generador puede sortear. `None` restaura los 100."""
    global POOL_NUM
    POOL_NUM = NUMEROS if k is None else NUMEROS[:k]
    return len(POOL_NUM)


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


def fijar_version(v):
    """Elige la version del idioma: 1 = «pertenece_a» (campania del 14-ago), 2 = «posee».

    Existe para poder MEDIR el efecto del arreglo contra las corridas que ya estaban hechas, en vez
    de cambiar el generador en silencio a mitad de camino y comparar contra numeros de otro idioma.

    Prediccion, escrita ANTES de correr: el cambio es un REEMPLAZO DE UN TOKEN por otro en la misma
    posicion de la misma plantilla — las secuencias tienen el mismo largo y la misma estructura, y
    el modelo no sabe castellano. Asi que no deberia mover la accuracy mas alla del ruido entre
    semillas. Si moviera mucho, lo que estaria mal es mi entendimiento de la tarea, no el arreglo.
    (Lo unico que cambia de verdad es que un humano que lee `dialogos.py` no tropieza.)

    El vocabulario mantiene su tamaño (242) y los bloques CONTROL/NUMEROS/NOMBRES/ENTIDADES sus
    indices: la permutacion queda contenida en el bloque de palabras de relaciones.

    La version 3 (2026-08-23) es OTRA cosa y por eso se aclara aparte: no permuta un token, agrega
    18 relaciones para que la COLISION DE CLAVE deje de dominar la tarea (ver `RELACIONES_V3`). El
    vocabulario crece, asi que una corrida v3 no es comparable con una v2 y el chequeo de identidad
    del checkpoint —que ya mira `idioma`— lo impide solo.
    """
    global ITOS, STOI, V, RELACIONES, PERSONALES
    if v == 3:
        RELACIONES = dict(RELACIONES_V3)
        PERSONALES = tuple(PERSONALES_V3)
    else:
        RELACIONES = dict(_RELACIONES_V12)
        PERSONALES = tuple(_PERSONALES_V12)
        sust, _, art = RELACIONES["duenio"]
        RELACIONES["duenio"] = (sust, VERBO_DUENIO[v], art)
    ITOS, STOI = construir_vocab()
    V = len(ITOS)
    return V


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


def correccion(rng, rel, ent, val, nivel):
    """Como se corrige un hecho ya dicho. En N3 la correccion NO nombra la entidad.

    OJO — 2026-08-14: esto usaba `np.random.choice`, o sea el generador GLOBAL de numpy, mientras
    todo el resto del generador de episodios usa el `rng` sembrado que se pasa por parametro. La
    consecuencia no era cosmetica: **la corrida no era reproducible a partir de su semilla** en los
    niveles 3 y 4 (los unicos que entran por esta rama), porque la forma de cada correccion eliptica
    dependia de un estado global que nadie controlaba ni guardaba. Lo delato el test de reanudacion:
    dos corridas con la misma semilla diferian ya en el paso 20, ANTES de reanudar nada.
    """
    sust, _, art = RELACIONES[rel]
    if nivel < 3:
        return f"no , {art} {sust} de {ent} es {val}"
    return str(rng.choice([f"no , es {val}", f"no , ahora es {val}", f"no , {val}"]))


def pregunta(rel, ent, cual="vigente"):
    sust, _, art = RELACIONES[rel]
    if cual == "vigente":
        return f"cual es {art} {sust} de {ent} ?"
    return f"cual era antes {art} {sust} de {ent} ?"


# --- episodios -------------------------------------------------------------------------------

def episodio(rng, nivel=4, n_hechos=4, n_sesiones=5, p_revision=0.5, p_pregunta_vieja=0.35,
             p_nose=0.0, con_meta=False, con_origen=False):
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
    # `origen` es paralelo a `sesiones`: por cada enunciado, DE QUE HECHO salio. Sirve para saber
    # que entrada del archivo corresponde al hecho preguntado, y asi distinguir «el hecho no se
    # escribio» de «se escribio y la lectura no lo alcanza» — que quedo sin resolver el 16-ago.
    # Es puramente contable: no consume una sola llamada al rng, asi que la reproducibilidad desde
    # la semilla no cambia.
    origen = [[] for _ in range(n_sesiones)]
    for i, (ent, rel) in enumerate(zip(ents, rels)):
        pool = NOMBRES if rel in PERSONALES else POOL_NUM
        v1 = rng.choice(pool)
        s = 0 if nivel < 4 else int(rng.integers(0, max(1, n_sesiones - 1)))
        sesiones[s].append(rng.choice(formas(rel, ent, v1, nivel)))
        origen[s].append(i)
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
                sesiones[s2].append(correccion(rng, rel, ent, v2, nivel=2))   # nombra la entidad
                origen[s2].append(i)
            else:
                sesiones[s].append(correccion(rng, rel, ent, v2, nivel))      # adyacente: puede elidir
                origen[s].append(i)
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
                if con_meta and con_origen:
                    return sesiones, consultas, vals_por_hecho, origen
                return (sesiones, consultas, vals_por_hecho) if con_meta else (sesiones, consultas)
            rel_q = str(rng.choice(libres))
            tipo = "nose_rel"
        consultas.append((pregunta(rel_q, ent_q, "vigente"), "NOSE", tipo))
    # `con_meta` devuelve ademas vals_por_hecho = [(rel, ent, [v1, v2...]), ...], que es lo que hace
    # falta para CLASIFICAR un error en vez de sólo contarlo: si el modelo contesta una version vieja
    # del hecho preguntado es un error de VERSION, y si contesta el valor de OTRA entidad es un error
    # de IDENTIDAD. La §6 del diseño pide esa desagregacion —es el corte propio frente a FAMA, que
    # penaliza el reuso de memoria invalidada sin separar los dos— y hasta ahora no se medía.
    # Es estrictamente aditivo: no toca ni una llamada al rng, así que las corridas siguen siendo
    # reproducibles bit a bit desde su semilla (verificado contra un hash de referencia).
    if con_meta and con_origen:
        return sesiones, consultas, vals_por_hecho, origen
    return (sesiones, consultas, vals_por_hecho) if con_meta else (sesiones, consultas)


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
