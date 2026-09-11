"""TRANSFERENCIA A LENGUAJE NATURAL · ¿la dilución es geométrica o es del banco sintético? · 2026-09-10

Replica el diseño de `dilucion.py` y `dilucion_topk.py` pero con **embeddings de frases reales en
castellano** en vez del archivo del micro-LM. Sin entrenar nada.

HIPÓTESIS, declarada ANTES de correr:

  H  · Si la dilución del VALOR es un fenómeno **geométrico** de leer por promedio ponderado —y no
       un artefacto del banco de 242 tokens— entonces con embeddings de lenguaje natural tiene que
       aparecer la misma firma: la exactitud cae al crecer N mientras el RANKING aguanta, y el
       top-k la recupera con un máximo en K chico.
  C1 · CONTROL NEGATIVO. Con distractores `real` —que pueden repetir (entidad, relación)— hay
       COLISIÓN y el top-k no debe rescatar.
  C2 · CONTROL DE IMPLEMENTACIÓN. Con K >= N el top-k es el softmax completo.

⚠️ LIMITACIÓN QUE HAY QUE DECLARAR: en el micro-LM las proyecciones `kw`/`vw`/`qr` están
CO-ENTRENADAS; acá se usan embeddings crudos, sin ninguna proyección aprendida. O sea esto prueba el
mecanismo SIN la parte aprendida. Si el fenómeno aparece igual, es geométrico. Si NO aparece, podría
deberse a la falta de proyección y no sería una refutación limpia.

⚠️ Y el encoder: `nomic-embed-text` colapsa todo token capitalizado a un mismo vector (memoria
`encoder-ollama-mayusculas`), así que TODO el texto va en minúscula.

    python3 transferencia_nl.py
"""
import json, os, sys, time, hashlib
import urllib.request
import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(AQUI, "cache_emb_nl.npz")
MODELO = "nomic-embed-text"
SEM = 20260910

NOMBRES = ("ana beto carla dario elena fabio gloria hugo irene julio karina leo marta nico olga "
           "pablo rosa said tania ulises vera walter ximena yamil zoe agustin brenda ciro delia "
           "emilio flor gaston helena ivan juana kevin lucia mateo nadia oscar paula quimey rocio "
           "sergio tomasa ulises2 valeria wanda xavier yanina zulema andres bruna camilo dora "
           "ezequiel fabiana gonzalo hilda ignacio").split()

# (relación, plantilla del hecho, plantilla de la pregunta, valores posibles)
RELS = [
    ("ciudad",  "{e} vive en {v}",           "donde vive {e}",
     "rosario cordoba mendoza salta neuquen parana corrientes tucuman jujuy posadas "
     "resistencia formosa rawson viedma ushuaia bariloche tandil olavarria pergamino junin".split()),
    ("oficio",  "{e} trabaja de {v}",        "de que trabaja {e}",
     "panadera herrero enfermera docente plomero contadora chofer jardinero costurera electricista "
     "carpintero veterinaria farmaceutica mecanico pintora soldador tornero ceramista relojero apicultor".split()),
    ("mascota", "{e} tiene un {v}",          "que mascota tiene {e}",
     "perro gato conejo loro canario hamster tortuga huron pez iguana erizo chinchilla raton jilguero calandria "
     "zorzal benteveo cardenal periquito agaporni".split()),
    ("comida",  "a {e} le gusta el {v}",     "que le gusta comer a {e}",
     "guiso locro puchero asado mondongo pastel chipa mate revuelto matambre "
     "escabeche bondiola arrollado budin flan panqueque alfajor bizcochuelo pionono strudel".split()),
]

def emb_lote(textos, tam=256):
    """Embeddings por la API de Ollama, en minúscula, por lotes."""
    out = []
    for i in range(0, len(textos), tam):
        pedazo = [t.lower() for t in textos[i:i+tam]]
        req = urllib.request.Request(
            "http://localhost:11434/api/embed",
            data=json.dumps({"model": MODELO, "input": pedazo}).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=300) as r:
            out.extend(json.load(r)["embeddings"])
        print(f"    embebidos {min(i+tam, len(textos)):5d}/{len(textos)}", end="\r", flush=True)
    print(" " * 40, end="\r")
    a = np.asarray(out, np.float32)
    return a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-9)

def construir():
    """Devuelve el universo de hechos, preguntas y valores, ya embebidos (con caché en disco)."""
    rng = np.random.default_rng(SEM)
    mitad = len(NOMBRES) // 2
    ENT_A, ENT_B = NOMBRES[:mitad], NOMBRES[mitad:]        # A = episodios · B = disjuntos

    hechos, meta = [], []                                   # meta: (entidad, i_rel, i_val)
    for ir, (_, pl, _, vals) in enumerate(RELS):
        for e in NOMBRES:
            for iv, v in enumerate(vals):
                hechos.append(pl.format(e=e, v=v)); meta.append((e, ir, iv))
    preguntas = [(e, ir, RELS[ir][2].format(e=e)) for ir in range(len(RELS)) for e in NOMBRES]
    valores = [v for _, _, _, vals in RELS for v in vals]

    firma = hashlib.sha1(("|".join(hechos[:50]) + str(len(hechos))).encode()).hexdigest()[:12]
    if os.path.exists(CACHE):
        z = np.load(CACHE, allow_pickle=True)
        if str(z["firma"]) == firma:
            print(f"  caché de embeddings reusada ({firma})")
            return (ENT_A, ENT_B, hechos, meta, preguntas, valores,
                    z["E_h"], z["E_q"], z["E_v"])
    t0 = time.time()
    print(f"  embebiendo {len(hechos):,} hechos · {len(preguntas)} preguntas · {len(valores)} valores")
    E_h = emb_lote(hechos)
    E_q = emb_lote([p[2] for p in preguntas])
    E_v = emb_lote(valores)
    print(f"  listo en {time.time()-t0:.0f}s")
    np.savez_compressed(CACHE, firma=firma, E_h=E_h, E_q=E_q, E_v=E_v)
    return ENT_A, ENT_B, hechos, meta, preguntas, valores, E_h, E_q, E_v

def celda(U, X, dist, K, n_mue=256, n_epi=40, rng=None, temp=8.0):
    """Una celda: n_mue consultas sobre un archivo de n_epi entradas propias + X distractores."""
    ENT_A, ENT_B, hechos, meta, preguntas, valores, E_h, E_q, E_v = U
    rng = rng or np.random.default_rng(SEM + 7)
    D = E_h.shape[1]
    base_val = np.cumsum([0] + [len(r[3]) for r in RELS])    # offset de cada relación en E_v
    idx_A = [i for i, m in enumerate(meta) if m[0] in ENT_A]
    idx_B = [i for i, m in enumerate(meta) if m[0] in ENT_B]
    idx_todos = list(range(len(meta)))
    mu, sd = E_h.mean(0), E_h.std(0)

    ok = rank0 = 0
    masas = []
    for _ in range(n_mue):
        # ── el episodio: n_epi hechos de entidades de A, uno de ellos es el preguntado
        propios = rng.choice(idx_A, n_epi, replace=False)
        elegido = int(propios[0])
        e_p, ir_p, iv_p = meta[elegido]
        # ningún otro hecho propio puede hablar de la misma (entidad, relación)
        propios = [elegido] + [i for i in propios[1:] if meta[i][:2] != (e_p, ir_p)]

        # ── los distractores
        if X > 0:
            if dist == "ruido":
                extra_e = rng.normal(mu, sd, (X, D)).astype(np.float32)
                extra_e /= np.linalg.norm(extra_e, axis=-1, keepdims=True) + 1e-9
                extra_v = rng.integers(0, len(valores), X)
            else:
                fuente = idx_B if dist == "disjunto" else idx_todos
                sel = rng.choice(fuente, X, replace=True)
                extra_e = E_h[sel]
                extra_v = np.array([base_val[meta[i][1]] + meta[i][2] for i in sel])
        else:
            extra_e = np.zeros((0, D), np.float32); extra_v = np.zeros(0, int)

        claves = np.concatenate([E_h[propios], extra_e], 0)
        vvals  = np.concatenate([[base_val[meta[i][1]] + meta[i][2] for i in propios], extra_v])

        # ── la lectura
        iq = [j for j, p in enumerate(preguntas) if p[0] == e_p and p[1] == ir_p][0]
        sim = claves @ E_q[iq]
        k = min(K, sim.size)
        top = np.argpartition(-sim, k-1)[:k] if k < sim.size else np.arange(sim.size)
        w = np.exp((sim[top] - sim[top].max()) * temp); w /= w.sum()
        leido = (E_v[vvals[top]] * w[:, None]).sum(0)

        # ── decodificar: el valor más cercano DENTRO de la relación preguntada
        a, b = base_val[ir_p], base_val[ir_p+1]
        pred = int(np.argmax(E_v[a:b] @ leido))
        ok += (pred == iv_p)
        orden = np.argsort(-sim)
        rank0 += (orden[0] == 0)
        wf = np.exp((sim - sim.max()) * temp); wf /= wf.sum()
        masas.append(float(wf[0]))
    return {"X": X, "dist": dist, "K": K, "temp": temp, "n": n_mue, "exactitud": ok/n_mue,
            "RECUP": rank0/n_mue, "masa_ganadora": float(np.mean(masas))}

if __name__ == "__main__":
    print("TRANSFERENCIA A LENGUAJE NATURAL · dilución y top-k sobre embeddings reales")
    U = construir()
    print(f"  universo: {len(U[2]):,} hechos · {len(U[4])} preguntas · {len(U[5])} valores "
          f"· dim {U[6].shape[1]}\n")
    # PARTE 0 · la temperatura es un parametro libre que en el micro-LM viene de proyecciones
    # APRENDIDAS. Se fija acá con archivo de 40 —el regimen donde la lectura tiene que andar— y NO
    # se vuelve a tocar. Declararlo importa: elegirla mirando las celdas grandes seria hacer trampa.
    print("PARTE 0 · calibrar la temperatura con archivo de 40 (y despues no tocarla)")
    mejor, T = -1, None
    for t in (2, 4, 8, 16, 32, 64):
        r = celda(U, 0, "ruido", 10**9, temp=t)
        print(f"  temp {t:>3} · exactitud {r['exactitud']:.4f} · masa gan {r['masa_ganadora']:.4f}",
              flush=True)
        if r["exactitud"] > mejor: mejor, T = r["exactitud"], t
    print(f"  -> temperatura FIJADA en {T} (exactitud {mejor:.4f})\n")

    PLAN = [("ruido",    [0, 360, 1120, 3280]),
            ("disjunto", [0, 360, 1120, 3280]),
            ("real",     [0, 360, 1120, 3280])]
    filas = []
    print("PARTE 1 · la curva de dilución (lectura densa, como hoy)")
    print(f"  {'dist':>9} {'X':>6} {'archivo':>8} {'exactitud':>10} {'RECUP':>8} {'masa gan':>9}")
    for dist, Xs in PLAN:
        for X in Xs:
            r = celda(U, X, dist, 10**9, temp=T)
            filas.append(r)
            print(f"  {dist:>9} {X:>6} {X+40:>8} {r['exactitud']:>10.4f} {r['RECUP']:>8.4f} "
                  f"{r['masa_ganadora']:>9.4f}", flush=True)
    print("\nPARTE 2 · el barrido de K con archivo de 3.280")
    print(f"  {'dist':>9} {'K':>6} {'exactitud':>10} {'RECUP':>8}")
    for dist in ("ruido", "disjunto", "real"):
        for K in (1, 2, 4, 8, 32, 10**9):
            r = celda(U, 3240, dist, K, temp=T)
            filas.append(r)
            print(f"  {dist:>9} {('N' if K > 1e8 else K)!s:>6} {r['exactitud']:>10.4f} "
                  f"{r['RECUP']:>8.4f}", flush=True)
    json.dump({"modelo": MODELO, "semilla": SEM, "temp": T, "filas": filas},
              open(os.path.join(AQUI, "transferencia_nl_20260910.json"), "w"), indent=1)
    print("\nguardado en transferencia_nl_20260910.json")
