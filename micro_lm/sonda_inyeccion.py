"""Sonda de inyeccion: la abstencion, ¿consulta el archivo o la forma de la pregunta?

    python sonda_inyeccion.py ckpts/x1_s0.pkl [n_muestras]

PREREGISTRADO en `../PREREG_INYECCION.md` (congelado antes de correr).

QUE MIDE. Para cada consulta de tipo NOSE se arman DOS tensorizaciones del MISMO episodio y la MISMA
consulta:

  A (ausente)    el episodio tal cual                      -> respuesta correcta = NOSE
  B (inyectada)  el mismo episodio + UN enunciado que dice
                 el hecho preguntado                       -> respuesta correcta = ese valor

La unica variable es si el hecho esta en el archivo. Si el modelo sigue contestando NOSE en B, la
abstencion esta enganchada a la FORMA de la pregunta y no a una consulta a memoria — y el 4-de-4 del
17-ago mide reconocimiento de distribucion, no abstencion.

Es CPU sobre checkpoints ya entrenados: no gasta GPU ni vuelve a entrenar nada.
"""
import json
import pickle
import sys

import numpy as np
import jax
import jax.numpy as jnp

import datos as DAT
import idioma as I
import entrenar as E

NOSE = I.STOI["NOSE"]
SUST2REL = {v[0]: k for k, v in I.RELACIONES.items()}


def parse_q(q):
    """`cual es <art> <sust> de <ent> ?` -> (rel, ent).

    Se parsea en vez de replicar el sorteo de `idioma.episodio` para no tocar ni una llamada al rng
    del generador: la sonda tiene que ver EXACTAMENTE los mismos episodios que la campania.
    """
    p = q.split()
    return SUST2REL[p[3]], p[5]


def tensorizar(muestras, rng_pad=None):
    """Replica la tensorizacion de `datos.lote` para (sesiones, pregunta, respuesta) ya armados.

    Va aparte a proposito: `datos.lote` sortea el episodio Y lo tensoriza en el mismo paso, y aca
    hace falta tensorizar DOS variantes del mismo episodio. Se copia la logica en vez de refactorizar
    `datos.py`, que es de donde salen todas las corridas de la campania.
    """
    B = len(muestras)
    S, N = 4, 4 * DAT.E_MAX
    ses = np.full((B, S, DAT.T_SES), DAT.PAD, np.int32)
    cortes = np.zeros((B, S, DAT.E_MAX), np.int32)
    mask = np.zeros((B, N), bool)
    turnos = np.zeros((B, N), np.int32)
    consulta = np.full((B, DAT.T_Q), DAT.PAD, np.int32)
    pos_q = np.zeros(B, np.int32)
    target = np.zeros(B, np.int32)
    truncados = 0
    total = 0

    for b, (sesiones, q, r) in enumerate(muestras):
        turno = 0
        for s, enunciados in enumerate(sesiones):
            toks = [I.STOI["BOS"]]
            puestos = 0
            for e, enunciado in enumerate(enunciados[:DAT.E_MAX]):
                ids = I.a_ids(enunciado)
                if len(toks) + len(ids) >= DAT.T_SES:
                    break
                toks += ids
                puestos += 1
                cortes[b, s, e] = len(toks) - 1
                mask[b, s * DAT.E_MAX + e] = True
                turnos[b, s * DAT.E_MAX + e] = turno
                turno += 1
            ses[b, s, :len(toks)] = toks
            total += len(enunciados)
            truncados += len(enunciados) - puestos
        ids_q = I.a_ids("BOS " + q)[:DAT.T_Q]
        consulta[b, :len(ids_q)] = ids_q
        pos_q[b] = len(ids_q) - 1
        target[b] = I.STOI[r]
    return ses, cortes, turnos, mask, consulta, pos_q, target, (truncados / max(1, total))


def juntar(rng, nivel, n, p_vieja=0.35):
    """Devuelve las muestras pareadas A/B/C. Solo entran episodios que produjeron consulta NOSE.

    C es la ENMIENDA E-1: en `nose_rel` la condicion B le deja DOS relaciones a la misma entidad, y
    `idioma.py:161` sortea las entidades con `replace=False`, o sea que en todo el entrenamiento cada
    entidad tuvo exactamente UNA. B saca de distribucion justo el caso dificil. C reemplaza el hecho
    que la entidad ya tenia en vez de agregarle otro, y asi el episodio queda dentro de distribucion.
    En `nose_ent` no hay hecho previo de esa entidad: C = B, y se reporta sólo para `nose_rel`.
    """
    A, B, C, tipos, otro = [], [], [], [], []
    intentos = 0
    while len(A) < n and intentos < n * 50:
        intentos += 1
        sesiones, consultas, vals, origen = I.episodio(
            rng, nivel=nivel, n_hechos=4, n_sesiones=4, p_pregunta_vieja=p_vieja, p_nose=1.0,
            con_meta=True, con_origen=True)
        sin_resp = [c for c in consultas if c[1] == "NOSE"]
        if not sin_resp:
            continue
        q, _, tipo = sin_resp[0]
        rel_q, ent_q = parse_q(q)

        # --- B: el mismo episodio con el hecho preguntado adentro -------------------------------
        pool = I.NOMBRES if rel_q in I.PERSONALES else I.NUMEROS
        v = str(rng.choice(pool))
        dicho = str(rng.choice(I.formas(rel_q, ent_q, v, nivel)))
        # la sesion mas corta, y en posicion aleatoria dentro de ella: pegarlo siempre al final
        # pondria el hecho en el turno mas alto, que es justo la señal que el sello usa para el
        # versionado — el contraste dejaria de ser sólo «esta / no esta».
        s_min = min(range(len(sesiones)), key=lambda s: len(sesiones[s]))
        inyectadas = [list(x) for x in sesiones]
        j = int(rng.integers(0, len(inyectadas[s_min]) + 1))
        inyectadas[s_min].insert(j, dicho)

        # --- C: el hecho que la entidad ya tenia, REEMPLAZADO por el preguntado -----------------
        if tipo == "nose_rel":
            i_h = next(k for k, (_, e, _) in enumerate(vals) if e == ent_q)
            reemp = [[] for _ in sesiones]
            puesto = False
            for s, enunciados in enumerate(sesiones):
                for e, enunciado in enumerate(enunciados):
                    del_hecho = e < len(origen[s]) and origen[s][e] == i_h
                    if not del_hecho:
                        reemp[s].append(enunciado)
                    elif not puesto:            # el primero se sustituye, el resto (revisiones) cae
                        reemp[s].append(dicho)
                        puesto = True
            if not puesto:                      # no deberia pasar; si pasa, la muestra no entra
                continue
            C.append((reemp, q, v))
            # el valor VIGENTE que esa entidad tenia bajo su otra relacion: es el candidato natural
            # a interferencia de identidad si el modelo indexa por entidad y no por (rel, ent).
            otro.append(I.STOI[str(vals[i_h][2][-1])])
        else:
            C.append((inyectadas, q, v))
            otro.append(-1)

        A.append((sesiones, q, "NOSE"))
        B.append((inyectadas, q, v))
        tipos.append(tipo)
    return A, B, C, np.array(tipos), np.array(otro)


def main():
    pesos = sys.argv[1] if len(sys.argv) > 1 else "ckpts/x1_s0.pkl"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
    with open(pesos, "rb") as f:
        bulto = pickle.load(f)
    params = jax.tree_util.tree_map(jnp.asarray, bulto["params"])
    cfg = bulto["config"]
    nivel = cfg["nivel"]

    rng = np.random.default_rng(180818)
    mA, mB, mC, tipos, otro = juntar(rng, nivel, n, p_vieja=cfg.get("p_vieja", 0.35))

    def predecir(muestras):
        out = []
        trunc = 0.0
        for i in range(0, len(muestras), 64):
            t = tensorizar(muestras[i:i + 64])
            trunc = max(trunc, t[7])
            pred = np.array(E.predecir(params, jnp.array(t[0]), jnp.array(t[1]), jnp.array(t[2]),
                                       jnp.array(t[3]), jnp.array(t[4]), jnp.array(t[5])))
            out.append((pred, t[6]))
        return (np.concatenate([o[0] for o in out]),
                np.concatenate([o[1] for o in out]), trunc)

    pA, tA, trA = predecir(mA)
    pB, tB, trB = predecir(mB)
    pC, tC, trC = predecir(mC)

    # COMPUERTA heredada del 14-ago: si el enunciado inyectado hace desbordar la sesion, el contraste
    # deja de ser pareado (B pierde enunciados que A tenia) y el numero no significa nada.
    if max(trA, trB, trC) > 0.0:
        print("ABORTA: truncamiento %.4f / %.4f / %.4f — el contraste dejo de ser pareado"
              % (trA, trB, trC))
        sys.exit(1)

    res = {"ckpt": pesos, "nivel": nivel, "n": len(pA), "truncados": [trA, trB, trC]}
    for nom, m in (("todo", np.ones(len(tipos), bool)),
                   ("nose_ent", tipos == "nose_ent"),
                   ("nose_rel", tipos == "nose_rel")):
        if not m.any():
            continue
        res[nom] = {
            "n": int(m.sum()),
            "A_nose": float((pA[m] == NOSE).mean()),      # abstiene cuando el hecho NO esta
            "B_nose": float((pB[m] == NOSE).mean()),      # abstiene cuando el hecho SI esta (agregado)
            "B_acierto": float((pB[m] == tB[m]).mean()),  # recupera el valor recien inyectado
            "caida": float((pA[m] == NOSE).mean() - (pB[m] == NOSE).mean()),
            "C_nose": float((pC[m] == NOSE).mean()),      # idem con el hecho REEMPLAZADO (E-1)
            "C_acierto": float((pC[m] == tC[m]).mean()),
            "caida_C": float((pA[m] == NOSE).mean() - (pC[m] == NOSE).mean()),
        }

    # Exploratorio (E-1, sin prediccion asociada): de lo que NO es NOSE en `nose_rel`, cuanto es el
    # valor que esa entidad tenia bajo su OTRA relacion — interferencia de identidad, no ruido.
    mr = (tipos == "nose_rel") & (otro >= 0)
    for nom, p in (("B", pB), ("C", pC)):
        vivo = mr & (p != NOSE)
        res["interferencia_%s" % nom] = {
            "n": int(vivo.sum()),
            "es_el_otro_valor": float((p[vivo] == otro[vivo]).mean()) if vivo.any() else None,
        }

    t = res["todo"]
    res["P1"] = bool(t["B_nose"] <= 0.20 and t["A_nose"] >= 0.60 and t["caida"] >= 0.40)
    res["P2"] = bool(t["B_acierto"] >= 0.50)
    r = res.get("nose_rel")
    if r:
        res["P4"] = bool(r["C_nose"] <= 0.20 and r["C_acierto"] >= 0.50)
        res["P5"] = bool(r["C_nose"] >= 0.50)
    print(json.dumps(res, indent=2, ensure_ascii=False))
    salida = "corridas_20260818/inyeccion_%s.json" % pesos.split("/")[-1].replace(".pkl", "")
    with open(salida, "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
