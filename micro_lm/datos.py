"""MICRO-LM · de episodios de texto a tensores.

Un ejemplo = un episodio (varias sesiones de enunciados) + UNA pregunta + UN token de respuesta.
Las sesiones se procesan por separado, con el estado reseteado entre ellas: lo que sobrevive de una
a otra es el archivo, no el estado. Eso es lo que hace que la prueba sea de memoria persistente y no
de contexto largo.
"""
import numpy as np

import idioma as I

PAD = I.STOI["."]          # relleno inerte
T_SES = 96                 # tokens por sesion
T_Q = 12                   # tokens de la pregunta
E_MAX = 10                 # enunciados por sesion

# OJO — 2026-08-14: con E_MAX=4 y T_SES=40 se truncaba el 34 % de los enunciados en los niveles 1-3,
# donde TODOS los hechos caen en la misma sesion. La accuracy tenia un techo de 1-0,34 = 0,66 y
# medimos 0,6707: no era una meseta de aprendizaje, era el padding. El nivel 4 reparte los hechos
# entre sesiones, truncaba el 1,5 % y por eso daba 0,988. Cualquier cambio de n_hechos o n_sesiones
# obliga a revisar estos dos numeros: mirar `truncados` antes de leer la accuracy.


# Contador de diagnostico: lo que la nota de arriba manda mirar ANTES de leer una accuracy.
# Se acumula sobre todas las llamadas a `lote`; `reset_truncados()` lo pone en cero.
TRUNC = {"enunciados": 0, "truncados": 0}


def reset_truncados():
    TRUNC["enunciados"] = TRUNC["truncados"] = 0


def tasa_truncados():
    return TRUNC["truncados"] / max(1, TRUNC["enunciados"])


def _tok(texto, largo):
    ids = I.a_ids(texto)[:largo]
    return ids + [PAD] * (largo - len(ids))


TIPOS = {"vigente": 0, "anterior": 1, "nose_ent": 2, "nose_rel": 3}


# ARCHIVO LARGO (2026-09-05). Base del turno del episodio PRINCIPAL cuando hay sesiones extra.
# `ord` tiene 64 filas y desde hoy la indexacion fuera de rango da NaN en vez de clampear en
# silencio (`modelo.sello`), asi que los turnos tienen que caber. Con `n_sesiones=4` y `E_MAX=10` el
# episodio principal usa 40 turnos como maximo, de 24 a 63, y las sesiones EXTRA sortean en 0..23.
#
# Eso no es una concesion al tope, es la HIPOTESIS: el archivo largo queda marcado como anterior, que
# es exactamente lo que el sello de orden deberia poder usar para descartarlo. Medido el 5-sep sobre
# checkpoints entrenados, hoy NO lo usa (0,0059 contra 0,0039, identico). La pregunta de la campania
# es si entrenando lo aprende.
TURNO_BASE = 24


def lote(rng, B, nivel=4, n_hechos=4, n_sesiones=4, p_vieja=0.35, p_nose=0.0, con_meta=False,
         con_origen=False, formas_q=("directa",), con_formas=False, n_ses_extra=0):
    """Devuelve sesiones, cortes, turnos, mask, consulta, target, tipo.

    Con `con_meta=True` agrega al final una lista de dicts, uno por muestra, con el hecho que se
    preguntó y los demas hechos del episodio. Sirve para CLASIFICAR el error —version contra
    identidad, la desagregacion que pide la §6— y no cambia nada del muestreo: `episodio` ya se
    llamaba igual, sólo se le pide que devuelva lo que ya tenia calculado.
    """
    # Las sesiones EXTRA son episodios completos de OTRAS conversaciones que se archivan y cuyas
    # consultas se descartan. Van por el MISMO forward que las del episodio, asi que sus entradas
    # las escribe el modelo con los pesos de ESE paso — que es la unica forma honesta de tener un
    # archivo largo durante el entrenamiento: un pool precomputado quedaria congelado en los pesos
    # del momento en que se genero.
    S, N = n_sesiones + n_ses_extra, (n_sesiones + n_ses_extra) * E_MAX
    ses = np.full((B, S, T_SES), PAD, np.int32)
    cortes = np.zeros((B, S, E_MAX), np.int32)
    mask = np.zeros((B, N), bool)
    turnos = np.zeros((B, N), np.int32)
    consulta = np.full((B, T_Q), PAD, np.int32)
    # Que FORMA de pregunta le toco a cada muestra (indice en `I.FORMAS_Q`). Es lo que permite
    # desagregar `nose_rel` y `nose_ent` POR FORMA, que es el cruce del 2-sep: la misma pregunta
    # reordenada cambia cual de las dos componentes entra en la ventana de la conv.
    forma_q = np.zeros(B, np.int32)
    pos_q = np.zeros(B, np.int32)          # ultima posicion real de la pregunta
    target = np.zeros(B, np.int32)
    tipo = np.zeros(B, np.int32)          # 0 = vigente, 1 = anterior

    meta = []
    # `origen_arch[b, k]` = de que hecho salio la entrada k del archivo (-1 = slot vacio), y
    # `hecho_q[b]` = que hecho se pregunto. Con eso se puede mirar EN QUE POSICION del ranking de
    # lectura quedo la entrada del hecho preguntado — lo unico que separa «no se escribio» de
    # «se escribio y la lectura no lo alcanza».
    origen_arch = np.full((B, N), -1, np.int32)
    hecho_q = np.full(B, -1, np.int32)
    b = 0
    while b < B:
        sesiones, consultas, vals, origen, formas_ep = I.episodio(
            rng, nivel=nivel, n_hechos=n_hechos, n_sesiones=n_sesiones, p_pregunta_vieja=p_vieja,
            p_nose=1.0 if p_nose > 0 else 0.0, con_meta=True, con_origen=True,
            formas_q=formas_q, con_formas=True)
        if not consultas:
            continue
        # La consulta sin respuesta se elige con probabilidad `p_nose` EXACTA, en vez de dejarla
        # competir en el sorteo uniforme con las otras n_hechos: asi `--p-nose 0,2` significa
        # «el 20 % de los ejemplos no tiene respuesta» y no «0,2 dividido cinco».
        sin_resp = [(c, i) for i, c in enumerate(consultas) if c[1] == "NOSE"]
        con_resp = [(c, i) for i, c in enumerate(consultas) if c[1] != "NOSE"]
        if sin_resp and rng.random() < p_nose:
            (q, r, t), i_q = sin_resp[0]
        else:
            if not con_resp:
                continue
            (q, r, t), i_q = con_resp[int(rng.integers(len(con_resp)))]
        forma_q[b] = I.FORMAS_Q.index(formas_ep[i_q]) if i_q < len(formas_ep) else 0
        turno = TURNO_BASE if n_ses_extra else 0
        for s, enunciados in enumerate(sesiones):
            toks = [I.STOI["BOS"]]
            puestos = 0
            for e, enunciado in enumerate(enunciados[:E_MAX]):
                ids = I.a_ids(enunciado)
                if len(toks) + len(ids) >= T_SES:
                    break
                toks += ids
                puestos += 1
                cortes[b, s, e] = len(toks) - 1        # ultimo token del enunciado
                mask[b, s * E_MAX + e] = True
                if e < len(origen[s]):
                    origen_arch[b, s * E_MAX + e] = origen[s][e]
                turnos[b, s * E_MAX + e] = turno
                turno += 1
            ses[b, s, :len(toks)] = toks
            TRUNC["enunciados"] += len(enunciados)
            TRUNC["truncados"] += len(enunciados) - puestos
        for k in range(n_ses_extra):
            s_ex = n_sesiones + k
            ses_ex, _ = I.episodio(rng, nivel=nivel, n_hechos=n_hechos, n_sesiones=1,
                                   p_pregunta_vieja=p_vieja, p_nose=0.0)
            toks = [I.STOI["BOS"]]
            for e, enunciado in enumerate(ses_ex[0][:E_MAX]):
                ids = I.a_ids(enunciado)
                if len(toks) + len(ids) >= T_SES:
                    break
                toks += ids
                cortes[b, s_ex, e] = len(toks) - 1
                mask[b, s_ex * E_MAX + e] = True
                # turno VIEJO: por debajo de TURNO_BASE, o sea anterior a todo el episodio
                turnos[b, s_ex * E_MAX + e] = int(rng.integers(0, TURNO_BASE))
                # origen -1: ninguna entrada extra es el hecho preguntado
            ses[b, s_ex, :len(toks)] = toks

        ids_q = I.a_ids("BOS " + q)[:T_Q]
        consulta[b, :len(ids_q)] = ids_q
        pos_q[b] = len(ids_q) - 1
        target[b] = I.STOI[r]
        tipo[b] = TIPOS[t]
        if con_meta:
            # `consultas` viene alineada con `vals` por indice para los hechos dichos; la consulta
            # sin respuesta, cuando existe, va al final y no tiene hecho asociado.
            try:
                idx = consultas.index((q, r, t))
            except ValueError:
                idx = -1
            hecho_q[b] = idx      # -1 cuando la consulta no tiene respuesta (NOSE)
            propio = vals[idx] if 0 <= idx < len(vals) else None
            meta.append({
                "tipo": t,
                "hecho": None if propio is None else {
                    "ent": str(propio[1]), "rel": str(propio[0]),
                    "versiones": [str(v) for v in propio[2]],
                },
                "otros": [{"ent": str(e), "rel": str(rl), "versiones": [str(v) for v in vs]}
                          for k, (rl, e, vs) in enumerate(vals) if k != idx],
            })
        b += 1
    salida = [ses, cortes, turnos, mask, consulta, pos_q, target, tipo]
    if con_meta:
        salida.append(meta)
    if con_meta and con_origen:
        salida += [origen_arch, hecho_q]
    if con_formas:                      # va AL FINAL: no mueve el desempaquetado de nadie
        salida.append(forma_q)
    return tuple(salida)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    ses, cortes, turnos, mask, consulta, pos_q, target, tipo = lote(rng, 4, nivel=4)
    print("sesiones", ses.shape, "| cortes", cortes.shape, "| archivo", mask.shape,
          "| consulta", consulta.shape)
    print("entradas archivadas por ejemplo:", mask.sum(1))
    print("turnos ej0:", turnos[0][mask[0]])
    print("\nejemplo 0, reconstruido:")
    for s in range(ses.shape[1]):
        toks = [I.ITOS[t] for t in ses[0, s] if t != PAD]
        if len(toks) > 1:
            print("  sesion", s + 1, ":", " ".join(toks))
    print("  pregunta:", " ".join(I.ITOS[t] for t in consulta[0] if t != PAD))
    print("  respuesta:", I.ITOS[target[0]], "| tipo:", "anterior" if tipo[0] else "vigente")
