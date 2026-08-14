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


def _tok(texto, largo):
    ids = I.a_ids(texto)[:largo]
    return ids + [PAD] * (largo - len(ids))


def lote(rng, B, nivel=4, n_hechos=4, n_sesiones=4):
    """Devuelve sesiones, cortes, turnos, mask, consulta, target, tipo."""
    S, N = n_sesiones, n_sesiones * E_MAX
    ses = np.full((B, S, T_SES), PAD, np.int32)
    cortes = np.zeros((B, S, E_MAX), np.int32)
    mask = np.zeros((B, N), bool)
    turnos = np.zeros((B, N), np.int32)
    consulta = np.full((B, T_Q), PAD, np.int32)
    pos_q = np.zeros(B, np.int32)          # ultima posicion real de la pregunta
    target = np.zeros(B, np.int32)
    tipo = np.zeros(B, np.int32)          # 0 = vigente, 1 = anterior

    b = 0
    while b < B:
        sesiones, consultas = I.episodio(rng, nivel=nivel, n_hechos=n_hechos,
                                         n_sesiones=n_sesiones)
        if not consultas:
            continue
        q, r, t = consultas[int(rng.integers(len(consultas)))]
        turno = 0
        for s, enunciados in enumerate(sesiones):
            pos = 0
            toks = [I.STOI["BOS"]]
            for e, enunciado in enumerate(enunciados[:E_MAX]):
                ids = I.a_ids(enunciado)
                if len(toks) + len(ids) >= T_SES:
                    break
                toks += ids
                cortes[b, s, e] = len(toks) - 1        # ultimo token del enunciado
                mask[b, s * E_MAX + e] = True
                turnos[b, s * E_MAX + e] = turno
                turno += 1
            ses[b, s, :len(toks)] = toks
        ids_q = I.a_ids("BOS " + q)[:T_Q]
        consulta[b, :len(ids_q)] = ids_q
        pos_q[b] = len(ids_q) - 1
        target[b] = I.STOI[r]
        tipo[b] = 1 if t == "anterior" else 0
        b += 1
    return ses, cortes, turnos, mask, consulta, pos_q, target, tipo


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
