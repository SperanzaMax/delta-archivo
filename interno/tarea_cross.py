"""Tarea CROSS-SECUENCIA: el hecho se escribe en una secuencia, se revisa en otra, y se consulta
en una tercera, con el estado RESETEADO entre ellas.

    S1  BOS (k v1)*L SEP          escribe
    S2  BOS (k v2)*r SEP          revisa r de las L claves
    S3  BOS SEP q*L               consulta -> target = ultimo valor escrito

Cada secuencia es un forward independiente: no hay estado que sobreviva. Por construccion, un modelo
sin memoria persistente NO PUEDE resolver S3 -- la informacion no esta en su entrada. El piso es el
azar (1/NV).

Esa es justamente la propiedad que hace util a la tarea: cualquier acierto por encima del azar viene
del archivo y de ningun otro lado. Es la version cross-secuencia de T2 (`gen_overwrite`), que ya esta
validada intra-secuencia en el harness de Ligamento.

`gen_intra` devuelve LA MISMA informacion concatenada en una sola secuencia. Es el CONTROL que puede
fallar: si un modelo no resuelve la version intra, tampoco se puede leer nada de que falle la cross.
Es la leccion del 11-ago -- un control de sanidad tiene que poder fallar.
"""
import sys

import numpy as np

sys.path.insert(0, "/home/maxi/Documentos/Nuevo Transformer/telar-ligamento/src")
from datos import IGNORE, V_E001


def _muestra(rng, B, L, r, voc):
    """Claves, valores iniciales y revisados. Compartido por las dos variantes."""
    assert L <= voc.NK and r <= L
    keys = np.argsort(rng.random((B, voc.NK)), axis=1)[:, :L]
    v1 = rng.integers(0, voc.NV, size=(B, L))
    # v2 garantizado distinto de v1 en las r claves revisadas
    v2 = (v1[:, :r] + 1 + rng.integers(0, voc.NV - 1, size=(B, r))) % voc.NV
    final = v1.copy()
    final[:, :r] = v2
    perm = np.argsort(rng.random((B, L)), axis=1)      # el orden de consulta no delata nada
    q = np.take_along_axis(keys, perm, axis=1)
    tgt = np.take_along_axis(final, perm, axis=1)
    return keys, v1, v2, q, tgt


def gen_cross(rng, B, L, r=None, voc=V_E001):
    """Devuelve (s1, s2, s3, y3). Cada s* es un batch de secuencias independientes."""
    r = L // 2 if r is None else r
    keys, v1, v2, q, tgt = _muestra(rng, B, L, r, voc)

    s1 = np.full((B, 2 * L + 2), voc.PAD, dtype=np.int32)
    s1[:, 0] = voc.BOS
    s1[:, 1:2 * L + 1:2] = voc.K0 + keys
    s1[:, 2:2 * L + 2:2] = voc.V0 + v1
    s1[:, -1] = voc.SEP

    s2 = np.full((B, 2 * r + 2), voc.PAD, dtype=np.int32)
    s2[:, 0] = voc.BOS
    s2[:, 1:2 * r + 1:2] = voc.K0 + keys[:, :r]
    s2[:, 2:2 * r + 2:2] = voc.V0 + v2
    s2[:, -1] = voc.SEP

    s3 = np.full((B, L + 2), voc.PAD, dtype=np.int32)
    y3 = np.full((B, L + 2), IGNORE, dtype=np.int32)
    s3[:, 0] = voc.BOS
    s3[:, 1] = voc.SEP
    s3[:, 2:] = voc.K0 + q
    y3[:, 2:] = voc.V0 + tgt
    return s1, s2, s3, y3


def gen_intra(rng, B, L, r=None, voc=V_E001):
    """CONTROL: la misma informacion, toda en una secuencia. Debe ser resoluble."""
    r = L // 2 if r is None else r
    keys, v1, v2, q, tgt = _muestra(rng, B, L, r, voc)
    E = L + r
    T = 2 * E + 2 + L
    x = np.full((B, T), voc.PAD, dtype=np.int32)
    y = np.full((B, T), IGNORE, dtype=np.int32)
    x[:, 0] = voc.BOS
    ek = np.concatenate([keys, keys[:, :r]], axis=1)
    ev = np.concatenate([v1, v2], axis=1)
    x[:, 1:2 * E + 1:2] = voc.K0 + ek
    x[:, 2:2 * E + 2:2] = voc.V0 + ev
    x[:, 2 * E + 1] = voc.SEP
    x[:, 2 * E + 2:] = voc.K0 + q
    y[:, 2 * E + 2:] = voc.V0 + tgt
    return x, y


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    s1, s2, s3, y3 = gen_cross(rng, 4, 6, 3)
    xi, yi = gen_intra(np.random.default_rng(0), 4, 6, 3)
    print("cross: s1", s1.shape, "s2", s2.shape, "s3", s3.shape)
    print("intra:", xi.shape)
    # el target de S3 debe ser identico al de la version intra: misma semilla, misma muestra
    assert (y3[:, 2:] == yi[:, -6:]).all(), "las dos variantes deben pedir lo mismo"
    # y S3 no puede contener ningun valor
    assert not ((s3 >= V_E001.V0) & (s3 < V_E001.V0 + V_E001.NV)).any(), "S3 filtra valores"
    print("OK · S3 no contiene valores · azar = 1/NV =", round(1 / V_E001.NV, 4))
