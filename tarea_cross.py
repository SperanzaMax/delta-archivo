"""Tarea CROSS-SECUENCIA: la unica donde la memoria persistente puede distinguirse de la nada.

Un episodio son TRES secuencias procesadas con el estado recurrente RESETEADO entre ellas:

  S1 escritura : BOS (k v)*L SEP          -> se escriben L pares
  S2 revision  : BOS (k v')*r SEP         -> se revisan r de esas claves con valor nuevo
  S3 consulta  : BOS SEP k*L              -> se pregunta por las L claves

Por construccion, en S3 un modelo sin memoria persistente NO PUEDE responder: el estado se
reseteo y nunca vio los pares. Delta puro y softmax dan azar (1/NV). Ese es el punto: la tarea
no mide "quien recuerda mejor", mide si hay memoria o no la hay.

Dos objetivos de consulta, y el contraste entre ellos es el experimento:

  VIGENTE   el ultimo valor escrito (v' si la clave se reviso, v si no).
            Un archivo con SOBRESCRITURA deberia resolverlo: guarda solo lo ultimo.
  ANTERIOR  el valor previo de las claves revisadas.
            La sobrescritura da 0 por construccion: lo piso. La GEMACION puede, porque
            deposito la version nueva al lado de la vieja en vez de encima.

Sin el objetivo ANTERIOR, gemacion y sobrescritura son indistinguibles y el experimento no
mide lo que dice medir.
"""
import os, sys
sys.path.insert(0, os.path.expanduser("~/Documentos/Nuevo Transformer/telar-ligamento/src"))

import numpy as np
from datos import V_E001, IGNORE


def gen_cross(rng, B, L, r=None, voc=V_E001):
    """Devuelve dict con las tres secuencias y los dos objetivos.

    x1 (B, 2L+2)      escritura
    x2 (B, 2r+2)      revision
    x3 (B, L+2)       consulta
    y_vig (B, L+2)    target = valor vigente        (IGNORE fuera de las columnas de query)
    y_ant (B, L+2)    target = valor anterior       (solo en claves revisadas; resto IGNORE)
    upd   (B, L)      mascara: esa columna de query corresponde a una clave revisada
    """
    if r is None:
        r = L // 2
    assert L <= voc.NK and r <= L

    keys = np.argsort(rng.random((B, voc.NK)), axis=1)[:, :L]          # L claves distintas
    v1 = rng.integers(0, voc.NV, size=(B, L))
    v2 = (v1[:, :r] + 1 + rng.integers(0, voc.NV - 1, size=(B, r))) % voc.NV   # != v1

    # --- S1: escritura ---
    x1 = np.full((B, 2 * L + 2), voc.PAD, np.int32)
    x1[:, 0] = voc.BOS
    x1[:, 1:2 * L + 1:2] = voc.K0 + keys
    x1[:, 2:2 * L + 2:2] = voc.V0 + v1
    x1[:, -1] = voc.SEP

    # --- S2: revision de las primeras r claves ---
    x2 = np.full((B, 2 * r + 2), voc.PAD, np.int32)
    x2[:, 0] = voc.BOS
    x2[:, 1:2 * r + 1:2] = voc.K0 + keys[:, :r]
    x2[:, 2:2 * r + 2:2] = voc.V0 + v2
    x2[:, -1] = voc.SEP

    # --- S3: consulta, en orden aleatorio ---
    perm = np.argsort(rng.random((B, L)), axis=1)
    qk = np.take_along_axis(keys, perm, axis=1)
    vig = v1.copy(); vig[:, :r] = v2
    ant = np.full((B, L), -1, np.int64); ant[:, :r] = v1[:, :r]        # -1 = no aplica

    x3 = np.full((B, L + 2), voc.PAD, np.int32)
    x3[:, 0] = voc.BOS
    x3[:, 1] = voc.SEP
    x3[:, 2:] = voc.K0 + qk

    y_vig = np.full((B, L + 2), IGNORE, np.int64)
    y_vig[:, 2:] = voc.V0 + np.take_along_axis(vig, perm, axis=1)

    y_ant = np.full((B, L + 2), IGNORE, np.int64)
    a = np.take_along_axis(ant, perm, axis=1)
    y_ant[:, 2:] = np.where(a >= 0, voc.V0 + a, IGNORE)

    upd = np.take_along_axis((np.arange(L)[None, :] < r).repeat(B, 0), perm, axis=1)
    return dict(x1=x1, x2=x2, x3=x3, y_vig=y_vig, y_ant=y_ant, upd=upd,
                azar=1.0 / voc.NV)


def sanidad():
    """Cuatro chequeos de que la tarea es lo que dice ser."""
    rng = np.random.default_rng(0)
    d = gen_cross(rng, 64, 8, 4)
    voc = V_E001
    ok = True

    # 1. formas
    B, L, r = 64, 8, 4
    assert d["x1"].shape == (B, 2 * L + 2), d["x1"].shape
    assert d["x2"].shape == (B, 2 * r + 2), d["x2"].shape
    assert d["x3"].shape == (B, L + 2), d["x3"].shape
    print(f"S1 {d['x1'].shape}  S2 {d['x2'].shape}  S3 {d['x3'].shape}   azar={d['azar']:.4f}")

    # 2. NINGUN valor aparece en S3: si apareciera, la tarea seria copiable sin memoria
    v_tokens = (d["x3"] >= voc.V0) & (d["x3"] < voc.V0 + voc.NV)
    print(f"valores presentes en la secuencia de consulta: {int(v_tokens.sum())}  "
          f"{'OK' if v_tokens.sum() == 0 else 'FALLA — la tarea es copiable'}")
    ok &= v_tokens.sum() == 0

    # 3. el target VIGENTE de una clave revisada != su valor original
    tv = d["y_vig"][:, 2:][d["upd"]] - voc.V0
    ta = d["y_ant"][:, 2:][d["upd"]] - voc.V0
    print(f"revisadas: vigente != anterior en {np.mean(tv != ta)*100:.1f}% "
          f"{'OK' if np.all(tv != ta) else 'FALLA'}")
    ok &= bool(np.all(tv != ta))

    # 4. el target ANTERIOR solo existe donde hubo revision
    tiene_ant = d["y_ant"][:, 2:] != IGNORE
    print(f"columnas con target anterior == revisadas: "
          f"{'OK' if np.array_equal(tiene_ant, d['upd']) else 'FALLA'}")
    ok &= bool(np.array_equal(tiene_ant, d["upd"]))

    print("\nSANIDAD:", "OK" if ok else "FALLA")
    return ok


if __name__ == "__main__":
    sanidad()
