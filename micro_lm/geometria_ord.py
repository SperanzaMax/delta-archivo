"""¿La tabla `ord` MUESTRA la bandera? · 2026-09-06

`PREREG_GEOMETRIA_ORD.md`, SHA `bedac0b5`.

Esta manana quedo medido que el sello de orden SE COMPORTA como una bandera y no como un reloj
(`INFORME_RELOJ_O_BANDERA_20260906.md`). Esto mira la contraparte de peso: `ord` son 64 vectores,
uno por turno, y si el modelo aprendio una particion de filas, la particion tiene que verse.

Mide, por checkpoint:
  · contraste de bloque   similitud media dentro de 0-23, dentro de 24-63, y entre bloques
  · estructura de orden   correlacion entre cos(ord[i],ord[j]) y |i-j|, por bloque
  · rampa                 correlacion de la 1a componente principal contra el indice de turno
  · normas                cuanta senal lleva cada fila

Uso:
    python3 geometria_ord.py ckpts/lg3_s0.pkl ckpts/lc3_s0.pkl --salida sal.json
"""
import argparse, json, pickle

import numpy as np

UMBRAL = 24          # la frontera que el entrenamiento vio SIEMPRE en el mismo lugar


def analizar(ord_tab):
    """`ord_tab` es (N_TURNOS, D). Devuelve el dict de metricas del prereg."""
    N = ord_tab.shape[0]
    normas = np.linalg.norm(ord_tab, axis=1)
    # filas de norma ~0 no tienen direccion: se excluyen de los cosenos para no meter ruido
    viva = normas > 1e-8
    u = ord_tab / np.maximum(normas, 1e-12)[:, None]
    C = u @ u.T

    bajo = np.array([i for i in range(N) if i < UMBRAL and viva[i]])
    alto = np.array([i for i in range(N) if i >= UMBRAL and viva[i]])

    def media_par(ia, ib, mismo):
        if len(ia) == 0 or len(ib) == 0:
            return float("nan")
        sub = C[np.ix_(ia, ib)]
        if mismo:                                   # fuera de la diagonal
            m = ~np.eye(len(ia), dtype=bool)
            return float(sub[m].mean())
        return float(sub.mean())

    d_bajo = media_par(bajo, bajo, True)
    d_alto = media_par(alto, alto, True)
    entre = media_par(bajo, alto, False)
    brecha = min(d_bajo, d_alto) - entre           # G-1: cuanto mas parecidas son dentro que entre

    def corr_distancia(idx):
        """G-2: un reloj hace que turnos cercanos se parezcan -> correlacion NEGATIVA fuerte."""
        if len(idx) < 4:
            return float("nan")
        xs, ys = [], []
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                xs.append(abs(idx[a] - idx[b])); ys.append(C[idx[a], idx[b]])
        if np.std(xs) < 1e-12 or np.std(ys) < 1e-12:
            return float("nan")
        return float(np.corrcoef(xs, ys)[0, 1])

    # G-4: la 1a componente principal, proyectada contra el indice de turno
    #
    # 2026-09-08 · EL SIGNO DE `Vt[0]` ES ARBITRARIO. `np.linalg.svd` no fija ninguna convencion:
    # `Vt[0]` y `-Vt[0]` son la misma componente y cual sale depende de detalles numericos de
    # LAPACK. La primera medicion de las seis unidades del sello relativo dio rampa POSITIVA en las
    # tres `rp` y NEGATIVA en las tres `rr`, y eso se leia como que el bit de pertenencia invierte
    # el gradiente de recencia. Es un ARTEFACTO: en valor absoluto las seis dan 0,091-0,133, o sea
    # lo mismo. Se fija el signo de forma determinista y ADEMAS se devuelve |rampa|, que es lo unico
    # que se puede comparar entre checkpoints.
    #
    # La convencion no puede depender del indice de turno, que es justo la variable de interes:
    # eso haria la rampa positiva por construccion. Se fija por la componente de mayor magnitud,
    # que es la convencion estandar y es ciega a lo que se mide.
    idx_viva = np.where(viva)[0]
    X = ord_tab[idx_viva] - ord_tab[idx_viva].mean(0)
    if len(idx_viva) >= 3:
        _, _, Vt = np.linalg.svd(X, full_matrices=False)
        pc1 = Vt[0]
        pc1 = pc1 * np.sign(pc1[np.argmax(np.abs(pc1))])
        proy = X @ pc1
        rampa = float(np.corrcoef(idx_viva, proy)[0, 1]) if np.std(proy) > 1e-12 else float("nan")
        # cuanta varianza explica esa 1a componente
        s = np.linalg.svd(X, compute_uv=False)
        var1 = float((s[0] ** 2) / (s ** 2).sum())
    else:
        rampa, var1 = float("nan"), float("nan")

    return {
        "n_filas": int(N), "n_vivas": int(viva.sum()),
        "cos_dentro_bajo": d_bajo, "cos_dentro_alto": d_alto, "cos_entre": entre,
        "brecha_G1": float(brecha),
        "corr_dist_bajo_G2": corr_distancia(bajo), "corr_dist_alto": corr_distancia(alto),
        "rampa_pc1_G4": rampa, "rampa_abs_G4": abs(rampa), "var_pc1": var1,
        "norma_media_bajo": float(normas[:UMBRAL].mean()),
        "norma_media_alto": float(normas[UMBRAL:].mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", nargs="+")
    ap.add_argument("--salida", default="")
    a = ap.parse_args()

    todo = {"prereg": "bedac0b5", "umbral": UMBRAL, "unidades": {}}
    print(f"{'unidad':>14} {'d_bajo':>8} {'d_alto':>8} {'entre':>8} {'brechaG1':>9} "
          f"{'r_distG2':>9} {'rampaG4':>8} {'varPC1':>7} {'|n|bajo':>8} {'|n|alto':>8}")
    for ruta in a.ckpt:
        b = pickle.load(open(ruta, "rb"))
        r = analizar(np.asarray(b["params"]["arch"]["ord"], dtype=np.float64))
        r["ses_extra_entrenado"] = b["config"].get("ses_extra", 0)
        r["pasos"] = b["config"].get("pasos")
        todo["unidades"][ruta] = r
        nom = ruta.split("/")[-1].replace(".pkl", "")
        print(f"{nom:>14} {r['cos_dentro_bajo']:8.4f} {r['cos_dentro_alto']:8.4f} "
              f"{r['cos_entre']:8.4f} {r['brecha_G1']:9.4f} {r['corr_dist_bajo_G2']:9.4f} "
              f"{r['rampa_abs_G4']:8.4f} {r['var_pc1']:7.4f} {r['norma_media_bajo']:8.4f} "
              f"{r['norma_media_alto']:8.4f}")
    if a.salida:
        json.dump(todo, open(a.salida, "w"), indent=1)
        print(f"\n-> {a.salida}")


if __name__ == "__main__":
    main()
