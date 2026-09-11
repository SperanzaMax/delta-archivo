"""EMPAQUETA LAS FOTOS DE UNA CORRIDA PARA EL VISOR DEL ARTEFACTO · 2026-09-11

`entrenar.py --fotos` deja `ckpts/<u>_fotos.json` con un cuadro cada N pasos: 26 submatrices
fijas de 12x12, la lectura del archivo sobre una muestra fija, taps, beta, perdida y exactitud.
Un cuadro son ~28 KB en JSON y una corrida de 8.000 pasos son 320 cuadros: 9 MB por unidad, que
no entra dos veces en un artefacto de 16 MB. Aca los pesos se CUANTIZAN a int8 respecto del maximo
absoluto de cada matriz a lo largo de TODA la pelicula (escala fija: el movimiento que se ve es el
del aprendizaje, no el de una renormalizacion por cuadro) y viajan como un blob base64:
26 x 144 = 3.744 bytes por cuadro, 1,2 MB por unidad. Lo demas queda en JSON chico.

    python3 armar_pelicula.py ckpts/ts3_s0_fotos.json ckpts/ds3_s0_fotos.json > pelicula.json
"""
import base64, json, sys
import numpy as np


def empaquetar(ruta):
    d = json.load(open(ruta, encoding="utf-8"))
    cuadros = d["cuadros"]
    # 11-sep, 09:20: las VMs reusadas para el plan B tenian todavia el `ck_fotos.json` de la corrida
    # en frio (tp3/dp3, cada 100 pasos), y `entrenar.py --fotos` lo continuo. La corrida propia
    # empieza en el ULTIMO cuadro cuyo paso es igual a `fotos`; lo anterior es de otra corrida.
    cada = (d.get("config") or {}).get("fotos")
    if cada:
        inicios = [i for i, c in enumerate(cuadros) if c["paso"] == cada]
        if inicios:
            cuadros = cuadros[inicios[-1]:]
    nombres = [c["nombre"] for c in d["capas"]]
    n, m = len(cuadros), len(nombres)
    W = np.zeros((n, m, 144), np.float32)
    for i, c in enumerate(cuadros):
        for j, nom in enumerate(nombres):
            v = c["pesos"][nom]
            W[i, j, :len(v)] = v
    escala = np.abs(W).max(axis=(0, 2)) + 1e-9                     # (m,) fija por matriz
    Q = np.clip(np.round(W / escala[None, :, None] * 127), -127, 127).astype(np.int8)
    meta = []
    for c in cuadros:
        meta.append({k: c.get(k) for k in ("paso", "perdida", "acc", "topk", "pred", "abst",
                                              "masa_relleno", "top_relleno", "masa_slot")}
                    | {"atencion": [round(x, 4) for x in c["atencion"]],
                       "taps": c["taps"], "beta": c["beta"]})
    # rms por matriz y cuadro, aparte (una fila por cuadro), para la curva de deriva
    rms = np.sqrt((W ** 2).mean(axis=2))                             # (n, m)
    cfg = d.get("config", {})
    return {
        "unidad": ruta.split("/")[-1].replace("_fotos.json", ""),
        "config": {k: cfg.get(k) for k in ("topk", "topk_desde", "relleno", "relleno_dist",
                                            "relleno_turnos", "fotos", "pasos", "sello", "pert",
                                            "kernel_q", "donde", "abst", "d", "capas", "semilla")},
        "params": d.get("params"), "k": d.get("k", 12),
        "capas": d["capas"], "muestra": d["muestra"], "respuesta_correcta": d["respuesta_correcta"],
        "escala": [round(float(x), 5) for x in escala],
        # 11-sep, pedido de Maxi al ver el bloque 1 -> bloque 2 casi sin lineas: `b2.wv` tiene UN peso
        # de 1,37 (el mayor de todas) y la mediana mas alta de todas; con el umbral en el 25 % del
        # maximo se dibujaba el 6 %. El visor corta por percentil de la matriz a lo largo de la
        # pelicula (p60) y normaliza el grosor por el p97, asi un peso dominante no apaga al resto.
        "p60": [round(float(np.percentile(np.abs(W[:, j]), 60)), 5) for j in range(m)],
        "p97": [round(float(np.percentile(np.abs(W[:, j]), 97)), 5) for j in range(m)],
        "rms": [[round(float(x), 4) for x in fila] for fila in rms],
        "cuadros": meta, "n": n, "m": m,
        "blob": base64.b64encode(Q.tobytes()).decode("ascii"),
    }


if __name__ == "__main__":
    salida = {"unidades": [empaquetar(r) for r in sys.argv[1:]]}
    json.dump(salida, sys.stdout, separators=(",", ":"), ensure_ascii=False)
