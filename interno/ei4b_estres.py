"""E-I4b — ESTRESAR el envejecimiento: llevar la deriva a la zona donde R5.1 predice daño.

E-I4 midió edades de 0 a 400 pasos y no encontró degradación: revisadas 0,9956 → 0,9911. Pero el
coseno del marco nunca bajó de **0,9374**, y la curva de tolerancia medida afuera (R5.1) dice que el
daño recién aparece por debajo de **0,7**. O sea que E-I4 midió justo la zona donde la propia teoría
del proyecto predice que no debe pasar nada. Un negativo con el estímulo demasiado débil no prueba
que el mecanismo aguante: prueba que no lo empujamos.

Acá se empuja: entrenamiento más largo y edades mucho mayores, para que el marco tenga tiempo de
moverse de verdad.

    entrenamiento 12000 pasos (la tarea converge cerca de 3000, así que el resto es antigüedad pura)
    edades 0 · 400 · 2000 · 8000

PREDICCIONES, comprometidas antes del dato:
  P-1  (bloqueante, control del instrumento) cos(8000) <= 0,80. Si NO se cumple, el modelo convergido
       simplemente no se mueve lo suficiente ni en 8000 pasos, y entonces **el envejecimiento por
       antigüedad no es la vía**: hay que forzar la deriva con cambio de distribución (lo de R6) y
       este experimento se declara sin poder de resolución en vez de leerse como robustez.
  P-2  si cos(8000) baja de 0,7, la accuracy en revisadas cae al menos 0,10 respecto de edad 0.
       Es la predicción que traslada el umbral de R5.1 —medido con encoder congelado y kNN— al índice
       co-entrenado. **Si cos baja de 0,7 y la accuracy NO cae, es un resultado positivo fuerte y
       nuevo: el índice co-entrenado tolera lo que el no paramétrico no toleraba**, y ahí sí se puede
       hablar de robustez con evidencia.
  P-3  el daño (si aparece) pega primero en las claves revisadas que en las de una sola versión:
       distinguir dos versiones del mismo hecho exige más precisión que distinguir hechos distintos.

Nota: la lectura de este experimento depende de P-1, y por eso P-1 es bloqueante. Sin deriva grande,
no hay nada que interpretar.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ei4_envejecimiento as E4
import ei3_orden as E3

E4.EDADES = (0, 400, 2000, 8000)
E4.PASOS = 12000
E4.SEMILLAS = (0, 1, 2)

if __name__ == "__main__":
    print(f"E-I4b · estres del envejecimiento · edades {E4.EDADES} · {E4.PASOS} pasos · "
          f"{len(E4.SEMILLAS)} semillas\n"
          f"referencia: E-I4 llegó a cos 0,9374 a 400 pasos sin daño · umbral R5.1 = 0,7\n", flush=True)
    acum = {e: [] for e in E4.EDADES}
    for semilla in E4.SEMILLAS:
        params, fotos = E4.entrenar_con_fotos(semilla, pasos=E4.PASOS)
        ev = np.random.default_rng(99000 + semilla)
        for edad in E4.EDADES:
            rs = [E4.evaluar(params, fotos[edad], E3.gen_lote(ev)) for _ in range(8)]
            m = np.mean(rs, axis=0)
            acum[edad].append(m)
            print(f"  s{semilla} edad {edad:5d} → acc {m[0]:.4f} (rev {m[1]:.4f} · "
                  f"no-rev {m[2]:.4f}) · cos {m[3]:.4f}", flush=True)
        json.dump({str(e): np.array(v).tolist() for e, v in acum.items()},
                  open("resultados_ei4b.json", "w"), indent=1)

    print("\n" + "=" * 74)
    res = {}
    for edad in E4.EDADES:
        a = np.array(acum[edad])
        res[edad] = {"rev": a[:, 1].mean(), "no_rev": a[:, 2].mean(), "cos": a[:, 3].mean(),
                     "sd_rev": a[:, 1].std(ddof=1)}
        print(f"  edad {edad:5d} → cos {res[edad]['cos']:.4f} · revisadas {res[edad]['rev']:.4f} "
              f"(sd {res[edad]['sd_rev']:.4f}) · una version {res[edad]['no_rev']:.4f}")
    print("-" * 74)
    cos_max = res[E4.EDADES[-1]]["cos"]
    caida = res[0]["rev"] - res[E4.EDADES[-1]]["rev"]
    print(f"  P-1 (bloqueante) cos(8000) = {cos_max:.4f}  "
          f"{'CUMPLE — hay deriva que interpretar' if cos_max <= 0.80 else 'NO CUMPLE — sin poder de resolucion: forzar deriva por cambio de distribucion'}")
    if cos_max <= 0.70:
        print(f"  P-2 caida en revisadas = {caida:+.4f}  "
              f"{'CUMPLE — el umbral de R5.1 se traslada adentro' if caida >= 0.10 else 'NO CUMPLE — el indice CO-ENTRENADO tolera lo que el no parametrico no toleraba (positivo fuerte)'}")
    else:
        print(f"  P-2 no evaluable: el coseno no bajo de 0,70 (quedo en {cos_max:.4f})")
    print(f"  P-3 rev {caida:+.4f} vs una version "
          f"{res[0]['no_rev'] - res[E4.EDADES[-1]]['no_rev']:+.4f}")
    print("=" * 74)
    json.dump({str(k): {kk: float(vv) for kk, vv in v.items()} for k, v in res.items()},
              open("resumen_ei4b.json", "w"), indent=1)
