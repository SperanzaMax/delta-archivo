"""Las magnitudes de los taps de `convq`. Control de la ablacion: si el daño ordena igual que la
norma, es inespecifico. Y responde el secundario del PREREG_KERNEL_Q5: ¿el modelo USO la ventana?

Control gratis que ya existe: `convq` se instancia en los CUATRO bloques y solo el bloque 0 recibe
gradiente (la lectura de `lat2` vive ahi). Los bloques 1-3 son weight decay puro desde [1,0,...,0].
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import medir_ratio_ce as R

for u in ["v3_s0", "v3_s1", "v3_s2", "kq3_s0", "kq3_s1", "kq3_s2"]:
    p = f"ckpts/{u}.pkl"
    if not os.path.exists(p):
        print(f"(falta {u})"); continue
    params, cfg, paso = R.cargar(p)
    cq0 = np.asarray(params["blocks"][0]["convq"])        # (K, D) con gradiente
    cq1 = np.asarray(params["blocks"][1]["convq"])        # (K, D) decay puro
    n0 = np.abs(cq0).mean(-1)
    n1 = np.abs(cq1).mean(-1)
    print(f"\n{u}  kernel {cfg.get('kernel_q',3)}  paso {paso}")
    print(f"  {'tap':>3s} {'|w| bloque0':>12s} {'|w| decay':>10s} {'razon':>8s}  distancia y token")
    que = {0: "la posicion de lectura", 1: "<ent>  ENTIDAD", 2: "«de»", 3: "<sust>  RELACION",
           4: "<art>  articulo", 5: "«es»", 6: "«cual»"}
    for k in range(cq0.shape[0]):
        r = n0[k] / n1[k] if n1[k] > 1e-12 else float("inf")
        print(f"  {k:3d} {n0[k]:12.6f} {n1[k]:10.6f} {r:8.2f}  d={k}  {que.get(k,'?')}")
