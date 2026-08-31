"""Sonda lineal por SOLUCION CERRADA (ridge). Sin lr, sin pasos, sin convergencia que verificar.

La v1 no servia: daba AUC 0,56 EN ENTRENAMIENTO con 242 features sobre 3072 muestras, o sea el
optimizador no habia llegado a ningun lado, y su nulo daba 0,5505 en vez de 0,50. Un veredicto tan
fuerte como «la senal no esta» no se puede dar con un instrumento que no converge (leccion del
13-ago: un negativo sin barrido de lr no es un negativo).

Ridge tiene solucion exacta  w = (X'X + lam I)^-1 X'y  y por eso no hay nada que verificar.
Se barre lam en cuatro ordenes de magnitud y se reporta el MEJOR held-out, que es la cota superior
honesta de lo que un lector lineal puede sacar del estado.

Controles, los dos obligatorios:
  * NULO con etiquetas barajadas -> tiene que dar ~0,50 held-out
  * TECHO con una etiqueta que SI esta en los logits por construccion (¿el argmax es un NOMBRE?)
    -> tiene que dar ~1,00. Si no, la sonda no sirve y no se concluye nada.
"""
import sys
sys.path.insert(0, "/home/maxi/Documentos/Nuevo Transformer/delta-archivo/micro_lm")
import numpy as np
import entrenar as E, idioma as I, medir_ratio_ce as R

def auc(s, pos):
    o = np.argsort(s); r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1)
    n1, n0 = pos.sum(), (~pos).sum()
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else float("nan")

def ridge_auc(Xtr, ytr, Xte, yte, lams=(1e-2, 1e0, 1e2, 1e4)):
    G = Xtr.T @ Xtr; b = Xtr.T @ ytr
    mejor = (-1, None)
    for lam in lams:
        w = np.linalg.solve(G + lam * np.eye(G.shape[0]), b)
        a = auc(Xte @ w, yte)
        if a > mejor[0]: mejor = (a, lam)
    return mejor

ids_nom = np.array([I.STOI[t] for t in I.NOMBRES])

for ruta in ("ckpts/t03_s3.pkl", "ckpts/n3_s0.pkl"):
    params, cfg, paso = R.cargar(ruta)
    lg, tgt = R.logits(params, cfg, 6144, 64, 54321, 0.4)
    lg = np.asarray(lg, dtype=np.float64); tgt = np.asarray(tgt)
    no = (tgt == E.NOSE)

    n = len(lg); idx = np.random.default_rng(0).permutation(n); tr, te = idx[:n//2], idx[n//2:]
    mu, sd = lg[tr].mean(0), lg[tr].std(0) + 1e-6
    X = np.hstack([(lg - mu) / sd, np.ones((n, 1))])
    Xtr, Xte = X[tr], X[te]

    a_señal, lam1 = ridge_auc(Xtr, no[tr].astype(float), Xte, no[te])
    yb = np.random.default_rng(1).permutation(no[tr].astype(float))
    a_nulo, _ = ridge_auc(Xtr, yb, Xte, no[te])
    # TECHO: una etiqueta que esta en los logits por construccion
    lg_v = lg.copy(); lg_v[:, E.NOSE] = -1e9
    es_nom = np.isin(lg_v.argmax(-1), ids_nom)
    a_techo, _ = ridge_auc(Xtr, es_nom[tr].astype(float), Xte, es_nom[te])

    print(f"\n--- {ruta}  paso={paso}  (n held-out {len(te)}) ---")
    print(f"  TECHO  «¿el argmax es un nombre?»   {a_techo:.4f}   <- si no da ~1, la sonda no sirve")
    print(f"  SENAL  «¿NO hay respuesta?»         {a_señal:.4f}   (lambda {lam1:g})")
    print(f"  NULO   etiquetas barajadas          {a_nulo:.4f}   <- tiene que dar ~0,50")
    ok = a_techo > 0.90 and abs(a_nulo - 0.5) < 0.05
    print(f"  [{'OK   ' if ok else 'FALLA'}] la sonda es confiable")
    if ok:
        v = ("LA SENAL ESTA -> falta el ACOPLE" if a_señal > 0.75 else
             "senal PARCIAL -> acople + mejora de deteccion" if a_señal > 0.60 else
             "LA SENAL NO ESTA -> falta DETECTAR la ausencia")
        print(f"  ** {v} **")
