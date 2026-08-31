"""¿El modelo TIENE la senal de ausencia, aunque su decision no la use?

Es LA pregunta que decide si estamos cerca o lejos de la abstencion calibrada:

  * si la senal ESTA (AUC alto) y la decision no la usa (acuerdo 0,50), falta el ACOPLE
    -> es lo que la campania de hoy ataca, y estamos cerca.
  * si la senal NO esta (AUC ~0,50), el modelo ni siquiera detecta la ausencia
    -> falta una capacidad, no un acople, y estamos lejos.

Se mide el AUC de separar «no hay respuesta» usando SOLO lo que el modelo ya calcula, sin entrenar
nada nuevo. Tres estadisticos, de menos a mas exigente.
"""
import sys
sys.path.insert(0, "/home/maxi/Documentos/Nuevo Transformer/delta-archivo/micro_lm")
import numpy as np
import entrenar as E, medir_ratio_ce as R

def auc(score, pos):
    """AUC por rangos. pos = booleano de la clase positiva (aca: NO hay respuesta)."""
    o = np.argsort(score); r = np.empty(len(score)); r[o] = np.arange(1, len(score) + 1)
    n1, n0 = pos.sum(), (~pos).sum()
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else float("nan")

for ruta in ("ckpts/t03_s3.pkl", "ckpts/n3_s0.pkl", "ckpts/b3_s3.pkl"):
    params, cfg, paso = R.cargar(ruta)
    lg, tgt = R.logits(params, cfg, 4096, 64, 54321, 0.4)
    lg = np.asarray(lg, dtype=np.float64); tgt = np.asarray(tgt)
    lg_v = lg.copy(); lg_v[:, E.NOSE] = -1e9
    p = np.exp(lg_v - lg_v.max(-1, keepdims=True)); p /= p.sum(-1, keepdims=True)
    hay = tgt != E.NOSE; no = ~hay

    mx    = p.max(-1)                                    # confianza en el mejor candidato
    ent   = -(p * np.log(np.maximum(p, 1e-30))).sum(-1)   # entropia sobre los valores
    marg  = np.sort(p, -1)[:, -1] - np.sort(p, -1)[:, -2]  # margen 1o contra 2o
    pall  = np.exp(lg - lg.max(-1, keepdims=True)); pall /= pall.sum(-1, keepdims=True)
    q     = pall[:, E.NOSE]

    print(f"\n--- {ruta}  paso={paso}  (sin respuesta: {no.mean():.4f}) ---")
    print(f"  AUC con la ENTROPIA sobre los valores   {auc(ent, no):.4f}")
    print(f"  AUC con 1 - max prob                    {auc(-mx, no):.4f}")
    print(f"  AUC con 1 - margen                      {auc(-marg, no):.4f}")
    print(f"  AUC con la propia q del modelo          {auc(q, no):.4f}   <- lo que DECIDE")
    mejor = max(auc(ent, no), auc(-mx, no), auc(-marg, no))
    print(f"  ** la senal MEJOR disponible: {mejor:.4f}   ·   la que usa: {auc(q, no):.4f}   "
          f"·  brecha {mejor - auc(q, no):+.4f} **")
