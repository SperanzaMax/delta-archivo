"""¿El termino de orden ORDENO de verdad? AUC del logit de NOSE contra la ausencia.

El juez automatico dijo «se cierra la linea» y sus propios numeros lo desmienten: con abstencion
0,0000 el acuerdo da 0,6003 contra un azar de 0,6003 y la pureza 1,0000 contra un nulo de 1,0000,
o sea los tres criterios son aritmetica de un modelo que no se calla nunca, no una medicion.

Lo que SI cambio: el termino de orden paso de 11,2053 (control) a 0,6295 y 0,6341, cruzando por
debajo de log 2 = 0,6931. Ordenar y calibrar el umbral son DOS cosas, y el termino solo pedia la
primera («toda constante da el mismo valor»).

Esto mide la primera, que es la que decide el veredicto:

  * AUC alto  -> el termino RESOLVIO la deteccion y solo falta mover el umbral, que es trivial.
                 El negativo seria de CALIBRACION, no de la via.
  * AUC ~0,5  -> no ordeno nada util y el 0,63 es otra cosa.

Y se reporta la exactitud que se lograria con el MEJOR umbral, que es la cota de lo que la via da.
"""
import sys
sys.path.insert(0, "/home/maxi/Documentos/Nuevo Transformer/delta-archivo/micro_lm")
import numpy as np
import entrenar as E, medir_ratio_ce as R

def auc(s, pos):
    o = np.argsort(s); r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1)
    n1, n0 = pos.sum(), (~pos).sum()
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else float("nan")

PISO = 0.4065
for ruta in ("ckpts/t03_s3.pkl", "ckpts/r03_s3.pkl", "ckpts/r03_s6.pkl"):
    params, cfg, paso = R.cargar(ruta)
    lg, tgt = R.logits(params, cfg, 4096, 64, 54321, 0.4)
    lg = np.asarray(lg, dtype=np.float64); tgt = np.asarray(tgt)
    lg_v = lg.copy(); lg_v[:, E.NOSE] = -1e9
    hay = tgt != E.NOSE; no = ~hay
    arg = lg_v.argmax(-1)
    # s = el logit de NOSE contra el resto, que es lo que el termino de orden entrena
    s = lg[:, E.NOSE] - (lg_v.max(-1) + np.log(np.exp(lg_v - lg_v.max(-1, keepdims=True)).sum(-1)))
    p = np.exp(lg - lg.max(-1, keepdims=True)); p /= p.sum(-1, keepdims=True)
    q = p[:, E.NOSE]

    a = auc(s, no)
    # exactitud con el MEJOR umbral posible sobre s (cota superior de lo que da esta via)
    ordenes = np.argsort(s)
    mejor, u_mejor = 0.0, None
    for k in range(0, len(s) + 1, max(1, len(s) // 400)):
        calla = np.zeros(len(s), bool)
        calla[ordenes[len(s) - k:]] = True
        ex = float(((calla & no) | (~calla & hay & (arg == tgt))).mean())
        if ex > mejor: mejor, u_mejor = ex, k / len(s)
    ex_actual = float(((q > 0.5) & no | (~(q > 0.5) & hay & (arg == tgt))).mean())

    print(f"\n--- {ruta}  paso={paso}  rec_rank={cfg.get('rec_rank')} ---")
    print(f"  abstencion real                 {(q > 0.5).mean():.4f}")
    print(f"  ** AUC del logit de NOSE vs la ausencia   {a:.4f} **")
    print(f"  exactitud con el umbral actual  {ex_actual:.4f}   (piso {PISO})")
    print(f"  exactitud con el MEJOR umbral   {mejor:.4f}   (abstiniendo el {u_mejor:.2%})")
    print(f"  -> {'SUPERA' if mejor > PISO else 'NO supera'} el piso trivial recalibrando solo el umbral")
