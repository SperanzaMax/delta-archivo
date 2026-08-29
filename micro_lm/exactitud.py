"""Exactitud GLOBAL: fraccion de preguntas donde el modelo hizo lo correcto,
sea dar la respuesta buena o decir NOSE cuando de verdad no estaba."""
import sys, os, collections, pickle
sys.path.insert(0, os.getcwd())
import idioma as I
from ser import clasificar
from ser_cobertura import sondear

print(f"{'unidad':9} {'paso':>6} {'EXACTITUD':>10} {'acierto':>8} {'noseOK':>8} {'diceNOSE':>9} {'inventa':>8}")
for u in sys.argv[1:]:
    ruta = f"ckpts/{u}.pkl"
    if not os.path.exists(ruta):
        continue
    paso = pickle.load(open(ruta,'rb')).get('paso')
    sc, pv, tg, mt, cfg = sondear(ruta, 2000, 64, None, None, 54321)
    c = collections.Counter()
    for i in range(len(sc)):
        tok = I.ITOS[int(pv[i])] if sc[i] <= 0.0 else "NOSE"
        c[clasificar(tok, I.ITOS[int(tg[i])], mt[i])] += 1
    n = len(sc)
    bien = c["acierto"] + c["acierto_nose"]
    dice_nose = (c["acierto_nose"] + c["abstencion"]) / n
    print(f"{u:9} {paso:>6} {bien/n:10.4f} {c['acierto']/n:8.4f} "
          f"{c['acierto_nose']/n:8.4f} {dice_nose:9.4f} {c['invento']/n:8.4f}")
