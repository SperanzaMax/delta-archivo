"""Antes de llamar RESULTADO al azar de E-I2, descartar las dos explicaciones aburridas.

(a) PACIENCIA. E-I1 ya nos enseno que en esta familia el aprendizaje es abrupto y tardio, y que
    cortar temprano fabrica negativos falsos. 20.000 pasos = 5x el presupuesto de E-I2.
(b) TASA DE APRENDIZAJE. 3e-3 viene del harness de Ligamento, calibrado para OTRA tarea. Si el
    problema es el paso y no la capacidad, se ve enseguida.

Solo si las dos fallan, el azar de E-I2 significa algo sobre el modelo.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ei2_consulta as E

print("=== (b) tasa de aprendizaje, 4000 pasos, bloque 0 ===", flush=True)
for lr in (1e-3, 1e-2):
    E.LR = lr
    r = E.entrenar(0, 0, pasos=4000)
    print(f"  lr={lr:g} → acc {r[0]:.4f} (rev {r[1]:.4f} · no-rev {r[2]:.4f})", flush=True)

print("\n=== (a) paciencia, 20000 pasos, lr por defecto ===", flush=True)
E.LR = 3e-3
for s in (0, 1):
    r = E.entrenar(0, s, pasos=20000)
    print(f"  s{s} · 20000 pasos → acc {r[0]:.4f} (rev {r[1]:.4f} · no-rev {r[2]:.4f})", flush=True)
