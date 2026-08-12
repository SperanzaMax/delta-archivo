"""¿Cuantos pasos necesita el CONTROL (intra) para resolver la tarea? Sin esto no se puede
dimensionar E-I0: hay que darle a las dos mitades el mismo presupuesto, y el presupuesto lo
fija la mitad que tiene que aprender algo."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ei0_piso as E
for pasos in (500, 1500, 3000):
    m, s = E.entrenar("delta", "intra", 0, pasos=pasos)
    print(f"  intra/delta {pasos:5d} pasos → acc {m:.4f}", flush=True)
    if m > 0.60:
        print(f"  ►► con {pasos} pasos alcanza"); break
