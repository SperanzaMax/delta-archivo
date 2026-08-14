"""COMPUERTA: cuanto se trunca con la config actual, por nivel.

La regla que dejo el 13-ago: mirar `truncados` ANTES de leer cualquier accuracy. Los niveles 1-3 del
dia 1 midieron el padding (33,9 % de enunciados que nunca llegaban al archivo) y no el modelo.
Esto no entrena nada: arma lotes y cuenta. Si alguna celda no da ~0, no se lanza la campaña.
"""
import numpy as np

import datos as DAT

LIMITE = 0.01          # 1 % -- por encima de esto la campania NO sale

if __name__ == "__main__":
    print(f"T_SES={DAT.T_SES}  E_MAX={DAT.E_MAX}  (dia 1: 40 y 4)\n")
    print(f"{'nivel':>6}  {'enunciados':>11}  {'truncados':>10}  {'tasa':>8}  {'largo max':>10}")
    peor = 0.0
    for nivel in (1, 2, 3, 4):
        DAT.reset_truncados()
        rng = np.random.default_rng(7)
        largos = []
        for _ in range(20):                       # 20 lotes x 64 = 1280 episodios por nivel
            ses, *_ = DAT.lote(rng, 64, nivel=nivel, n_hechos=4, n_sesiones=4)
            largos.append(int((ses != DAT.PAD).sum(-1).max()))
        t = DAT.tasa_truncados()
        peor = max(peor, t)
        print(f"{nivel:>6}  {DAT.TRUNC['enunciados']:>11}  {DAT.TRUNC['truncados']:>10}  "
              f"{t:>8.4f}  {max(largos):>10}")
    print(f"\ncompuerta {'ABRE' if peor <= LIMITE else 'NO ABRE'}  "
          f"(peor tasa {peor:.4f} vs limite {LIMITE})")
