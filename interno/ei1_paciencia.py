"""¿La celda que falla FALLA, o solo tarda? Test obligatorio antes de afirmar nada.

En E-I1 la celda (tardio, topk) se queda en el azar a 1500 pasos. Pero en esta tarea el aprendizaje
es ABRUPTO y TARDIO: las celdas que resuelven tienen la perdida plana 900-1200 pasos y colapsan de
golpe. Cortar a 1500 y concluir "no aprende" puede ser impaciencia disfrazada de mecanismo.

Se le dan 6000 pasos -- cuatro veces el presupuesto -- a la celda que fallo, y a una que funciona
como referencia de que el codigo es el mismo.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ei1_lectura as E

for etiqueta, bloque, sel in (("tardio/topk  (la que fallo)", 3, "topk"),
                              ("tardio/densa (referencia)  ", 3, "densa")):
    for s in (0, 1):
        m = E.entrenar(bloque, sel, s, pasos=6000)
        print(f"  {etiqueta} s{s} · 6000 pasos → acc {m:.4f}", flush=True)
