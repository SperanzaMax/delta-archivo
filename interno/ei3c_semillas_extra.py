"""E-I3c · SEMILLAS EXTRA, porque el resultado salio apretado.

La corrida de 3 semillas dio ANTERIOR(sello) = 0,8316 con sd 0,2399: 0,9766 · 0,9635 · 0,5547. La
media queda apenas arriba del umbral de 0,80 que fija P-2, sostenida por dos semillas buenas y una
mala. El prereg de E-I3c dice, textual: "si el resultado sale apretado, se agregan semillas antes de
escribir nada". Esto es eso.

Y la semilla mala tiene diagnostico, hecho ANTES de correr esto: su curva va retrasada, no plana.

    s2   paso 8000 ANT 0,3125 · 10000 ANT 0,2500 · 12000 ANT 0,5625   (vig llega a 1,000 al final)
    s0   paso 6000 ANT 0,7083 ·  8000 ANT 0,9583 · 12000 ANT 1,0000

La s2 a 12000 esta donde la s0 estaba a ~5000, y su ultimo tramo sube. Eso es convergencia parcial,
no meseta. Se prueba directamente dandole el doble de presupuesto.

QUE SE CORRE:
  - semillas 3 y 4 de `sello` a 12000 pasos, para llevar la condicion a 5 semillas como el resto del
    brazo interno;
  - la semilla 2 a 24000 pasos, para separar "corte prematuro" de "inestabilidad del mecanismo".

CRITERIO, fijado antes de mirar:
  - si s2 a 24000 sube por encima de 0,90 -> el 0,5547 era corte prematuro; se reporta la media de
    las 5 semillas a 12000 Y se declara que una semilla necesito el doble de presupuesto.
  - si s2 a 24000 se queda debajo de 0,70 -> es inestabilidad real entre semillas, y ESO es el
    resultado: el mecanismo funciona pero no siempre converge, lo que hay que reportar como tal en
    vez de esconderlo en un promedio.
  - la media de las 5 semillas a 12000 es el numero que va al informe; el de 24000 va como nota.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ei3c_orden_limpio as C

salida = {}

print("E-I3c · semillas extra de `sello`\n", flush=True)
for s in (3, 4):
    r = C.entrenar("sello", s, pasos=12000)
    salida[f"s{s}_12000"] = [float(v) for v in r]
    print(f"  sello s{s} (12000) → vig {r[1]:.4f} · ANT {r[2]:.4f} · una {r[3]:.4f}", flush=True)
    json.dump(salida, open("resultados_ei3c_extra.json", "w"), indent=1)

print("\n  ahora la semilla 2 con el doble de presupuesto\n", flush=True)
r2 = C.entrenar("sello", 2, pasos=24000)
salida["s2_24000"] = [float(v) for v in r2]
print(f"  sello s2 (24000) → vig {r2[1]:.4f} · ANT {r2[2]:.4f} · una {r2[3]:.4f}", flush=True)
json.dump(salida, open("resultados_ei3c_extra.json", "w"), indent=1)

previas = {0: 0.9766, 1: 0.9635, 2: 0.5547}
cinco = list(previas.values()) + [salida["s3_12000"][2], salida["s4_12000"][2]]
m, sd = float(np.mean(cinco)), float(np.std(cinco, ddof=1))
ant2 = salida["s2_24000"][2]

print("\n" + "=" * 74)
print(f"  ANTERIOR(sello), 5 semillas a 12000 pasos: {m:.4f} (sd {sd:.4f})")
print(f"  por semilla: {[round(v, 4) for v in cinco]}")
print(f"  semilla 2 con 24000 pasos: {ant2:.4f} (era {previas[2]:.4f})")
print("-" * 74)
if ant2 > 0.90:
    print("  → el 0,5547 era CORTE PREMATURO. Se reporta la media de 5 semillas y se declara que")
    print("    una semilla necesito el doble de presupuesto para converger.")
elif ant2 < 0.70:
    print("  → INESTABILIDAD REAL entre semillas: el mecanismo funciona pero no siempre converge.")
    print("    Eso es el resultado y va reportado como tal, no promediado.")
else:
    print(f"  → zona gris ({ant2:.4f}): ni corte prematuro claro ni meseta clara. Hace falta mas")
    print("    presupuesto o mas semillas antes de afirmar nada.")
print("=" * 74)
