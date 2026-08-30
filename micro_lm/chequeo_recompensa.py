"""Compuerta W-0 de `PREREG_RECOMPENSA.md` (SHA f1f7bb66). BLOQUEANTE.

    python chequeo_recompensa.py

Verifica en segundos, sin GPU y sin entrenar, que la recompensa esperada hace lo que el §2 del
pre-registro dice que hace. Si algo falla, la campaña no se lanza.

  R-1  el gradiente en q tiene el SIGNO que predice la fórmula derivada, en todo el rango de c.
  R-2  con los pesos elegidos (L=M=0,5, F=1,5) «abstenerse de todo» NO es el óptimo, ni siquiera con
       un modelo que no sabe nada (c=0). Es la razón de ser de los pesos.
  R-3  el punto donde conviene contestar coincide con el umbral analítico c* = (L+M)/(1+M)... no:
       se verifica contra la raíz de dE/dq = 0 calculada numéricamente, sin suponer la fórmula.
  R-4  la pérdida es finita y derivable en los bordes (q=0, q=1, c=0, c=1), que es donde un
       `log` mal puesto explotaría.
  R-5  el cableado: el flag existe, llega a la global, y las DOS interfaces (token y cabeza) usan la
       misma función. Es el bug de `--abst slot` del 24-ago, que ya mordió una vez.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import jax, jax.numpy as jnp

FALLAS = []
PI = 0.4065


def check(nombre, ok, detalle=""):
    print(f"  [{'OK ' if ok else 'FALLA'}] {nombre}   {detalle}")
    if not ok:
        FALLAS.append(nombre)


import entrenar as E
L, M, F = E._REC_L, E._REC_M, E._REC_F
print(f"\npesos del pre-registro: L={L}  M={M}  F={F}   ·   pi={PI}\n")


def E_R(q, c):
    """Recompensa esperada por muestra, promediando los dos tipos de pregunta."""
    r_hay = q * (-F) + (1 - q) * (c - (1 - c) * M)
    r_no = q * L + (1 - q) * (-M)
    return (1 - PI) * r_hay + PI * r_no


def dEdq(q, c):
    return (1 - PI) * (-F - c + (1 - c) * M) + PI * (L + M)


print("R-1 · el gradiente en q tiene el signo que predice la formula")
malos = []
for c in np.linspace(0, 1, 21):
    num = (E_R(0.5 + 1e-5, c) - E_R(0.5 - 1e-5, c)) / 2e-5
    if abs(num - dEdq(0.5, c)) > 1e-4:
        malos.append(c)
check("la derivada numerica coincide con la analitica", not malos,
      f"maximo desvio en {len(malos)} de 21 puntos" if malos else "en los 21 puntos de c")

print("\nR-2 · con estos pesos, abstenerse de todo NO es el optimo (ni con c=0)")
for c in (0.0, 0.1, 0.35):
    qs = np.linspace(0, 1, 1001)
    v = E_R(qs, c)
    q_opt = qs[int(np.argmax(v))]
    check(f"c={c}: el optimo no es q=1", q_opt < 0.999,
          f"q optimo = {q_opt:.3f}, E[R] en q=1 vale {E_R(1.0, c):+.4f} vs {v.max():+.4f} en el optimo")

print("\n  y el minimo de F que la formula del prereg exige, para contrastar:")
for c in (0.0, 0.1, 0.35):
    f_min = (PI * (L + M) + (1 - PI) * ((1 - c) * M - c)) / (1 - PI)
    check(f"c={c}: F={F} supera el minimo {f_min:.3f}", F > f_min,
          f"margen {100*(F-f_min)/f_min:.0f} %")

print("\nR-3 · el c a partir del cual conviene contestar, resuelto numericamente")
raiz = None
for c in np.linspace(0, 1, 100001):
    if dEdq(0.0, c) < 0:
        raiz = c
        break
check("existe un c donde deja de convenir callarse", raiz is not None,
      f"c* = {raiz:.5f}" if raiz is not None else "no existe en [0,1]")
check("y ese c* es 0, o sea vale desde el arranque", raiz is not None and raiz < 1e-4,
      f"c* = {raiz:.5f}")

print("\nR-4 · finito y derivable en los bordes")
bordes = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0), (0.5, 0.5)]
ok = all(np.isfinite(E_R(q, c)) and np.isfinite(dEdq(q, c)) for q, c in bordes)
check("E[R] y su derivada son finitas en los 5 bordes", ok)

lg = jnp.array(np.random.default_rng(0).normal(size=(8, 242)) * 3, dtype=jnp.float32)
tgt = jnp.array([E.NOSE, 5, 7, E.NOSE, 11, 13, 17, E.NOSE])
for nom, q in (("token", jax.nn.softmax(lg, -1)[:, E.NOSE]),
               ("cabeza", jax.nn.sigmoid(jnp.array(np.random.default_rng(1).normal(size=8))))):
    val, acc = E._recompensa(lg, tgt, q)
    g = jax.grad(lambda x: E._recompensa(x, tgt, q)[0])(lg)
    check(f"_recompensa con interfaz {nom} da valor y gradiente finitos",
          bool(np.isfinite(float(val))) and bool(np.all(np.isfinite(np.asarray(g)))),
          f"perdida {float(val):+.4f}, |grad| max {float(np.abs(np.asarray(g)).max()):.3e}")

print("\nR-6 · el optimo POR MUESTRA, que es lo que el modelo puede elegir de verdad")
# R-2 mira un q GLOBAL unico, y con estos pesos su optimo es 0. Eso NO es el fracaso que parece:
# el modelo elige q por muestra, no uno solo para todas. Lo que hay que verificar es que el optimo
# por muestra sea el comportamiento deseado, y ese es este chequeo.
for c in (0.0, 0.5, 0.9):
    contestar, callarse = c - (1 - c) * M, -F
    check(f"con respuesta y c={c}: conviene CONTESTAR", contestar > callarse,
          f"{contestar:+.3f} vs {callarse:+.3f}")
check("sin respuesta: conviene CALLARSE", L > -M, f"{L:+.3f} vs {-M:+.3f}")
umbral = (M - F) / (1 + M)
check("el umbral c* es negativo, o sea conviene contestar desde el arranque", umbral < 0,
      f"c* = {umbral:+.3f}")
print(f"  -> optimo por muestra = contestar donde hay respuesta, callarse donde no. El deseado.")
print(f"  -> PRECIO, y define que fracaso vigilar: con F={F} > M={M}, callarse teniendo la")
print(f"     respuesta ({-F:+.1f}) es peor que errar ({-M:+.1f}). Si el modelo no logra distinguir,")
print(f"     su mejor politica es contestar TODO. El riesgo de esta campania es la locuacidad,")
print(f"     no la mudez. Es W-6, y W-4 (invento < 0,05) es lo que lo detecta.")

print("\nR-5 · el cableado (el bug de --abst slot del 24-ago)")
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "entrenar.py")).read()
check("el flag acepta 'recompensa'", '"recompensa"' in src and 'choices=("bce", "balance", "ranking", "recompensa")' in src)
check("la global se declara", "global _DONDE, _ABST, _BLANCO, _PERDIDA_CABEZA" in src)
check("se asigna desde el argparse", "_PERDIDA_CABEZA = a.perdida_cabeza" in src)
check("la rama TOKEN la usa (perdida, sin cabeza)",
      'if _PERDIDA_CABEZA == "recompensa":\n        return _recompensa(lg, tgt, q=jax.nn.softmax(lg, -1)[:, NOSE])' in src)
check("la rama CABEZA la usa (perdida_cabeza)",
      "return _recompensa(lg, tgt, q=jax.nn.sigmoid(a))" in src)
check("las dos interfaces llaman a la MISMA funcion", src.count("_recompensa(lg, tgt, q=") == 2)

print()
if FALLAS:
    print(f"compuerta W-0 CERRADA · fallan {len(FALLAS)}: {', '.join(FALLAS)}")
    sys.exit(1)
print("compuerta W-0 ABRE · la recompensa hace lo que el pre-registro dice, y el flag llega")
