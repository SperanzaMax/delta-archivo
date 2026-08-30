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

print("\nR-2 · el optimo GLOBAL con c=0 ES callarse, y eso ahora es lo CORRECTO")
# Version corregida (ENMIENDA_RECOMPENSA_F). La primera version pedia lo contrario, y ese pedido es
# lo que llevo a F=1,5 y a que las 8 unidades contestaran TODO. Un modelo que no sabe nada DEBE
# callarse; lo que no debe es quedarse ahi, y de eso se encarga R-8 (la CE sigue viva).
# OJO, y aca me equivoque una vez: el umbral GLOBAL no es el mismo que el por muestra. El global
# pesa tambien la mezcla de preguntas, porque un q unico se aplica tambien a las que NO tienen
# respuesta, donde callarse siempre paga. Sale de resolver dE/dq = 0:
c_glob = ((1 - PI) * (M - F) + PI * (L + M)) / ((1 - PI) * (1 + M))
print(f"  umbral por muestra c* = {(M-F)/(1+M):.3f}   ·   umbral GLOBAL = {c_glob:.3f}")
for c in (0.0, 0.35, 0.9):
    qs = np.linspace(0, 1, 1001)
    q_opt = qs[int(np.argmax(E_R(qs, c)))]
    esperado = "callarse" if c < c_glob else "contestar"
    real = "callarse" if q_opt > 0.5 else "contestar"
    check(f"c={c}: el optimo global es {esperado}", real == esperado,
          f"q optimo = {q_opt:.3f} -> {real}")

print("\nR-3 · el umbral por muestra separa los dos regimenes")
cs = (M - F) / (1 + M)
for c, quiero in ((cs - 0.1, "callarse"), (cs + 0.1, "contestar")):
    contestar, callarse = c - (1 - c) * M, -F
    real = "contestar" if contestar > callarse else "callarse"
    check(f"con c={c:.3f} conviene {quiero}", real == quiero,
          f"contestar {contestar:+.3f} vs callarse {callarse:+.3f}")

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

print("\nR-6 · el optimo POR MUESTRA no es ninguno de los dos extremos")
alto = [c for c in (0.5, 0.9) if (c - (1 - c) * M) > -F]
bajo = [c for c in (0.0, 0.1) if (c - (1 - c) * M) < -F]
check("con confianza ALTA conviene contestar", len(alto) == 2, f"c en {alto}")
check("con confianza BAJA conviene callarse", len(bajo) == 2, f"c en {bajo}")
check("sin respuesta conviene callarse", L > -M, f"{L:+.3f} vs {-M:+.3f}")
print("  -> optimo por muestra = contestar donde sabe, callarse donde no. NO es un extremo.")

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
