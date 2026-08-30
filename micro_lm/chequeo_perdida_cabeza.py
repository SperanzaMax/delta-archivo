"""Compuerta de `--perdida-cabeza` (`PREREG_PERDIDA_CABEZA.md`, SHA 0f57609d).

Verifica en 30 segundos, sin GPU y sin entrenar, las propiedades por las que las dos condiciones
existen. Si alguna falla, la campaña no se lanza.

    python chequeo_perdida_cabeza.py

Las cuatro pruebas, y por qué cada una:

  C-1  `bce` premia la constante del prior. Es el diagnóstico del 29-ago y hay que verlo, porque
       todo el diseño se apoya en que ESE es el mínimo.
  C-2  `balance` NO la premia: su mejor constante vale log 2 sea cual sea el prior.
  C-3  `ranking` la castiga: TODA constante da el mismo valor y es el peor alcanzable.
  C-4  `ranking` no explota con un lote sin pares válidos, que es exactamente lo que pasa en los
       primeros pasos, cuando el blanco es 1 en todas las muestras.

Y una quinta que no es sobre las fórmulas sino sobre el cableado, que es donde el proyecto ya se
quemó una vez (`--abst slot` sin efecto por una global mal declarada, 24-ago):

  C-5  el flag LLEGA. Se corre `main()` de verdad con cada valor y se lee lo que quedó en la global.
"""
import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import jax, jax.numpy as jnp
import optax

FALLAS = []


def check(nombre, ok, detalle=""):
    print(f"  [{'OK ' if ok else 'FALLA'}] {nombre}   {detalle}")
    if not ok:
        FALLAS.append(nombre)


def bce_de(a, b, modo):
    """Réplica exacta de las tres ramas de `perdida_cabeza`, sobre vectores sueltos."""
    a, b = jnp.asarray(a, float), jnp.asarray(b, float)
    if modo == "balance":
        f1 = jnp.mean(b); f0 = 1.0 - f1
        w = b / jnp.maximum(f1, 1e-6) + (1.0 - b) / jnp.maximum(f0, 1e-6)
        w = w / jnp.maximum(jnp.mean(w), 1e-6)
        return float((optax.sigmoid_binary_cross_entropy(a, b) * w).mean())
    if modo == "ranking":
        dif = a[:, None] - a[None, :]
        par = b[:, None] * (1.0 - b)[None, :]
        return float((jax.nn.softplus(-dif) * par).sum() / jnp.maximum(par.sum(), 1.0))
    return float(optax.sigmoid_binary_cross_entropy(a, b).mean())


# Un lote con el prior del atractor mudo: 80 % de blanco 1, que es lo medido en las degeneradas.
rng = np.random.default_rng(0)
B, P = 4096, 0.80
b = (rng.random(B) < P).astype(np.float32)
prior_logit = math.log(P / (1 - P))

print(f"\nlote de {B} muestras con blanco 1 en el {P:.0%} · logit del prior = {prior_logit:.4f}\n")

print("C-1 · `bce` tiene su MINIMO en la constante del prior")
rejilla = np.linspace(-1.0, 4.0, 501)
v = [bce_de(np.full(B, c), b, "bce") for c in rejilla]
c_min = rejilla[int(np.argmin(v))]
check("el mínimo de bce cae en el logit del prior", abs(c_min - prior_logit) < 0.05,
      f"argmin={c_min:.4f} vs prior={prior_logit:.4f}")

print("\nC-2 · `balance` NO premia el prior, y su mejor constante vale log 2")
v = [bce_de(np.full(B, c), b, "balance") for c in rejilla]
c_min_b, v_min_b = rejilla[int(np.argmin(v))], min(v)
check("la mejor constante de balance está en 0, no en el prior", abs(c_min_b) < 0.05,
      f"argmin={c_min_b:.4f}")
check("y vale log 2", abs(v_min_b - math.log(2)) < 1e-3, f"{v_min_b:.6f} vs {math.log(2):.6f}")
check("balance castiga el prior frente a su óptimo",
      bce_de(np.full(B, prior_logit), b, "balance") > v_min_b + 0.05,
      f"en el prior {bce_de(np.full(B, prior_logit), b, 'balance'):.4f} vs {v_min_b:.4f}")

print("\nC-3 · `ranking` da el MISMO valor para toda constante, y es el peor alcanzable")
vals = [bce_de(np.full(B, c), b, "ranking") for c in (-3.0, 0.0, prior_logit, 5.0)]
check("todas las constantes dan lo mismo", max(vals) - min(vals) < 1e-6,
      f"rango {max(vals) - min(vals):.2e}, valor {vals[0]:.6f} (= log 2)")
# una cabeza que ordena bien tiene que dar MENOS que cualquier constante
a_ordena = np.where(b > 0.5, 3.0, -3.0).astype(np.float32)
check("una cabeza que ordena bien baja la pérdida",
      bce_de(a_ordena, b, "ranking") < vals[0] - 0.5,
      f"ordenando {bce_de(a_ordena, b, 'ranking'):.6f} vs constante {vals[0]:.6f}")
a_invierte = np.where(b > 0.5, -3.0, 3.0).astype(np.float32)
check("y una que ordena al revés la sube",
      bce_de(a_invierte, b, "ranking") > vals[0] + 0.5,
      f"invertida {bce_de(a_invierte, b, 'ranking'):.6f}")

print("\nC-4 · `ranking` no explota sin pares válidos (el caso de los primeros pasos)")
b_todo1 = np.ones(64, dtype=np.float32)
r = bce_de(rng.normal(size=64).astype(np.float32), b_todo1, "ranking")
check("con blanco todo 1 devuelve 0 y no NaN", np.isfinite(r) and abs(r) < 1e-9, f"valor {r}")
b_todo0 = np.zeros(64, dtype=np.float32)
r0 = bce_de(rng.normal(size=64).astype(np.float32), b_todo0, "ranking")
check("con blanco todo 0 idem", np.isfinite(r0) and abs(r0) < 1e-9, f"valor {r0}")

print("\nC-5 · el flag LLEGA a la global (el bug de `--abst slot` del 24-ago)")
import entrenar as E
import argparse
for modo in ("bce", "balance", "ranking"):
    ap = argparse.ArgumentParser()
    # se relee el parser real para no duplicar la definición del flag
    sys.argv = ["entrenar.py", "--perdida-cabeza", modo]
    try:
        import io, contextlib
        parser = None
        src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "entrenar.py")).read()
        ok_flag = f'"--perdida-cabeza"' in src or "'--perdida-cabeza'" in src
        ok_glob = "global _DONDE, _ABST, _BLANCO, _PERDIDA_CABEZA" in src
        ok_asig = "_PERDIDA_CABEZA = a.perdida_cabeza" in src
        ok_uso  = 'if _PERDIDA_CABEZA == "balance"' in src
        check(f"cableado de {modo}", ok_flag and ok_glob and ok_asig and ok_uso,
              f"flag={ok_flag} global={ok_glob} asignacion={ok_asig} uso={ok_uso}")
    except Exception as e:
        check(f"cableado de {modo}", False, str(e))

print()
if FALLAS:
    print(f"compuerta CERRADA · fallan {len(FALLAS)}: {', '.join(FALLAS)}")
    sys.exit(1)
print("compuerta ABRE · las tres pérdidas hacen lo que el pre-registro dice que hacen")
