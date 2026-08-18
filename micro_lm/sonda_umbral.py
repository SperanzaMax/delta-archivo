#!/usr/bin/env python3
"""¿La cabeza no alcanza, o esta mal CALIBRADA? — barrido del umbral de abstencion.

EXPLORATORIO POST-HOC. No es un test de PREREG_CABEZA_ABSTENCION.md: el prereg fijo la regla de
decision en sigma(a) > 0,5 y por ahi se juzgan P-1..P-4. Esto pregunta otra cosa, mirando los mismos
checkpoints ya entrenados: si el umbral fuera otro, ¿la compuerta se pasaria?

Las dos respuestas posibles son muy distintas:

  - **AUC alto y existe un umbral que pasa** -> la cabeza SEPARA bien lo que esta de lo que no, y
    0,5 simplemente no es el punto de corte correcto. Es un problema de calibracion: barato.
  - **AUC bajo** -> ningun umbral salva nada, porque la senal no distingue. Es capacidad, y ahi el
    resultado de la campaña se sostiene tal cual.

`a` es el logit crudo de la cabeza binaria; sigma(a) > 0,5 equivale a a > 0. El barrido va sobre `a`.
"""
import os, sys, pickle, argparse
import numpy as np
import jax, jax.numpy as jnp

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
import idioma as I, datos as DAT, modelo as M

NOSE = I.STOI["NOSE"]
UNIDADES = ["1_s0", "2_s0", "3_s0", "3_s1", "3_s2", "4_s0", "4_s1"]


def partes(params, ses, cortes, turnos, mask, cons, pos):
    archivo = M.escribir(params, ses, cortes)
    lg, a = M.responder_con_abst(params, archivo, turnos, cons, mask)
    lg = jnp.take_along_axis(lg, pos[:, None, None], axis=1)[:, 0, :]
    a = jnp.take_along_axis(a, pos[:, None], axis=1)[:, 0]
    return lg, a


def juntar(ck, nivel, semilla, n, B, p_nose, p_vieja=0.35):
    """Corre el modelo y devuelve, por muestra: el logit de la cabeza, si acierta el valor, y el tipo."""
    with open(ck, "rb") as f:
        params = jax.tree_util.tree_map(jnp.asarray, pickle.load(f)["params"])
    if "abst" not in params:
        return None
    fn = jax.jit(partes)
    rng = np.random.default_rng(90000 + semilla)          # el MISMO rng que usa `evaluar`
    A, OKV, TIPO = [], [], []
    for _ in range(n):
        ses, cortes, turnos, mask, cons, pos, tgt, tipo = DAT.lote(
            rng, B, nivel=nivel, n_hechos=4, n_sesiones=4, p_vieja=p_vieja, p_nose=p_nose)
        lg, a = fn(params, jnp.array(ses), jnp.array(cortes), jnp.array(turnos),
                   jnp.array(mask), jnp.array(cons), jnp.array(pos))
        lg = np.asarray(lg).copy()
        lg[:, NOSE] = -np.inf                              # NOSE sale del argmax de valores
        A.append(np.asarray(a))
        OKV.append(lg.argmax(-1) == tgt)                   # ¿acierta el valor, si contestara?
        TIPO.append(tipo)
    return np.concatenate(A), np.concatenate(OKV), np.concatenate(TIPO)


def metricas(a, okv, tipo, umbral):
    """Replica exacto las metricas de `evaluar`, pero decidiendo con `a > umbral`."""
    abst = a > umbral
    sin_resp = tipo >= 2                                   # la respuesta NO esta en el archivo
    hay = ~sin_resp
    nose = abst[sin_resp].mean() if sin_resp.any() else np.nan
    falsa = abst[hay].mean() if hay.any() else np.nan
    # `vigente`: acierta el valor Y no se abstuvo
    vig = tipo == 0
    vigente = (okv[vig] & ~abst[vig]).mean() if vig.any() else np.nan
    return nose, falsa, vigente


def auc(a, tipo):
    """AUC de `a` separando «no esta» de «si esta». 0,5 = no distingue; 1,0 = separacion perfecta."""
    pos, neg = a[tipo >= 2], a[tipo < 2]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    r = np.argsort(np.argsort(np.concatenate([pos, neg]))) + 1
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=8, help="lotes por unidad")
ap.add_argument("--batch", type=int, default=32, help="chico a proposito: esto corre en CPU")
ap.add_argument("--p-nose", type=float, default=0.4)
a_ = ap.parse_args()

print("EXPLORATORIO POST-HOC — el prereg juzga con a > 0 (sigma > 0,5). Esto pregunta si otro corte\n"
      "pasaria la compuerta, sobre los MISMOS checkpoints ya entrenados.\n")
print(f"{'unidad':<8} {'AUC':>6} {'a>0 (prereg)':>26} {"umbral: elegido en A, medido en B":>34}")
print(f"{'':8} {'':>6} {'f_abst':>8} {'nose':>7} {'':>9} {'a*':>7} {'f_abst':>8} {'nose':>7} {'pasa?':>9}")
print("-" * 78)

for u in UNIDADES:
    ck = os.path.join(AQUI, "ckpts", f"c{u}.pkl")
    if not os.path.exists(ck):
        continue
    nivel, semilla = int(u[0]), int(u.split("_s")[1])
    r = juntar(ck, nivel, semilla, a_.n, a_.batch, a_.p_nose)
    if r is None:
        print(f"c{u:<7} (sin cabeza en el checkpoint)")
        continue
    A, OKV, TIPO = r
    ar = auc(A, TIPO)
    n0, f0, _ = metricas(A, OKV, TIPO, 0.0)

    # El umbral se ELIGE en una mitad y se EVALUA en la otra. Elegirlo y juzgarlo sobre las mismas
    # muestras da un numero optimista por construccion —es ajustar un parametro y reportar el error
    # de entrenamiento— y con 6 checkpoints y 400 cortes el sobreajuste no es despreciable.
    rs = np.random.default_rng(1234)
    idx = rs.permutation(len(A))
    ajuste, prueba = idx[: len(A) // 2], idx[len(A) // 2:]
    # Se pide MARGEN al elegir (0,07) aunque se juzgue con el criterio real (0,10). Sin margen, el
    # barrido elige siempre el punto pegado al limite —el mas fragil— y al cambiar de muestra lo
    # cruza: con el criterio al borde pasaban 2 de 6, y las dos unidades faciles, que ya pasaban
    # holgadas con a>0, quedaban reprobadas por su propio umbral "optimo".
    MARGEN = 0.07
    mejor = None
    for t in np.quantile(A[ajuste], np.linspace(0.001, 0.999, 400)):
        nn, ff, _ = metricas(A[ajuste], OKV[ajuste], TIPO[ajuste], t)
        if ff <= MARGEN and nn >= 0.50 and (mejor is None or nn > mejor[1]):
            mejor = (t, nn, ff)
    if mejor:
        t = mejor[0]
        nn, ff, _ = metricas(A[prueba], OKV[prueba], TIPO[prueba], t)   # el numero que se reporta
        ok = "SI" if (ff <= 0.10 and nn >= 0.50) else "no"
        print(f"c{u:<7} {ar:>6.3f} {f0:>8.4f} {n0:>7.4f} {'':>9} {t:>7.2f} {ff:>8.4f} {nn:>7.4f} {ok:>9}")
    else:
        print(f"c{u:<7} {ar:>6.3f} {f0:>8.4f} {n0:>7.4f} {'':>9} {'—':>7} {'—':>8} {'—':>7} {'NO':>9}")

print("\nComo se lee: si un ckpt que FALLA con a>0 tiene un umbral que pasa, el problema es de\n"
      "calibracion, no de capacidad. Si el AUC es bajo (~0,5-0,7), ningun umbral lo salva.")
