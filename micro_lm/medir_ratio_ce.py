"""Mide el RATIO DE GRADIENTES que decide `--rec-ce`, sobre checkpoints ya en disco.

Por que existe (2026-08-31). El `PRECISION_RECOMPENSA_L_CE.md` (SHA `4b61894e`) midio que con
`--rec-ce 1.0` la recompensa es el 7,3 % de la perdida y el logit de `NOSE` recibe 3,5 veces MENOS
gradiente que un token de valor cualquiera, y dejo escrita la regla:

    antes de contrastar dos valores de un peso, medir cuanto gradiente mueve ese peso contra el
    resto de la perdida. Un contraste sobre el 3 % de la perdida no es un contraste.

Su §4 dice ademas COMO se elige el reemplazo, y el punto es que no se elige mirando resultados:

    se elige igualando el gradiente en la columna de `NOSE` con el gradiente medio del resto del
    vocabulario, que es una cantidad medible ANTES de entrenar.

Esto lo mide. Cero GPU, cero pasos de entrenamiento, sobre los pesos que ya estan en disco.

QUE NO SE PUEDE DERIVAR A MANO, y es la razon de que esto sea un barrido y no una division.
`_recompensa` hace `lg_v = lg.at[:, NOSE].set(-1e9)` y la CE se calcula sobre `lg_v`, asi que el
gradiente de la CE hacia la columna de `NOSE` es CERO por construccion (el `.set()` corta el camino).
Consecuencia:

  * el gradiente en la columna `NOSE` NO depende de `rec_ce`  ->  bajar CE no le agrega senal;
  * el del resto es  |g_rec + rec_ce * g_ce|,  que NO es lineal en `rec_ce` porque los dos terminos
    tienen signo propio por muestra y se miden en valor absoluto.

O sea que la intervencion es RELATIVA: no sube la senal de callarse, baja la que compite con ella.
Eso hay que decirlo antes, porque cambia lo que el experimento puede afirmar.

Uso:
    python3 medir_ratio_ce.py ckpts/t03_s3.pkl ckpts/b3_s3.pkl --n 4000
    python3 medir_ratio_ce.py ckpts/t03_s3.pkl --rec-l 0.0 --objetivo 1.0
"""

import argparse
import pickle

import jax
import jax.numpy as jnp
import numpy as np

import datos as DAT
import entrenar as E

# Rejilla del barrido. Cubre dos ordenes de magnitud hacia abajo desde el 1,0 que se uso hasta ayer,
# porque el ratio a batir es 3,5 y no hay razon a priori para que la correccion sea exactamente 1/3,5.
REJILLA = (1.0, 0.75, 0.5, 0.4, 0.3, 0.2857, 0.2, 0.15, 0.1, 0.05, 0.02, 0.0)


def cargar(ruta):
    with open(ruta, "rb") as f:
        bulto = pickle.load(f)
    params, cfg = bulto["params"], bulto["config"]
    E._DONDE = cfg.get("donde", "pre")
    E._ABST = cfg.get("abst", "token")
    # Misma razon que en `medir_confianza.py:46`: una base con p_nose=0 no tiene cabeza en `params` y
    # `modelo.py` calcula su logit igual. Se rellena aca, en el script de medicion, y queda constante 0.
    if "abst" not in params:
        d = params["ln_f"]["g"].shape[-1]
        params = dict(params)
        params["abst"] = {"w": jnp.zeros((d, 1)), "b": jnp.zeros((1,))}
        print(f"  [aviso] {ruta}: sin cabeza de abstencion (base con p_nose=0). Se mide como `token`.")
        E._ABST = "token"
    return params, cfg, bulto.get("paso")


def logits(params, cfg, n, B, semilla, p_nose_cli):
    """Junta logits y targets de `n` muestras reales. Es la entrada de todas las mediciones."""
    nivel = cfg["nivel"]
    p_nose = p_nose_cli if p_nose_cli is not None else cfg.get("p_nose", 0.0)

    @jax.jit
    def partes(params, ses, cortes, turnos, mask, cons, pos):
        return E._partes(params, ses, cortes, turnos, mask, cons, pos)

    rng = np.random.default_rng(semilla)
    LG, TGT = [], []
    vistos = 0
    while vistos < n:
        b = min(B, n - vistos)
        ses, cortes, turnos, mask, cons, pos, tgt, tipo, meta = DAT.lote(
            rng, b, nivel=nivel, n_hechos=4, n_sesiones=4, p_nose=p_nose, con_meta=True)
        lg, a = partes(params, jnp.array(ses), jnp.array(cortes), jnp.array(turnos),
                       jnp.array(mask), jnp.array(cons), jnp.array(pos))
        LG.append(np.asarray(lg))
        TGT.append(np.asarray(tgt))
        vistos += b
    return jnp.array(np.concatenate(LG)), jnp.array(np.concatenate(TGT))


def medir(lg, tgt, rec_ce, rec_l):
    """|grad| en la columna NOSE contra |grad| medio del resto, y el peso de cada termino.

    El gradiente se toma respecto de los LOGITS, que es donde el `PRECISION` lo midio y lo que hace
    comparables los numeros contra su tabla (7,09e-06 vs 2,49e-05).
    """
    ce_previo, l_previo = E._REC_CE, E._REC_L
    E._REC_CE, E._REC_L = rec_ce, rec_l
    try:
        def f(x):
            return E._recompensa(x, tgt, q=jax.nn.softmax(x, -1)[:, E.NOSE])[0]

        val = float(f(lg))
        g = np.abs(np.asarray(jax.grad(f)(lg), dtype=np.float64))

        # Los dos terminos por separado, sobre el MISMO lote y los mismos pesos, para poder decir
        # que fraccion de la perdida es cada uno sin volver a correr el modelo.
        es_nose = (np.asarray(tgt) == E.NOSE)
        hay = ~es_nose
        lg_v = np.asarray(lg, dtype=np.float64).copy()
        lg_v[:, E.NOSE] = -1e9
        p = np.exp(lg_v - lg_v.max(-1, keepdims=True))
        p /= p.sum(-1, keepdims=True)
        ce = -np.log(np.maximum(p[np.arange(len(tgt)), np.asarray(tgt)], 1e-30))
        ce = float((ce * hay).sum() / max(hay.sum(), 1.0))
    finally:
        E._REC_CE, E._REC_L = ce_previo, l_previo

    col = float(g[:, E.NOSE].mean())
    resto = float(np.delete(g, E.NOSE, axis=1).mean())
    return dict(perdida=val, ce=ce, ce_pesada=rec_ce * ce, rec=val - rec_ce * ce,
                col=col, resto=resto, ratio=resto / col if col > 0 else float("inf"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--lote", type=int, default=64)
    ap.add_argument("--semilla", type=int, default=54321)   # la misma de las Fases 1 y L, pareada
    ap.add_argument("--p-nose", type=float, default=0.4)
    ap.add_argument("--rec-l", type=float, default=0.0,
                    help="L de la campania que viene. 0,0 es el valor derivado en PREREG_RECOMPENSA_L "
                         "(el mudo cobra negativo). Se mide tambien con 0,5 como control.")
    ap.add_argument("--objetivo", type=float, default=1.0,
                    help="ratio buscado. 1,0 = el logit de NOSE recibe tanto gradiente como un token "
                         "de valor cualquiera, que es lo que pide el §4 del PRECISION.")
    a = ap.parse_args()

    print("=" * 92)
    print("RATIO DE GRADIENTES  |grad| resto del vocabulario  /  |grad| columna NOSE")
    print(f"n={a.n}  semilla {a.semilla} (pareada)  p_nose={a.p_nose}  ·  objetivo ratio = {a.objetivo}")
    print("referencia del 30-ago (t03_s3, L=0, CE=1,0): col 7,09e-06 · resto 2,49e-05 · ratio 3,5")
    print("=" * 92)

    for ruta in a.ckpts:
        params, cfg, paso = cargar(ruta)
        lg, tgt = logits(params, cfg, a.n, a.lote, a.semilla, a.p_nose)
        print(f"\n--- {ruta}   paso={paso}  abst={cfg.get('abst')}  nivel={cfg.get('nivel')} ---")

        for rec_l in (a.rec_l, 0.5) if a.rec_l != 0.5 else (0.5,):
            print(f"\n  L = {rec_l}")
            print(f"  {'rec_ce':>7}  {'perdida':>9}  {'CE pesada':>9}  {'%CE':>6}  "
                  f"{'|g| NOSE':>10}  {'|g| resto':>10}  {'ratio':>7}")
            filas = []
            for ce_w in REJILLA:
                m = medir(lg, tgt, ce_w, rec_l)
                filas.append((ce_w, m))
                pct = 100.0 * m["ce_pesada"] / m["perdida"] if m["perdida"] else 0.0
                print(f"  {ce_w:>7.4f}  {m['perdida']:>9.4f}  {m['ce_pesada']:>9.4f}  {pct:>5.1f}%  "
                      f"{m['col']:>10.3e}  {m['resto']:>10.3e}  {m['ratio']:>7.3f}")

            # El valor que cruza el objetivo, por interpolacion lineal entre las dos celdas vecinas.
            # No se extrapola: si ni con CE=0 el ratio baja al objetivo, se dice y no se inventa.
            orden = sorted(filas, key=lambda t: t[0])
            cruce = None
            for (w0, m0), (w1, m1) in zip(orden, orden[1:]):
                if (m0["ratio"] - a.objetivo) * (m1["ratio"] - a.objetivo) <= 0:
                    r0, r1 = m0["ratio"], m1["ratio"]
                    cruce = w0 + (w1 - w0) * (a.objetivo - r0) / (r1 - r0) if r1 != r0 else w0
                    break
            piso = orden[0][1]["ratio"]
            if cruce is not None:
                print(f"  ->  rec_ce = {cruce:.4f}  lleva el ratio a {a.objetivo}")
            else:
                print(f"  ->  NO hay cruce en la rejilla. Con rec_ce=0 el ratio es {piso:.3f} "
                      f"(la recompensa sola ya reparte asi). El objetivo {a.objetivo} no es "
                      f"alcanzable bajando la CE, y eso ACOTA la intervencion.")

            # Control de la afirmacion del encabezado: el gradiente en NOSE no puede depender de
            # rec_ce, porque el `.set(-1e9)` corta el camino de la CE. Si esto varia, el diagnostico
            # del PRECISION esta mal y la campania no se lanza.
            cols = [m["col"] for _, m in filas]
            disp = (max(cols) - min(cols)) / max(cols) if max(cols) > 0 else 0.0
            estado = "OK" if disp < 1e-9 else "FALLA"
            print(f"  [{estado}] |g| en NOSE es constante en rec_ce (dispersion relativa {disp:.2e}) "
                  f"-> la CE no le llega, la intervencion es RELATIVA")


if __name__ == "__main__":
    main()
