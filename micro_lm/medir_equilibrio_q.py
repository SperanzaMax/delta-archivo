"""¿El gradiente chico en `NOSE` es FALTA de senal o CANCELACION entre dos fuerzas opuestas?

2026-08-31. El `PRECISION_RECOMPENSA_L_CE.md` midio |grad| en la columna de `NOSE` contra |grad| medio
del resto y concluyo que «la decision de callarse recibe 3,5 veces MENOS gradiente», o sea que el
bloqueo es de MAGNITUD. Este script prueba la explicacion alternativa, que invierte la causalidad:

    el gradiente en `NOSE` es chico porque `q` YA ESTA en el optimo de la perdida, con las preguntas
    con respuesta tirando hacia abajo y las sin respuesta tirando hacia arriba, y las dos fuerzas
    cancelandose. Gradiente chico como CONSECUENCIA del equilibrio, no como su causa.

Las dos hipotesis predicen cosas distintas y separables sobre el MISMO lote:

  * falta de magnitud  ->  |media del gradiente| ~ media de |gradiente|. La fuerza neta es chica
                          porque cada muestra aporta poco.
  * equilibrio         ->  |media del gradiente| << media de |gradiente|. Cada muestra aporta MUCHO
                          y la suma se cancela. Y las dos poblaciones tienen SIGNO OPUESTO.

El instrumento del 30-ago no podia distinguirlas: promedio |g| por muestra, y el valor absoluto
borra justamente el signo que separa las dos hipotesis.

Signos, derivados de `_recompensa` y fijados antes de mirar ningun numero (L=0, M=0,5, F=0,2):

    con respuesta:  d(rec)/dq = -F - c + (1-c)M = 0,3 - 1,5c   -> negativo si c > c*=0,2
    sin respuesta:  d(rec)/dq = L + M = +0,5                   -> siempre positivo

y como la perdida es -E[rec], la perdida empuja `q` HACIA ABAJO (contestar) en las que tienen
respuesta con c > c*, y HACIA ARRIBA (callarse) en las que no la tienen. En el equilibrio la suma da
cero, y ahi el modelo se queda aunque cada muestra por separado tenga mucho para decir.

Uso:  python3 medir_equilibrio_q.py ckpts/t03_s3.pkl ckpts/t53_s3.pkl --n 2048
"""

import argparse
import pickle

import jax
import jax.numpy as jnp
import numpy as np

import datos as DAT
import entrenar as E
import medir_ratio_ce as R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--n", type=int, default=2048)
    ap.add_argument("--lote", type=int, default=64)
    ap.add_argument("--semilla", type=int, default=54321)
    ap.add_argument("--p-nose", type=float, default=0.4)
    ap.add_argument("--rec-ce", type=float, default=1.0)
    ap.add_argument("--rec-l", type=float, default=0.0)
    a = ap.parse_args()

    print("=" * 100)
    print("FUERZA NETA contra MAGNITUD BRUTA en la columna de NOSE")
    print(f"n={a.n}  semilla {a.semilla}  p_nose={a.p_nose}  rec_ce={a.rec_ce}  rec_l={a.rec_l}")
    print("  cancelacion = 1 - |media| / media(|.|)   ·   1,000 = las dos fuerzas se anulan del todo")
    print("=" * 100)

    for ruta in a.ckpts:
        params, cfg, paso = R.cargar(ruta)
        lg, tgt = R.logits(params, cfg, a.n, a.lote, a.semilla, a.p_nose)

        ce_previo, l_previo = E._REC_CE, E._REC_L
        E._REC_CE, E._REC_L = a.rec_ce, a.rec_l
        try:
            def f(x):
                return E._recompensa(x, tgt, q=jax.nn.softmax(x, -1)[:, E.NOSE])[0]
            g = np.asarray(jax.grad(f)(lg), dtype=np.float64)
        finally:
            E._REC_CE, E._REC_L = ce_previo, l_previo

        col = g[:, E.NOSE]
        tgt_np = np.asarray(tgt)
        hay = tgt_np != E.NOSE
        no = ~hay

        # `q` real del modelo y `c`, para poder decir DONDE esta parado respecto del umbral por muestra
        p_all = np.exp(np.asarray(lg, dtype=np.float64) - np.asarray(lg, dtype=np.float64).max(-1, keepdims=True))
        p_all /= p_all.sum(-1, keepdims=True)
        q = p_all[:, E.NOSE]
        lg_v = np.asarray(lg, dtype=np.float64).copy()
        lg_v[:, E.NOSE] = -1e9
        p_val = np.exp(lg_v - lg_v.max(-1, keepdims=True))
        p_val /= p_val.sum(-1, keepdims=True)
        c = p_val[np.arange(len(tgt_np)), tgt_np] * hay

        neto, bruto = float(col.mean()), float(np.abs(col).mean())
        canc = 1.0 - abs(neto) / bruto if bruto > 0 else float("nan")

        print(f"\n--- {ruta}   paso={paso}  abst={cfg.get('abst')} ---")
        print(f"  q media {q.mean():.4f}  ·  q desvio {q.std():.4f}  ·  "
              f"c medio (con respuesta) {c[hay].mean():.4f}  ·  frac c>0,200 {(c[hay]>0.200).mean():.4f}")
        print(f"  {'poblacion':<22} {'n':>6}  {'media con signo':>16}  {'media de |g|':>13}")
        for nom, m in (("con respuesta", hay), ("sin respuesta", no), ("TODAS", np.ones_like(hay))):
            m = m.astype(bool)
            print(f"  {nom:<22} {m.sum():>6}  {col[m].mean():>+16.3e}  {np.abs(col[m]).mean():>13.3e}")
        print(f"  fuerza NETA {neto:+.3e}   magnitud BRUTA {bruto:.3e}   "
              f"**cancelacion {canc:.4f}**")

        s_hay, s_no = float(col[hay].mean()), float(col[no].mean())
        opuestos = (s_hay * s_no) < 0
        print(f"  [{'OK ' if opuestos else 'NO '}] las dos poblaciones tiran en sentidos OPUESTOS "
              f"({s_hay:+.2e} contra {s_no:+.2e})")
        if opuestos:
            print(f"  -> el gradiente chico es EQUILIBRIO, no falta de senal. Bajar la CE no mueve "
                  f"esto: no hay una colina que subir, `q` esta en el fondo.")


if __name__ == "__main__":
    main()
