"""LA PRUEBA: ¿la perdida es PLANA respecto de QUE preguntas callar?

2026-08-31. Tres hipotesis mias cayeron hoy sobre el mismo corte nombre/numero. No lo explica la
ausencia (+0,0000 de ganancia), ni la dificultad (RECUP -0,0200, al reves), ni el conteo de
candidatos (la brecha esta en l_NOSE, 38 nats, y truncar a k=58 no mueve nada), ni la confianza
(c nombre 0,2916 contra numero 0,2680 en s3, y -0,0029 en s6, con SIGNOS OPUESTOS entre semillas).

Queda una sola explicacion, y a diferencia de las otras tres es DEMOSTRABLE. La recompensa esperada
por muestra es

    R = hay * [ q(-F) + (1-q)(c - (1-c)M) ]  +  (1-hay) * [ qL + (1-q)(-M) ]

que es **LINEAL en q**. Al promediar sobre el lote, si `q` es independiente de (hay, c), entonces
E[R] depende de `q` SOLO A TRAVES DE SU MEDIA. Dos modelos con la misma tasa global de abstencion y
particiones COMPLETAMENTE DISTINTAS tienen exactamente la misma perdida.

> Si eso es asi, la perdida no tiene ninguna preferencia por CUALES preguntas callar. Hay una
> VARIEDAD de optimos equivalentes, el modelo cae en el mas accesible —una feature que ya esta en el
> token de entrada, como la relacion— y satura, porque moverse dentro de la variedad no cuesta nada.

LA PRUEBA. Se toma el `q` real del modelo y se lo BARAJA entre muestras. Barajar destruye cualquier
relacion entre callarse y la pregunta, pero **conserva exactamente la distribucion de `q`** y por lo
tanto su media. Si la perdida no cambia, es plana y esta probado.

Y el control que puede fallar, que es lo que hace valida la prueba: se compara contra un `q` ORACULO
—el que se callaria exactamente donde no hay respuesta, con la misma tasa— que SI tiene que bajar la
perdida. Si el oraculo tampoco la baja, la prueba esta mal construida y no se concluye nada.

Uso:  python3 prueba_perdida_plana.py ckpts/t03_s3.pkl ckpts/t03_s6.pkl --n 3072
"""

import argparse

import numpy as np

import entrenar as E
import medir_ratio_ce as R


def recompensa(q, c, hay, L, M, F):
    """-E[recompensa], identica a `entrenar._recompensa` pero en numpy y sin la CE (que no toca q)."""
    r_hay = q * (-F) + (1.0 - q) * (c - (1.0 - c) * M)
    r_no = q * L + (1.0 - q) * (-M)
    return -float((hay * r_hay + (1.0 - hay) * r_no).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--n", type=int, default=3072)
    ap.add_argument("--lote", type=int, default=64)
    ap.add_argument("--semilla", type=int, default=54321)
    ap.add_argument("--p-nose", type=float, default=0.4)
    ap.add_argument("--rec-l", type=float, default=0.0)
    ap.add_argument("--rec-m", type=float, default=0.5)
    ap.add_argument("--rec-f", type=float, default=0.2)
    ap.add_argument("--reps", type=int, default=200)
    a = ap.parse_args()

    L, M, F = a.rec_l, a.rec_m, a.rec_f
    print("=" * 100)
    print("¿LA PERDIDA ES PLANA RESPECTO DE *QUE* PREGUNTAS CALLAR?")
    print(f"  L={L}  M={M}  F={F}   ·   n={a.n}   ·   {a.reps} barajadas")
    print("  R es LINEAL en q -> si q es independiente de (hay, c), E[R] depende solo de la MEDIA")
    print("=" * 100)

    for ruta in a.ckpts:
        params, cfg, paso = R.cargar(ruta)
        lg, tgt = R.logits(params, cfg, a.n, a.lote, a.semilla, a.p_nose)
        lg = np.asarray(lg, dtype=np.float64)
        tgt = np.asarray(tgt)

        lg_v = lg.copy()
        lg_v[:, E.NOSE] = -1e9
        p_val = np.exp(lg_v - lg_v.max(-1, keepdims=True))
        p_val /= p_val.sum(-1, keepdims=True)
        hay = (tgt != E.NOSE).astype(np.float64)
        c = p_val[np.arange(len(tgt)), tgt] * hay

        p_all = np.exp(lg - lg.max(-1, keepdims=True))
        p_all /= p_all.sum(-1, keepdims=True)
        q = p_all[:, E.NOSE]

        real = recompensa(q, c, hay, L, M, F)

        rng = np.random.default_rng(20260831)
        baraj = [recompensa(rng.permutation(q), c, hay, L, M, F) for _ in range(a.reps)]
        bm, bs = float(np.mean(baraj)), float(np.std(baraj))

        # ORACULO con la MISMA tasa: se callan las `k` muestras con menos c, k = q.sum() redondeado.
        # Es el q binario, con la misma media, que mejor separa. Si la perdida fuera plana en serio,
        # este tampoco bajaria -- y por eso es el control que puede hacer fallar la prueba.
        k = int(round(q.sum()))
        orden = np.argsort(c)                      # las sin respuesta tienen c=0, van primero
        q_or = np.zeros_like(q)
        q_or[orden[:k]] = 1.0
        oraculo = recompensa(q_or, c, hay, L, M, F)

        # Y el peor caso con la misma tasa, para tener la escala completa del eje
        q_pe = np.zeros_like(q)
        q_pe[orden[-k:]] = 1.0
        peor = recompensa(q_pe, c, hay, L, M, F)

        print(f"\n--- {ruta}  paso={paso}   tasa de abstencion {q.mean():.4f} ---")
        print(f"  perdida con el q REAL del modelo      {real:+.6f}")
        print(f"  perdida con el q BARAJADO             {bm:+.6f}  ± {bs:.6f}")
        print(f"  perdida con el q ORACULO (misma tasa) {oraculo:+.6f}")
        print(f"  perdida con el q PEOR    (misma tasa) {peor:+.6f}")
        dif = abs(real - bm)
        rango = abs(peor - oraculo)
        print(f"\n  |real - barajado| = {dif:.6f}   ({dif/bs if bs else float('nan'):.2f} desvios)")
        print(f"  rango total del eje (peor - oraculo) = {rango:.6f}")
        print(f"  ** el modelo aprovecha el {100*(peor-real)/rango if rango else float('nan'):.2f} % "
              f"del margen disponible **")

        plana = dif < 3 * bs
        util = (peor - oraculo) > 20 * bs
        print(f"  [{'OK ' if util else 'FALLA'}] el control ORACULO si baja la perdida "
              f"({peor - oraculo:.6f}, {rango/bs if bs else float('nan'):.1f} desvios) -> la prueba "
              f"{'puede' if util else 'NO puede'} distinguir")
        if util:
            print(f"  [{'PLANA' if plana else 'NO plana'}] barajar a que preguntas se calla "
                  f"{'NO cambia' if plana else 'CAMBIA'} la perdida")
            if plana:
                print("     -> hay una VARIEDAD de optimos equivalentes. La perdida premia la TASA")
                print("        de abstencion, no la DISCRIMINACION. El modelo cae en la particion")
                print("        mas accesible (la relacion, que ya esta en el token de entrada) y")
                print("        satura, porque moverse dentro de la variedad no cuesta nada.")


if __name__ == "__main__":
    main()
