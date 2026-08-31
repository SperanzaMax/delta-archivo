"""¿`q` es una CONSTANTE, o una decision TAJANTE por muestra que sigue la politica optima?

2026-08-31. El `INFORME_RECOMPENSA_L_20260830.md` concluyo que «`q` es una CONSTANTE ~0,50 robusta a
semilla, origen y L». Al medir el equilibrio de fuerzas aparecio un numero que no encaja con eso:

    q media 0,4902  ·  q DESVIO 0,4906

Un desvio de 0,49 sobre una variable acotada en [0,1] con media 0,49 no es una constante: es la firma
de una Bernoulli(0,5), o sea de una decision TAJANTE por muestra que cae en 0 o en 1. Una constante
real tendria desvio ~0, y una uniforme en [0,1] tendria 0,289.

Si eso se confirma, el diagnostico cambia de raiz. No es «el modelo no puede condicionar la decision
a la pregunta»; es «decide tajante en cada pregunta, y lo que hay que explicar es SEGUN QUE».

Y hay una segunda coincidencia que pide explicacion, del mismo volcado:

    frac c > c*=0,200  =  0,4856        abstencion  =  0,4902

Si el modelo se callara exactamente cuando c < c*, estaria implementando la politica OPTIMA POR
MUESTRA que la recompensa le pide, y entonces el cuello de botella no seria la DECISION sino la
CONFIANZA que la alimenta, o sea la recuperacion. Es justo el desenlace que el §6 del
`PREREG_RECOMPENSA_L` anticipo por escrito.

Las dos coincidencias de media NO alcanzan para afirmar nada: dos variables pueden tener la misma
tasa marginal y no coincidir en una sola muestra. Lo que decide es el ACUERDO PAREADO, y eso es lo
que mide esto.

  P-1  histograma de `q`: ¿bimodal en los extremos, o concentrado en 0,5?
  P-2  acuerdo pareado entre «se calla» (q>0,5) y «deberia callarse» (c<c*), muestra por muestra.
  P-3  el control que puede FALLAR: acuerdo contra la VERDAD (no hay respuesta). Si el modelo sigue
       su propia confianza pero su confianza no sabe, P-2 da alto y P-3 da azar. Si las dos dan
       alto, el modelo esta calibrado y el problema es otro.

Uso:  python3 medir_politica_q.py ckpts/t03_s3.pkl ckpts/t03_s6.pkl --n 2048
"""

import argparse

import numpy as np

import entrenar as E
import medir_ratio_ce as R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--n", type=int, default=2048)
    ap.add_argument("--lote", type=int, default=64)
    ap.add_argument("--semilla", type=int, default=54321)
    ap.add_argument("--p-nose", type=float, default=0.4)
    ap.add_argument("--c-est", type=float, default=0.200)
    a = ap.parse_args()

    print("=" * 100)
    print("¿CONSTANTE O DECISION TAJANTE? · distribucion de `q` y acuerdo con la politica optima")
    print(f"n={a.n}  semilla {a.semilla}  p_nose={a.p_nose}  c*={a.c_est}")
    print("=" * 100)

    for ruta in a.ckpts:
        params, cfg, paso = R.cargar(ruta)
        lg, tgt = R.logits(params, cfg, a.n, a.lote, a.semilla, a.p_nose)

        lgn = np.asarray(lg, dtype=np.float64)
        tgt_np = np.asarray(tgt)
        hay = tgt_np != E.NOSE

        p_all = np.exp(lgn - lgn.max(-1, keepdims=True))
        p_all /= p_all.sum(-1, keepdims=True)
        q = p_all[:, E.NOSE]

        lg_v = lgn.copy()
        lg_v[:, E.NOSE] = -1e9
        p_val = np.exp(lg_v - lg_v.max(-1, keepdims=True))
        p_val /= p_val.sum(-1, keepdims=True)
        c = p_val[np.arange(len(tgt_np)), tgt_np] * hay

        print(f"\n--- {ruta}   paso={paso} ---")
        print(f"  q media {q.mean():.4f}   desvio {q.std():.4f}   "
              f"(constante -> ~0,000 · uniforme -> 0,289 · Bernoulli(0,5) -> 0,500)")

        # P-1 · histograma
        bordes = [0.0, 0.01, 0.1, 0.3, 0.7, 0.9, 0.99, 1.0]
        h, _ = np.histogram(q, bins=bordes)
        print("  P-1 histograma de q:")
        for i in range(len(bordes) - 1):
            print(f"       [{bordes[i]:.2f}, {bordes[i+1]:.2f})  {h[i]:>5}  {h[i]/len(q):>6.3f}")
        extremos = (h[0] + h[-1]) / len(q)
        centro = h[3] / len(q)
        print(f"       -> en los extremos {extremos:.4f}  ·  en el centro [0,3-0,7) {centro:.4f}")
        print(f"       [{'BIMODAL' if extremos > 0.8 else 'no bimodal'}]")

        # P-2 · acuerdo con la politica optima segun SU PROPIA confianza
        calla = q > 0.5
        deberia = c < a.c_est          # incluye las sin respuesta, donde c=0 y callarse es correcto
        ac_pol = float((calla == deberia).mean())

        # P-3 · el control que puede fallar: acuerdo con la VERDAD
        ac_ver = float((calla == ~hay).mean())

        # Y el azar de cada uno, para que el numero se pueda leer
        az_pol = float(calla.mean() * deberia.mean() + (1 - calla.mean()) * (1 - deberia.mean()))
        az_ver = float(calla.mean() * (~hay).mean() + (1 - calla.mean()) * hay.mean())

        print(f"  P-2 acuerdo  «se calla» vs «c < c*»       {ac_pol:.4f}   (azar {az_pol:.4f})")
        print(f"  P-3 acuerdo  «se calla» vs «NO hay resp»  {ac_ver:.4f}   (azar {az_ver:.4f})")
        print(f"      tasas marginales: se calla {calla.mean():.4f} · deberia {deberia.mean():.4f} · "
              f"sin respuesta {(~hay).mean():.4f}")

        sigue = ac_pol - az_pol > 0.15
        sabe = ac_ver - az_ver > 0.15
        if sigue and not sabe:
            print("      -> SIGUE su propia confianza y su confianza NO SABE. El cuello de botella "
                  "es la RECUPERACION, no la decision (§6 del PREREG_RECOMPENSA_L).")
        elif sigue and sabe:
            print("      -> sigue su confianza Y su confianza sabe. El bloqueo no esta aca.")
        elif not sigue:
            print("      -> NO sigue la politica optima por muestra. Las medias coincidian por "
                  "casualidad marginal y el acuerdo pareado lo desmiente.")


if __name__ == "__main__":
    main()
