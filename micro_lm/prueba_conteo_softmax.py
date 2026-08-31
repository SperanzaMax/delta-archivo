"""¿La abstencion de la interfaz `token` la decide el NUMERO DE CANDIDATOS? Prueba algebraica exacta.

2026-08-31. El 31 se midio que las unidades `t*` se callan en las relaciones cuyo valor es un NUMERO
(altura, precio, clave) y contestan en las de NOMBRE (director, dueño, guardia), con pureza 0,98 y
replica exacta en dos semillas. Ese corte NO sigue la ausencia (ganancia +0,0000) ni la dificultad
(brecha de RECUP -0,0200, y del lado contrario).

La hipotesis mecanica es que no es una decision sino aritmetica del softmax. Con `--abst token`:

    q = softmax(lg)[NOSE] = exp(l_NOSE) / (exp(l_NOSE) + sum_j exp(l_j))

o sea, exactamente,

    q > 0,5   <=>   l_NOSE > logsumexp(logits de los valores)

`NOSE` no compite contra el mejor candidato: compite contra la SUMA de todos. Y el idioma tiene
**100 numeros contra 58 nombres**, asi que la suma tiene 1,72x mas terminos del lado numerico.

DOS CAUSAS POSIBLES, Y SE SEPARAN:

  (a) CONTEO   - hay mas terminos en la suma, y logsumexp crece con la cantidad aunque cada logit
                 sea igual. Seria un defecto de la INTERFAZ, no algo que el modelo aprendio.
  (b) APRENDIDO - el modelo pone logits mas altos a los numeros. Seria una decision suya.

El contrafactico las separa sin entrenar nada: se recalcula `q` truncando la suma a los **58 mejores**
candidatos de cada clase, que iguala la cantidad de terminos y deja los logits intactos.

  * si con 58 terminos el corte SE DA VUELTA  -> la causa es el CONTEO (a)
  * si se mantiene                            -> el modelo aprendio a preferir NOSE ahi (b)

Y un control que puede fallar: se compara tambien el logit MAXIMO por clase. Si el maximo es igual
entre clases pero el logsumexp difiere, lo que separa es la cantidad y no la calidad del candidato.

Uso:  python3 prueba_conteo_softmax.py ckpts/t03_s3.pkl ckpts/t03_s6.pkl --n 3072
"""

import argparse

import numpy as np

import entrenar as E
import idioma as I
import medir_ratio_ce as R

PERSONALES = ("director", "dueño", "guardia")


def lse(x, axis=-1):
    m = x.max(axis, keepdims=True)
    return (m + np.log(np.exp(x - m).sum(axis, keepdims=True))).squeeze(axis)


def lse_top(x, k, axis=-1):
    """logsumexp sobre los k logits MAS ALTOS. Iguala la cantidad de terminos sin tocar los valores."""
    idx = np.argsort(x, axis=axis)[..., -k:]
    return lse(np.take_along_axis(x, idx, axis=axis), axis=axis)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--n", type=int, default=3072)
    ap.add_argument("--lote", type=int, default=64)
    ap.add_argument("--semilla", type=int, default=54321)
    ap.add_argument("--p-nose", type=float, default=0.4)
    a = ap.parse_args()

    ids_nom = np.array([I.STOI[t] for t in I.NOMBRES])
    ids_num = np.array([I.STOI[t] for t in I.NUMEROS])
    k = min(len(ids_nom), len(ids_num))
    print("=" * 100)
    print("¿LA ABSTENCION LA DECIDE EL NUMERO DE CANDIDATOS? · prueba algebraica, sin entrenar")
    print(f"  nombres {len(ids_nom)}  ·  numeros {len(ids_num)}  ·  se truncan los dos a k={k}")
    print(f"  identidad exacta:  q > 0,5  <=>  l_NOSE > logsumexp(valores)")
    print("=" * 100)

    for ruta in a.ckpts:
        params, cfg, paso = R.cargar(ruta)
        lg, tgt = R.logits(params, cfg, a.n, a.lote, a.semilla, a.p_nose)
        lg = np.asarray(lg, dtype=np.float64)
        l_nose = lg[:, E.NOSE]

        # La consulta no se necesita: la clase la da el TARGET cuando hay respuesta. Para las `nose`
        # no hay clase, asi que se excluyen; el corte se midio sobre la relacion, no sobre el target,
        # y acá alcanza con las que tienen respuesta para probar el mecanismo.
        hay = tgt != E.NOSE
        es_nom = np.isin(tgt, ids_nom) & hay
        es_num = np.isin(tgt, ids_num) & hay

        print(f"\n--- {ruta}  paso={paso} ---")
        print(f"  {'clase':<10} {'n':>6} {'l_NOSE':>9} {'LSE todos':>11} {'q real':>8} "
              f"{'max logit':>10} {'LSE k=58':>10} {'q con k':>8}")

        vueltas = {}
        for nom, m in (("NOMBRE", es_nom), ("numero", es_num)):
            if m.sum() == 0:
                continue
            ids = ids_nom if nom == "NOMBRE" else ids_num
            sub = lg[m][:, ids]                       # logits de los candidatos de SU clase
            lse_todos = lse(lg[m][:, np.concatenate([ids_nom, ids_num])])
            q_real = 1.0 / (1.0 + np.exp(-(l_nose[m] - lse_todos)))
            lse_k = lse_top(sub, k)
            # contrafactico: la misma comparacion pero con k terminos de la clase propia
            q_k = 1.0 / (1.0 + np.exp(-(l_nose[m] - lse_k)))
            vueltas[nom] = ((q_real > 0.5).mean(), (q_k > 0.5).mean())
            print(f"  {nom:<10} {m.sum():>6} {l_nose[m].mean():>9.3f} {lse_todos.mean():>11.3f} "
                  f"{(q_real > 0.5).mean():>8.4f} {sub.max(-1).mean():>10.3f} {lse_k.mean():>10.3f} "
                  f"{(q_k > 0.5).mean():>8.4f}")

        if len(vueltas) == 2:
            (qr_nom, qk_nom), (qr_num, qk_num) = vueltas["NOMBRE"], vueltas["numero"]
            print(f"\n  brecha REAL      (numero - nombre) = {qr_num - qr_nom:+.4f}")
            print(f"  brecha con k={k} terminos iguales   = {qk_num - qk_nom:+.4f}")
            cae = abs(qk_num - qk_nom) < 0.5 * abs(qr_num - qr_nom)
            print(f"  [{'CONTEO' if cae else 'APRENDIDO'}] al igualar la cantidad de candidatos la "
                  f"brecha {'CAE' if cae else 'SE MANTIENE'}")

        # El control que puede fallar: ¿el mejor candidato es igual de bueno en las dos clases?
        if es_nom.sum() and es_num.sum():
            mx_nom = lg[es_nom][:, ids_nom].max(-1).mean()
            mx_num = lg[es_num][:, ids_num].max(-1).mean()
            print(f"  control · logit MAXIMO  nombre {mx_nom:.3f}  contra  numero {mx_num:.3f}  "
                  f"(dif {mx_num - mx_nom:+.3f})")
            print("    -> si el maximo es parecido y el logsumexp no, lo que separa es la CANTIDAD")


if __name__ == "__main__":
    main()
