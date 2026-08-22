"""COMPUERTA de la politica de escritura (`DISENO_POLITICA_ESCRITURA.md` §4).

La pregunta, y es bloqueante: **¿la sorpresa separa un hecho de un relleno?**

Si no separa, la eviction sorpresa-gated de [[vigia03-capacity-scheduling]] no tiene de donde
agarrarse y la campania se cae ACA, sin gastar una sola unidad de GPU. Se corre antes de tocar el
generador y antes de escribir el pre-registro, que es el orden que el §6 del diseño fija.

«Sorpresa» = el residuo comprometido de la regla delta, `beta * ||v - S k||`, medido en la posicion
del ultimo token de cada enunciado —que es exactamente donde `modelo.escribir` toma el vector que
archiva—. Es la señal de CENTINELA-01 y la que HOLA usa para decidir escritura (`beta*||e||`), asi
que el estadistico no es una invencion de este experimento.

El episodio se arma a mano, sin pasar por `datos.py`: los cinco archivos del generador estan
congelados mientras la campania de la query conjunta rota entre cuentas (§7 del diseño), y ademas
para esta compuerta alcanza con una secuencia de enunciados y sus cortes.

    python compuerta_sorpresa.py ckpts/c3_s0.pkl

**Lo que puede fallar, y es el modo de falla declarado:** que un enunciado de charla tenga residuo
alto por tener tokens poco frecuentes en vez de por ser informativo. Por eso hay dos controles, no
uno solo, y por eso la charla se construyo con el mismo vocabulario que los hechos.
"""
import argparse
import json
import pickle

import jax
import jax.numpy as jnp
import numpy as np

import idioma as I
import modelo as M
import relleno as R


def auc(x, y):
    """P(x > y) con empates a 0,5."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if not len(x) or not len(y):
        return float("nan")
    todo = np.concatenate([x, y])
    orden = np.argsort(todo, kind="mergesort")
    r = np.empty(len(todo))
    i = 0
    ordenado = todo[orden]
    while i < len(ordenado):
        j = i
        while j + 1 < len(ordenado) and ordenado[j + 1] == ordenado[i]:
            j += 1
        r[orden[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((r[:len(x)].sum() - len(x) * (len(x) + 1) / 2.0) / (len(x) * len(y)))


def residuos(params, x, bloque=0):
    """(T,) — `beta * ||v - S k||` en cada posicion del bloque pedido.

    Se recalcula la regla delta con el mismo scan que `modelo.delta_mixer`, exponiendo el residuo
    en vez de la lectura. La entrada al mixer es la misma: `conv3(conv, ln1(h))`.
    """
    h = params["emb"][x][None, :]
    for i, blk in enumerate(params["blocks"]):
        u = M.conv3(blk["conv"], M.ln(blk["ln1"], h))[0]
        if i == bloque:
            q, k, v = u @ blk["wq"], u @ blk["wk"], u @ blk["wv"]
            k = k / (jnp.linalg.norm(k, axis=-1, keepdims=True) + 1e-6)
            beta = jax.nn.sigmoid(blk["beta"])

            def paso(S, ent):
                ki, vi, bi = ent
                e = bi * (vi - S @ ki)
                return S + jnp.outer(e, ki), jnp.linalg.norm(e)

            _, res = jax.lax.scan(paso, jnp.zeros((h.shape[-1], h.shape[-1])),
                                  (k, v, jnp.broadcast_to(beta, k.shape)))
            return res
        h = h + jax.vmap(M.delta_mixer, in_axes=(None, 0))(blk, M.conv3(blk["conv"],
                                                                       M.ln(blk["ln1"], h)))
        h2 = M.ln(blk["ln2"], h)
        h = h + jax.nn.gelu(h2 @ blk["m1"]["w"] + blk["m1"]["b"]) @ blk["m2"]["w"] + blk["m2"]["b"]
    raise ValueError("bloque fuera de rango")


def episodio_mixto(rng, n_hechos=4, n_charla=4, n_repes=2, nivel=3):
    """Una sesion con hechos, repeticiones y charla, barajados. Devuelve (texto, etiquetas)."""
    ents = list(rng.choice(R.ENTIDADES_OK, size=n_hechos, replace=False))
    rels = list(rng.choice(list(I.RELACIONES), size=n_hechos, replace=False))
    enunciados = []
    dichos = []
    for ent, rel in zip(ents, rels):
        val = (str(rng.choice(R.NOMBRES_OK)) if rel in I.PERSONALES
               else str(rng.integers(0, 100)))
        f = I.formas(rel, ent, val, nivel)[0]
        enunciados.append((f, "hecho"))
        dichos.append((f, rel, ent, val))
    for _ in range(n_charla):
        enunciados.append((R.charla(rng), "charla"))
    idx = rng.permutation(len(enunciados))
    enunciados = [enunciados[i] for i in idx]
    # `repeticion` tiene que llegar DESPUES de su original —si no, no es redundante— pero NO siempre
    # al final. Ponerlas todas al final las media en posiciones tardias, donde el estado S ya esta
    # cargado y el residuo baja por si solo: el confound habria producido justo el resultado que se
    # espera. Se inserta en una posicion sorteada entre la del original y el final.
    for _ in range(min(n_repes, len(dichos))):
        f, _rel, _e, _v = dichos[int(rng.integers(0, len(dichos)))]
        orig = next(i for i, (t, _c) in enumerate(enunciados) if t == f)
        pos = int(rng.integers(orig + 1, len(enunciados) + 1))
        enunciados.insert(pos, (f, "repeticion"))
    return enunciados


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pesos")
    ap.add_argument("--n", type=int, default=400, help="episodios")
    ap.add_argument("--semilla", type=int, default=31337)
    ap.add_argument("--salida", default="compuerta_sorpresa_20260822.json")
    a = ap.parse_args()

    with open(a.pesos, "rb") as f:
        bulto = pickle.load(f)
    params = jax.tree_util.tree_map(jnp.asarray, bulto["params"])
    cfg = bulto.get("config", {})
    print(f"checkpoint: {a.pesos} · nivel {cfg.get('nivel')} · paso {bulto.get('paso')} · "
          f"lectura {cfg.get('donde', 'pre')}")

    rng = np.random.default_rng(a.semilla)
    fn = jax.jit(residuos)
    por_clase = {"hecho": [], "charla": [], "repeticion": []}
    largos = {"hecho": [], "charla": [], "repeticion": []}
    estrato = {"hecho": [], "charla": [], "repeticion": []}

    for _ in range(a.n):
        enun = episodio_mixto(rng)
        ids, cortes, etiq = [], [], []
        ids.append(I.STOI["BOS"])
        for txt, cl in enun:
            ids += I.a_ids(txt)
            ids.append(I.STOI["SEP"])
            cortes.append(len(ids) - 1)          # el ultimo token del enunciado (el SEP)
            etiq.append(cl)
            largos[cl].append(len(txt.split()))
        res = np.asarray(fn(params, jnp.array(ids)))
        for j, (c, cl) in enumerate(zip(cortes, etiq)):
            por_clase[cl].append(float(res[c]))
            estrato[cl].append((j, len(enun[j][0].split())))    # (posicion, largo en tokens)

    print(f"\n{a.n} episodios · residuo comprometido en la posicion de archivo\n")
    print(f"{'clase':<12} {'n':>6} {'media':>9} {'desvio':>9} {'largo medio':>12}")
    for cl in ("hecho", "charla", "repeticion"):
        v = np.array(por_clase[cl])
        print(f"{cl:<12} {len(v):>6} {v.mean():>9.4f} {v.std():>9.4f} "
              f"{np.mean(largos[cl]):>12.1f}")

    a_hc = auc(por_clase["hecho"], por_clase["charla"])
    a_hr = auc(por_clase["hecho"], por_clase["repeticion"])
    print(f"\nC-1 · AUC(hecho > charla)      {a_hc:.4f}")
    print(f"C-2 · AUC(hecho > repeticion)  {a_hr:.4f}")

    # --- EL CONTROL QUE DA EL VEREDICTO, y no es permutar etiquetas -------------------------------
    # Permutar la etiqueta destruye TODA la estructura, asi que da 0,50 pase lo que pase y no dice
    # nada: es el error que el `INFORME_SIN_ETIQUETAS` del 20-ago dejo escrito («el nulo correcto NO
    # era permutar etiquetas»). Las dos explicaciones alternativas de esta separacion son concretas
    # y hay que matarlas de frente:
    #
    #   · LARGO    — una charla tiene ~4 tokens y un hecho ~6, y el residuo podria seguir al largo.
    #   · POSICION — el estado S se va cargando, asi que el residuo baja solo a medida que avanza la
    #     secuencia; si una clase cae sistematicamente mas tarde, gana por eso.
    #
    # Se comparan entonces SOLO pares con el mismo largo y la misma posicion, y se promedia el AUC
    # dentro de cada estrato pesado por su tamaño. Si la separacion es del contenido, sobrevive.
    def auc_estratificado(c1, c2):
        e1 = {}
        for v, k in zip(por_clase[c1], estrato[c1]):
            e1.setdefault(k, []).append(v)
        e2 = {}
        for v, k in zip(por_clase[c2], estrato[c2]):
            e2.setdefault(k, []).append(v)
        num, den = 0.0, 0
        for k in set(e1) & set(e2):
            a = auc(e1[k], e2[k])
            if not np.isnan(a):
                peso = min(len(e1[k]), len(e2[k]))
                num += a * peso
                den += peso
        return (num / den, den) if den else (float("nan"), 0)

    a_hc_e, n_hc = auc_estratificado("hecho", "charla")
    a_hr_e, n_hr = auc_estratificado("hecho", "repeticion")
    print(f"\nC-1e · AUC(hecho > charla)     estratificado por (posicion, largo)  "
          f"{a_hc_e:.4f}  [n={n_hc}]")
    print(f"C-2e · AUC(hecho > repeticion) estratificado por (posicion, largo)  "
          f"{a_hr_e:.4f}  [n={n_hr}]")

    ok = (not np.isnan(a_hc_e)) and (a_hc_e >= 0.65 or a_hr_e >= 0.65)
    print("\n" + "-" * 70)
    if ok:
        print("COMPUERTA ABRE: la sorpresa distingue. La politica sorpresa-gated tiene senial")
        print("de la cual agarrarse y el experimento puede seguir al pre-registro.")
    else:
        print("COMPUERTA NO ABRE: la sorpresa NO distingue hechos de relleno en este checkpoint.")
        print("La eviction sorpresa-gated se cae aca, sin gastar GPU. Es el resultado que el §4")
        print("del diseño pedia poder obtener.")

    with open(a.salida, "w") as f:
        json.dump({"pesos": a.pesos, "auc_hecho_charla": a_hc, "auc_hecho_repeticion": a_hr,
                   "auc_hecho_charla_estrat": a_hc_e, "auc_hecho_repeticion_estrat": a_hr_e,
                   "abre": bool(ok),
                   "medias": {k: float(np.mean(v)) for k, v in por_clase.items()},
                   "n": {k: len(v) for k, v in por_clase.items()}}, f, indent=1)
    print(f"\n-> {a.salida}")


if __name__ == "__main__":
    main()
