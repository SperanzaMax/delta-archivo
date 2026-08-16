"""MICRO-LM · Fase 0: ¿el score de matcheo del archivo separa PRESENCIA de AUSENCIA?

    python score_archivo.py ckpts/n4_s0.pkl --n 4000 --p-nose 0.4

Pre-registrado en `../PREREG_SCORE_ARCHIVO.md` (SHA fea5e061...), congelado antes de correr esto.

QUE MIDE, y en que se diferencia de `mitigar.py`:

  mitigar.py  ->  AUC(aciertos, errores).           Eje: ¿el modelo sabe cuando SE EQUIVOCA?
  este        ->  AUC(con respuesta, sin respuesta). Eje: ¿el modelo sabe cuando NO SABE?

No son el mismo numero y no se comparan. Por eso aca se recomputan TAMBIEN las tres señales de
salida sobre ESTE eje: la comparacion legitima es score-de-archivo contra salida sobre el mismo eje,
no contra el 0,7397 del 15-ago, que vive en el otro.

DONDE SE TOMA EL SCORE: `modelo.responder` inyecta la lectura en el bloque 0 y la calcula sobre
`ln(blocks[0].ln1, emb[consulta])`. Se replica exactamente ese calculo y se corta ANTES del softmax,
que es donde la magnitud del matcheo todavia existe: despues del softmax la masa suma 1 siempre y la
informacion de «nada matchea» ya se perdio. Ese detalle es, ademas, el argumento mecanico a favor
del slot nulo.
"""
import argparse
import pickle

import numpy as np
import jax
import jax.numpy as jnp

import datos as DAT
import idioma as I
import modelo as M
import entrenar as E

TIPOS = {0: "vigente", 1: "anterior", 2: "nose_ent", 3: "nose_rel"}


def auc(pos, neg):
    """AUC de Mann-Whitney: P(un positivo puntue mas alto que un negativo)."""
    if not len(pos) or not len(neg):
        return float("nan")
    todo = np.concatenate([pos, neg])
    r = np.argsort(np.argsort(todo)) + 1
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def scores_archivo(params, ses, cortes, turnos, mask, consulta, pos_q):
    """Replica la lectura del bloque 0 y devuelve el `sim` PRE-softmax en la posicion de la pregunta.

    Devuelve (B, N): un score por entrada del archivo, con las vacias ya en -1e9.
    """
    a = params["arch"]
    archivo = M.escribir(params, ses, cortes)                       # (B, N, D)
    ak = archivo @ a["kw"] + a["ord"][turnos]
    penal = jnp.where(mask, 0.0, -1e9)

    h = params["emb"][consulta]                                     # (B, Tq, D) — igual que tronco
    h_ln = M.ln(params["blocks"][0]["ln1"], h)
    q = h_ln @ a["qr"]
    sim = jnp.einsum("btd,bnd->btn", q, ak) / jnp.sqrt(h.shape[-1]) + penal[:, None, :]
    # Solo interesa la posicion desde la que se decide la respuesta.
    return jnp.take_along_axis(sim, pos_q[:, None, None], axis=1)[:, 0, :]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pesos")
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--B", type=int, default=64)
    ap.add_argument("--nivel", type=int, default=None)
    ap.add_argument("--p-nose", type=float, default=0.4)
    ap.add_argument("--semilla", type=int, default=31415)
    a = ap.parse_args()

    with open(a.pesos, "rb") as f:
        bulto = pickle.load(f)
    params = jax.tree_util.tree_map(jnp.asarray, bulto["params"])
    cfg = bulto["config"]
    nivel = a.nivel if a.nivel is not None else cfg["nivel"]

    rng = np.random.default_rng(a.semilla)
    col = {k: [] for k in ("s_max", "s_margen", "s_lse", "c_prob", "c_margen", "c_entropia")}
    tipos, vistos = [], 0

    while vistos < a.n:
        B = min(a.B, a.n - vistos)
        ses, cortes, turnos, mask, cons, pos, tgt, tipo = DAT.lote(
            rng, B, nivel=nivel, n_hechos=4, n_sesiones=4, p_nose=a.p_nose)
        ses, cortes, turnos = jnp.array(ses), jnp.array(cortes), jnp.array(turnos)
        mask, cons, pos = jnp.array(mask), jnp.array(cons), jnp.array(pos)

        s = np.asarray(scores_archivo(params, ses, cortes, turnos, mask, cons, pos))
        s_ord = np.sort(s, -1)
        col["s_max"].extend(s_ord[:, -1])
        col["s_margen"].extend(s_ord[:, -1] - s_ord[:, -2])
        # logsumexp sobre las entradas validas; las vacias estan en -1e9 y no aportan masa.
        col["s_lse"].extend(jax.scipy.special.logsumexp(jnp.array(s), axis=-1))

        lg = E.logits_de(params, ses, cortes, turnos, mask, cons, pos)
        p = np.asarray(jax.nn.softmax(lg, -1))
        p_ord = np.sort(p, -1)
        col["c_prob"].extend(p_ord[:, -1])
        col["c_margen"].extend(p_ord[:, -1] - p_ord[:, -2])
        col["c_entropia"].extend((p * np.log(p + 1e-12)).sum(-1))     # negada: mas alto = mas seguro

        tipos.extend(np.asarray(tipo).tolist())
        vistos += B

    tipos = np.array(tipos)
    col = {k: np.array(v) for k, v in col.items()}

    con = tipos <= 1                    # vigente o anterior: la respuesta ESTA en el archivo
    ent = tipos == 2                    # la entidad nunca se nombro
    rel = tipos == 3                    # la entidad si, la relacion no
    sin = ent | rel

    print(f"pesos: {a.pesos}")
    print(f"nivel {nivel} · paso {bulto.get('paso','?')} · p_nose {a.p_nose} · n={len(tipos)}")
    print(f"con respuesta {con.sum()} · sin respuesta {sin.sum()} "
          f"(nose_ent {ent.sum()} · nose_rel {rel.sum()})")

    # §6 del prereg: si el reparto ent/rel no esta cerca de 50/50, P-3 no es interpretable.
    if sin.sum():
        prop = ent.sum() / sin.sum()
        estado = "OK" if 0.40 <= prop <= 0.60 else "FUERA DE RANGO -> P-3 no interpretable"
        print(f"control de reparto: nose_ent/sin_resp = {prop:.4f}   {estado}")

    print("\nAUC(con respuesta, sin respuesta) — mas alto = mejor separa PRESENCIA de AUSENCIA")
    print(f"  {'señal':<12} {'AUC':>8}   {'nose_ent':>9} {'nose_rel':>9}")
    res = {}
    for k in ("s_max", "s_margen", "s_lse", "c_prob", "c_margen", "c_entropia"):
        v = col[k]
        A = auc(v[con], v[sin])
        Ae = auc(v[con], v[ent])
        Ar = auc(v[con], v[rel])
        res[k] = (A, Ae, Ar)
        marca = "  <- archivo" if k.startswith("s_") else ""
        print(f"  {k:<12} {A:8.4f}   {Ae:9.4f} {Ar:9.4f}{marca}")

    print("\nPREDICCIONES (comprometidas en PREREG_SCORE_ARCHIVO.md, SHA fea5e061...)")
    a_max = res["s_max"][0]
    a_prob = res["c_prob"][0]
    p1 = a_max >= 0.60
    p2 = a_max > a_prob + 0.03
    p3 = res["s_max"][1] > res["s_max"][2] + 0.10
    print(f"  P-1  AUC(s_max) >= 0,60                    {a_max:.4f}            "
          f"{'CUMPLE' if p1 else 'NO CUMPLE'}")
    print(f"  P-2  AUC(s_max) > AUC(c_prob) + 0,03       {a_max:.4f} vs {a_prob:.4f}   "
          f"{'CUMPLE' if p2 else 'NO CUMPLE'}")
    print(f"  P-3  ent > rel + 0,10                      {res['s_max'][1]:.4f} vs "
          f"{res['s_max'][2]:.4f}   {'CUMPLE' if p3 else 'NO CUMPLE'}")
    print("\nNo se imprime veredicto global a proposito: el 13-ago un veredicto automatico dijo lo")
    print("contrario de lo que mostraban los numeros. Las predicciones se leen de a una.")


if __name__ == "__main__":
    main()
