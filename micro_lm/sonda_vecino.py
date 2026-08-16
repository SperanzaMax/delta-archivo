"""MICRO-LM · sonda del vecino: el error de identidad, ¿se escribe mal o se lee mal?

    python sonda_vecino.py ckpts/n3_s2.pkl --n 4000

Pre-registrado en `../PREREG_SONDA_VECINO.md` (SHA faebb671...), congelado antes de correr.

IDEA: cuando el modelo contesta el valor de OTRA entidad, hay dos historias posibles y no se
distinguen mirando la respuesta.

  · fallo al LEER      el archivo esta bien; bajo la consulta original compitieron dos claves.
                       -> tiene señal interna -> CONVERTIBLE en abstencion.
  · fallo al ESCRIBIR  la correccion eliptica se ligo al vecino: el archivo contiene un hecho FALSO.
                       -> al leer es indistinguible de uno verdadero -> NINGUNA abstencion lo agarra.

Se separan preguntandole al VECINO, con una consulta que lo nombra explicitamente, sobre el MISMO
episodio y el MISMO archivo. Si el vecino devuelve su propio valor, el archivo esta intacto. Si
devuelve el valor de la correccion del hecho propio, la corrupcion ocurrio al escribir.
"""
import argparse
import pickle

import numpy as np
import jax
import jax.numpy as jnp

import datos as DAT
import idioma as I
import entrenar as E
from ser import clasificar


def tensor_consulta(texto):
    ids = I.a_ids("BOS " + texto)[:DAT.T_Q]
    q = np.full(DAT.T_Q, DAT.PAD, np.int32)
    q[:len(ids)] = ids
    return q, len(ids) - 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pesos")
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--B", type=int, default=64)
    ap.add_argument("--semilla", type=int, default=31415)
    a = ap.parse_args()

    with open(a.pesos, "rb") as f:
        bulto = pickle.load(f)
    params = jax.tree_util.tree_map(jnp.asarray, bulto["params"])
    nivel = bulto["config"]["nivel"]

    rng = np.random.default_rng(a.semilla)
    # cuentas: por cada grupo (error de identidad / acierto), que hizo el vecino
    cuenta = {g: {"propio": 0, "corrupto": 0, "otro": 0, "n": 0, "rescate": 0}
              for g in ("err_identidad", "acierto")}
    vistos = 0

    while vistos < a.n:
        B = min(a.B, a.n - vistos)
        ses, cortes, turnos, mask, cons, pos, tgt, tipo, meta = DAT.lote(
            rng, B, nivel=nivel, n_hechos=4, n_sesiones=4, p_nose=0.0, con_meta=True)
        jses, jcor, jtur = jnp.array(ses), jnp.array(cortes), jnp.array(turnos)
        jmask = jnp.array(mask)
        pred = np.asarray(E.predecir(params, jses, jcor, jtur, jmask,
                                     jnp.array(cons), jnp.array(pos)))

        # Consultas nuevas, NO ambiguas, sobre el MISMO episodio: solo cambia el tensor de consulta.
        q_prop = np.full((B, DAT.T_Q), DAT.PAD, np.int32); p_prop = np.zeros(B, np.int32)
        q_vec = np.full((B, DAT.T_Q), DAT.PAD, np.int32); p_vec = np.zeros(B, np.int32)
        casos = []
        for i in range(B):
            m = meta[i]
            cat = clasificar(I.ITOS[int(pred[i])], I.ITOS[int(tgt[i])], m)
            if cat not in ("err_identidad", "acierto") or not m["hecho"]:
                casos.append(None); continue
            # el vecino: el hecho de `otros` que contiene el valor contestado (para el acierto, se
            # toma el primer `otro` como vecino de referencia, que es el control de P-3)
            dicho = I.ITOS[int(pred[i])]
            vec = next((o for o in m["otros"] if dicho in o["versiones"]), None)
            if vec is None:
                vec = m["otros"][0] if m["otros"] else None
            if vec is None:
                casos.append(None); continue
            qp, pp = tensor_consulta(I.pregunta(m["hecho"]["rel"], m["hecho"]["ent"], "vigente"))
            qv, pv = tensor_consulta(I.pregunta(vec["rel"], vec["ent"], "vigente"))
            q_prop[i], p_prop[i] = qp, pp
            q_vec[i], p_vec[i] = qv, pv
            casos.append((cat, m["hecho"], vec))

        r_prop = np.asarray(E.predecir(params, jses, jcor, jtur, jmask,
                                       jnp.array(q_prop), jnp.array(p_prop)))
        r_vec = np.asarray(E.predecir(params, jses, jcor, jtur, jmask,
                                      jnp.array(q_vec), jnp.array(p_vec)))

        for i, caso in enumerate(casos):
            if caso is None:
                continue
            cat, hecho, vec = caso
            c = cuenta[cat]; c["n"] += 1
            rv = I.ITOS[int(r_vec[i])]
            if rv == vec["versiones"][-1]:
                c["propio"] += 1                       # el vecino devuelve LO SUYO -> archivo intacto
            elif rv == hecho["versiones"][-1]:
                c["corrupto"] += 1                     # devuelve la correccion ajena -> escritura
            else:
                c["otro"] += 1
            if I.ITOS[int(r_prop[i])] == hecho["versiones"][-1]:
                c["rescate"] += 1                      # la consulta no ambigua recupera el hecho
        vistos += B

    print(f"pesos: {a.pesos}")
    print(f"nivel {nivel} · paso {bulto.get('paso','?')} · n={vistos}\n")
    for g in ("err_identidad", "acierto"):
        c = cuenta[g]; n = max(1, c["n"])
        print(f"{g}  (n={c['n']})")
        print(f"   vecino intacto (devuelve lo suyo)   {c['propio']/n:.4f}   -> clase 2 · LECTURA")
        print(f"   vecino CORRUPTO (valor ajeno)       {c['corrupto']/n:.4f}   -> clase 3 · ESCRITURA")
        print(f"   vecino devuelve otra cosa           {c['otro']/n:.4f}")
        print(f"   rescate por consulta no ambigua     {c['rescate']/n:.4f}\n")

    e = cuenta["err_identidad"]; ne = max(1, e["n"])
    ok = cuenta["acierto"]; no = max(1, ok["n"])
    p1 = e["rescate"] / ne >= 0.30
    p2 = e["corrupto"] / ne >= 0.25
    p3 = ok["corrupto"] / no <= 0.10
    print("PREDICCIONES (PREREG_SONDA_VECINO.md, SHA faebb671...)")
    print(f"  P-1  rescate >= 0,30                 {e['rescate']/ne:.4f}   {'CUMPLE' if p1 else 'NO CUMPLE'}")
    print(f"  P-2  vecino corrupto >= 0,25         {e['corrupto']/ne:.4f}   {'CUMPLE' if p2 else 'NO CUMPLE'}")
    print(f"  P-3  control: corrupto en aciertos <= 0,10   {ok['corrupto']/no:.4f}   "
          f"{'CUMPLE' if p3 else 'NO CUMPLE -> P-2 NO interpretable'}")


if __name__ == "__main__":
    main()
