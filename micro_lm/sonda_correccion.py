"""MICRO-LM · ¿la correccion eliptica se pierde al escribir?

    python sonda_correccion.py ckpts/n4_s0.pkl --n 4000

Pre-registrado en `../PREREG_CORRECCION_PERDIDA.md` (SHA c51b36b4...), congelado antes de correr.

LA TERCERA HISTORIA. La sonda del vecino dejo vecino intacto 0,83 Y rescate 0,10 a la vez: el archivo
del vecino no esta corrupto, pero el hecho propio tampoco se recupera. Eso admite algo que no estaba
en ninguna de las dos hipotesis: que la correccion no se ligue A NADIE y se pierda al escribir. El
hecho propio quedaria con su version VIEJA y el modelo contesta el valor de otra entidad porque el
suyo nuevo no existe en ningun lado.

Se testea preguntando por la version ANTERIOR del hecho propio. Si la v1 esta bien y la v2 no
aparece, el hecho SI se archivo y lo que falto fue ligar la revision.
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
from sonda_vecino import tensor_consulta


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
    cuenta = {g: {"v1": 0, "vigente": 0, "otro": 0, "n": 0}
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

        q_ant = np.full((B, DAT.T_Q), DAT.PAD, np.int32)
        p_ant = np.zeros(B, np.int32)
        casos = []
        for i in range(B):
            m = meta[i]
            cat = clasificar(I.ITOS[int(pred[i])], I.ITOS[int(tgt[i])], m)
            # Solo hechos REVISADOS: sin dos versiones, «la anterior» no existe y la pregunta no
            # tiene respuesta. Es la mitad del punto: se mide si la revision quedo ligada.
            if (cat not in ("err_identidad", "acierto") or not m["hecho"]
                    or len(m["hecho"]["versiones"]) < 2):
                casos.append(None); continue
            q, p = tensor_consulta(I.pregunta(m["hecho"]["rel"], m["hecho"]["ent"], "anterior"))
            q_ant[i], p_ant[i] = q, p
            casos.append((cat, m["hecho"]))

        r_ant = np.asarray(E.predecir(params, jses, jcor, jtur, jmask,
                                      jnp.array(q_ant), jnp.array(p_ant)))

        for i, caso in enumerate(casos):
            if caso is None:
                continue
            cat, hecho = caso
            c = cuenta[cat]; c["n"] += 1
            r = I.ITOS[int(r_ant[i])]
            if r == hecho["versiones"][-2]:
                c["v1"] += 1            # la version vieja esta -> el hecho se archivo bien
            elif r == hecho["versiones"][-1]:
                c["vigente"] += 1       # devuelve la NUEVA cuando se pide la vieja -> invierte orden
            else:
                c["otro"] += 1
        vistos += B

    print(f"pesos: {a.pesos}")
    print(f"nivel {nivel} · paso {bulto.get('paso','?')} · n={vistos}\n")
    for g in ("err_identidad", "acierto"):
        c = cuenta[g]; n = max(1, c["n"])
        print(f"{g}  (hechos revisados n={c['n']})")
        print(f"   acierta la ANTERIOR (v1)        {c['v1']/n:.4f}   -> el hecho SI esta archivado")
        print(f"   devuelve la vigente             {c['vigente']/n:.4f}   -> tiene las dos, invierte orden")
        print(f"   otra cosa                       {c['otro']/n:.4f}\n")

    e = cuenta["err_identidad"]; ne = max(1, e["n"])
    ok = cuenta["acierto"]; no = max(1, ok["n"])
    p1 = e["v1"] / ne >= 0.50
    p2 = ok["v1"] / no >= 0.50
    p3 = ok["v1"] / no > e["v1"] / ne
    print("PREDICCIONES (PREREG_CORRECCION_PERDIDA.md, SHA c51b36b4...)")
    print(f"  P-1  anterior en errores >= 0,50    {e['v1']/ne:.4f}   {'CUMPLE' if p1 else 'NO CUMPLE'}")
    print(f"  P-2  control: en aciertos >= 0,50   {ok['v1']/no:.4f}   "
          f"{'CUMPLE' if p2 else 'NO CUMPLE -> P-1 no interpretable'}")
    print(f"  P-3  aciertos > errores             {ok['v1']/no:.4f} vs {e['v1']/ne:.4f}   "
          f"{'CUMPLE' if p3 else 'NO CUMPLE'}")


if __name__ == "__main__":
    main()
