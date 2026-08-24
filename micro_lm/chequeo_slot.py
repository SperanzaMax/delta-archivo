"""Chequeo de instrumento del SLOT NULO (`DISENO_ATRIBUCION.md` §7).

Se corre ANTES del pre-registro y antes de gastar un minuto de GPU, por la regla que dejo el monitor
v1 del 20-ago: lo primero que se verifica de una reparacion es que la reparacion HAGA algo, y que lo
que hace sea lo que dice.

  A-1  Con el slot agregado la lectura sigue sumando 1, y la masa del nulo con pesos al azar esta en
       el orden de 1/(N+1). Si diera 0 el slot no compite; si diera ~1 se come toda la atencion y el
       modelo no leeria el archivo.

  A-2  El slot NO cambia nada cuando no se lo usa. Con `abst != "slot"` el modelo tiene que dar
       EXACTAMENTE lo mismo que antes de agregarlo. Es lo que protege a `token`, `escala` y `cabeza`,
       que son los controles ya corridos — misma funcion que K-5 en `lat2`.

  A-3  **La que importa.** La masa del nulo responde al CONTENIDO: al tapar del archivo la entrada
       del hecho preguntado, la masa del nulo tiene que SUBIR. Se mide sobre un checkpoint entrenado
       (`p3_s0`), que es gratis y esta en disco.

       Ojo con lo que A-3 puede y no puede decir: `p3_s0` se entreno SIN slot, asi que su `k_nulo`
       esta en su valor inicial y nunca recibio gradiente. Un efecto positivo aca seria una sorpresa
       fuerte; uno nulo NO refuta el diseño, solo dice que el mecanismo depende enteramente de la
       supervision. Las dos lecturas van escritas antes de mirar el numero.

  A-3b **EL CONTROL QUE HACE VALER A A-3, y sin el A-3 miente.** Al tapar CUALQUIER entrada su masa
       se reparte entre las que quedan, el slot incluido, asi que el nulo sube por construccion del
       softmax sin detectar nada. Se tapa una entrada IRRELEVANTE —viva y distinta de la del hecho—
       y se compara. Corrido el 24-ago: tapar el hecho da +0,02779 y tapar una irrelevante +0,03128,
       o sea el efecto especifico es **NEGATIVO** (-0,0035). **A-3 era redistribucion, no
       deteccion.** Sin este control se habria anotado un falso positivo en el pre-registro.

  A-4  El gradiente llega a `k_nulo` y `v_nulo` (el chequeo que en `lat2` destapo el weight decay).

Costo: CPU, segundos. No toca checkpoints ni corridas.
"""
import json
import os
import pickle

import jax
import jax.numpy as jnp
import numpy as np

import datos as DAT
import idioma as I
import modelo as M

AQUI = os.path.dirname(os.path.abspath(__file__))
SEMILLA = 24
D, NB = 128, 4


def masa_del_nulo(params, ses, cortes, turnos, mask, cons, pos, abst="slot"):
    """Devuelve la masa de atencion que se va al slot, en la posicion de respuesta."""
    archivo = M.escribir(params, ses, cortes)
    _, a = M.responder_con_abst(params, archivo, turnos, cons, mask, donde="pre", abst=abst)
    a = jnp.take_along_axis(a, pos[:, None], axis=1)[:, 0]
    return jax.nn.sigmoid(a)          # el logit vuelve a masa


def main():
    I.fijar_version(2)
    res = {}
    print("=" * 78)
    print("CHEQUEO DE INSTRUMENTO · slot nulo")
    print("=" * 78)

    rng = np.random.default_rng(SEMILLA)
    sal = DAT.lote(rng, 64, nivel=3, n_hechos=4, n_sesiones=4, p_vieja=0.35, p_nose=0.4,
                   con_meta=True, con_origen=True)
    ses, cortes, turnos, mask, cons, pos, tgt = sal[:7]
    ses, cortes, turnos = jnp.array(ses), jnp.array(cortes), jnp.array(turnos)
    cons, pos = jnp.array(cons), jnp.array(pos)
    maskj = jnp.array(mask)
    N = np.asarray(mask).shape[1]

    p = M.init_params(SEMILLA, I.V, D=D, NB=NB)

    # --- A-1 · el slot compite, y no se come todo -------------------------------------------------
    m = np.asarray(masa_del_nulo(p, ses, cortes, turnos, maskj, cons, pos))
    esperado = 1.0 / (N + 1)
    a1 = 0.2 * esperado < m.mean() < 5 * esperado
    print(f"\nA-1 · masa del nulo con pesos al azar")
    print(f"     media {m.mean():.5f} · min {m.min():.5f} · max {m.max():.5f}")
    print(f"     referencia 1/(N+1) = 1/{N + 1} = {esperado:.5f}   (N = {N} entradas de archivo)")
    print(f"     A-1: {'CUMPLE' if a1 else 'NO CUMPLE'}")
    res["A-1"] = {"media": float(m.mean()), "esperado": esperado, "cumple": bool(a1)}

    # --- A-2 · sin `--abst slot`, nada cambia -----------------------------------------------------
    arch = M.escribir(p, ses, cortes)
    lg_sin, a_sin = M.responder_con_abst(p, arch, turnos, cons, maskj, donde="pre", abst="cabeza")
    p2 = {k: v for k, v in p.items() if k != "arch"}
    p2["arch"] = {k: v for k, v in p["arch"].items() if k not in ("k_nulo", "v_nulo")}
    arch2 = M.escribir(p2, ses, cortes)
    lg_ref, a_ref = M.responder_con_abst(p2, arch2, turnos, cons, maskj, donde="pre", abst="cabeza")
    d_lg = float(jnp.max(jnp.abs(lg_sin - lg_ref)))
    d_a = float(jnp.max(jnp.abs(a_sin - a_ref)))
    a2 = d_lg == 0.0 and d_a == 0.0
    print(f"\nA-2 · con abst != slot, el arbol con slot da lo mismo que el arbol sin slot")
    print(f"     logits maxabs {d_lg:.3e} · cabeza maxabs {d_a:.3e}   (hace falta 0,0 EXACTO)")
    print(f"     A-2: {'CUMPLE' if a2 else 'NO CUMPLE'}")
    res["A-2"] = {"logits": d_lg, "abst": d_a, "cumple": bool(a2)}

    # --- A-4 · el gradiente llega al slot ---------------------------------------------------------
    def perd(pp):
        archivo = M.escribir(pp, ses, cortes)
        lg, a = M.responder_con_abst(pp, archivo, turnos, cons, maskj, donde="pre", abst="slot")
        a = jnp.take_along_axis(a, pos[:, None], axis=1)[:, 0]
        es_nose = (jnp.array(tgt) == I.STOI["NOSE"]).astype(jnp.float32)
        return jnp.mean(jnp.maximum(a, 0) - a * es_nose + jnp.log1p(jnp.exp(-jnp.abs(a))))

    g = jax.grad(perd)(p)
    gk = float(jnp.max(jnp.abs(g["arch"]["k_nulo"])))
    gv = float(jnp.max(jnp.abs(g["arch"]["v_nulo"])))
    a4 = gk > 0.0
    print(f"\nA-4 · el gradiente llega al slot")
    print(f"     |grad k_nulo| max {gk:.3e}   (hace falta > 0)")
    print(f"     |grad v_nulo| max {gv:.3e}   (puede ser 0: v_nulo arranca en cero y solo entra por")
    print(f"                                   la salida, no por la decision de abstenerse)")
    print(f"     A-4: {'CUMPLE' if a4 else 'NO CUMPLE'}")
    res["A-4"] = {"k_nulo": gk, "v_nulo": gv, "cumple": bool(a4)}

    # --- A-3 · LA QUE IMPORTA · la masa responde al contenido -------------------------------------
    ck = os.path.join(AQUI, "ckpts", "p3_s0.pkl")
    print(f"\nA-3 · la masa del nulo sube cuando se tapa la entrada del hecho preguntado")
    if not os.path.exists(ck):
        print("     sin checkpoint p3_s0: A-3 no se puede correr")
        res["A-3"] = {"cumple": None}
    else:
        with open(ck, "rb") as f:
            d = pickle.load(f)
        pe = jax.tree_util.tree_map(jnp.asarray, d["params"])
        # el checkpoint es anterior al slot: se le agrega en su valor inicial
        pe["arch"]["k_nulo"] = p["arch"]["k_nulo"]
        pe["arch"]["v_nulo"] = p["arch"]["v_nulo"]
        origen = np.asarray(sal[10])                 # hecho_q -> indice de su entrada en el archivo
        mk = np.asarray(mask).copy()
        base = np.asarray(masa_del_nulo(pe, ses, cortes, turnos, jnp.array(mk), cons, pos))
        val = origen[:, 0] if origen.ndim > 1 else origen
        ok = (val >= 0) & (val < mk.shape[1])
        mk2 = mk.copy()
        mk2[np.arange(len(val))[ok], val[ok]] = False
        tap = np.asarray(masa_del_nulo(pe, ses, cortes, turnos, jnp.array(mk2), cons, pos))
        dif = float((tap[ok] - base[ok]).mean())
        a3 = dif > 0
        print(f"     masa media  sin tapar {base[ok].mean():.5f}  ->  tapando {tap[ok].mean():.5f}"
              f"   ({dif:+.5f})")
        print(f"     medido sobre {ok.sum()} de {len(val)} consultas")
        print(f"     A-3: {'sube' if a3 else 'NO sube'}")
        print(f"     LECTURA declarada antes de mirar: p3_s0 se entreno SIN slot, asi que k_nulo")
        print(f"     nunca recibio gradiente. Que suba seria una sorpresa fuerte; que no suba NO")
        print(f"     refuta el diseño, dice que el mecanismo depende enteramente de la supervision.")
        # --- A-3b · el control de redistribucion ---------------------------------------------
        r = np.random.default_rng(999)
        mk3 = mk.copy()
        eleg = np.full(len(val), -1)
        for i in range(len(val)):
            cand = [j for j in range(mk.shape[1]) if mk[i, j] and j != val[i]]
            if cand:
                j = int(r.choice(cand)); mk3[i, j] = False; eleg[i] = j
        ok2 = eleg >= 0
        irr = np.asarray(masa_del_nulo(pe, ses, cortes, turnos, jnp.array(mk3), cons, pos))
        d_irr = float((irr[ok2] - base[ok2]).mean())
        espec = dif - d_irr
        print(f"\nA-3b · CONTROL de redistribucion: tapar una entrada IRRELEVANTE")
        print(f"     tapando el HECHO        {dif:+.5f}")
        print(f"     tapando una IRRELEVANTE {d_irr:+.5f}   <- control")
        print(f"     efecto especifico       {espec:+.5f}")
        print(f"     A-3b: {'hay efecto especifico' if espec > 0 else 'A-3 es REDISTRIBUCION, no deteccion'}")
        res["A-3"] = {"base": float(base[ok].mean()), "tapado": float(tap[ok].mean()),
                      "dif": dif, "sube": bool(a3),
                      "irrelevante": d_irr, "especifico": espec,
                      "veredicto": "redistribucion" if espec <= 0 else "especifico"}

    with open(os.path.join(AQUI, "chequeo_slot_20260824.json"), "w") as f:
        json.dump(res, f, indent=1)
    print("\n-> chequeo_slot_20260824.json")
    duras = [k for k in ("A-1", "A-2", "A-4") if not res[k]["cumple"]]
    print(f"\nRESUMEN: {'A-1, A-2 y A-4 CUMPLEN' if not duras else 'FALLAN ' + ', '.join(duras)}")


if __name__ == "__main__":
    main()
