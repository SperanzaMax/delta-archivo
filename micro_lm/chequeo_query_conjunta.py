"""Chequeo de instrumento del experimento de la QUERY CONJUNTA (2026-08-22).

Se corre ANTES de escribir el pre-registro y antes de gastar un minuto de GPU. La regla sale del
monitor v1 (20-ago): aquel perturbaba permutando el archivo, la atencion softmax es invariante a eso
por construccion, y el instrumento quedo VACIO —cero en todo— sin que nada lo avisara hasta el smoke.
Asi que lo primero que se verifica de una reparacion es que la reparacion HAGA algo, y que lo que
hace sea lo que dice.

Dos afirmaciones, las dos falsables con pesos al azar y sin entrenar nada:

  C-1  En `pre` la query de lectura es funcion PURA del token de su posicion. Se interviene el
       CONTEXTO (se cambian los tokens anteriores, se deja fijo el token de la posicion medida) y la
       query no se mueve NADA. Predicho: delta relativo < 1e-6 (cero de punto flotante).
       Esto no es un supuesto del experimento nuevo: es la afirmacion del diagnostico del 21-ago,
       que hasta hoy estaba leida del codigo y nunca medida. Si falla, el que se cae es el
       diagnostico de ayer, no el arreglo de hoy.

  C-2  En `post` la misma intervencion SI mueve la query. Predicho: delta relativo > 0,01.
       Es la condicion de que el arreglo no sea un instrumento vacio.

  C-3  Corolario, la forma en que el defecto se ve sin intervenir: en `pre`, dos posiciones con el
       MISMO token tienen queries identicas aunque esten en contextos distintos. En `post`, no.
       C-3 es lo que conecta el mecanismo con la colision de clave: si la query del token de la
       relacion es la misma en todos lados, matchea a todas las entradas que comparten esa relacion
       y el empate esta garantizado por construccion.

Costo: CPU, segundos, pesos al azar. No toca checkpoints ni corridas.
"""
import json

import jax
import jax.numpy as jnp
import numpy as np

import modelo as M

V, D, NB = 242, 128, 4          # el tamaño de la campania de abstencion (config de c3_s0)
T = 24
SEMILLA = 22


def queries(params, x, donde):
    """Devuelve (T, D): la query de lectura en cada posicion, para la condicion `donde`.

    Se pasa una `lectura` que GRABA su entrada y devuelve ceros: asi el forward queda intacto y el
    diagnostico no puede alterar lo que mide.
    """
    capturado = {}

    def lectura(h):
        capturado["h"] = h
        return jnp.zeros_like(h)

    M.tronco(params, x, lectura, 0, donde)
    return capturado["h"][0] @ params["arch"]["qr"]


def delta_relativo(a, b):
    """Cuanto se movio la query, en unidades de su propio tamaño."""
    return float(jnp.linalg.norm(a - b) / (jnp.linalg.norm(a) + 1e-9))


def main():
    params = M.init_params(SEMILLA, V, D=D, NB=NB)
    rng = np.random.default_rng(SEMILLA)

    # Dos secuencias que COINCIDEN en la segunda mitad y difieren en la primera. La posicion que se
    # mide es la ultima: mismo token, mismo sufijo inmediato, contexto lejano distinto.
    base = rng.integers(0, V, size=T)
    otra = base.copy()
    otra[:T // 2] = rng.integers(0, V, size=T // 2)
    p = T - 1
    assert base[p] == otra[p], "la intervencion tiene que dejar fijo el token de la posicion medida"

    x1 = jnp.array(base)[None, :]
    x2 = jnp.array(otra)[None, :]

    res = {}
    for donde in ("pre", "post", "lat"):
        q1, q2 = queries(params, x1, donde), queries(params, x2, donde)
        res[donde] = {"delta_contexto": delta_relativo(q1[p], q2[p])}

    # L-2 · el contexto LEJANO. `lat` tiene que depender de las dos posiciones anteriores y de nada
    # mas: la conv de kernel 3 no ve mas atras. Si dependiera del contexto lejano seria `post` con
    # otro nombre —habria reintroducido el mixer— y el experimento volveria a mezclar los dos
    # factores que el informe de la mañana pide separar.
    lejos = base.copy()
    lejos[p - 5] = (int(base[p - 5]) + 7) % V          # un solo token, cinco posiciones atras
    cerca = base.copy()
    cerca[p - 1] = (int(base[p - 1]) + 7) % V          # un solo token, la posicion anterior
    for donde in ("pre", "post", "lat"):
        q0 = queries(params, jnp.array(base)[None, :], donde)
        res[donde]["delta_lejano"] = delta_relativo(
            q0[p], queries(params, jnp.array(lejos)[None, :], donde)[p])
        res[donde]["delta_vecino"] = delta_relativo(
            q0[p], queries(params, jnp.array(cerca)[None, :], donde)[p])

    # C-3: dos posiciones distintas con el MISMO token, dentro de una sola secuencia.
    rep = rng.integers(0, V, size=T)
    tok = int(rep[3])
    rep[3], rep[T - 2] = tok, tok
    xr = jnp.array(rep)[None, :]
    for donde in ("pre", "post", "lat"):
        q = queries(params, xr, donde)
        res[donde]["delta_mismo_token"] = delta_relativo(q[3], q[T - 2])

    c1 = res["pre"]["delta_contexto"] < 1e-6
    c2 = res["post"]["delta_contexto"] > 0.01
    c3 = res["pre"]["delta_mismo_token"] < 1e-6 and res["post"]["delta_mismo_token"] > 0.01

    print(f"{'':6} {'contexto':>12} {'vecino p-1':>12} {'lejano p-5':>12} {'mismo token':>12}")
    for donde in ("pre", "post", "lat"):
        r = res[donde]
        print(f"{donde:6} {r['delta_contexto']:12.8f} {r['delta_vecino']:12.8f} "
              f"{r['delta_lejano']:12.8f} {r['delta_mismo_token']:12.8f}")
    print()
    print(f"C-1 (pre es funcion pura del token)   : {'CUMPLE' if c1 else 'NO CUMPLE'}")
    print(f"C-2 (post depende del contexto)       : {'CUMPLE' if c2 else 'NO CUMPLE'}")
    print(f"C-3 (mismo token -> misma query en pre): {'CUMPLE' if c3 else 'NO CUMPLE'}")

    # --- el camino lateral (22-ago, tarde) --------------------------------------------------------
    l1 = res["lat"]["delta_vecino"] > 0.01      # SI depende del token anterior
    l2 = res["lat"]["delta_lejano"] < 1e-6      # NO depende del lejano: es contexto local, no global
    l3 = res["pre"]["delta_vecino"] < 1e-6      # y `pre` sigue siendo funcion pura del token
    print(f"L-1 (lat depende del vecino p-1)      : {'CUMPLE' if l1 else 'NO CUMPLE'}")
    print(f"L-2 (lat NO depende del lejano p-5)   : {'CUMPLE' if l2 else 'NO CUMPLE'}")
    print(f"L-3 (pre no depende ni del vecino)    : {'CUMPLE' if l3 else 'NO CUMPLE'}")
    ok = c1 and c2 and c3 and l1 and l2 and l3
    print()
    print("VEREDICTO:", "instrumento VALIDO" if ok else "REVISAR — no seguir")

    res["veredicto"] = {"C1": bool(c1), "C2": bool(c2), "C3": bool(c3),
                        "L1": bool(l1), "L2": bool(l2), "L3": bool(l3), "valido": bool(ok)}
    with open("chequeo_query_conjunta.json", "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
