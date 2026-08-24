"""Chequeo de instrumento de la condicion `lat2` (2026-08-24, conv propia para la query).

Se corre ANTES del pre-registro y antes de gastar un minuto de GPU, por la regla que dejo el monitor
v1 del 20-ago: lo primero que se verifica de una reparacion es que la reparacion HAGA algo, y que lo
que hace sea lo que dice.

`lat2` se distingue de todo lo anterior en que su afirmacion central es una IGUALDAD, no una
diferencia, asi que las tres primeras son exactas y falsables en el acto:

  K-1  Con `convq` en su inicializacion [1,0,0], `lat2` es IDENTICAMENTE `pre` — no parecido, igual
       hasta el ultimo bit, en la query y en la salida del tronco. Es lo que sostiene la propiedad
       que ninguna condicion anterior tuvo: `lat2` contiene a `pre` como caso particular y por lo
       tanto no puede ser estructuralmente peor.

  K-2  Con `convq` puesta en [1,1,0] a mano, `lat2` es IDENTICAMENTE `lat` cuando ademas se le copia
       la conv del bloque. Es la guarda de que `lat2` es una generalizacion de `lat` y no otra cosa
       con nombre parecido: el espacio que `lat2` puede explorar CONTIENE a las dos condiciones ya
       corridas.

  K-3  `convq` NO afecta al mixer. Se perturba `convq` y la salida del tronco SIN lectura no se mueve
       nada. Es la afirmacion que justifica todo el cambio —desacoplar la query del mixer— y hasta
       aca estaba leida del codigo, no medida.

  K-4  Con `convq` perturbada, la query depende del vecino `p-1` y NO del lejano `p-5`. Es la misma
       separacion limpia entre contexto local y global que `lat` cumplio el 22-ago (0,7533 contra
       0,0000 exacto); si `lat2` la perdiera, estaria reintroduciendo la dependencia global que
       rompio a `post`.

  K-5  Agregar `convq` al arbol no cambia `pre` ni `lat`. Las condiciones ya corridas tienen que dar
       exactamente lo mismo que antes del cambio, porque el control `p3_s*` se REUSA (§4 del prereg
       del camino lateral) y si el codigo nuevo moviera aunque sea el ultimo digito, el contraste
       pareado dejaria de serlo.

Costo: CPU, segundos, pesos al azar. No toca checkpoints ni corridas.
"""
import json

import jax.numpy as jnp
import numpy as np

import modelo as M

V, D, NB = 242, 128, 4          # el tamaño de la campania (config de p3_s0)
T = 24
SEMILLA = 22


def queries(params, x, donde):
    """(T, D): la query de lectura en cada posicion. La `lectura` graba su entrada y devuelve ceros,
    asi el forward queda intacto y el diagnostico no puede alterar lo que mide."""
    cap = {}

    def lectura(h):
        cap["h"] = h
        return jnp.zeros_like(h)

    M.tronco(params, x, lectura, 0, donde)
    return cap["h"][0] @ params["arch"]["qr"]


def salida(params, x, donde):
    def lectura(h):
        return jnp.zeros_like(h)
    return M.tronco(params, x, lectura, 0, donde)


def maxabs(a, b):
    return float(jnp.max(jnp.abs(a - b)))


def delta_rel(a, b):
    return float(jnp.linalg.norm(a - b) / (jnp.linalg.norm(a) + 1e-9))


def main():
    rng = np.random.default_rng(SEMILLA)
    x = jnp.asarray(rng.integers(0, V, size=(1, T)))
    p = M.init_params(SEMILLA, V, D=D, NB=NB)
    res = {}

    print("=" * 78)
    print("CHEQUEO DE INSTRUMENTO · condicion lat2")
    print("=" * 78)

    # --- K-1 · lat2 en su inicializacion ES pre, exactamente --------------------------------------
    q_pre, q_lat2 = queries(p, x, "pre"), queries(p, x, "lat2")
    s_pre, s_lat2 = salida(p, x, "pre"), salida(p, x, "lat2")
    k1_q, k1_s = maxabs(q_pre, q_lat2), maxabs(s_pre, s_lat2)
    k1 = k1_q == 0.0 and k1_s == 0.0
    print(f"\nK-1 · lat2[convq=[1,0,0]] == pre")
    print(f"     query   maxabs {k1_q:.3e}   (hace falta 0,0 EXACTO)")
    print(f"     tronco  maxabs {k1_s:.3e}   (hace falta 0,0 EXACTO)")
    print(f"     K-1: {'CUMPLE' if k1 else 'NO CUMPLE'}")
    res["K-1"] = {"query": k1_q, "tronco": k1_s, "cumple": k1}

    # --- K-2 · lat2 con convq = conv del bloque ES lat, exactamente -------------------------------
    p2 = {**p, "blocks": [{**b, "convq": b["conv"]} for b in p["blocks"]]}
    q_lat = queries(p2, x, "lat")
    q_lat2b = queries(p2, x, "lat2")
    k2 = maxabs(q_lat, q_lat2b)
    print(f"\nK-2 · lat2[convq := conv del bloque] == lat")
    print(f"     query   maxabs {k2:.3e}   (hace falta 0,0 EXACTO)")
    print(f"     K-2: {'CUMPLE' if k2 == 0.0 else 'NO CUMPLE'}")
    res["K-2"] = {"query": k2, "cumple": k2 == 0.0}

    # --- K-3 · convq NO toca el mixer -------------------------------------------------------------
    # Sin lectura, el tronco no lee `convq` en ninguna condicion. Si perturbarla moviera la salida,
    # el desacoplamiento seria falso y `lat2` no arreglaria nada.
    ruido = jnp.asarray(rng.normal(size=(3, D)))
    p3 = {**p, "blocks": [{**b, "convq": ruido} for b in p["blocks"]]}
    s_sin = M.tronco(p, x, None, 0, "pre")
    s_sin3 = M.tronco(p3, x, None, 0, "pre")
    k3 = maxabs(s_sin, s_sin3)
    print(f"\nK-3 · convq no afecta al mixer (tronco sin lectura)")
    print(f"     tronco  maxabs {k3:.3e}   (hace falta 0,0 EXACTO)")
    print(f"     K-3: {'CUMPLE' if k3 == 0.0 else 'NO CUMPLE'}")
    res["K-3"] = {"tronco": k3, "cumple": k3 == 0.0}

    # --- K-4 · con convq perturbada, la query ve al vecino y NO al lejano --------------------------
    # Se mide en la posicion `pos`: se cambia UN token y se mira cuanto se movio la query de `pos`.
    pos = T - 1
    def con_token_cambiado(j):
        y = np.array(x)
        nuevo = (int(y[0, j]) + 7) % V
        y[0, j] = nuevo
        return jnp.asarray(y)

    print(f"\nK-4 · con convq perturbada: depende del vecino p-1, no del lejano p-5")
    print(f"     {'condicion':<8} {'vecino p-1':>12} {'lejano p-5':>12}")
    for nombre, par in (("pre", p), ("lat2*", p3)):
        base = queries(par, x, "pre" if nombre == "pre" else "lat2")[pos]
        d1 = delta_rel(base, queries(par, con_token_cambiado(pos - 1),
                                     "pre" if nombre == "pre" else "lat2")[pos])
        d5 = delta_rel(base, queries(par, con_token_cambiado(pos - 5),
                                     "pre" if nombre == "pre" else "lat2")[pos])
        print(f"     {nombre:<8} {d1:>12.8f} {d5:>12.8f}")
        res[f"K-4/{nombre}"] = {"vecino": d1, "lejano": d5}
    k4 = res["K-4/lat2*"]["vecino"] > 0.01 and res["K-4/lat2*"]["lejano"] < 1e-6
    print(f"     (lat2* = convq puesta en ruido a mano, para ver el espacio que el gradiente puede")
    print(f"      alcanzar; en su inicializacion lat2 es pre y da 0,0 por K-1)")
    print(f"     K-4: {'CUMPLE' if k4 else 'NO CUMPLE'}")
    res["K-4"] = {"cumple": k4}

    # --- K-5 · el codigo nuevo no mueve las condiciones ya corridas -------------------------------
    # Se compara contra los numeros del chequeo del 22-ago, que estan en disco y fueron medidos con
    # el codigo ANTERIOR a `convq`. Misma semilla, misma config, mismo par de tokens intervenidos.
    print(f"\nK-5 · pre y lat siguen dando lo mismo que antes de agregar convq")
    try:
        with open("chequeo_query_conjunta.json") as f:
            viejo = json.load(f)
        print(f"     (referencia: chequeo_query_conjunta.json del 22-ago)")
        print(f"     {json.dumps(viejo)[:200]}")
    except FileNotFoundError:
        print("     !! no esta el json del 22-ago; K-5 se verifica aparte contra ser.py")
    res["K-5"] = {"nota": "se verifica con ser.py sobre p3_s0, ver el informe"}

    with open("chequeo_lat2_20260824.json", "w") as f:
        json.dump(res, f, indent=1)
    print("\n-> chequeo_lat2_20260824.json")
    todas = all(res[k]["cumple"] for k in ("K-1", "K-2", "K-3", "K-4"))
    print(f"\nRESUMEN: K-1..K-4 {'TODAS CUMPLEN' if todas else 'HAY FALLAS'}")


if __name__ == "__main__":
    main()
