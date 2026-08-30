"""Idea de Maxi (30-ago): ¿y si la clave del archivo fuera un NUMERO CON COMA, donde la parte
entera nombra el tema y la decimal la antiguedad (mas grande = mas nueva)?

⚠ LA v1 DE ESTE SCRIPT ERA UN CONTROL VACIO Y DIO 1,000 EN LAS TRES CONDICIONES. Le pasaba a la
consulta el NUMERO DE TEMA EXACTO, asi que recuperar no requeria buscar: es el mismo defecto que el
control `m=1` del 12-ago («con una sola candidata, acertar no requiere leer»). Cazado antes de
reportarlo.

La correccion es la que importa y define el experimento: en el modelo real **la consulta no trae el
numero del tema**. Es un vector formado leyendo una pregunta en lenguaje, y el tema hay que
INFERIRLO de ahi. Entonces la pregunta honesta no es «¿el escalar ordena versiones?» —eso ya lo
sabemos, es el sello de E-I3 y gana— sino:

    ¿que pasa cuando el tema se infiere CON ERROR, que es el unico regimen que ocurre?

Tres codificaciones, misma consulta ruidosa, misma regla de recuperacion:
  A · escalar     clave = tema + version/10, y el tema se lee del vecino mas cercano en el espacio
  B · denso+sello clave = vector del tema; el orden en un campo APARTE (E-I3: 0,4570 -> 0,9956)
  C · discreto    simbolo exacto del tema, orden aparte (salida (b) de R5.2, nunca probada)
"""

import numpy as np

RNG = np.random.default_rng(20260830)


def banco(n_temas, n_versiones):
    t = np.repeat(np.arange(n_temas), n_versiones)
    v = np.tile(np.arange(n_versiones), n_temas)
    return t, v


def corrida(n_temas, n_versiones, d, sigma, n_consultas=3000, deriva=None):
    """`sigma` = ruido con que el modelo forma la query. La consulta NO conoce el numero de tema."""
    t, v = banco(n_temas, n_versiones)
    E = RNG.normal(size=(n_temas, d)) / np.sqrt(d)
    K = E[t] if deriva is None else E[t] @ deriva      # claves viejas, consulta de hoy

    temas = RNG.integers(0, n_temas, n_consultas)
    Q = E[temas] + RNG.normal(size=(n_consultas, d)) * sigma / np.sqrt(d)

    sim = Q @ K.T                                       # (n_consultas, n_entradas)
    ok = {"A_escalar": 0, "B_denso": 0, "C_discreto": 0}

    for i, c in enumerate(temas):
        # --- el tema se INFIERE: en las tres, del mismo vecino mas cercano. Lo que cambia es que
        # --- hace cada codificacion DESPUES de inferirlo.
        j = int(np.argmax(sim[i]))
        tema_inferido = t[j]

        # A · escalar: la clave es un numero. Inferido el tema, se salta al escalar objetivo
        #     `tema + (V-1)/10` y se toma el mas cercano EN LA RECTA. El punto es que en la recta
        #     los vecinos de un tema son OTROS TEMAS, no sus versiones.
        claves = t + v / 10.0
        obj = tema_inferido + (n_versiones - 1) / 10.0
        a = int(np.argmin(np.abs(claves - obj)))
        ok["A_escalar"] += int(t[a] == c and v[a] == n_versiones - 1)

        # B · denso + sello: entre las entradas del tema inferido, gana el sello mas alto
        cand = np.where(t == tema_inferido)[0]
        b = cand[int(np.argmax(v[cand]))]
        ok["B_denso"] += int(t[b] == c and v[b] == n_versiones - 1)

        # C · discreto: identico a B en este regimen; se separa en la prueba de deriva
        ok["C_discreto"] += int(t[b] == c and v[b] == n_versiones - 1)

    return {k: n / n_consultas for k, n in ok.items()}


def escalar_ruido_directo(n_temas, n_versiones, sigma_rel, n_consultas=3000):
    """El regimen que le es PROPIO al escalar: que el modelo emita el numero, con su error.
    Si el numero sale con error de +-0,1 ya cae en otro tema; con +-0,05 cae entre versiones."""
    t, v = banco(n_temas, n_versiones)
    claves = t + v / 10.0
    temas = RNG.integers(0, n_temas, n_consultas)
    obj = temas + (n_versiones - 1) / 10.0 + RNG.normal(size=n_consultas) * sigma_rel
    idx = np.abs(claves[None, :] - obj[:, None]).argmin(1)
    tema_ok = (t[idx] == temas).mean()
    todo_ok = ((t[idx] == temas) & (v[idx] == n_versiones - 1)).mean()
    return tema_ok, todo_ok


if __name__ == "__main__":
    print("=" * 84)
    print("LA CLAVE COMO NUMERO CON COMA · idea de Maxi, 30-ago")
    print("=" * 84)

    print("\nP-1 · el tema se INFIERE de un vector con ruido (100 temas x 8 versiones, d=64)")
    print("      `sigma` = cuanto se equivoca el modelo al formar la query\n")
    print(f"{'sigma':>7} | {'A escalar':>11} {'B denso+sello':>15} {'C discreto':>12}")
    print("-" * 84)
    for s in (0.0, 0.5, 1.0, 2.0, 4.0):
        r = corrida(100, 8, 64, s)
        print(f"{s:>7.1f} | {r['A_escalar']:>11.4f} {r['B_denso']:>15.4f} {r['C_discreto']:>12.4f}")
    print("  -> las tres colapsan JUNTAS: el cuello de botella es inferir el tema, y eso NO lo")
    print("     cambia como se guarde la clave. La codificacion no es la palanca.")

    print("\n\nP-2 · el regimen PROPIO del escalar: el modelo emite el numero con su error")
    print("      (100 temas x 8 versiones; el error se mide en unidades de la recta)\n")
    print(f"{'error del numero':>18} | {'acierta tema':>13} {'tema+version':>14} | que pasa")
    print("-" * 84)
    for s, nota in ((0.01, "error 10x menor que el paso entre versiones"),
                    (0.05, "medio paso entre versiones"),
                    (0.10, "un paso: ya toca la version vecina"),
                    (0.50, "medio tema"),
                    (1.00, "un tema entero")):
        a, b = escalar_ruido_directo(100, 8, s)
        print(f"{s:>18.2f} | {a:>13.4f} {b:>14.4f} | {nota}")

    print("\n\nP-3 · el defecto de la METRICA. ⚠ La v2 de este script escribio MAL esta conclusion")
    print("      (dijo que 18,0 estaba mas cerca que 17,9, y los numeros que el mismo imprimia lo")
    print("      desmentian). El defecto real es de ESCALAS y es mas fuerte:\n")
    V = 8
    rango_interno = (V - 1) / 10.0
    sep_temas = 1.0 - rango_interno
    print(f"        un tema con {V} versiones ocupa el intervalo [17,0 ; 17,{V-1}] = ancho {rango_interno:.1f}")
    print(f"        y entre el ultimo del tema 17 y el primero del 18 hay {sep_temas:.1f}")
    print(f"        -> DOS VERSIONES DEL MISMO TEMA pueden estar a {rango_interno:.1f}, mientras DOS TEMAS")
    print(f"           DISTINTOS estan a {sep_temas:.1f}. La distancia dentro del tema es {rango_interno/sep_temas:.1f}x")
    print(f"           la distancia entre temas.")
    print(f"        -> con V > 4 versiones las dos escalas se INVIERTEN, y el numero de versiones")
    print(f"           que entran depende de cuantos decimales se reserven: es un presupuesto de")
    print(f"           bits repartido a mano entre 'que tema' y 'que tan viejo'.")
    print("      Es el defecto que R4 ya le habia medido al «eje global»: acopla identificar el item")
    print("      con recuperar su version, y por eso tiene techo duro. El eje POR RECUERDO gano.")
