"""EXPLORATORIO · ¿que variable del input gobierna el volado de `q`?

2026-08-31. Declarado EXPLORATORIO antes de correr: no hay pre-registro, no adjudica ninguna
prediccion y no elige ninguna celda ganadora. Sirve para saber que preguntar despues.

Lo que se sabe hasta aca, medido hoy sobre los cuatro `t*` a 3000 pasos:

  * `q` NO es una constante: es BIMODAL (85-99 % de la masa en los extremos, <1 % en el centro).
    El modelo decide 0 o 1 tajante en cada pregunta.
  * esa decision es independiente de si hay respuesta (acuerdo 0,4985 contra azar 0,5004)
  * y tambien de su propia confianza (acuerdo 0,5156 contra azar 0,4992)

El forward es determinista, asi que el volado ES una funcion del input. La pregunta es de cual.

METODO. Para cada variable candidata se agrupa por su valor y se mide la PUREZA:

    pureza = suma_g  (n_g / N) * max(p_g, 1 - p_g)      con p_g = fraccion que se calla en el grupo

0,50 = la variable no explica nada · 1,00 = dentro de cada grupo la decision es unanime.

EL CONTROL ES OBLIGATORIO Y ES LO QUE HACE VALIDA LA MEDIDA. Agrupar por una variable con muchos
valores infla la pureza por puro tamano de grupo: en el limite, un grupo por muestra da pureza 1,000
sin explicar nada. Por eso cada candidata se compara contra una variable ALEATORIA con la MISMA
cantidad de grupos y el mismo reparto de tamanos. Lo que se lee es la DIFERENCIA, nunca la pureza
cruda. Es el mismo defecto que el `m=1` del 12-ago y que el nulo de la clave discreta del 30-ago.

Uso:  python3 sonda_volado.py ckpts/t03_s3.pkl ckpts/t03_s6.pkl --n 4096
"""

import argparse

import numpy as np

import datos as DAT
import entrenar as E
import medir_ratio_ce as R

NOMBRES_TIPO = {0: "vigente", 1: "anterior", 2: "nose_ent", 3: "nose_rel"}


def pureza(calla, grupos):
    """Media ponderada de max(p, 1-p) por grupo. 0,50 = no explica; 1,00 = unanime dentro del grupo."""
    tot, acc = len(calla), 0.0
    for v in np.unique(grupos):
        m = grupos == v
        p = calla[m].mean()
        acc += m.sum() / tot * max(p, 1.0 - p)
    return acc


def nulo(calla, grupos, semilla, reps=20):
    """La MISMA estructura de grupos, barajada. Es el piso que hay que superar."""
    rng = np.random.default_rng(semilla)
    vals = []
    for _ in range(reps):
        vals.append(pureza(calla, rng.permutation(grupos)))
    return float(np.mean(vals)), float(np.std(vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--n", type=int, default=4096)
    ap.add_argument("--lote", type=int, default=64)
    ap.add_argument("--semilla", type=int, default=54321)
    ap.add_argument("--p-nose", type=float, default=0.4)
    a = ap.parse_args()

    print("=" * 100)
    print("EXPLORATORIO · que variable del input gobierna el volado de `q`")
    print(f"n={a.n}  semilla {a.semilla}  p_nose={a.p_nose}")
    print("  se lee la COLUMNA `dif`, nunca la pureza cruda")
    print("=" * 100)

    for ruta in a.ckpts:
        params, cfg, paso = R.cargar(ruta)

        # Se rehace el lote guardando tambien la consulta y el tipo, que `R.logits` no devuelve.
        nivel = cfg["nivel"]
        import jax, jax.numpy as jnp

        @jax.jit
        def partes(params, ses, cortes, turnos, mask, cons, pos):
            return E._partes(params, ses, cortes, turnos, mask, cons, pos)

        rng = np.random.default_rng(a.semilla)
        LG, TGT, CONS, TIPO = [], [], [], []
        vistos = 0
        while vistos < a.n:
            b = min(a.lote, a.n - vistos)
            ses, cortes, turnos, mask, cons, pos, tgt, tipo, meta = DAT.lote(
                rng, b, nivel=nivel, n_hechos=4, n_sesiones=4, p_nose=a.p_nose, con_meta=True)
            lg, _ = partes(params, jnp.array(ses), jnp.array(cortes), jnp.array(turnos),
                           jnp.array(mask), jnp.array(cons), jnp.array(pos))
            LG.append(np.asarray(lg)); TGT.append(np.asarray(tgt))
            CONS.append(np.asarray(cons)); TIPO.append(np.asarray(tipo))
            vistos += b
        lg = np.concatenate(LG).astype(np.float64)
        tgt = np.concatenate(TGT); cons = np.concatenate(CONS); tipo = np.concatenate(TIPO)

        p_all = np.exp(lg - lg.max(-1, keepdims=True))
        p_all /= p_all.sum(-1, keepdims=True)
        calla = p_all[:, E.NOSE] > 0.5

        print(f"\n--- {ruta}   paso={paso}   se calla {calla.mean():.4f} ---")
        print(f"  {'variable':<28} {'grupos':>7} {'pureza':>8} {'nulo':>8} {'±':>7} {'dif':>8}")

        cands = [("tipo de pregunta", tipo)]
        for j in range(cons.shape[1]):
            if len(np.unique(cons[:, j])) > 1:
                cands.append((f"consulta, posicion {j}", cons[:, j]))

        filas = []
        for nom, g in cands:
            pu = pureza(calla, g)
            nu, sd = nulo(calla, g, a.semilla)
            filas.append((nom, len(np.unique(g)), pu, nu, sd, pu - nu))

        for nom, ng, pu, nu, sd, dif in filas:
            marca = "  <===" if dif > 5 * max(sd, 1e-9) and dif > 0.05 else ""
            print(f"  {nom:<28} {ng:>7} {pu:>8.4f} {nu:>8.4f} {sd:>7.4f} {dif:>+8.4f}{marca}")

        mejor = max(filas, key=lambda f: f[5])
        if mejor[5] > 0.05:
            print(f"  -> la que mas explica es «{mejor[0]}» con dif {mejor[5]:+.4f}")
        else:
            print(f"  -> NINGUNA candidata supera el nulo por mas de 0,05 (maximo {mejor[5]:+.4f} en "
                  f"«{mejor[0]}»). El volado no lo gobierna ninguna variable simple de la consulta.")

        # Desglose por tipo, que es interpretable aunque no gane
        print("     por tipo de pregunta:  ", end="")
        for v in sorted(np.unique(tipo)):
            print(f"{NOMBRES_TIPO.get(int(v), v)} {calla[tipo == v].mean():.3f}   ", end="")
        print()


if __name__ == "__main__":
    main()
