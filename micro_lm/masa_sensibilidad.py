"""¿Por qué `at3` gana si NO es mas selectivo? · post-hoc declarado, 2026-09-08

H2' quedo REFUTADA con las tres semillas: `sel = TV(d3)/TV(d5)` da 1,00 / 1,01 / 0,94, o sea
exactamente lo mismo que cuatro controles que nunca entrenaron con acceso global (0,94-1,05). Y sin
embargo `nose_rel` de `at3` va 0,94-0,97 contra 0,86-0,88 del kernel 5 y 0,64 del kernel 3.

La enmienda 1 pre-registro esta rama como «el resultado mas informativo de los tres, y el que obliga
a buscar la causa alternativa ANTES de escribir nada». Esto es esa busqueda, y es POST-HOC, no una
hipotesis pre-registrada. Se declara asi.

La candidata: la razon de selectividad compara DOS distancias y por eso no ve lo que cambio. El
kernel 5 concentra mucha sensibilidad en cuatro posiciones y da CERO EXACTO en el resto; la atencion
reparte poca en TODAS. Si lo que decide es la masa ACUMULADA sobre la consulta entera y no el pico
por token, `attn` gana por area aunque pierda por altura, y la selectividad medida en dos puntos no
puede verlo.

Predice algo falsable. La suma de TV sobre las 20 distancias tiene que ser MAYOR en `at3` que en
`kq3`, aunque el maximo por distancia sea MENOR.

    python masa_sensibilidad.py <ckpt attn> <ckpt lat2>
"""
import os
import sys

import numpy as np
import jax, jax.numpy as jnp
import pickle

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
import modelo as M
import conf_ckpt
from control_attn import distribucion, T, N_ARCH

N = int(os.environ.get("N", "300"))
DMAX = int(os.environ.get("DMAX", "20"))


def perfil(params, donde, semilla=0):
    rng = np.random.default_rng(1000 + semilla)
    V, D = params["emb"].shape
    cons = jnp.array(rng.integers(0, V, (N, T)))
    arch = jnp.array(rng.normal(size=(N, N_ARCH, D)), dtype=jnp.float32)
    tur = jnp.array(rng.integers(0, 3, (N, N_ARCH)))
    msk = jnp.ones((N, N_ARCH), bool)
    pos = T - 1
    base = distribucion(params, arch, tur, msk, cons, pos, donde)
    out = {}
    for d in range(1, DMAX + 1):
        if pos - d < 0:
            break
        alt = np.asarray(cons).copy()
        col = pos - d
        alt[:, col] = (alt[:, col] + 1 + rng.integers(0, V - 2, N)) % V
        pd = distribucion(params, arch, tur, msk, jnp.array(alt), pos, donde)
        out[d] = float((0.5 * np.abs(base - pd).sum(-1)).mean())
    return out


if __name__ == "__main__":
    rutas = sys.argv[1:]
    print(f"masa de sensibilidad acumulada · N={N} · distancias 1..{DMAX}\n")
    res = {}
    for ruta in rutas:
        b = pickle.load(open(ruta, "rb"))
        params, cfg = b["params"], b["config"]
        conf_ckpt.aplicar(cfg)
        donde = cfg["donde"]
        p = perfil(params, donde)
        nom = os.path.basename(ruta)
        res[nom] = (donde, p)
        print(f"{nom}  ({conf_ckpt.descripcion(cfg)})")
        print("   d " + " ".join(f"{d:>7}" for d in sorted(p)))
        print("  TV " + " ".join(f"{p[d]:7.4f}" for d in sorted(p)))
        nz = [d for d in p if p[d] > 0]
        print(f"   pico {max(p.values()):.4f}   distancias con TV>0 {len(nz)} de {len(p)}"
              f"   MASA (suma) {sum(p.values()):.4f}\n")

    if len(res) == 2:
        (na, (da, pa)), (nb, (db, pb)) = res.items()
        print(f"{'':>16}{'pico':>10}{'masa':>10}{'alcance':>10}")
        for n, (dd, p) in res.items():
            print(f"{n[:15]:>16}{max(p.values()):10.4f}{sum(p.values()):10.4f}"
                  f"{len([d for d in p if p[d] > 0]):>7} de {len(p)}")
        print(f"\nLa candidata predice masa MAYOR en attn y pico MENOR.")
        pico_menor = max(pa.values()) < max(pb.values())
        masa_mayor = sum(pa.values()) > sum(pb.values())
        print(f"  pico menor en {na}: {pico_menor}   masa mayor en {na}: {masa_mayor}")
        print("  -> " + ("CANDIDATA SOSTENIDA" if (pico_menor and masa_mayor) else
                         "CANDIDATA NO SOSTENIDA, hay que buscar otra"))
