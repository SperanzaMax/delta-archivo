"""Compuerta W-0 de `PREREG_SLOT_ORDEN.md` · el slot nulo con gradiente de ORDEN.

Se corre ANTES de congelar el pre-registro y antes de gastar un minuto de GPU, por la regla del
monitor v1 del 20-ago: lo primero que se verifica de una reparacion es que la reparacion HAGA algo, y
que lo que hace sea lo que dice.

  W-0(a)  `k_nulo` y `v_nulo` existen en los checkpoints de siembra `b3_s3` / `b3_s6` y NUNCA
          RECIBIERON GRADIENTE, porque solo los lee `--abst slot`.

          **El criterio se corrige antes de aplicarlo, y la correccion es del 31 a la noche.** La
          primera version pedia distancia 0,0 EXACTA al valor inicial y fallaba en las dos semillas
          por 1,4e-02. La explicacion alternativa —que es la que resulto cierta— es el WEIGHT DECAY:
          26000 pasos de decay encogen `k_nulo` aunque no reciba un solo gradiente, y sobre `v_nulo`,
          que vale cero, no dejan marca. Lo que distingue las dos lecturas es la DIRECCION: si el
          slot nunca recibio gradiente su vector solo pudo encogerse, asi que
          **coseno = 1,0 exacto con el `init_params` de su propia semilla, y razon de normas < 1**.
          Un gradiente lo habria girado. Se mide asi.

  W-0(b)  Con `--abst slot` la masa del slot NO es constante entre muestras. Si lo fuera, el logit de
          abstencion seria un numero fijo y el termino de orden no tendria nada que ordenar.

          **Y el criterio «desvio > 0» no alcanza, tambien corregido antes de aplicarlo.** Una masa
          que vale 0 o 1 y nada en el medio tiene desvio ALTO y no gradua nada: cumpliria el criterio
          siendo justo la degeneracion que se quiere evitar. Es la misma clase de defecto que el
          informe de esta tarde encontro en O-6 —un criterio que no mide lo que su nombre dice— y por
          eso aca se mide ademas la SATURACION: la fraccion de muestras pegadas al clip de
          `modelo.py:306` (|logit| = 13,8155) y cuantos valores distintos toma el logit. Se declara
          antes: **si mas de la mitad de las muestras estan pegadas al clip, el logit de abstencion
          es de hecho una variable de dos valores** y hay que decirlo en el pre-registro, porque
          cambia lo que el experimento puede medir.

  W-0(c)  LA QUE DECIDE SI EL DISEÑO HACE LO QUE DICE. El gradiente del termino de orden tiene que
          llegar a `k_nulo`, a `qr` y a `kw` —el mecanismo de BUSQUEDA— y NO a `head` —la salida—.
          Es la diferencia con la interfaz `token` corrida hoy, donde el termino empujaba el logit de
          `NOSE` en `head`, o sea despues de que el softmax del archivo ya aplasto la evidencia.
          **Si (c) falla, el diseño no hace lo que dice y no se lanza.**

  W-0(d)  EL PESO, DERIVADO Y NO ELEGIDO, y por eso el criterio va escrito ANTES de correr nada:

          > `--rec-rank 0,008` fue derivado para la interfaz `token` igualando el gradiente en la
          > columna de `NOSE` de `head` con el gradiente medio del RESTO de las columnas. Reusar ese
          > numero aca seria repetir el error del 30-ago —medir en un lugar y aplicar en otro—,
          > porque con `slot` el termino ni siquiera toca `head`.
          >
          > **Criterio, homologo y declarado antes del dato:** el parametro especifico de la
          > abstencion pasa a ser `k_nulo`, y el «resto» pasa a ser `kw`, las claves de las entradas
          > del archivo, que son las que compiten con el slot DENTRO DEL MISMO SOFTMAX. Se fija
          >
          >     rec_rank* = |g| medio de (recompensa + CE) en `kw`
          >                 ---------------------------------------
          >                     |g| medio del termino de ORDEN en `k_nulo`
          >
          > medido en el CHECKPOINT DE SIEMBRA —no a mitad de corrida, que fue el error del 30— y
          > promediado entre las dos semillas, igual que el 0,008 de hoy.

          Se reportan ademas los ratios homologos sobre `qr` y sobre el conjunto entero de la
          busqueda, para que se vea de cuanto es la ambiguedad de la eleccion; pero el peso sale del
          criterio de arriba, que es el que quedo escrito primero.

Costo: CPU, segundos. No toca checkpoints ni corridas.
"""
import json
import os
import pickle

import jax
import jax.numpy as jnp
import numpy as np

import idioma as I

I.fijar_version(2)

import datos as DAT      # noqa: E402
import entrenar as E     # noqa: E402
import modelo as M       # noqa: E402

AQUI = os.path.dirname(os.path.abspath(__file__))
SEMILLA = 31
SEMBRADAS = ("b3_s3", "b3_s6")

# Configuracion de la campaña, identica a `lanzar_orden_nose.sh` salvo la interfaz de abstencion.
E._DONDE = "pre"
E._ABST = "slot"
E._BLANCO = "error"
E._PERDIDA_CABEZA = "recompensa"
E._REC_L, E._REC_M, E._REC_F, E._REC_CE = 0.0, 0.5, 0.2, 1.0
E._REC_RANK = 0.0        # el termino de orden se mide APARTE, no dentro de la perdida base


def lote():
    rng = np.random.default_rng(SEMILLA)
    ses, cortes, turnos, mask, cons, pos, tgt, _ = DAT.lote(
        rng, 64, nivel=3, n_hechos=4, n_sesiones=4, p_vieja=0.35, p_nose=0.4)
    return tuple(jnp.array(x) for x in (ses, cortes, turnos, mask, cons, pos, tgt))


def cargar(nombre):
    with open(os.path.join(AQUI, "ckpts", nombre + ".pkl"), "rb") as f:
        d = pickle.load(f)
    return jax.tree_util.tree_map(jnp.asarray, d["params"])


def medio(g):
    return float(jnp.mean(jnp.abs(g)))


def main():
    L = lote()
    ses, cortes, turnos, mask, cons, pos, tgt = L
    p0 = M.init_params(3, I.V, D=128, NB=4)     # referencia del valor inicial del slot
    res = {}

    print("=" * 78)
    print("COMPUERTA W-0 · slot nulo + termino de ORDEN")
    print("=" * 78)

    for nombre in SEMBRADAS:
        p = cargar(nombre)
        r = {}
        print(f"\n{'-' * 78}\n{nombre}\n{'-' * 78}")

        # --- W-0(a) · el slot existe y NUNCA recibio gradiente ------------------------------------
        hay_k = "k_nulo" in p["arch"] and "v_nulo" in p["arch"]
        semilla_ck = int(nombre.split("_s")[1])
        pi = M.init_params(semilla_ck, I.V, D=128, NB=4)
        k_ck, k_0 = p["arch"]["k_nulo"], pi["arch"]["k_nulo"]
        cos = float(k_ck @ k_0 / (jnp.linalg.norm(k_ck) * jnp.linalg.norm(k_0)))
        razon = float(jnp.linalg.norm(k_ck) / jnp.linalg.norm(k_0))
        d_v = float(jnp.max(jnp.abs(p["arch"]["v_nulo"])))
        a_ok = bool(hay_k and abs(cos - 1.0) < 1e-6 and razon <= 1.0 and d_v == 0.0)
        print(f"W-0(a) · k_nulo/v_nulo presentes: {hay_k}")
        print(f"         coseno con init(seed={semilla_ck}) {cos:+.6f}   (hace falta 1,0: sin giro)")
        print(f"         razon de normas {razon:.6f}   (< 1: encogido por weight decay, no por"
              f" gradiente)")
        print(f"         |v_nulo| max {d_v:.3e}   (0,0: el decay sobre cero no deja marca)")
        print(f"         W-0(a): {'CUMPLE' if a_ok else 'NO CUMPLE'}")
        r["a"] = {"presentes": hay_k, "coseno": cos, "razon_normas": razon, "v_nulo": d_v,
                  "cumple": a_ok}

        # --- W-0(b) · la masa del slot no es constante, y no esta saturada ------------------------
        lg, s = E._partes(p, ses, cortes, turnos, mask, cons, pos)
        m = jax.nn.sigmoid(s)
        CLIP = float(np.log((1 - 1e-6) / 1e-6))          # 13,8155: el tope de modelo.py:306
        pegadas = float(jnp.mean((jnp.abs(jnp.abs(s) - CLIP) < 1e-3).astype(jnp.float32)))
        distintos = int(len(np.unique(np.round(np.asarray(s), 3))))
        b_ok = bool(float(jnp.std(m)) > 0.0)
        b_sat = bool(pegadas <= 0.5)
        print(f"\nW-0(b) · masa del slot: media {float(jnp.mean(m)):.5f} · desvio "
              f"{float(jnp.std(m)):.5f} · min {float(jnp.min(m)):.5f} · max {float(jnp.max(m)):.5f}")
        print(f"         logit: media {float(jnp.mean(s)):+.4f} · desvio {float(jnp.std(s)):.4f}")
        print(f"         W-0(b): {'CUMPLE' if b_ok else 'NO CUMPLE'}   (hace falta desvio > 0)")
        print(f"         SATURACION · muestras pegadas al clip (|s| = {CLIP:.4f}): {pegadas:.4f}")
        print(f"                      valores distintos del logit: {distintos} de {len(s)}")
        print(f"                      {'OK' if b_sat else '** el logit es de DOS VALORES **'}")
        r["b"] = {"masa_media": float(jnp.mean(m)), "masa_desvio": float(jnp.std(m)),
                  "logit_media": float(jnp.mean(s)), "logit_desvio": float(jnp.std(s)),
                  "pegadas_al_clip": pegadas, "valores_distintos": distintos,
                  "cumple": b_ok, "no_saturado": b_sat}

        # --- el termino de orden en el punto de partida ------------------------------------------
        es_nose = (tgt == E.NOSE).astype(jnp.float32)
        hay = 1.0 - es_nose
        orden0 = float(E._orden_nose(s, es_nose, hay))
        sv = np.asarray(s)
        x, y = sv[np.asarray(es_nose) == 1], sv[np.asarray(hay) == 1]
        auc0 = float((x[:, None] > y[None, :]).mean() + 0.5 * (x[:, None] == y[None, :]).mean())
        print(f"\n         termino de orden en la siembra: {orden0:.4f}"
              f"   (constante = log 2 = 0,6931 · oraculo = 0)")
        print(f"         AUC del logit vs la ausencia en la siembra: {auc0:.4f}   (azar = 0,5)")
        r["orden_siembra"] = orden0
        r["auc_siembra"] = auc0

        # --- gradientes: termino de orden SOLO, y perdida base SOLA ------------------------------
        def solo_orden(pp):
            _, sa = E._partes(pp, ses, cortes, turnos, mask, cons, pos)
            return E._orden_nose(sa, es_nose, hay)

        def solo_base(pp):
            return E.perdida_cabeza(pp, ses, cortes, turnos, mask, cons, pos, tgt)[0]

        g_ord = jax.grad(solo_orden)(p)
        g_bas = jax.grad(solo_base)(p)

        # --- W-0(c) · el gradiente del orden va a la BUSQUEDA y no a la SALIDA -------------------
        print(f"\nW-0(c) · gradiente del termino de orden, |g| medio por elemento")
        filas = [("kw (keys del archivo)", g_ord["arch"]["kw"]),
                 ("qr (query de busqueda)", g_ord["arch"]["qr"]),
                 ("k_nulo (el slot)", g_ord["arch"]["k_nulo"]),
                 ("v_nulo", g_ord["arch"]["v_nulo"]),
                 ("head.w (la salida)", g_ord["head"]["w"])]
        gm = {}
        for et, gg in filas:
            gm[et.split(" ")[0]] = medio(gg)
            print(f"         {et:26s} {medio(gg):.4e}")
        c_ok = bool(gm["k_nulo"] > 0.0 and gm["kw"] > 0.0 and gm["qr"] > 0.0
                    and gm["head.w"] == 0.0)
        print(f"         W-0(c): {'CUMPLE' if c_ok else 'NO CUMPLE'}"
              f"   (busqueda > 0 y salida = 0,0 exacto)")
        r["c"] = {**gm, "cumple": c_ok}

        # --- W-0(d) · el peso -------------------------------------------------------------------
        base_kw = medio(g_bas["arch"]["kw"])
        base_qr = medio(g_bas["arch"]["qr"])
        base_bus = medio(jnp.concatenate([g_bas["arch"]["kw"].ravel(),
                                          g_bas["arch"]["qr"].ravel()]))
        ord_bus = medio(jnp.concatenate([g_ord["arch"]["kw"].ravel(),
                                         g_ord["arch"]["qr"].ravel()]))
        w_princ = base_kw / gm["k_nulo"]
        print(f"\nW-0(d) · peso derivado")
        print(f"         |g| base (recompensa+CE) en kw     {base_kw:.4e}   <- el «resto»")
        print(f"         |g| orden en k_nulo                {gm['k_nulo']:.4e}   <- el especifico")
        print(f"         rec_rank* (criterio principal)     {w_princ:.5f}")
        print(f"         --- referencias, NO se usan para fijar el peso ---")
        print(f"         base qr {base_qr:.4e} -> qr/k_nulo      {base_qr / gm['k_nulo']:.5f}")
        print(f"         base busqueda {base_bus:.4e} / orden busqueda {ord_bus:.4e}"
              f" -> {base_bus / ord_bus:.5f}")
        r["d"] = {"base_kw": base_kw, "base_qr": base_qr, "base_busqueda": base_bus,
                  "orden_k_nulo": gm["k_nulo"], "orden_busqueda": ord_bus,
                  "rec_rank_principal": w_princ,
                  "ref_qr": base_qr / gm["k_nulo"], "ref_busqueda": base_bus / ord_bus}
        res[nombre] = r

    # --- el peso que se fija ---------------------------------------------------------------------
    ws = [res[n]["d"]["rec_rank_principal"] for n in SEMBRADAS]
    w = float(np.mean(ws))
    disp = max(ws) / min(ws)
    print(f"\n{'=' * 78}")
    print(f"PESO · {SEMBRADAS[0]} {ws[0]:.5f} · {SEMBRADAS[1]} {ws[1]:.5f} · media {w:.5f}")
    # Se redondea a dos cifras significativas, igual que el 0,008 de la campania `token`.
    if w > 0:
        exp = int(np.floor(np.log10(w)))
        w_fijo = float(round(w, -exp + 1))
    else:
        w_fijo = 0.0
    print(f"       media redondeada: {w_fijo:g}")
    # Referencia declarada antes: con la interfaz `token`, el MISMO criterio dio 0,00805 y 0,00720
    # sobre estos mismos dos checkpoints, o sea una dispersion de 1,12x. Si aca la dispersion es
    # mucho mayor, el peso NO esta determinado por el criterio y promediar seria inventar un numero
    # que no describe a ninguna de las dos unidades.
    print(f"       dispersion entre semillas {disp:.2f}x   (con `token`, el mismo criterio: 1,12x)")
    if disp > 2.0:
        print(f"       ** el peso NO queda determinado: hay que elegir peso POR SEMILLA o cambiar")
        print(f"          el punto de partida. No se fija un unico numero. **")
    res["peso"] = {"por_semilla": ws, "media": w, "media_redondeada": w_fijo,
                   "dispersion": disp, "determinado": bool(disp <= 2.0)}

    duras = [f"{n}:{k}" for n in SEMBRADAS for k in ("a", "b", "c") if not res[n][k]["cumple"]]
    veredicto = "COMPUERTA ABIERTA" if not duras else "COMPUERTA CERRADA: " + ", ".join(duras)
    print(f"\n{veredicto}")
    res["veredicto"] = veredicto

    sal = os.path.join(AQUI, "compuerta_slot_orden_20260831.json")
    with open(sal, "w") as f:
        json.dump(res, f, indent=1)
    print(f"-> {os.path.basename(sal)}")


if __name__ == "__main__":
    main()
