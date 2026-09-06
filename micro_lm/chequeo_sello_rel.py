"""¿El sello relativo y el bit de pertenencia dejan intacto lo publicado? · 2026-09-06

Compuerta del cambio de arquitectura del 6-sep. Antes de preregistrar nada hay que probar que, en la
configuracion de siempre (`SELLO="abs"`, sin `pertenece`), el modelo da EXACTAMENTE los mismos
numeros que HEAD. Si no, cualquier comparacion contra los resultados ya publicados queda invalidada.

Cuatro comprobaciones, y las cuatro pueden fallar:

  A. logits identicos bit a bit contra el `modelo.py` de HEAD, sobre un checkpoint ya entrenado.
  B. `init_params` identico bit a bit fuera del parametro nuevo, con la misma semilla.
  C. `pert` arranca en cero exacto, o sea la condicion nueva CONTIENE a la vieja.
  D. el sello relativo CAMBIA algo (si no, no se estaria probando nada) y es invariante al
     corrimiento del turno, que es justamente lo que le falta al absoluto.

Uso:
    python3 chequeo_sello_rel.py --head /ruta/modelo_head.py
"""
import argparse, importlib.util, pickle, sys

import numpy as np, jax, jax.numpy as jnp

import datos as DAT, idioma as I, modelo as M


def cargar_head(ruta):
    spec = importlib.util.spec_from_file_location("modelo_head", ruta)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["modelo_head"] = mod
    spec.loader.exec_module(mod)
    return mod


def lote_fijo(cfg, semilla=4242, B=8, ses_extra=26):
    rng = np.random.default_rng(semilla)
    s, c, t, mk, q, pq, tg, tp = DAT.lote(rng, B, nivel=cfg["nivel"], p_vieja=0.35, p_nose=0.2,
                                          n_ses_extra=ses_extra)
    return [jnp.array(x) for x in (s, c, t, mk, q)] + [np.array(pq)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", required=True, help="modelo.py de HEAD (git show HEAD:micro_lm/modelo.py)")
    ap.add_argument("--ckpt", default="ckpts/lg3_s0.pkl")
    a = ap.parse_args()

    H = cargar_head(a.head)
    b = pickle.load(open(a.ckpt, "rb"))
    cfg, params = b["config"], b["params"]
    M.KQ = H.KQ = cfg.get("kernel_q", 3)
    M.SELLO = "abs"
    ses, cortes, turnos, mask, cons, pos = lote_fijo(cfg)
    ok = True

    # ---- A. logits identicos contra HEAD -------------------------------------------------------
    arch_n = M.escribir(params, ses, cortes)
    arch_h = H.escribir(params, ses, cortes)
    d_arch = float(jnp.abs(arch_n - arch_h).max())
    lg_n = M.responder(params, arch_n, turnos, cons, mask, donde=cfg["donde"])
    lg_h = H.responder(params, arch_h, turnos, cons, mask, donde=cfg["donde"])
    d_lg = float(jnp.abs(lg_n - lg_h).max())
    ln_n, ab_n = M.responder_con_abst(params, arch_n, turnos, cons, mask, donde=cfg["donde"],
                                      abst=cfg.get("abst", "token"))
    ln_h, ab_h = H.responder_con_abst(params, arch_h, turnos, cons, mask, donde=cfg["donde"],
                                      abst=cfg.get("abst", "token"))
    d_ab = max(float(jnp.abs(ln_n - ln_h).max()), float(jnp.abs(ab_n - ab_h).max()))
    print(f"A. contra HEAD, con SELLO='abs' y sin `pertenece`:")
    print(f"     escribir        max|dif| = {d_arch:.6e}")
    print(f"     responder       max|dif| = {d_lg:.6e}")
    print(f"     responder_abst  max|dif| = {d_ab:.6e}")
    a_ok = (d_arch == 0.0 and d_lg == 0.0 and d_ab == 0.0)
    print(f"   -> {'IDENTICO bit a bit' if a_ok else 'DIFIERE'}")
    ok &= a_ok

    # ---- B. init_params identico fuera del parametro nuevo -------------------------------------
    pn = M.init_params(7, I.V, D=cfg["d"], NB=cfg["capas"])
    ph = H.init_params(7, I.V, D=cfg["d"], NB=cfg["capas"])
    nuevas = set(pn["arch"]) - set(ph["arch"])
    peor = 0.0
    hojas_n = jax.tree_util.tree_leaves_with_path(pn)
    for ruta, val in hojas_n:
        clave = "/".join(str(getattr(k, "key", getattr(k, "idx", k))) for k in ruta)
        if any(f"/{nv}" in f"/{clave}" for nv in nuevas):
            continue
        otra = pn
        try:
            ref = ph
            for k in ruta:
                kk = getattr(k, "key", getattr(k, "idx", None))
                ref = ref[kk]
            peor = max(peor, float(jnp.abs(val - ref).max()))
        except Exception as e:
            print(f"     (no comparable: {clave} · {e})"); ok = False
    print(f"\nB. init_params(semilla 7) contra HEAD, fuera de {sorted(nuevas)}:")
    print(f"     max|dif| = {peor:.6e}  -> {'IDENTICO' if peor == 0.0 else 'DIFIERE'}")
    ok &= (peor == 0.0)

    # ---- C. `pert` arranca en cero exacto ------------------------------------------------------
    z = float(jnp.abs(pn["arch"]["pert"]).max())
    print(f"\nC. `pert` inicial: max|pert| = {z:.6e}  -> {'CERO exacto' if z == 0.0 else 'NO es cero'}")
    print(f"     params nuevos en el arbol: {pn['arch']['pert'].size} sobre {M.contar(pn)} "
          f"({100 * pn['arch']['pert'].size / M.contar(pn):.4f} %)")
    ok &= (z == 0.0)

    # ---- D. el sello relativo cambia algo, y es invariante al corrimiento ----------------------
    # OJO, defecto del primer intento de esta prueba (6-sep): el corrimiento era
    # `maximum(turnos - 5, 0)`, que APLASTA a cero todo turno menor que 5 y por lo tanto destruye el
    # orden relativo que la prueba dice conservar. Con el orden roto el sello relativo TIENE que
    # moverse, asi que el test fallaba por estar mal especificado, no por el sello. Es la misma
    # forma de error que la deriva modelada como ruido independiente del 8-ago.
    #
    # El corrimiento correcto SUMA, sobre un lote sin sesiones extra (turnos 0..39), de modo que
    # nada se aplaste abajo ni sature arriba de las 64 filas.
    ses0, cortes0, turnos0, mask0, cons0, _ = lote_fijo(cfg, semilla=99, ses_extra=0)
    arch0 = M.escribir(params, ses0, cortes0)
    t_alto = jnp.where(mask0, turnos0 + 10, 0)
    print(f"\n   (control del corrimiento: turnos {int(turnos0[mask0].min())}..{int(turnos0[mask0].max())}"
          f" -> {int(t_alto[mask0].min())}..{int(t_alto[mask0].max())}, sin aplastar ni saturar)")

    M.SELLO = "rel"
    lg_rel = M.responder(params, arch0, turnos0, cons0, mask0, donde=cfg["donde"])
    lg_abs0 = None
    M.SELLO = "abs"
    lg_abs0 = M.responder(params, arch0, turnos0, cons0, mask0, donde=cfg["donde"])
    dif_rel = float(jnp.abs(lg_rel - lg_abs0).max())
    lg_abs_c = M.responder(params, arch0, t_alto, cons0, mask0, donde=cfg["donde"])
    d_abs_corr = float(jnp.abs(lg_abs_c - lg_abs0).max())
    M.SELLO = "rel"
    lg_rel_c = M.responder(params, arch0, t_alto, cons0, mask0, donde=cfg["donde"])
    d_rel_corr = float(jnp.abs(lg_rel_c - lg_rel).max())
    M.SELLO = "abs"
    print(f"\nD. sello relativo:")
    print(f"     rel contra abs, mismos turnos      max|dif| = {dif_rel:.6e}  "
          f"-> {'cambia algo' if dif_rel > 0 else 'NO CAMBIA NADA (sospechoso)'}")
    print(f"     abs, turnos corridos +10           max|dif| = {d_abs_corr:.6e}  (deberia moverse)")
    print(f"     rel, turnos corridos +10           max|dif| = {d_rel_corr:.6e}  "
          f"-> {'INVARIANTE' if d_rel_corr == 0.0 else 'se mueve'}")
    d_ok = (dif_rel > 0) and (d_abs_corr > 0) and (d_rel_corr == 0.0)
    ok &= d_ok

    print(f"\n{'=' * 70}\nCOMPUERTA: {'PASA — nada publicado cambia' if ok else 'NO PASA'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
