#!/usr/bin/env python3
"""Consistencia de ida y vuelta — PREREG_ROUNDTRIP.md (SHA 55ba857a...).

IDA: con el archivo intacto, el modelo responde `X` (argmax sin NOSE, como toda la campaña).
VUELTA: se rearma LA MISMA pregunta cambiando la entidad por cada candidata `E'` y se mide cuan
probable es `X` bajo cada una. Cierra si el maximo cae en la entidad que se pregunto.

Por que asi y no preguntandole «de que entidad es X»: `idioma.pregunta()` genera un solo formato y el
modelo NUNCA vio una pregunta inversa. Preguntarsela seria medir fuera de distribucion, o sea el
instrumento vacio del monitor v1 pero del lado de la entrada. Sustituir un token no inventa formato.

La prediccion que ninguna vía anterior podia hacer (RT-5): si el error de identidad es la
marginalizacion sobre la entidad de origen —la atencion promedia entradas que son mutuamente
excluyentes—, entonces en los errores la vuelta tiene que apuntar al DUEÑO REAL del valor emitido,
no a una entidad cualquiera. Eso separa «anclado en la entrada correcta» de «anclado en cualquier
entrada», que es lo que el logit y el desacuerdo no separaban.
"""
import os, sys, json, pickle, argparse, shutil
import numpy as np
import jax, jax.numpy as jnp

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
import idioma as I, datos as DAT, modelo as M

NOSE = I.STOI["NOSE"]
ENT_IDS = np.array([I.STOI[e] for e in I.ENTIDADES])
ES_ENT = np.zeros(I.V, bool)
ES_ENT[ENT_IDS] = True

UNIDADES = ["1_s0", "2_s0", "3_s0", "3_s1", "3_s2", "4_s0", "4_s1", "4_s2"]
SEM_PRUEBA = 77000          # el mismo rng de prueba del monitor y del corte sin etiquetas
CAND_MAX = 5                # 4 entidades del episodio + la preguntada (PREREG §1)


# --- el instrumento ---------------------------------------------------------------------------

def pos_entidad(cons):
    """Indice del token de entidad en cada consulta. Tiene que haber exactamente uno."""
    hay = ES_ENT[cons]
    n = hay.sum(1)
    if not np.all(n == 1):
        raise AssertionError(f"consultas con {sorted(set(n.tolist()))} entidades, se espera 1")
    return hay.argmax(1)


def candidatas(cons, pe, meta):
    """(B, CAND_MAX) tokens y (B, CAND_MAX) validez. El slot 0 es SIEMPRE la entidad preguntada.

    Que el slot 0 sea la preguntada es lo que hace que `cierra` sea `argmax == 0`, y que un empate
    exacto —que no deberia existir, porque se deduplica— caiga del lado de cerrar.
    """
    B = cons.shape[0]
    tok = np.zeros((B, CAND_MAX), np.int32)
    val = np.zeros((B, CAND_MAX), bool)
    for b in range(B):
        eq = int(cons[b, pe[b]])
        vistos, lista = {eq}, [eq]
        m = meta[b]
        ents = ([m["hecho"]["ent"]] if m["hecho"] else []) + [o["ent"] for o in m["otros"]]
        for e in ents:
            t = I.STOI[e]
            if t not in vistos:
                vistos.add(t)
                lista.append(t)
        lista = lista[:CAND_MAX]
        tok[b, :len(lista)] = lista
        val[b, :len(lista)] = True
    return tok, val


def candidatas_ausentes(cons, pe, meta, rp):
    """RT-3 · el nulo: rivales que NO aparecen en el episodio. El slot 0 sigue siendo la preguntada."""
    B = cons.shape[0]
    tok = np.zeros((B, CAND_MAX), np.int32)
    val = np.ones((B, CAND_MAX), bool)
    for b in range(B):
        eq = int(cons[b, pe[b]])
        m = meta[b]
        usadas = {eq} | {I.STOI[o["ent"]] for o in m["otros"]}
        if m["hecho"]:
            usadas.add(I.STOI[m["hecho"]["ent"]])
        libres = np.array([t for t in ENT_IDS if t not in usadas])
        tok[b, 0] = eq
        tok[b, 1:] = rp.choice(libres, size=CAND_MAX - 1, replace=False)
    return tok, val


def variantes(cons, pe, tok):
    """(CAND_MAX, B, T): la misma consulta con la entidad reemplazada. La 0 es la original."""
    J = tok.shape[1]
    out = np.repeat(cons[None], J, axis=0).copy()
    for j in range(J):
        out[j, np.arange(cons.shape[0]), pe] = tok[:, j]
    return out


def ida_y_vuelta(params, ses, cortes, turnos, cons_var, mask, pos):
    """Un solo `escribir` por lote; la vuelta son CAND_MAX forwards de la consulta (T_Q = 12)."""
    archivo = M.escribir(params, ses, cortes)

    def responde(cj):
        lg, a = M.responder_con_abst(params, archivo, turnos, cj, mask)
        lg = jnp.take_along_axis(lg, pos[:, None, None], axis=1)[:, 0, :]
        a = jnp.take_along_axis(a, pos[:, None], axis=1)[:, 0]
        return lg, a

    lg0, a0 = responde(cons_var[0])
    X = lg0.at[:, NOSE].set(-jnp.inf).argmax(-1)

    def s_j(cj):
        lg, _ = responde(cj)
        lp = jax.nn.log_softmax(lg, -1)
        return (jnp.take_along_axis(lp, X[:, None], axis=1)[:, 0],
                jnp.take_along_axis(lg, X[:, None], axis=1)[:, 0])

    S, Scrudo = jax.vmap(s_j)(cons_var)
    return X, a0, S.T, Scrudo.T          # (B,), (B,), (B, J), (B, J)


def cerrar(S, val):
    """`cierra`, `p_E` y la entidad a la que apunta la vuelta (indice de slot)."""
    s = np.where(val, S, -np.inf)
    j = s.argmax(1)
    m = s.max(1, keepdims=True)
    e = np.where(val, np.exp(s - m), 0.0)
    p = e / e.sum(1, keepdims=True)
    ent = -np.log(np.clip(p, 1e-12, None)) * p
    return j == 0, p[:, 0], j, ent.sum(1)


# --- clasificacion de la respuesta ------------------------------------------------------------

def clasificar(X, tgt, tipo, meta, cand_tok):
    """acierto / err_version / err_identidad / err_fuera para las que TIENEN respuesta.

    Las preguntas sin respuesta van a `sin_resp` aparte —ahi no hay version ni identidad correcta que
    errar—, pero se les busca igual el dueño del valor emitido, que es lo que dice si el modelo se
    ancló en alguna entrada real del archivo.

    `duenio` es el SLOT de candidata dueña real del valor emitido, o -1 si no es de nadie.
    """
    B = len(X)
    cls = np.empty(B, dtype="<U14")
    duenio = np.full(B, -1)
    for b in range(B):
        m, x = meta[b], I.ITOS[X[b]]
        for o in m["otros"]:
            if x in o["versiones"]:
                w = np.flatnonzero(cand_tok[b] == I.STOI[o["ent"]])
                duenio[b] = int(w[0]) if len(w) else -1
                break
        if tipo[b] >= 2:
            cls[b] = "sin_resp"
        elif X[b] == tgt[b]:
            cls[b] = "acierto"
        elif x in set(m["hecho"]["versiones"] if m["hecho"] else []):
            cls[b] = "err_version"
        elif duenio[b] >= 0:
            cls[b] = "err_identidad"
        else:
            cls[b] = "err_fuera"
    return cls, duenio


def auc(x, pos_mask):
    p, n = x[pos_mask], x[~pos_mask]
    if len(p) == 0 or len(n) == 0:
        return np.nan
    r = np.argsort(np.argsort(np.concatenate([p, n]))) + 1
    return float((r[:len(p)].sum() - len(p) * (len(p) + 1) / 2) / (len(p) * len(n)))


def pasa(nose, falsa):
    return bool(falsa <= 0.10 and nose >= 0.50)


# --- corrida ----------------------------------------------------------------------------------

def main():
    """Todo lo que CORRE va acá adentro.

    2026-08-20, D-3: esto estaba a nivel de modulo, asi que `import sonda_roundtrip` —que hacen los
    dos diagnosticos para reusar `candidatas` y `variantes`— EJECUTABA la sonda entera con los
    argumentos del diagnostico, y le sobrescribia el JSON a la corrida principal. Un analisis que
    contamina su propia entrada, la misma familia que la D-1 del dia pero del lado del proceso.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--p-nose", type=float, default=0.4)
    ap.add_argument("--unidades", default="")
    ap.add_argument("--ckpt-alt", action="append", default=[], metavar="UNIDAD=RUTA")
    ap.add_argument("--smoke", action="store_true", help="RT-0: una unidad, 64 muestras, y para")
    ap.add_argument("--salida", default=os.path.join(AQUI, "roundtrip_20260820.json"))
    A = ap.parse_args()
    UNI = A.unidades.split(",") if A.unidades else UNIDADES
    ALT = dict(x.split("=", 1) for x in A.ckpt_alt)
    if A.smoke:
        A.n, A.batch = 1, 64

    print("CONSISTENCIA DE IDA Y VUELTA — PREREG_ROUNDTRIP.md (SHA 55ba857a...)")
    print(f"{'SMOKE (RT-0)' if A.smoke else f'{A.n}x{A.batch} = {A.n * A.batch} muestras por unidad'} · "
          f"p_nose={A.p_nose} · rng {SEM_PRUEBA}+s · |C|={CAND_MAX}\n")

    res = {}
    for u in UNI:
        ck = ALT.get(u, os.path.join(AQUI, "ckpts", f"c{u}.pkl"))
        if not os.path.isabs(ck):
            ck = os.path.join(AQUI, ck)
        if not os.path.exists(ck):
            print(f"c{u}: sin checkpoint")
            continue
        with open(ck, "rb") as f:
            d = pickle.load(f)
        paso = d.get("paso", "?")
        print(f"c{u}: {os.path.basename(ck)} · paso {paso}", flush=True)
        params = jax.tree_util.tree_map(jnp.asarray, d["params"])
        nivel, semilla = int(u[0]), int(u.split("_s")[1])
        rng = np.random.default_rng(SEM_PRUEBA + semilla)
        rp = np.random.default_rng(4321 + semilla)
        fn = jax.jit(ida_y_vuelta)

        ac = {k: [] for k in ("cierra", "pE", "ent", "cls", "duenio", "apunta", "tipo", "a",
                              "cierra_aus", "cierra_nul", "pE_nul", "dlogit", "cierra_crudo")}
        for _ in range(A.n):
            sal = DAT.lote(rng, A.batch, nivel=nivel, n_hechos=4, n_sesiones=4, p_vieja=0.35,
                           p_nose=A.p_nose, con_meta=True, con_origen=True)
            ses, cortes, turnos, mask, cons, pos, tgt, tipo, meta, _origen, _hq = sal
            mask = np.asarray(mask)
            pe = pos_entidad(cons)
            tok, val = candidatas(cons, pe, meta)
            cv = variantes(cons, pe, tok)

            args = (jnp.array(ses), jnp.array(cortes), jnp.array(turnos))
            X, a, S, Sc = fn(params, *args, jnp.array(cv), jnp.array(mask), jnp.array(pos))
            X, a, S, Sc = np.asarray(X), np.asarray(a), np.asarray(S), np.asarray(Sc)

            cierra, pE, j, ent = cerrar(S, val)
            # D-1 · el mismo cierre con el logit CRUDO, para que se vea si la conclusion depende de
            # haber normalizado. Cuesta cero: sale del mismo forward.
            cierra_c, _, _, _ = cerrar(Sc, val)
            cls, duenio = clasificar(X, np.asarray(tgt), np.asarray(tipo), meta, tok)

            # RT-0a · la sustitucion tiene que MOVER los logits. Con el slot 0 = la original, la
            # distancia se mide contra las otras candidatas validas.
            dl = np.max(np.where(val[:, 1:], np.abs(Sc[:, 1:] - Sc[:, :1]), 0.0), axis=1)

            # RT-3 · rivales ausentes del episodio
            tok_a, val_a = candidatas_ausentes(cons, pe, meta, rp)
            _, _, S_a, _ = fn(params, *args, jnp.array(variantes(cons, pe, tok_a)),
                              jnp.array(mask), jnp.array(pos))
            cierra_a, _, _, _ = cerrar(np.asarray(S_a), val_a)

            # RT-4 · el nulo de archivo: todo tapado, misma consulta
            m0 = np.zeros_like(mask)
            _, _, S_0, _ = fn(params, *args, jnp.array(cv), jnp.array(m0), jnp.array(pos))
            cierra_0, pE_0, _, _ = cerrar(np.asarray(S_0), val)

            ac["cierra"].append(cierra); ac["pE"].append(pE); ac["ent"].append(ent)
            ac["cls"].append(cls); ac["duenio"].append(duenio); ac["apunta"].append(j)
            ac["tipo"].append(np.asarray(tipo)); ac["a"].append(a); ac["dlogit"].append(dl)
            ac["cierra_aus"].append(cierra_a); ac["cierra_crudo"].append(cierra_c)
            ac["cierra_nul"].append(cierra_0); ac["pE_nul"].append(pE_0)

        G = {k: np.concatenate(v) for k, v in ac.items()}
        cls, tipo, cierra = G["cls"], G["tipo"], G["cierra"]
        hay, sin_r, vig = tipo < 2, tipo >= 2, tipo == 0
        okv = cls == "acierto"
        ident = cls == "err_identidad"

        # RT-0
        rt0a = float((G["dlogit"] > 1e-3).mean())
        rt0b = float(cierra[okv].mean()) if okv.any() else np.nan

        # RT-1 · AUC de p_E separando aciertos de errores de identidad (etiquetas)
        sel = okv | (ident & hay)
        a1 = auc(G["pE"][sel], okv[sel]) if sel.any() else np.nan

        # RT-2 · el corte estructural, y sigma>0,5 sobre la MISMA muestra
        abst = ~cierra
        nose = float(abst[sin_r].mean()) if sin_r.any() else np.nan
        falsa = float(abst[hay].mean()) if hay.any() else np.nan
        vigente = float((okv[vig] & ~abst[vig]).mean()) if vig.any() else np.nan
        s05 = G["a"] > 0.0
        nose5 = float(s05[sin_r].mean()) if sin_r.any() else np.nan
        falsa5 = float(s05[hay].mean()) if hay.any() else np.nan
        domina = bool(nose > nose5 and falsa <= falsa5)

        # D-1 · el corte con el logit crudo, al lado
        abst_c = ~G["cierra_crudo"]
        nose_c = float(abst_c[sin_r].mean()) if sin_r.any() else np.nan
        falsa_c = float(abst_c[hay].mean()) if hay.any() else np.nan
        concuerda = float((G["cierra_crudo"] == cierra).mean())

        # RT-5 · en los errores de identidad, ¿la vuelta apunta al dueño real?
        ei = ident & (G["duenio"] >= 0)
        rt5 = float((G["apunta"][ei] == G["duenio"][ei]).mean()) if ei.any() else np.nan
        # el mismo estadistico sobre las preguntas SIN respuesta: ahi «anclarse en otra entrada real»
        # no es un error de identidad, es todo lo que el modelo puede hacer. Se reporta, no decide.
        sr = (cls == "sin_resp") & (G["duenio"] >= 0)
        rt5_sr = float((G["apunta"][sr] == G["duenio"][sr]).mean()) if sr.any() else np.nan

        # secundaria · entropia de la posterior
        h_ok = float(G["ent"][okv].mean()) if okv.any() else np.nan
        h_er = float(G["ent"][ident & hay].mean()) if (ident & hay).any() else np.nan

        res[u] = {"ckpt": os.path.basename(ck), "paso": paso, "n": int(len(cls)),
                  "RT0a": rt0a, "RT0b": rt0b, "RT1_auc": a1,
                  "cierra": float(cierra.mean()), "falsa": falsa, "nose": nose, "vigente": vigente,
                  "RT2_pasa": pasa(nose, falsa), "s05_falsa": falsa5, "s05_nose": nose5,
                  "s05_pasa": pasa(nose5, falsa5), "RT2_domina": domina,
                  "RT3_cierra_ausentes": float(G["cierra_aus"].mean()),
                  # desglose informativo, no criterio: el nulo mezcla preguntas con respuesta y sin
                  # respuesta, y en estas ultimas `X` no es de nadie en particular, asi que no hay razon
                  # a priori para que la preguntada gane. El criterio declarado sigue siendo el global.
                  "RT3_en_acierto": float(G["cierra_aus"][okv].mean()) if okv.any() else np.nan,
                  "RT3_en_sin_resp": (float(G["cierra_aus"][sin_r].mean()) if sin_r.any() else np.nan),
                  "RT4_auc": auc(G["pE_nul"][sel], okv[sel]) if sel.any() else np.nan,
                  "RT5_al_duenio": rt5, "RT5_n": int(ei.sum()), "RT5_sin_resp": rt5_sr,
                  "H_acierto": h_ok, "H_error": h_er,
                  "crudo": {"falsa": falsa_c, "nose": nose_c, "pasa": pasa(nose_c, falsa_c),
                            "concuerda_con_normalizado": concuerda},
                  "tasas": {k: float((cls[hay] == k).mean()) for k in
                            ("acierto", "err_version", "err_identidad", "err_fuera")}}
        r = res[u]
        print(f"   RT-0a Δlogit>1e-3 {rt0a:.3f} · RT-0b cierra|acierto {rt0b:.3f} · "
              f"RT-1 AUC {a1:.3f}")
        print(f"   corte: f_abst {falsa:.4f} nose {nose:.4f} pasa {'SI' if r['RT2_pasa'] else 'no'} | "
              f"σ>0,5: f_abst {falsa5:.4f} nose {nose5:.4f} pasa {'SI' if r['s05_pasa'] else 'no'} | "
              f"domina {'SI' if domina else 'no'}")
        print(f"   RT-3 ausentes {r['RT3_cierra_ausentes']:.3f} · RT-4 AUC nulo {r['RT4_auc']:.3f} · "
              f"RT-5 al dueño {rt5:.3f} (n={r['RT5_n']}) · H ok/err {h_ok:.3f}/{h_er:.3f}", flush=True)
        if A.smoke:
            print("\nRT-0 (a) criterio >= 0,95 · (b) criterio >= 0,90 — el veredicto lo escribe una "
                  "persona, no este print.")
            break

    if not A.smoke:
        n = len(res)
        n1 = sum(1 for r in res.values() if r["RT1_auc"] >= 0.70)
        n2 = sum(1 for r in res.values() if r["RT2_pasa"])
        nd = sum(1 for r in res.values() if r["RT2_domina"])
        n3 = sum(1 for r in res.values() if r["RT3_cierra_ausentes"] >= 0.95)
        n4 = sum(1 for r in res.values() if 0.45 <= r["RT4_auc"] <= 0.55)
        n5 = sum(1 for r in res.values() if r["RT5_al_duenio"] >= 0.50)
        print("\n" + "-" * 78)
        print(f"RT-1 · AUC >= 0,70 en {n1}/{n}   (criterio >= 6/8)")
        print(f"RT-2 · compuerta en {n2}/{n} (>= 6/8) y domina a σ>0,5 en {nd}/{n} (>= 5/8)")
        print(f"        referencias del 20-ago: U-1 = 2/8 · σ>0,5 = 6/8 · U-2 = 7/8 · monitor = 0/8")
        print(f"RT-3 · cierra >= 0,95 con rivales ausentes en {n3}/{n}")
        print(f"RT-4 · AUC del nulo en 0,45-0,55 en {n4}/{n}")
        print(f"RT-5 · apunta al dueño real >= 0,50 en {n5}/{n}   (azar ≈ 1/3)")
        print("\nEl veredicto lo escribe una persona leyendo esta tabla (regla del 13-ago).")

    json.dump({"prereg": "PREREG_ROUNDTRIP.md", "sha": "55ba857a",
               "config": {"n": A.n, "batch": A.batch, "p_nose": A.p_nose, "cand_max": CAND_MAX,
                          "rng_prueba": SEM_PRUEBA, "smoke": A.smoke},
               "unidades": res}, open(A.salida, "w"), indent=1, default=float)
    print(f"\n-> {A.salida}")


if __name__ == "__main__":
    main()
