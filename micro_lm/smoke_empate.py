#!/usr/bin/env python3
"""SMOKE del EMPATE DE CLAVE — ¿la colision es visible en los pesos de lectura, SIN etiquetas?

    python smoke_empate.py --unidades 3_s0,4_s0 --n 8

NO es un experimento: es el chequeo de instrumento que va ANTES de congelar el prereg. Existe
porque el monitor v1 (20-ago) fue un instrumento vacio —perturbaba permutando un archivo al que la
atencion es invariante por construccion— y lo cazo un smoke de 64 muestras. La regla que dejo: la
señal se verifica que EXISTA antes de escribir predicciones sobre ella.

DE DONDE SALE LA HIPOTESIS (`INFORME_ROUNDTRIP_20260820.md`): `err_identidad` no es
marginalizacion sobre la entidad, es COLISION DE CLAVE. El modelo encuentra el hecho por la
RELACION; con relacion unica en el episodio acierta 0,94-0,99 y el error es 0,005-0,014, con
relacion repetida acierta 0,45-0,58 y el error es 0,38-0,54 = el azar entre las dos que empatan.

QUE MIDE, y por que no es ninguna de las tres vias ya cerradas: las tres buscaban la señal en la
SALIDA (el logit, la forma de su densidad, su estabilidad bajo perturbacion). Esta la busca en la
ENTRADA — la distribucion de lectura sobre el archivo, que existe antes de que haya respuesta:

    p   = softmax(sim) sobre las entradas validas, en la posicion desde la que se responde
    r21 = p2 / p1      razon entre el segundo peso y el primero  -> 1 = empate perfecto
    gap = p1 - p2      la misma cosa por la otra cara

LO QUE ESTA MEDICION *NO* REPITE. El 16-ago `foco_lectura.py` midio entropia y masa del top-1 sobre
esta misma distribucion. No es lo mismo y el caso que las separa es justo el que interesa: dos picos
de 0,45 dan masa top-1 BAJA y entropia MEDIA —indistinguible de «disperso»— y sin embargo son el
empate. La entropia mezcla «no matchea nada» con «matchean dos»; `r21` los separa.

EL CONFOUND QUE HAY QUE MIRAR, y es la razon principal de correr esto antes: un hecho REVISADO tiene
sus dos versiones en el archivo con la misma entidad y la misma relacion, asi que empatan en la
clave de contenido y se desempatan por el sello de orden (E-I3: 0,4570 -> 0,9956). Si `r21` sube ahi
tambien, la señal marca casos que el modelo resuelve BIEN y seria una alarma falsa. Por eso el
desglose va cruzado: relacion unica/repetida x hecho revisado/no revisado.

CRITERIO DEL SMOKE, escrito antes de correr (no es un pre-registro; es la condicion para que valga
la pena escribir uno):

  A. AUC(`r21`; relacion repetida vs unica) >= 0,60  -> la señal existe en crudo.
  B. El efecto de la relacion repetida sobrevive DENTRO de los hechos no revisados -> no es el
     confound de versiones disfrazado.

Si A falla, la via no se prueba con esta metrica y el prereg no se escribe.

--------------------------------------------------------------------------------------------------
SEGUNDA PASADA (2026-08-21, declarada tras la primera corrida, con su motivo)

La primera corrida sobre `c3_s0` dio A = 0,5293 y, sobre todo, dio el diagnostico de POR QUE:
`r21 ~ 0,92` y `gap ~ 0,016` en LOS OCHO grupos por igual, con ~6 entradas validas. La lectura del
archivo es **casi uniforme** (uniforme seria 1/6 = 0,167 y el top-1 mide ~0,20). No existe «un top-1
y un top-2»: existen seis pesos parecidos. `r21` esta saturado contra 1 por construccion y no puede
separar nada — el mismo defecto de forma que el monitor v1, en otra parte del instrumento.

Que la distribucion sea plana NO significa que no haya estructura: el 16-ago `rank_hecho.py` midio
que en los aciertos la entrada correcta encabeza 47-50 % de las veces, muy por encima del 1/6 que
daria una lectura sin informacion. O sea la señal vive en el ORDEN de los scores, no en su magnitud
normalizada.

Por eso se agrega UNA metrica, y una sola, sobre los mismos datos:

    z12 = (s1 - s2) / std(s validos)     margen entre los dos primeros scores CRUDOS, en unidades
                                         de la dispersion del propio episodio

Es la lectura correcta de «dos pesos altos y parecidos» cuando la temperatura efectiva aplana todo:
lo que dice si dos entradas empatan ENTRE SI no es cuanto valen, es cuanto se separan comparado con
lo que se separan las demas. El criterio A se re-evalua sobre `z12` con el mismo umbral 0,60.

Se deja escrito para que no sea un rescate silencioso: la metrica original queda reportada al lado,
con su fallo, y si `z12` tampoco llega la via se cierra con las dos medidas a la vista.
"""
import argparse
import os
import pickle
import sys

import numpy as np
import jax
import jax.numpy as jnp

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
import datos as DAT
import idioma as I
import modelo as M
from score_archivo import scores_archivo, auc
from ser import clasificar

NOSE = I.STOI["NOSE"]
SEM_PRUEBA = 77000          # el mismo generador de prueba que la campaña de presupuesto


def scores_todas(params, ses, cortes, turnos, mask, consulta):
    """Como `scores_archivo`, pero devuelve (B, Tq, N): TODAS las posiciones de la consulta.

    TERCERA PASADA (2026-08-21), y el motivo es mecanico, no un rescate. En `modelo.tronco` la
    lectura se inyecta en el bloque 0 ANTES de la conv y del mixer, sobre `h = emb[x]`. Por lo tanto
    la query que consulta el archivo es `ln(emb[token]) @ qr`: **funcion pura del token de esa
    posicion**, sin una sola operacion de contexto delante. La posicion de la respuesta —que es
    donde miraban `scores_archivo` y la primera pasada de este smoke— es el ULTIMO token de la
    pregunta, y ahi no hay nada que matchear: por eso salia plana en las ocho celdas.

    Consecuencia, y es la que da la hipotesis: el modelo NO puede formar una query conjunta
    entidad x relacion en el bloque 0. Consulta el archivo token por token y la conjuncion la
    resuelve aguas abajo, integrando. Eso explica mecanicamente el atajo de la relacion del
    `INFORME_ROUNDTRIP`: en la posicion del token de la relacion, la query matchea a TODAS las
    entradas que comparten esa relacion, y el empate es entre ellas.
    """
    a = params["arch"]
    archivo = M.escribir(params, ses, cortes)
    ak = archivo @ a["kw"] + a["ord"][turnos]
    penal = jnp.where(mask, 0.0, -1e9)
    h = params["emb"][consulta]
    q = M.ln(params["blocks"][0]["ln1"], h) @ a["qr"]
    return jnp.einsum("btd,bnd->btn", q, ak) / jnp.sqrt(h.shape[-1]) + penal[:, None, :]


def responder(params, ses, cortes, turnos, cons, mask, pos):
    archivo = M.escribir(params, ses, cortes)
    lg, _ = M.responder_con_abst(params, archivo, turnos, cons, mask)
    lg = jnp.take_along_axis(lg, pos[:, None, None], axis=1)[:, 0, :]
    return lg.at[:, NOSE].set(-jnp.inf).argmax(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--unidades", default="3_s0,4_s0")
    ap.add_argument("--n", type=int, default=8, help="lotes por unidad")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--dir-ckpt", default=os.path.join(AQUI, "ckpts", "rt_congelados"))
    a = ap.parse_args()

    print("SMOKE · EMPATE DE CLAVE en la distribucion de lectura")
    print(f"{a.n * a.batch} muestras por unidad · solo preguntas CON respuesta (p_nose=0)\n")

    for u in a.unidades.split(","):
        ck = os.path.join(a.dir_ckpt, f"c{u}.pkl")
        if not os.path.exists(ck):
            print(f"c{u}: sin checkpoint congelado, se saltea")
            continue
        with open(ck, "rb") as f:
            d = pickle.load(f)
        params = jax.tree_util.tree_map(jnp.asarray, d["params"])
        nivel = int(u[0])
        semilla = int(u.split("_s")[1].split("_")[0])
        rng = np.random.default_rng(SEM_PRUEBA + semilla)
        fn = jax.jit(responder)

        R21, GAP, Z12, ZFOC, ZMIN, REP, REV, OK, ID, NVAL = (
            [], [], [], [], [], [], [], [], [], [])
        for _ in range(a.n):
            sal = DAT.lote(rng, a.batch, nivel=nivel, n_hechos=4, n_sesiones=4,
                           p_vieja=0.35, p_nose=0.0, con_meta=True)
            ses, cortes, turnos, mask, cons, pos, tgt, tipo, meta = sal
            jses, jcor = jnp.array(ses), jnp.array(cortes)
            jtur, jmask = jnp.array(turnos), jnp.array(np.asarray(mask))
            jcons, jpos = jnp.array(cons), jnp.array(pos)

            X = np.asarray(fn(params, jses, jcor, jtur, jcons, jmask, jpos))
            s = np.asarray(scores_archivo(params, jses, jcor, jtur, jmask, jcons, jpos))
            p = np.asarray(jax.nn.softmax(jnp.array(s), -1))
            stodas = np.asarray(scores_todas(params, jses, jcor, jtur, jmask, jcons))
            tgt = np.asarray(tgt)
            tipo = np.asarray(tipo)

            for b in range(len(X)):
                m = meta[b]
                if tipo[b] >= 2 or not m["hecho"]:
                    continue
                orden = np.sort(p[b])[::-1]
                p1, p2 = float(orden[0]), float(orden[1])
                # `z12` va sobre los scores CRUDOS y solo sobre las entradas validas: las vacias
                # estan en -1e9 y arruinarian cualquier desviacion estandar.
                sv = s[b][np.asarray(mask)[b].astype(bool)]
                so = np.sort(sv)[::-1]
                sd = float(sv.std())
                rel = m["hecho"]["rel"]
                x = I.ITOS[X[b]]
                R21.append(p2 / (p1 + 1e-12))
                GAP.append(p1 - p2)
                Z12.append(float(so[0] - so[1]) / (sd + 1e-12) if len(so) > 1 else np.nan)

                # Las mismas dos cuentas, pero barriendo las posiciones de la consulta hasta la de
                # la respuesta (lo de mas alla es relleno). `z_foco` mira donde el matcheo es mas
                # fuerte; `z_min` pregunta si EXISTE alguna posicion con dos entradas empatadas,
                # que es el detector directo.
                val = np.asarray(mask)[b].astype(bool)
                st = stodas[b][: int(pos[b]) + 1][:, val]              # (T util, N validas)
                if st.shape[1] > 1 and st.shape[0] > 0:
                    ordt = np.sort(st, axis=1)[:, ::-1]
                    sdt = st.std(axis=1) + 1e-12
                    zt = (ordt[:, 0] - ordt[:, 1]) / sdt
                    ZFOC.append(float(zt[int(np.argmax(st.max(axis=1)))]))
                    ZMIN.append(float(zt.min()))
                else:
                    ZFOC.append(np.nan); ZMIN.append(np.nan)
                REP.append(any(o["rel"] == rel for o in m["otros"]))
                REV.append(len(m["hecho"]["versiones"]) > 1)
                OK.append(X[b] == tgt[b])
                ID.append((X[b] != tgt[b]) and any(x in o["versiones"] for o in m["otros"]))
                NVAL.append(int(np.asarray(mask)[b].sum()))

        R21 = np.array(R21); GAP = np.array(GAP); Z12 = np.array(Z12)
        ZFOC = np.array(ZFOC); ZMIN = np.array(ZMIN)
        REP = np.array(REP); REV = np.array(REV)
        OK = np.array(OK); ID = np.array(ID)

        print(f"=== c{u} · nivel {nivel} · paso {d.get('paso', '?')} · n={len(R21)} · "
              f"entradas validas ~{np.mean(NVAL):.0f}")
        print(f"    acierto {OK.mean():.4f} · err_identidad {ID.mean():.4f} · "
              f"P(relacion repetida) {REP.mean():.4f} · P(revisado) {REV.mean():.4f}")
        print(f"    {'grupo':<28} {'n':>5} {'r21':>8} {'gap':>8} {'z12':>8} "
              f"{'z_foco':>8} {'z_min':>8}")
        for nom, sel in (("relacion unica", ~REP), ("relacion repetida", REP),
                         ("  unica  · no revisado", ~REP & ~REV),
                         ("  unica  · revisado", ~REP & REV),
                         ("  repet. · no revisado", REP & ~REV),
                         ("  repet. · revisado", REP & REV),
                         ("acierto", OK), ("err_identidad", ID)):
            if sel.sum() == 0:
                continue
            print(f"    {nom:<28} {int(sel.sum()):>5} {R21[sel].mean():>8.4f} "
                  f"{GAP[sel].mean():>8.4f} {Z12[sel].mean():>8.4f} "
                  f"{np.nanmean(ZFOC[sel]):>8.4f} {np.nanmean(ZMIN[sel]):>8.4f}")

        # En `z12` el empate es margen CHICO, asi que se le da vuelta el signo para que la AUC
        # siga leyendose igual que la de `r21`: > 0,5 = «la repetida se ve mas empatada».
        a_rep = auc(R21[REP], R21[~REP])
        a_err = auc(R21[ID], R21[OK])
        a_rep_nr = (auc(R21[REP & ~REV], R21[~REP & ~REV])
                    if (REP & ~REV).sum() and (~REP & ~REV).sum() else float("nan"))
        z_rep = auc(-Z12[REP], -Z12[~REP])
        z_err = auc(-Z12[ID], -Z12[OK])
        z_rep_nr = (auc(-Z12[REP & ~REV], -Z12[~REP & ~REV])
                    if (REP & ~REV).sum() and (~REP & ~REV).sum() else float("nan"))
        f_rep = auc(-ZFOC[REP], -ZFOC[~REP])
        f_err = auc(-ZFOC[ID], -ZFOC[OK])
        f_rep_nr = (auc(-ZFOC[REP & ~REV], -ZFOC[~REP & ~REV])
                    if (REP & ~REV).sum() and (~REP & ~REV).sum() else float("nan"))
        m_rep = auc(-ZMIN[REP], -ZMIN[~REP])
        m_err = auc(-ZMIN[ID], -ZMIN[OK])
        m_rep_nr = (auc(-ZMIN[REP & ~REV], -ZMIN[~REP & ~REV])
                    if (REP & ~REV).sum() and (~REP & ~REV).sum() else float("nan"))
        mejor = max(z_rep, f_rep, m_rep)
        print(f"\n    {'':<42}{'r21':>8} {'z12':>8} {'z_foco':>8} {'z_min':>8}")
        print(f"    A · AUC(repetida vs unica)                {a_rep:>8.4f} {z_rep:>8.4f} "
              f"{f_rep:>8.4f} {m_rep:>8.4f}   "
              f"{'ALGUNA PASA' if mejor >= 0.60 else 'ninguna llega a 0,60'}")
        print(f"    B · idem, solo hechos NO revisados        {a_rep_nr:>8.4f} {z_rep_nr:>8.4f} "
              f"{f_rep_nr:>8.4f} {m_rep_nr:>8.4f}")
        print(f"    -   AUC(err_identidad vs acierto)         {a_err:>8.4f} {z_err:>8.4f} "
              f"{f_err:>8.4f} {m_err:>8.4f}   (informativo: usa etiquetas)\n")


if __name__ == "__main__":
    main()
