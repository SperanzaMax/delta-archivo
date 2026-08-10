"""Tarea de hechos versionados: genera embeddings LOCALES y corre las cuatro condiciones.

Implementa exactamente lo congelado en PREREG_HECHOS.md + D1 + E1 + ENMIENDA_E2_INDEXACION.md.
Nada acá decide nada: todas las decisiones de diseño están en esos documentos, con hash.

Orden de ejecución, y es el orden por una razón:
  1. compuerta (compuerta_encoder.py) — si falla, ABORTA sin generar datos
  2. embeddings en MINÚSCULA y en local (E1 + hallazgo del tokenizador)
  3. las cuatro condiciones × 10 semillas
  4. P1–P4 con IC por t de Student, 9 gl
"""
import json
import os
import sys
import time
import urllib.request

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tarea_hechos import gen_hechos
import compuerta_encoder as CG

MODELO = "nomic-embed-text"
N = 3000                  # §6 del prereg
N_SEMILLAS = 10           # §6
N_SUB = 1000              # D1: submuestreo por semilla
K = 5                     # §5: cobertura en top-k
EPS = 0.30                # §6, de R2 sin re-ajustar
MARGEN = 0.02             # §6, absoluto, para P1 y P3
SALIDA = "hechos_min.npz"
INFORME = "INFORME_HECHOS_FINAL.md"


def log(m):
    print(m, flush=True)


def embed_lote(textos, lote=64):
    """Embeddings normalizados. MINÚSCULA: ver HALLAZGO_TOKENIZADOR_20260810.md."""
    out, t0 = [], time.time()
    for i in range(0, len(textos), lote):
        payload = {"model": MODELO, "input": [t.lower() for t in textos[i:i + lote]]}
        req = urllib.request.Request("http://localhost:11434/api/embed",
                                     data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        for intento in range(3):
            try:
                out.extend(json.load(urllib.request.urlopen(req, timeout=300))["embeddings"])
                break
            except Exception as e:
                if intento == 2:
                    raise RuntimeError(f"endpoint caído en el lote {i}: {e}")
        if (i // lote) % 10 == 0:
            log(f"    {len(out)}/{len(textos)} — {time.time()-t0:.0f}s")
    X = np.array(out, dtype=np.float32)
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)


def tangente(t, x):
    """Componente de t ortogonal a x, normalizada (exp_gemacion.py:72-74)."""
    t = t - (t * x).sum(-1, keepdims=True) * x
    return t / (np.linalg.norm(t, axis=-1, keepdims=True) + 1e-8)


def construir_archivo(cond, E1, E2, idx, rng):
    """Índice de la condición, según §2.1 de la enmienda E2.

    Devuelve (dirs, contenido, revision, dueno):
      dirs      (M,d) direcciones contra las que se compara la consulta
      contenido (M,)  índice de entidad cuyo valor devuelve la entrada; +0 = v1, +N = v2
      revision  (M,)  contador entero: 0 = original, 1 = revisión (metadato, no geometría)
      dueno     (M,)  a qué entidad pertenece la entrada
    """
    dirs, cont, rev, due = [], [], [], []
    for i in idx:
        if cond == "sin":
            continue
        # entrada original (v1) — existe en las tres condiciones con archivo
        if cond != "sobrescritura":
            dirs.append(E1[i]); cont.append(("v1", i)); rev.append(0); due.append(i)
        if cond == "sobrescritura":
            # reemplaza: la versión vieja se pierde
            dirs.append(E2[i]); cont.append(("v2", i)); rev.append(1); due.append(i)
        elif cond == "duplicados":
            dirs.append(E2[i]); cont.append(("v2", i)); rev.append(1); due.append(i)
        elif cond == "gemacion":
            # dirección ANCLADA a v1, contenido de v2 (§2.1 de la enmienda)
            eje = tangente(rng.normal(size=E1.shape[1]).astype(np.float32), E1[i])
            nueva = E1[i] + EPS * eje
            nueva = nueva / (np.linalg.norm(nueva) + 1e-8)
            dirs.append(nueva); cont.append(("v2", i)); rev.append(1); due.append(i)
    if not dirs:
        return None
    return (np.stack(dirs), cont, np.array(rev), np.array(due))


def evaluar(cond, E1, E2, EQ, idx, rng):
    """VIGENTE, ANTERIOR y COBERTURA según §2.4 de la enmienda."""
    arch = construir_archivo(cond, E1, E2, idx, rng)
    if arch is None:                       # condición `sin`: no hay archivo
        return dict(vigente=0.0, anterior=0.0, cobertura=0.0)
    dirs, cont, rev, due = arch
    Q = EQ[idx]                            # (n,d)
    S = Q @ dirs.T                          # (n,M)
    top = np.argsort(-S, axis=1)[:, :K]

    vig = ant = cob = 0
    for fila, i in enumerate(idx):
        cand = top[fila]
        # clúster recuperado = entradas del top-k que pertenecen a la entidad consultada
        mio = [j for j in cand if due[j] == i]
        if mio:
            r = rev[mio]
            # VIGENTE: mayor contador de revisión
            if cont[mio[int(np.argmax(r))]][0] == "v2":
                vig += 1
            # ANTERIOR: penúltimo por contador; con un solo elemento, falla
            if len(mio) >= 2:
                orden = np.argsort(r)
                if cont[mio[orden[-2]]][0] == "v1":
                    ant += 1
        # COBERTURA: ambas versiones del recuerdo entre los top-k
        vers = {cont[j][0] for j in cand if due[j] == i}
        if vers == {"v1", "v2"}:
            cob += 1
    n = len(idx)
    return dict(vigente=vig / n, anterior=ant / n, cobertura=cob / n)


def ic_t(v):
    """Media ± IC95 por t de Student con 9 gl (§6 del prereg)."""
    v = np.asarray(v, float)
    m, s = v.mean(), v.std(ddof=1)
    h = 2.262 * s / np.sqrt(v.size)        # t(0.975, 9) = 2.262
    return m, m - h, m + h


def main():
    log("=== 1. COMPUERTA (aborta si falla) ===\n")
    if CG.correr(MODELO, 400) != 0:
        log("\nCOMPUERTA CERRADA — no se genera ningún dato.")
        return 1

    log(f"\n=== 2. embeddings · {N} entidades × 3 textos, en MINÚSCULA y local ===")
    items = gen_hechos(np.random.default_rng(0), N)          # D1: un corpus, semilla 0
    if os.path.exists(SALIDA):
        d = np.load(SALIDA)
        E1, E2, EQ = d["E1"], d["E2"], d["EQ"]
        log(f"  reusando {SALIDA}")
    else:
        X = embed_lote([x["v1"] for x in items] + [x["v2"] for x in items]
                       + [x["consulta"] for x in items])
        E1, E2, EQ = X[:N], X[N:2 * N], X[2 * N:]
        np.savez(SALIDA, E1=E1, E2=E2, EQ=EQ)
        log(f"  guardado {SALIDA} — {X.shape}")

    # chequeo de discriminación sobre los datos DEFINITIVOS, no sólo sobre la muestra
    ident = int((E1 == E2).all(1).sum())
    unicos = len(np.unique(E1, axis=0))
    log(f"  v1==v2 idénticos: {ident}/{N} · vectores únicos: {unicos}/{N}")
    if ident or unicos < 0.95 * N:
        log("  ABORTA: los datos definitivos no pasan el chequeo de discriminación.")
        return 1

    log(f"\n=== 3. las cuatro condiciones × {N_SEMILLAS} semillas ===")
    conds = ("sin", "sobrescritura", "duplicados", "gemacion")
    res = {c: {"vigente": [], "anterior": [], "cobertura": []} for c in conds}
    for s in range(N_SEMILLAS):
        rng = np.random.default_rng(1000 + s)                 # D1: submuestreo y ejes
        idx = rng.choice(N, N_SUB, replace=False)
        for c in conds:
            r = evaluar(c, E1, E2, EQ, idx, np.random.default_rng(2000 + s))
            for k, v in r.items():
                res[c][k].append(v)
        log(f"  semilla {s}: " + " · ".join(
            f"{c} vig {res[c]['vigente'][-1]:.3f}/ant {res[c]['anterior'][-1]:.3f}"
            f"/cob {res[c]['cobertura'][-1]:.3f}" for c in conds[1:]))

    log(f"\n=== 4. veredictos ===\n")
    L = [f"# Tarea de hechos versionados — resultados\n",
         f"N = {N} entidades · {N_SEMILLAS} semillas × {N_SUB} · k = {K} · ε = {EPS} · "
         f"margen {MARGEN} · encoder `{MODELO}` en minúscula\n",
         "Prereg + D1 + E1 + enmienda E2, todos congelados con hash antes del dato.\n",
         "## Métricas por condición (media, IC95 por t de Student, 9 gl)\n",
         "| condición | VIGENTE | ANTERIOR | COBERTURA |", "|---|---|---|---|"]
    M = {}
    for c in conds:
        M[c] = {k: ic_t(res[c][k]) for k in ("vigente", "anterior", "cobertura")}
        L.append(f"| `{c}` | " + " | ".join(
            f"{M[c][k][0]:.4f} [{M[c][k][1]:.4f}, {M[c][k][2]:.4f}]"
            for k in ("vigente", "anterior", "cobertura")) + " |")

    # P1 — gemacion > duplicados en COBERTURA, por encima del margen
    d1 = np.array(res["gemacion"]["cobertura"]) - np.array(res["duplicados"]["cobertura"])
    m1, lo1, hi1 = ic_t(d1)
    p1 = bool(m1 > MARGEN and lo1 > 0)
    # P2 — sobrescritura ≈ azar en ANTERIOR
    m2, lo2, hi2 = M["sobrescritura"]["anterior"]
    p2 = bool(hi2 < 0.05)
    # P3 — gemacion ≥ sobrescritura − margen en VIGENTE
    d3 = np.array(res["gemacion"]["vigente"]) - np.array(res["sobrescritura"]["vigente"])
    m3, lo3, hi3 = ic_t(d3)
    p3 = bool(lo3 > -MARGEN)

    L += ["\n## Predicciones pre-registradas\n",
          f"**P1 (principal)** cobertura `gemacion` − `duplicados` = **{m1:+.4f}** "
          f"IC95 [{lo1:+.4f}, {hi1:+.4f}] · margen {MARGEN} → "
          f"**{'CONFIRMA' if p1 else 'NO CONFIRMA'}**",
          f"\n**P2 (control)** ANTERIOR de `sobrescritura` = {m2:.4f} "
          f"IC95 [{lo2:.4f}, {hi2:.4f}] → "
          f"**{'OK, la tarea mide lo que dice' if p2 else 'FUGA — el experimento sería inválido'}**",
          f"\n**P3** VIGENTE `gemacion` − `sobrescritura` = {m3:+.4f} "
          f"IC95 [{lo3:+.4f}, {hi3:+.4f}] · piso −{MARGEN} → "
          f"**{'CUMPLE' if p3 else 'NO CUMPLE'}** (anclar no cuesta precisión sobre el valor al día)",
          "\n**P4** (ley de escala en K) — no corrida en esta tanda; requiere K ∈ {2,4,8}.",
          "\n**P5** VersionRAG reporta 58 % en consultas versionadas. Se cita como contexto; "
          "el prereg prohíbe declarar superioridad porque la tarea no es idéntica."]

    txt = "\n".join(L) + "\n"
    open(INFORME, "w").write(txt)
    print(txt)
    json.dump({c: {k: list(map(float, v)) for k, v in res[c].items()} for c in conds},
              open("resultados_hechos.json", "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
