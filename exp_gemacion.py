"""Gemacion en un modelo chico: ¿preserva informacion que la sobrescritura pierde?

Diseno deliberadamente minimo. El modelo se PREENTRENA en MQAR normal y despues se CONGELA: no
se aprende nada nuevo. El archivo es no parametrico. Asi la pregunta queda aislada — no se mide
"si el modelo aprende a usar una memoria", se mide si la geometria de la gemacion **preserva
informacion recuperable** que la sobrescritura destruye.

Procedimiento por episodio (estado recurrente RESETEADO entre secuencias):
  S1 escritura -> se recorren los pares y cada (clave_latente, token_valor) entra al archivo
  S2 revision  -> idem, con los valores nuevos
  S3 consulta  -> se busca en el archivo por similitud y se devuelve el token guardado

Tres condiciones de archivo:
  SIN        no hay archivo (control). Por construccion debe dar AZAR: el estado se reseteo.
  SOBRESC.   si la clave entrante se parece a una guardada (cos > umbral), REEMPLAZA esa entrada.
  GEMACION   siempre agrega una entrada nueva; si se parece a una guardada, la deposita
             desplazada eps en el eje temporal del recuerdo (t_hat por recuerdo, R4).

Lectura: se recupera el clúster por similitud y se desempata por posicion en el eje temporal —
mas avanzado = mas reciente. Para el objetivo ANTERIOR se pide el penultimo del clúster.

Se usa la FUSION de las 4 cabezas (R8/R10.3: una sola medicion en d=16 colapsa).
"""
import os, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "2")
sys.path.insert(0, os.path.expanduser("~/Documentos/Nuevo Transformer/telar-ligamento/src"))

import numpy as np, jax, jax.numpy as jnp, optax
import modelos as M
from modelos import split_heads, l2n, ln, conv3, H, DH
from datos import V_E001, IGNORE, gen_mqar
from entrenar import loss_fn
from tarea_cross import gen_cross

CKPT = "modelo_base.npz"
EPS_GEM = 0.30          # radio de gemacion (R2: la ventana util)
UMBRAL = 0.90           # cos por encima del cual se considera "la misma clave"


def claves_latentes(params, x):
    """(B, T, H, DH) normalizadas — el espacio donde vive el archivo.

    SIN conv3, a proposito. El PASO A midio que `W_k sobre conv3` separa "misma clave" de
    "clave distinta" con AUC 0.789 — insuficiente para un archivo persistente, porque conv3
    mezcla cada token con sus vecinos y la misma clave cambia de representacion segun este
    rodeada de pares (S1) o de consultas (S3). `W_k sobre ln1` da AUC 1.000 (misma 1.000 vs
    distinta 0.247) y sigue siendo una representacion interna del modelo, no el embedding crudo.
    """
    blk = params["blocks"][0]
    hx = params["emb"][x]
    xin = ln(blk["ln1"], hx)
    k = l2n(jax.nn.silu(split_heads(xin @ blk["k"])))      # (B,H,T,DH)
    return np.asarray(k.transpose(0, 2, 1, 3))             # (B,T,H,DH)


def preentrenar(pasos=1500, batch=16, carga=8, lr=3e-3, seed=0):
    if os.path.exists(CKPT):
        z = np.load(CKPT, allow_pickle=True)
        return jax.tree_util.tree_map(jnp.asarray, z["p"].item())
    gv = jax.jit(jax.value_and_grad(loss_fn, has_aux=True), static_argnums=3)
    opt = optax.adam(lr); p = M.init_params(seed, "delta"); st = opt.init(p)
    t0 = time.time()
    for s in range(pasos + 1):
        x, y = gen_mqar(np.random.default_rng(s), batch, carga)
        (l, a), g = gv(p, jnp.asarray(x), jnp.asarray(y), "delta")
        upd, st = opt.update(g, st, p); p = optax.apply_updates(p, upd)
    print(f"preentrenado: loss {float(l):.3f} acc {float(a):.3f} [{time.time()-t0:.0f}s]")
    np.savez(CKPT, p=jax.tree_util.tree_map(np.asarray, p))
    return p


def tangente(t, x):
    t = t - (t * x).sum(-1, keepdims=True) * x
    return t / (np.linalg.norm(t, axis=-1, keepdims=True) + 1e-8)


class Archivo:
    """Indice persistente. Cada entrada: direccion por cabeza (H,DH), token, y paso temporal."""

    def __init__(self, modo, d=DH, seed=0):
        self.modo = modo
        self.dirs = []       # lista de (H, DH)
        self.tok = []
        self.paso = []       # cuantas gemaciones lleva ese recuerdo
        self.eje = []        # eje temporal por recuerdo (H, DH)
        self.rng = np.random.default_rng(seed)

    def _sim(self, k):
        if not self.dirs:
            return None, -1.0
        A = np.stack(self.dirs)                       # (N,H,DH)
        s = np.einsum("nhd,hd->n", A, k) / H          # fusion de las 4 cabezas
        j = int(np.argmax(s))
        return j, float(s[j])

    def escribir(self, k, token):
        j, s = self._sim(k)
        if self.modo == "sobrescritura":
            if j is not None and s > UMBRAL:
                self.dirs[j] = k; self.tok[j] = token   # pisa: la version vieja se pierde
            else:
                self._nuevo(k, token)
        else:                                            # gemacion
            if j is not None and s > UMBRAL:
                e = self.eje[j]
                nueva = self.dirs[j] + EPS_GEM * tangente(e, self.dirs[j])
                nueva = nueva / (np.linalg.norm(nueva, axis=-1, keepdims=True) + 1e-8)
                self.dirs.append(nueva); self.tok.append(token)
                self.paso.append(self.paso[j] + 1); self.eje.append(e)
            else:
                self._nuevo(k, token)

    def _nuevo(self, k, token):
        e = self.rng.normal(size=(H, DH)).astype(np.float32)
        e = e / np.linalg.norm(e, axis=-1, keepdims=True)
        self.dirs.append(k.copy()); self.tok.append(token)
        self.paso.append(0); self.eje.append(tangente(e, k))

    def leer(self, q, cual="vigente", topk=6):
        """Recupera el clúster por similitud y desempata por avance en el eje temporal."""
        if not self.dirs:
            return None
        A = np.stack(self.dirs)
        s = np.einsum("nhd,hd->n", A, q) / H
        idx = np.argsort(-s)[:topk]
        idx = [i for i in idx if s[i] > UMBRAL - 0.15] or [int(np.argmax(s))]
        pasos = [self.paso[i] for i in idx]
        if cual == "vigente":
            return self.tok[idx[int(np.argmax(pasos))]]
        orden = np.argsort(pasos)                      # anterior = penultimo del clúster
        return self.tok[idx[orden[-2]]] if len(orden) >= 2 else None


def episodio(params, d, b, modo):
    """Procesa un episodio (fila b) y devuelve (aciertos_vigente, aciertos_anterior, n_ant)."""
    voc = V_E001
    arch = Archivo(modo, seed=b) if modo != "sin" else None
    if arch is not None:
        for x, en_pares in ((d["x1"], True), (d["x2"], True)):
            K = claves_latentes(params, jnp.asarray(x[b:b + 1]))[0]     # (T,H,DH)
            for t in range(1, x.shape[1] - 1, 2):
                tokv = int(x[b, t + 1])
                if voc.V0 <= tokv < voc.V0 + voc.NV:
                    arch.escribir(K[t], tokv)

    Kq = claves_latentes(params, jnp.asarray(d["x3"][b:b + 1]))[0]
    ok_v = ok_a = n_a = 0
    L = d["x3"].shape[1] - 2
    for i in range(L):
        col = 2 + i
        tgt_v = int(d["y_vig"][b, col])
        pred = arch.leer(Kq[col], "vigente") if arch else None
        if pred is None:
            pred = voc.V0 + np.random.default_rng(b * 100 + i).integers(0, voc.NV)
        ok_v += int(pred == tgt_v)
        tgt_a = int(d["y_ant"][b, col])
        if tgt_a != IGNORE:
            n_a += 1
            pa = arch.leer(Kq[col], "anterior") if arch else None
            if pa is None:
                pa = voc.V0 + np.random.default_rng(b * 200 + i).integers(0, voc.NV)
            ok_a += int(pa == tgt_a)
    return ok_v, L, ok_a, n_a


def main(B=48, L=8, r=4, seed=7):
    p = preentrenar()
    d = gen_cross(np.random.default_rng(seed), B, L, r)
    print(f"\nepisodios={B}  L={L} pares  r={r} revisados  azar={d['azar']:.4f}\n")
    print(f"{'archivo':>16} {'VIGENTE':>18} {'ANTERIOR':>18} {'entradas':>9}")
    for modo in ("sin", "sobrescritura", "gemacion"):
        t0 = time.time(); V = Vn = A = An = 0; ents = []
        for b in range(B):
            a, an, c, cn = episodio(p, d, b, modo)
            V += a; Vn += an; A += c; An += cn
        print(f"{modo:>16} {V/Vn:18.3f} {A/max(An,1):18.3f} "
              f"{'-' if modo=='sin' else '':>9}   [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
