"""COMPUERTA del experimento en modelo real · decide ANTES de gastar una sola GPU · 2026-09-02

Tres cosas, y si cualquiera falla el experimento NO se hace.

  1. el ALCANCE REAL de la conv de `mamba-370m`, medido por intervencion y no leido del config.
     En `mamba-130m` el tap mas viejo vale cero exacto en las 24 capas y el alcance real es 2, no 3.
  2. las DISTANCIAS REALES de cada componente de la pregunta, contadas en TOKENS del BPE y no en
     palabras. «Zephyra» puede ser tres tokens y arruinar toda la geometria en silencio.
  3. que EXISTA el contraste: al menos una plantilla con la relacion ADENTRO de la ventana y otra con
     la relacion AFUERA, con la entidad en el mismo lugar en las dos.

Es exactamente el error que cometi hoy dos veces, escribir un criterio sobre un numero SUPUESTO en
vez del MEDIDO. Aca se mide primero.
"""
import os
import sys

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODELO = os.environ.get("MODELO", "state-spaces/mamba-370m-hf")

# Entidades y valores INVENTADOS, para que el modelo no pueda contestar de memoria previa.
ENTIDADES = ["Zephyra", "Quandor", "Velmara", "Torrin", "Nyxalis", "Grendale", "Cassoway",
             "Umbriel", "Fenwick", "Aldrath", "Bexley", "Corvane"]
VALORES = ["Kalen", "Bryse", "Doran", "Elvire", "Fyrn", "Gastel", "Hollis", "Imre"]
RELACIONES = ["director", "warden", "founder", "keeper", "curator", "steward"]

# Plantillas candidatas. La posicion de lectura es el ULTIMO token, que es donde el modelo genera.
# `{r}` es la relacion y `{e}` la entidad.
PLANTILLAS = {
    "directa":    "What is the {r} of {e}?",
    "invertida":  "For {e}, what is the {r}?",
    "lejana":     "What is the {r} that {e} has?",
    "corta":      "The {r} of {e} is",
    "inv_corta":  "For {e}, the {r} is",
}


def alcance_real(modelo):
    conv = modelo.backbone.layers[0].mixer.conv1d
    k = conv.kernel_size[0]
    w = conv.weight[:, 0, :]
    vivos = [t for t in range(k) if float(w[:, t].abs().max()) > 0]
    print(f"  kernel nominal {k} · taps con peso NO nulo en la capa 0: {vivos}")
    nulos = sum(1 for c in modelo.backbone.layers
                if float(c.mixer.conv1d.weight[:, 0, 0].abs().max()) == 0.0)
    print(f"  capas con el tap 0 en CERO EXACTO: {nulos} de {len(modelo.backbone.layers)}")

    # medicion por INTERVENCION, que es la que manda
    capt = {}
    h = conv.register_forward_hook(lambda m, i, o: capt.__setitem__("o", o.detach().clone()))
    ids = torch.arange(60, 90).unsqueeze(0)
    modelo(ids)
    base = capt["o"][0].clone()
    otro = ids.clone(); otro[0, 10] = 999
    modelo(otro)
    d = (capt["o"][0] - base).abs().max(0).values
    movidas = [int(p) for p in torch.nonzero(d > 0).flatten()]
    h.remove()
    alc = max(movidas) - 10 if movidas else -1
    print(f"  cambiando el token 10, la conv se mueve en las posiciones {movidas}")
    print(f"  >> ALCANCE REAL MEDIDO = {alc} tokens hacia atras")
    return alc


def distancias(tok):
    """Para cada plantilla, la distancia en TOKENS de la relacion y de la entidad al ultimo token."""
    print(f"\n  {'plantilla':11s} {'d_rel':>17s} {'d_ent':>17s}   ejemplo tokenizado")
    out = {}
    for nom, plt in PLANTILLAS.items():
        drs, des = [], []
        for e in ENTIDADES:
            for r in RELACIONES:
                texto = plt.format(r=r, e=e)
                ids = tok(texto).input_ids
                piezas = [tok.decode([i]) for i in ids]
                n = len(ids)
                # la ULTIMA pieza que contiene la relacion / la entidad
                ir = max((j for j, p in enumerate(piezas) if r.startswith(p.strip())
                          and p.strip()), default=None)
                ie = max((j for j, p in enumerate(piezas) if p.strip()
                          and e.lower().startswith(p.strip().lower())), default=None)
                if ir is None or ie is None:
                    continue
                drs.append(n - 1 - ir); des.append(n - 1 - ie)
        if not drs:
            print(f"  {nom:11s}  (no se pudo alinear)"); continue
        ej = tok(PLANTILLAS[nom].format(r=RELACIONES[0], e=ENTIDADES[0])).input_ids
        out[nom] = (min(drs), max(drs), min(des), max(des))
        print(f"  {nom:11s} {min(drs):3d}..{max(drs):<3d} {'(fijo)' if min(drs)==max(drs) else '(VARIA)':>8s}"
              f" {min(des):3d}..{max(des):<3d} {'(fijo)' if min(des)==max(des) else '(VARIA)':>8s}"
              f"   {[tok.decode([i]) for i in ej]}")
    return out


def main():
    torch.set_grad_enabled(False)
    print("=" * 104)
    print(f"COMPUERTA DEL EXPERIMENTO EN MODELO REAL · {MODELO}")
    print("=" * 104)
    tok = AutoTokenizer.from_pretrained(MODELO)
    modelo = AutoModelForCausalLM.from_pretrained(MODELO, dtype=torch.float32).eval()
    print(f"  parametros {sum(p.numel() for p in modelo.parameters()):,}\n")

    print("1 · ALCANCE REAL de la conv que forma la query")
    print("-" * 104)
    alc = alcance_real(modelo)

    print("\n2 · DISTANCIAS REALES en tokens del BPE (no en palabras)")
    print("-" * 104)
    d = distancias(tok)

    print("\n3 · ¿EXISTE EL CONTRASTE?")
    print("-" * 104)
    dentro = [n for n, (rmin, rmax, _, _) in d.items() if rmax <= alc]
    afuera = [n for n, (rmin, rmax, _, _) in d.items() if rmin > alc]
    print(f"  con la relacion SIEMPRE ADENTRO de la ventana (d_rel <= {alc}): {dentro or 'NINGUNA'}")
    print(f"  con la relacion SIEMPRE AFUERA (d_rel > {alc}):                 {afuera or 'NINGUNA'}")
    fijas = [n for n, v in d.items() if v[0] == v[1] and v[2] == v[3]]
    print(f"  con distancias FIJAS, o sea sin depender de cuantos tokens ocupe el nombre: {fijas}")

    ok = bool(dentro) and bool(afuera)
    print("\n" + "=" * 104)
    print(f"COMPUERTA {'ABRE' if ok else 'NO ABRE'}"
          f"{'' if ok else '  <- sin las dos familias no hay experimento, no se gasta GPU'}")
    print("=" * 104)
    if ok:
        print(f"  el contraste es {dentro[0]!r} (relacion adentro) contra {afuera[0]!r} (afuera).")


if __name__ == "__main__":
    main()
