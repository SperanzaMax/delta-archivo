"""Fine-tune de mamba-370m · ¿la DIVERSIDAD DE FORMAS compra abstencion en un modelo REAL? · 2-sep

Es el escalon que le faltaba al proyecto entero: todo estaba medido en un micro-LM de 3,5 MB.

    condicion `una`  se entrena SOLO con la forma directa, donde la relacion cae AFUERA (d=3 > 2)
    condicion `dos`  se entrena con directa + invertida, y en la invertida la relacion cae ADENTRO

Las dos se EVALUAN siempre en la forma DIRECTA, que es donde la relacion es invisible para la query.
La prediccion, medida en el micro-LM el 2-sep, es que `dos` levanta `nose_rel` en la forma directa
sin tocar un solo parametro de la arquitectura.

La perdida va SOLO sobre el ultimo token. El baseline en el paso 0 se mide y se informa, porque un
modelo preentrenado ya sabe decir «unknown» y eso no nos lo podemos atribuir.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
import tarea_real as T


def evaluar(modelo, tok, rng, n=8, B=16, forma="directa", p_nose=0.4, largo=64):
    modelo.eval()
    id_abst = tok(" " + T.ABST).input_ids[0]
    P, G, TI = [], [], []
    with torch.no_grad():
        for _ in range(n):
            ids, lab, tipos, _ = T.lote(rng, tok, B, (forma,), p_nose=p_nose, largo=largo)
            ids = ids.to(modelo.device)
            lg = modelo(ids[:, :-1]).logits[:, -1, :]
            P.append(lg.argmax(-1).cpu().numpy())
            G.append(ids[:, -1].cpu().numpy())
            TI.append(tipos)
    modelo.train()
    return T.metricas(np.concatenate(P), np.concatenate(G), np.concatenate(TI), id_abst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", default="state-spaces/mamba-370m-hf")
    ap.add_argument("--condicion", choices=("una", "dos", "ciega"), required=True,
                    help="una = solo directa · dos = directa+invertida · "
                         "ciega = directa+lejana, el control donde la relacion nunca entra")
    ap.add_argument("--semilla", type=int, default=0)
    ap.add_argument("--pasos", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--acum", type=int, default=2,
                    help="pasos de acumulacion; el batch EFECTIVO es batch*acum")
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--largo", type=int, default=64,
                    help="el ejemplo mas largo del generador son ~36 tokens, asi que 64 sobra. "
                         "128 causaba OOM en T4 (medido, no estimado).")
    ap.add_argument("--cada", type=int, default=250)
    ap.add_argument("--p-nose", type=float, default=0.4)
    ap.add_argument("--salida", default="salida.json")
    a = ap.parse_args()

    FORMAS = {"una": ("directa",), "dos": ("directa", "invertida"),
              "ciega": ("directa", "lejana")}[a.condicion]

    from transformers import AutoTokenizer, AutoModelForCausalLM
    torch.manual_seed(a.semilla)
    tok = AutoTokenizer.from_pretrained(a.modelo)
    modelo = AutoModelForCausalLM.from_pretrained(a.modelo, dtype=torch.float32)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    modelo.to(dev).train()
    # OOM medido en T4 con batch 8 y largo 128: Mamba guarda estados intermedios de las 48 capas y
    # las activaciones pesan mucho mas de lo que sugiere el conteo de parametros. El checkpointing
    # las recalcula en el backward, cuesta ~30 % de tiempo y es lo que hace entrar el modelo.
    if dev == "cuda":
        modelo.gradient_checkpointing_enable()
        modelo.config.use_cache = False
        print("gradient checkpointing ACTIVADO", flush=True)

    conv = modelo.backbone.layers[0].mixer.conv1d
    with torch.no_grad():
        vivos = [t for t in range(conv.kernel_size[0])
                 if float(conv.weight[:, 0, t].abs().max()) > 0]
    alcance = conv.kernel_size[0] - 1 - min(vivos)
    print(f"modelo {a.modelo} · {sum(p.numel() for p in modelo.parameters()):,} params · {dev}",
          flush=True)
    print(f"conv kernel {conv.kernel_size[0]} · taps vivos {vivos} · ALCANCE REAL {alcance}",
          flush=True)
    print(f"condicion {a.condicion} · formas de entrenamiento {FORMAS} · se EVALUA en `directa`",
          flush=True)

    opt = torch.optim.AdamW(modelo.parameters(), lr=a.lr, weight_decay=0.01)
    rng = np.random.default_rng(1000 + a.semilla)
    rng_ev = lambda: np.random.default_rng(90000 + a.semilla)

    hist = []
    base = evaluar(modelo, tok, rng_ev(), largo=a.largo, p_nose=a.p_nose)
    base["paso"] = 0
    hist.append(base)
    print(f"  BASELINE paso 0 · vigente {base['vigente']:.4f} · nose {base['nose']:.4f} "
          f"(ent {base['nose_ent']:.4f}/rel {base['nose_rel']:.4f}) · "
          f"falsa {base['falsa_abst']:.4f}", flush=True)

    t0 = time.time()
    for paso in range(1, a.pasos + 1):
        for _ in range(a.acum):
            ids, lab, _, _ = T.lote(rng, tok, a.batch, FORMAS, p_nose=a.p_nose, largo=a.largo)
            ids, lab = ids.to(dev), lab.to(dev)
            out = modelo(ids, labels=lab)
            (out.loss / a.acum).backward()
        torch.nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
        opt.step(); opt.zero_grad(set_to_none=True)

        if paso % 50 == 0:
            print(f"  paso {paso:5d}  loss {float(out.loss):.4f}  ({time.time()-t0:.0f}s)",
                  flush=True)
        if paso % a.cada == 0 or paso == a.pasos:
            m = evaluar(modelo, tok, rng_ev(), largo=a.largo, p_nose=a.p_nose)
            m["paso"] = paso
            hist.append(m)
            print(f"  ── eval {paso}: vigente {m['vigente']:.4f} · nose {m['nose']:.4f} "
                  f"(ent {m['nose_ent']:.4f}/rel {m['nose_rel']:.4f}) · "
                  f"falsa {m['falsa_abst']:.4f}", flush=True)
            json.dump({"config": vars(a), "alcance_real": alcance, "historia": hist},
                      open(a.salida, "w"), indent=1)
    print("listo", flush=True)


if __name__ == "__main__":
    main()
