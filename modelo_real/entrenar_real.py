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


def evaluar(modelo, tok, rng, n=32, B=16, forma="directa", p_nose=0.4, largo=64, n_hechos=4):
    """n*B ejemplos. Con n=8 quedaban ~40 de `nose_rel` y el error tipico de la proporcion era
    0,063: la diferencia de 0,15 que abrio la compuerta del 3-sep daba 2,4 sigma, demasiado poco
    para leer tres semillas. Con n=32 son ~160 y el error baja a ~0,03. La evaluacion va por el
    camino lento de HF (el pscan solo entra en training) pero cuesta una fraccion del paso."""
    modelo.eval()
    id_abst = tok(" " + T.ABST).input_ids[0]
    P, G, TI = [], [], []
    with torch.no_grad():
        for _ in range(n):
            ids, lab, tipos, _ = T.lote(rng, tok, B, (forma,), n_hechos=n_hechos,
                                        p_nose=p_nose, largo=largo)
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
    ap.add_argument("--condicion",
                    choices=("una", "dos", "ciega", "cerca", "lejos", "lejos_dos", "lejos_relleno",
                             "muylejos", "muylejos_dos", "muylejos_relleno"),
                    required=True,
                    help="LAS DE ARRIBA quedan del 2-sep y su geometria estaba mal contada: "
                         "una = solo directa · dos = directa+invertida · ciega = directa+lejana. "
                         "LAS DE ABAJO son las del 3-sep, contadas sobre la distancia relacion<->"
                         "entidad, que es la que decide en un modelo recurrente: "
                         "cerca = d2 (comparten ventana) · lejos = d5 (0,0 exacto en la capa 0) · "
                         "lejos_dos = d5+d2, la diversidad · lejos_relleno = d5+d5b, el control "
                         "que adjudica, con diversidad y SIN que la relacion entre nunca")
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
    ap.add_argument("--n-eval", type=int, default=32,
                    help="lotes de 16 en cada evaluacion; 32 da ~160 ejemplos de nose_rel")
    ap.add_argument("--n-hechos", type=int, default=4,
                    help="hechos en el contexto. MEDIDO el 2-sep: con 4 la tarea SATURA en "
                         "mamba-130m, las dos condiciones dan nose_rel 1,0000 y no queda margen "
                         "para medir nada. Es efecto techo, no ausencia de efecto.")
    ap.add_argument("--sin-pscan", action="store_true",
                    help="fuerza el camino secuencial de HF. Solo para control de equivalencia.")
    ap.add_argument("--salida", default="salida.json")
    a = ap.parse_args()

    FORMAS = {"una": ("directa",), "dos": ("directa", "invertida"),
              "ciega": ("directa", "lejana"),
              "cerca": ("d2",), "lejos": ("d5",),
              "lejos_dos": ("d5", "d2"), "lejos_relleno": ("d5", "d5b"),
              # d=9, donde la atenuacion medida en la capa 1 es ~2x la de d=5. Para el caso de que
              # con d=5 las 24 capas alcancen a pagar el impuesto y las dos condiciones saturen.
              "muylejos": ("d9",), "muylejos_dos": ("d9", "d2"),
              "muylejos_relleno": ("d9", "d9b")}[a.condicion]
    # Cada condicion se EVALUA en la forma mas lejana con la que entrena, que es donde la relacion
    # no entra en la ventana de la capa 0.
    F_EVAL = {"una": "directa", "dos": "directa", "ciega": "directa", "cerca": "d2"}.get(
        a.condicion, "d9" if a.condicion.startswith("muylejos") else "d5")

    from transformers import AutoTokenizer, AutoModelForCausalLM
    torch.manual_seed(a.semilla)
    tok = AutoTokenizer.from_pretrained(a.modelo)
    modelo = AutoModelForCausalLM.from_pretrained(a.modelo, dtype=torch.float32)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    modelo.to(dev).train()

    # 2026-09-03. Sin los kernels CUDA de `mamba-ssm` HF recorre la secuencia token por token en
    # Python y el paso costaba 279 s en la PC y 9,7 s en T4. El propio mensaje de error nombraba la
    # salida: `use_mambapy is set to False`. `mambapy` es un scan asociativo en PyTorch PURO, se
    # instala con pip sin compilar nada y se verifico EQUIVALENTE en `bench_mambapy.py` (logits
    # 3e-6 relativo, gradientes 9e-6, o sea ruido de fp32). En CPU acelera 9,6x.
    # Ojo con modeling_mamba.py:418: el pscan solo entra si `use_mambapy and self.training and
    # cache_params is None`, asi que la EVALUACION sigue yendo por el camino lento.
    if not a.sin_pscan:
        from transformers.utils.import_utils import is_mambapy_available
        if is_mambapy_available():
            modelo.config.use_mambapy = True
            for capa in modelo.backbone.layers:
                capa.mixer.use_mambapy = True
            print("scan PARALELO (mambapy) ACTIVADO", flush=True)
        else:
            print("*** mambapy NO instalado: se cae al scan secuencial, ~10x mas lento", flush=True)
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
    print(f"condicion {a.condicion} · formas de entrenamiento {FORMAS} · se EVALUA en `{F_EVAL}`",
          flush=True)

    opt = torch.optim.AdamW(modelo.parameters(), lr=a.lr, weight_decay=0.01)
    rng = np.random.default_rng(1000 + a.semilla)
    rng_ev = lambda: np.random.default_rng(90000 + a.semilla)

    hist = []
    base = evaluar(modelo, tok, rng_ev(), n=a.n_eval, largo=a.largo, p_nose=a.p_nose,
                   n_hechos=a.n_hechos, forma=F_EVAL)
    base["paso"] = 0
    hist.append(base)
    print(f"  BASELINE paso 0 · vigente {base['vigente']:.4f} · nose {base['nose']:.4f} "
          f"(ent {base['nose_ent']:.4f}/rel {base['nose_rel']:.4f}) · "
          f"falsa {base['falsa_abst']:.4f}", flush=True)

    t0 = time.time()
    for paso in range(1, a.pasos + 1):
        for _ in range(a.acum):
            ids, lab, _, _ = T.lote(rng, tok, a.batch, FORMAS, n_hechos=a.n_hechos,
                                    p_nose=a.p_nose, largo=a.largo)
            ids, lab = ids.to(dev), lab.to(dev)
            out = modelo(ids, labels=lab)
            (out.loss / a.acum).backward()
        torch.nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
        opt.step(); opt.zero_grad(set_to_none=True)

        if paso % 50 == 0:
            print(f"  paso {paso:5d}  loss {float(out.loss):.4f}  ({time.time()-t0:.0f}s)",
                  flush=True)
        if paso % a.cada == 0 or paso == a.pasos:
            m = evaluar(modelo, tok, rng_ev(), n=a.n_eval, largo=a.largo, p_nose=a.p_nose,
                        n_hechos=a.n_hechos, forma=F_EVAL)
            m["paso"] = paso
            hist.append(m)
            print(f"  ── eval {paso}: vigente {m['vigente']:.4f} · nose {m['nose']:.4f} "
                  f"(ent {m['nose_ent']:.4f}/rel {m['nose_rel']:.4f}) · "
                  f"falsa {m['falsa_abst']:.4f}", flush=True)
            json.dump({"config": vars(a), "alcance_real": alcance, "historia": hist},
                      open(a.salida, "w"), indent=1)

    # 2026-09-03. Lo mecanicista se venia midiendo en el modelo PREENTRENADO y lo conductual en el
    # AJUSTADO, o sea sobre dos objetos distintos, y eso deja el puente entre las dos mitades sin
    # cerrar. Aca se mide lo mismo que `escalera_v2.py` sobre ESTE modelo, al final: cuanto se mueve
    # `conv1d` en la posicion de la entidad al cambiar el token de la relacion. El fine-tune puede
    # haber movido los pesos de la conv, y si el alcance cambio hay que saberlo.
    # Va envuelto: el json con las metricas ya se escribio en el ultimo eval, y una medicion de
    # yapa no puede tumbar una unidad que costo su tiempo de GPU.
    try:
        sens = sensibilidad(modelo, tok, F_EVAL, dev)
        print(f"  SENSIBILIDAD final en `{F_EVAL}` · alcance {sens['alcance']} · "
              f"conv@ent capa 0 {sens['conv_ent'][0]:.4e} · capa 1 {sens['conv_ent'][1]:.4e} · "
              f"capa 12 {sens['conv_ent'][12]:.4e}", flush=True)
        json.dump({"config": vars(a), "alcance_real": alcance, "historia": hist,
                   "sensibilidad_final": sens}, open(a.salida, "w"), indent=1)
    except Exception as e:
        print(f"  (la sensibilidad final fallo: {type(e).__name__}: {e}) — las metricas ya estan",
              flush=True)
    print("listo", flush=True)


def sensibilidad(modelo, tok, forma, dev, n_textos=4):
    """max|dif| de la salida de `conv1d` en la posicion de la ENTIDAD al cambiar la RELACION.

    Es la medicion de `sonda_combinacion.py` corrida sobre el modelo ya ajustado. Devuelve tambien
    el alcance MEDIDO de la conv despues del entrenamiento: el tap mas viejo valia cero exacto en el
    preentrenado y no hay garantia de que siga valiendo cero.
    """
    modelo.eval()
    capas = modelo.backbone.layers
    plantilla = T.PLANTILLAS[forma]
    rng = np.random.default_rng(12345)
    acum = []
    for _ in range(n_textos):
        es = list(rng.choice(T.ENTIDADES, size=5, replace=False)); ent = es.pop()
        rs = list(rng.choice(T.RELACIONES, size=6, replace=False))
        vs = list(rng.choice(T.VALORES, size=4, replace=False))
        ctx = " ".join(f"The {rs[2+i]} of {es[i]} is {vs[i]}." for i in range(4))
        pares = []
        for r in (rs[0], rs[1]):
            ids = tok(f"{ctx} {plantilla.format(r=r, e=ent)}", return_tensors="pt").input_ids
            g, hs = {}, []
            for i, capa in enumerate(capas):
                hs.append(capa.mixer.conv1d.register_forward_hook(
                    (lambda i: lambda _m, _i, o:
                        g.__setitem__(i, o.detach()[0].transpose(0, 1).float().cpu()))(i)))
            with torch.no_grad():
                modelo(ids.to(dev))
            for h in hs:
                h.remove()
            pares.append((ids[0], g))
        (i1, g1), (i2, g2) = pares
        dif = [k for k in range(len(i1)) if int(i1[k]) != int(i2[k])]
        if len(dif) != 1:
            continue
        p_e = [k for k in range(len(i1)) if int(i1[k]) == tok(" " + ent).input_ids[0]][-1]
        acum.append([float((g1[i][p_e] - g2[i][p_e]).abs().max()) for i in range(len(capas))])
    modelo.train()
    conv = capas[0].mixer.conv1d
    with torch.no_grad():
        vivos = [t for t in range(conv.kernel_size[0])
                 if float(conv.weight[:, 0, t].abs().max()) > 0]
    return {"conv_ent": np.array(acum).mean(0).tolist() if acum else [],
            "n_textos": len(acum), "taps_vivos": vivos,
            "alcance": conv.kernel_size[0] - 1 - min(vivos)}


if __name__ == "__main__":
    main()
