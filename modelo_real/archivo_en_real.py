"""¿El archivo co-entrenado funciona sobre un transformer REAL preentrenado? · 2026-09-06

Todo lo que la linea sabe del archivo persistente esta medido sobre el micro-LM: 3,5 MB, idioma
cerrado de 242 tokens, entrenado desde cero. La pregunta que faltaba es si el mecanismo es una
propiedad del mecanismo o del banco.

Esto lo monta sobre TinyLlama-1.1B **congelado**: 22 capas, d=2048, vocabulario natural de 32.000
tokens, preentrenado por otros. Se entrenan SOLO las proyecciones del archivo.

## El diseño, que es el de E-I0/E-I2 del brazo interno

  S1  se DICE un hecho            ("Ana vive en Cordoba")   -> se archiva un vector
  S2  se PREGUNTA por el hecho    ("Ana vive en")           -> forward INDEPENDIENTE

Los dos forwards no comparten estado: el hecho no esta en el contexto de la consulta. Sin archivo,
el modelo no puede saber la respuesta y el piso es lo que el preentrenamiento adivine — que es
justamente el control que hace interpretable cualquier subida.

## Que se entrena, y que no

Congelado: TinyLlama entero (1,10 B params).
Entrenable: `kw`, `qr`, `vw` (2048 -> R), `wo` (R -> 2048) y `ord` (T x R). Con R=256 son ~2,1 M,
el 0,19 % del modelo. La lectura se inyecta en el residual de una capa TEMPRANA por hook, que es
lo que E-I1 midio como 4x mas rapido y con techo mas alto.

`wo` arranca en CERO: al paso 0 la lectura aporta exactamente el vector nulo al residual, asi que
el modelo empieza siendo TinyLlama sin tocar y todo lo que aparezca lo fue a buscar el gradiente.
Es el mismo criterio que `convq` en [1,0,0] y `v_nulo` en cero en el micro-LM.
"""
import argparse, time

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

MID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Hechos: entidad -> valor. Vocabulario natural, no un idioma inventado.
ENTIDADES = ["Ana", "Beto", "Carla", "Dario", "Elena", "Franco", "Gabi", "Hugo"]
VALORES = ["Cordoba", "Salta", "Rosario", "Mendoza", "Neuquen", "Ushuaia", "Parana", "Tucuman"]


class Archivo(nn.Module):
    """Las piezas nuevas. Lo unico que tiene gradiente."""

    def __init__(self, d, r=256, n_turnos=64):
        super().__init__()
        self.kw = nn.Linear(d, r, bias=False)
        self.qr = nn.Linear(d, r, bias=False)
        self.vw = nn.Linear(d, r, bias=False)
        self.wo = nn.Linear(r, d, bias=False)
        self.ord = nn.Parameter(torch.zeros(n_turnos, r))
        nn.init.normal_(self.ord, std=0.02)
        # CERO: al paso 0 la lectura aporta el vector nulo y el modelo es TinyLlama intacto.
        nn.init.zeros_(self.wo.weight)
        self.r = r

    def leer(self, h, archivo, turnos):
        """h: (B, T, d) estado de la consulta · archivo: (B, N, d) · turnos: (B, N)"""
        ak = self.kw(archivo) + self.ord[turnos]           # (B, N, r)
        av = self.vw(archivo)                              # (B, N, r)
        q = self.qr(h)                                     # (B, T, r)
        sim = torch.einsum("btr,bnr->btn", q, ak) / (self.r ** 0.5)
        p = torch.softmax(sim, -1)
        return self.wo(torch.einsum("btn,bnr->btr", p, av)), p


def escribir(modelo, tok, frases, capa, estado=None):
    """Un vector por frase, tomado en su ULTIMO token. Es la politica del micro-LM.

    BUG del 6-sep: el hook de lectura seguia armado durante la escritura, asi que desde el paso 2
    los vectores archivados se calculaban CON la lectura del archivo del paso anterior. Se apaga
    explicitamente: escribir tiene que ser el modelo limpio.
    """
    ids = tok(frases, return_tensors="pt", padding=True)
    guardado = estado.get("archivo") if estado is not None else None
    if estado is not None:
        estado["archivo"] = None
    with torch.no_grad():
        o = modelo(**ids, output_hidden_states=True)
    if estado is not None:
        estado["archivo"] = guardado
    h = o.hidden_states[capa]                              # (B, T, d)
    ultimo = ids["attention_mask"].sum(1) - 1              # (B,)
    return h[torch.arange(len(frases)), ultimo]            # (B, d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pasos", type=int, default=60)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--capa-escritura", type=int, default=11)
    ap.add_argument("--capa-lectura", type=int, default=2, help="inyeccion TEMPRANA (E-I1)")
    ap.add_argument("--rango", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--semilla", type=int, default=0)
    ap.add_argument("--sin-archivo", action="store_true",
                    help="CONTROL 1 (el piso): sin lectura no hay NADA con gradiente —el modelo esta "
                         "congelado—, asi que no se puede entrenar: se EVALUA. Mide lo que el "
                         "preentrenamiento adivina sin haber visto el hecho.")
    ap.add_argument("--barajar", type=int, default=0,
                    help="CONTROL 2, el que decide: cada N pasos evalua con el archivo BARAJADO "
                         "(mismos vectores, asignados a la pregunta equivocada). Si el acierto se "
                         "sostiene barajando, el modelo no esta leyendo el archivo sino adivinando.")
    a = ap.parse_args()
    torch.manual_seed(a.semilla)

    tok = AutoTokenizer.from_pretrained(MID)
    tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    modelo = AutoModelForCausalLM.from_pretrained(MID, dtype=torch.float32)
    modelo.eval()
    for p in modelo.parameters():
        p.requires_grad_(False)
    d = modelo.config.hidden_size

    arch = Archivo(d, a.rango)
    n_entr = sum(p.numel() for p in arch.parameters())
    n_cong = sum(p.numel() for p in modelo.parameters())
    print(f"{MID} · {n_cong/1e9:.2f} B CONGELADOS · piezas nuevas {n_entr/1e6:.2f} M "
          f"({100*n_entr/n_cong:.3f} %) · rango {a.rango}")
    print(f"escritura en la capa {a.capa_escritura} · lectura inyectada en la {a.capa_lectura}")
    assert not any(p.requires_grad for p in modelo.parameters()), "el modelo NO esta congelado"

    opt = torch.optim.AdamW(arch.parameters(), lr=a.lr)
    capa = modelo.model.layers[a.capa_lectura]
    estado = {}

    def hook(mod, args, kwargs, salida):
        if estado.get("archivo") is None:
            return salida
        h = salida[0] if isinstance(salida, tuple) else salida
        lect, masa = arch.leer(h, estado["archivo"], estado["turnos"])
        estado["masa"] = masa
        h = h + lect
        return (h,) + salida[1:] if isinstance(salida, tuple) else h

    capa.register_forward_hook(hook, with_kwargs=True)
    g = torch.Generator().manual_seed(a.semilla)
    t0 = time.time()

    for paso in range(1, a.pasos + 1):
        # S1: se dicen los hechos que se van a preguntar, MAS distractores de otras entidades.
        #
        # TERCER intento de este control (6-sep). Los dos primeros fallaron por la misma razon en dos
        # formas distintas: mientras el archivo de CADA muestra contenga TODOS los hechos, barajar no
        # saca nada —la respuesta sigue adentro, sólo en otra posicion— y el control da 1,0000 sin
        # poder fallar. Ahora el archivo de la muestra i es [su hecho] + distractores de entidades
        # que NADIE pregunta, asi que darle el archivo de otra muestra SI le saca su respuesta.
        perm = torch.randperm(len(ENTIDADES), generator=g)
        idx = perm[:a.batch]                       # entidades que se preguntan
        dis = perm[a.batch:]                       # entidades distractoras, nunca preguntadas
        vals = torch.randint(0, len(VALORES), (len(perm),), generator=g)
        frases_q = [f"{ENTIDADES[i]} vive en {VALORES[vals[k]]}." for k, i in enumerate(idx)]
        frases_d = [f"{ENTIDADES[i]} vive en {VALORES[vals[a.batch + k]]}."
                    for k, i in enumerate(dis)]
        vq = escribir(modelo, tok, frases_q, a.capa_escritura, estado)          # (B, d)
        vd = escribir(modelo, tok, frases_d, a.capa_escritura, estado) if frases_d else None
        # archivo[i] = [hecho de i] + distractores comunes
        partes = [vq.unsqueeze(1)] + ([vd.unsqueeze(0).expand(a.batch, -1, -1)] if vd is not None else [])
        archivo = torch.cat(partes, dim=1)                              # (B, 1+D, d)
        turnos = torch.arange(archivo.shape[1]).unsqueeze(0).expand(a.batch, -1)

        # S2: se pregunta por cada una, en un forward INDEPENDIENTE.
        preg = [f"{ENTIDADES[i]} vive en" for i in idx]
        ids = tok(preg, return_tensors="pt", padding=True)
        # BUG del 6-sep, y el que invalidaba todo el experimento: `tok(" Cordoba")[0]` en Llama
        # devuelve **29871, el token de espacio**, igual para los ocho valores. El objetivo era
        # «predecir un espacio», que es trivial, se acierta 1,0000 y —lo importante— NO depende del
        # archivo: por eso el control barajado no podia fallar por mas que se rediseñara el banco.
        # Tres rediseños del control persiguiendo un defecto que estaba en el tokenizador.
        # Se toma el PRIMER token de la palabra (indice 1), que son 8 distintos para 8 valores.
        objetivo = torch.tensor([tok(f" {VALORES[vals[k]]}", add_special_tokens=False)["input_ids"][1]
                                 for k in range(a.batch)])
        estado["archivo"] = None if a.sin_archivo else archivo
        estado["turnos"] = turnos
        salida = modelo(**ids)
        ultimo = ids["attention_mask"].sum(1) - 1
        logits = salida.logits[torch.arange(len(preg)), ultimo]         # (B, V)
        perdida = nn.functional.cross_entropy(logits, objetivo)

        if not a.sin_archivo:
            opt.zero_grad()
            perdida.backward()
            opt.step()

        if paso == 1 or paso % 10 == 0 or paso == a.pasos:
            acierto = (logits.argmax(-1) == objetivo).float().mean().item()
            extra = ""
            if a.barajar and not a.sin_archivo:
                # Mismos vectores, rotados una posicion: cada pregunta recibe el archivo de otra.
                with torch.no_grad():
                    # cada muestra recibe el archivo de la SIGUIENTE: su hecho ya no esta primero
                    estado["archivo"] = archivo.roll(1, dims=0)
                    lg2 = modelo(**ids).logits[torch.arange(len(preg)), ultimo]
                    ac2 = (lg2.argmax(-1) == objetivo).float().mean().item()
                    estado["archivo"] = archivo
                extra = f" · BARAJADO {ac2:.4f}"
            print(f"  paso {paso:4d} · perdida {perdida.item():7.4f} · acierto {acierto:.4f}"
                  f"{extra} · {(time.time()-t0)/paso:.1f} s/paso")

    # DIAGNOSTICO (6-sep): el control barajado no discriminaba en tres diseños seguidos. Antes de
    # volver a tocar el diseño hay que MIRAR: donde cae la masa de lectura, y que pasa si se apaga
    # la lectura sobre el modelo ya entrenado. Si apagarla no cambia el acierto, la respuesta nunca
    # vino del archivo y toda la discusion del barajado era sobre un canal que no se usa.
    with torch.no_grad():
        estado["archivo"] = archivo
        lg_con = modelo(**ids).logits[torch.arange(len(preg)), ultimo]
        m = estado.get("masa")
        estado["archivo"] = None
        lg_sin = modelo(**ids).logits[torch.arange(len(preg)), ultimo]
    ac_con = (lg_con.argmax(-1) == objetivo).float().mean().item()
    ac_sin = (lg_sin.argmax(-1) == objetivo).float().mean().item()
    print(f"\n  DIAGNOSTICO sobre el modelo ya entrenado:")
    print(f"    con lectura        acierto {ac_con:.4f}")
    print(f"    lectura APAGADA    acierto {ac_sin:.4f}   <- si es igual, el archivo no se usa")
    # ¿el barajado cambia realmente el tensor, y cambian las predicciones?
    with torch.no_grad():
        baraj = archivo.roll(1, dims=0)
        dif = (archivo - baraj).abs().max().item()
        estado["archivo"] = baraj
        lg_b = modelo(**ids).logits[torch.arange(len(preg)), ultimo]
        estado["archivo"] = archivo
    ac_b = (lg_b.argmax(-1) == objetivo).float().mean().item()
    iguales = (lg_b.argmax(-1) == lg_con.argmax(-1)).float().mean().item()
    print(f"    barajado           acierto {ac_b:.4f}")
    print(f"    max|archivo - barajado| = {dif:.4f}  (0 = el barajado NO cambia nada)")
    print(f"    predicciones identicas con y sin barajar: {iguales:.4f}")
    print(f"    objetivos: {objetivo.tolist()}")
    print(f"    pred con: {lg_con.argmax(-1).tolist()} · pred barajado: {lg_b.argmax(-1).tolist()}")
    if m is not None:
        mm = m[torch.arange(len(preg)), ultimo]          # (B, N) en el ultimo token
        print(f"    masa en la entrada 0 (la propia): {mm[:, 0].mean().item():.4f}")
        print(f"    masa maxima por muestra:          {mm.max(-1).values.mean().item():.4f}")
        print(f"    entrada mas leida por muestra:    {mm.argmax(-1).tolist()}  (0 = la propia)")
    print(f"\nlisto en {(time.time()-t0)/60:.1f} min"
          f"{'  (CONTROL sin archivo)' if a.sin_archivo else ''}")


if __name__ == "__main__":
    main()
