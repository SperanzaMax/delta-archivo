"""El banco de TinyLlama, escalado · 2026-09-09 · ejecuta `PLAN_ESCALADO_BANCO.md`

Que cambia respecto de `archivo_en_real.py`, en el orden de valor que fijo el plan.

  1. VERSIONES     un hecho se dice, despues se CORRIGE, y se pregunta. La respuesta correcta es
                   la ultima. Es la tarea canonica del programa y nunca se probo sobre un modelo
                   real. En el micro-LM el sello de orden la lleva de 0,4570 a 0,9956.
  2. ABSTENCION    se pregunta por algo que no esta. Sin esto no hay `nose` ni `invento`, o sea la
                   mitad de la vara del proyecto no se puede medir. Y aca el vocabulario es
                   ABIERTO, asi que el invento tiene casos reales por primera vez.
  3. ESCALA        8 entidades -> 60 · 8 valores -> 100 · batch 4 -> 32.
  4. 2a RELACION   `lives in` y `works in`. Con una sola no hay consulta compuesta entidad x
                   relacion, que es donde vive la ley de la ventana.

Lo que NO se toca, porque es el resultado del 6-sep: modelo congelado, escritura en la capa 11,
lectura inyectada en la capa 2, y los dos controles que dieron cero exacto.

## Dos cosas que este banco arregla y no estaban en el plan

`turnos` era `arange(N)`, o sea el sello de orden estaba CONFUNDIDO con la posicion en el archivo:
no se podia saber si leia el sello o contaba lugares. Aca las entradas se BARAJAN de posicion y el
turno viaja con el hecho.

Y el objetivo pasa a ser una palabra de UN token (ver `pool_tinyllama.py`). Con transformers 5.3.0
el indice 1 de `tok(" Cordoba")` ya no es '▁Cord' sino 'oba', asi que el objetivo del banco viejo
depende de la version de la libreria. Aca no.

## Riesgo declarado antes de correr (del plan)

TinyLlama ya sabe cosas del mundo. El control `--vacio` es el que lo separa: mismo camino de
lectura, archivo en CERO. Si el acierto se sostiene, la respuesta no venia del archivo.
"""
import argparse, json, time

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

MID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
RELACIONES = ["lives in", "works in"]

# 2026-09-09: medido, las 161 piezas del banco son de UN token en los tres tokenizadores
# (TinyLlama, Qwen3-0.6B y Qwen3.5-0.8B), asi que el MISMO banco corre en los tres sin tocar nada
# y la comparacion de vehiculos sale casi gratis. Igual se verifica pieza por pieza en `objtok`.
VEHICULOS = {
    "tiny": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",   # denso, 22 capas, atencion completa
    "q3":   "Qwen/Qwen3-0.6B",                      # denso, 28 capas, atencion completa
    "q35":  "Qwen/Qwen3.5-0.8B",                    # HIBRIDO, 24 capas, 18 lineales + 6 completas
}


class Archivo(nn.Module):
    """Las piezas nuevas. Lo unico que tiene gradiente. Identico al del 6-sep."""

    def __init__(self, d, r=256, n_turnos=64):
        super().__init__()
        self.kw = nn.Linear(d, r, bias=False)
        self.qr = nn.Linear(d, r, bias=False)
        self.vw = nn.Linear(d, r, bias=False)
        self.wo = nn.Linear(r, d, bias=False)
        self.ord = nn.Parameter(torch.zeros(n_turnos, r))
        nn.init.normal_(self.ord, std=0.02)
        nn.init.zeros_(self.wo.weight)   # al paso 0 la lectura aporta el vector nulo
        self.r = r

    def leer(self, h, archivo, turnos):
        ak = self.kw(archivo) + self.ord[turnos]
        av = self.vw(archivo)
        q = self.qr(h)
        sim = torch.einsum("btr,bnr->btn", q, ak) / (self.r ** 0.5)
        p = torch.softmax(sim, -1)
        return self.wo(torch.einsum("btn,bnr->btr", p, av)), p


def escribir(modelo, tok, frases, capa, estado):
    """Un vector por frase, en su ULTIMO token, con la lectura APAGADA (bug del 6-sep)."""
    ids = tok(frases, return_tensors="pt", padding=True).to(next(modelo.parameters()).device)
    guardado = estado.get("archivo")
    estado["archivo"] = None
    with torch.no_grad():
        o = modelo(**ids, output_hidden_states=True)
    estado["archivo"] = guardado
    h = o.hidden_states[capa]
    ultimo = ids["attention_mask"].sum(1) - 1
    return h[torch.arange(len(frases), device=h.device), ultimo]


def lote(a, pool, g, calentando=False, verdad=None):
    """Arma un paso. Devuelve las frases a escribir, los indices por muestra, turnos y objetivos.

    Cada muestra i es una de cuatro clases, y las cuatro conviven en el mismo lote:

      una       la entidad tiene UNA version de la relacion preguntada     -> ese valor
      dos       tiene DOS, dicha y corregida                               -> la ULTIMA
      nose_rel  esta en el archivo pero con la OTRA relacion               -> abstencion
      nose_aus  no esta en el archivo                                      -> abstencion

    `nose_rel` es la dura y es la que mide consulta compuesta: la entidad SI esta, lo que no esta
    es el par entidad x relacion. Contestar ahi con el valor de la otra relacion es exactamente
    ignorar la relacion, o sea el fallo que la ley de la ventana predice.
    """
    ENT, VAL = pool["entidades"], pool["valores"]
    perm = torch.randperm(len(ENT), generator=g)
    preguntadas = perm[:a.batch].tolist()
    distractoras = perm[a.batch:].tolist()

    frases, turnos_f = [], []          # las frases del paso, y el turno de cada una
    def sortear_val():
        return int(torch.randint(0, len(VAL), (1,), generator=g))

    def sortear_val_para(ent):
        """Excluye la respuesta VERDADERA de esa entidad, para que el archivo contradiga siempre."""
        if not verdad or ent not in verdad:
            return sortear_val()
        for _ in range(50):
            v = sortear_val()
            if VAL[v] != verdad[ent]:
                return v
        return sortear_val()

    def agregar(ent, rel, val, turno):
        frases.append(f"{ENT[ent]} {RELACIONES[rel]} {VAL[val]}.")
        turnos_f.append(turno)
        return len(frases) - 1

    # Distractores: entidades que NADIE pregunta. Son comunes a todas las muestras, asi que rotar
    # el archivo le saca a cada muestra SU hecho y le deja los ajenos (la propiedad del 6-sep).
    dis_idx = []
    for k, e in enumerate(distractoras[:a.distractores]):
        r = int(torch.randint(0, len(RELACIONES), (1,), generator=g))
        v = int(torch.randint(0, len(VAL), (1,), generator=g))
        dis_idx.append(agregar(e, r, v, int(torch.randint(0, a.turnos, (1,), generator=g))))

    propios, objetivos, clases, pos_correcta = [], [], [], []
    for e in preguntadas:
        rel = int(torch.randint(0, len(RELACIONES), (1,), generator=g))
        u = float(torch.rand(1, generator=g))
        if calentando:
            # Sin casos de abstencion: `u` se remapea al tramo que tiene respuesta, conservando la
            # proporcion relativa entre `dos` y `una`.
            u *= (a.p_dos + a.p_una)
        mios = []
        if u < a.p_dos:                                    # DOS versiones: dicha y corregida
            clase = "dos"
            v1, v2 = torch.randperm(len(VAL), generator=g)[:2].tolist()
            t1 = int(torch.randint(0, a.turnos - 1, (1,), generator=g))
            t2 = int(torch.randint(t1 + 1, a.turnos, (1,), generator=g))
            i1 = agregar(e, rel, v1, t1)
            i2 = agregar(e, rel, v2, t2)
            mios = [i1, i2]
            obj, corr = VAL[v2], i2
        elif u < a.p_dos + a.p_una:                        # UNA version
            clase = "una"
            v = sortear_val_para(e)
            t = int(torch.randint(0, a.turnos, (1,), generator=g))
            i1 = agregar(e, rel, v, t)
            mios = [i1]
            obj, corr = VAL[v], i1
        elif u < a.p_dos + a.p_una + a.p_nose_rel:         # esta, pero con la OTRA relacion
            clase = "nose_rel"
            otra = 1 - rel
            v = int(torch.randint(0, len(VAL), (1,), generator=g))
            t = int(torch.randint(0, a.turnos, (1,), generator=g))
            i1 = agregar(e, otra, v, t)
            mios = [i1]
            obj, corr = pool["abstencion"], -1
        else:                                              # no esta en el archivo
            clase = "nose_aus"
            obj, corr = pool["abstencion"], -1
        propios.append((e, rel, mios))
        objetivos.append(obj)
        clases.append(clase)
        pos_correcta.append(corr)
    return frases, turnos_f, dis_idx, propios, objetivos, clases, pos_correcta


def armar(propios, dis_idx, turnos_f, n_arch, g):
    """Indices y turnos por muestra, con las entradas BARAJADAS de posicion.

    El turno viaja con el hecho, no con el lugar. Sin esto `ord` queda confundido con la posicion
    y no se puede decir si el modelo lee el sello o cuenta lugares.
    """
    idx, tur, donde = [], [], []
    for (_, _, mios) in propios:
        # Cada muestra tiene 0, 1 o 2 hechos propios, asi que el relleno que pide varia. Para que
        # el tensor salga rectangular TODAS las filas tienen que quedar en `n_arch`, y para eso
        # tiene que haber al menos `n_arch` distractores. Lo garantiza la guarda de `main`.
        relleno = dis_idx[: n_arch - len(mios)]
        assert len(mios) + len(relleno) == n_arch, (
            f"fila de {len(mios)+len(relleno)} en un archivo de {n_arch}: "
            f"faltan distractores ({len(dis_idx)} disponibles)")
        fila = mios + relleno
        orden = torch.randperm(len(fila), generator=g).tolist()
        fila_b = [fila[j] for j in orden]
        idx.append(fila_b)
        tur.append([turnos_f[j] for j in fila_b])
        donde.append({f: p for p, f in enumerate(fila_b)})
    return torch.tensor(idx), torch.tensor(tur), donde


def main():
    global RELACIONES
    ap = argparse.ArgumentParser()
    ap.add_argument("--pasos", type=int, default=600)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--distractores", type=int, default=28)
    ap.add_argument("--arch", type=int, default=24, help="entradas por muestra (propias+relleno)")
    ap.add_argument("--turnos", type=int, default=64)
    ap.add_argument("--modelo", default="tiny",
                    help="tiny | q3 | q35, o un id de HuggingFace")
    ap.add_argument("--capa-escritura", type=int, default=None,
                    help="por defecto la MITAD de la profundidad (11 de 22 en TinyLlama)")
    ap.add_argument("--capa-lectura", type=int, default=2,
                    help="inyeccion TEMPRANA (E-I1). En un modelo hibrido conviene mirar si cae "
                         "antes o despues de la primera capa de atencion completa.")
    ap.add_argument("--rango", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--semilla", type=int, default=0)
    ap.add_argument("--calentar", type=int, default=0,
                    help="pasos iniciales SIN casos de abstencion. El atractor mudo esta medido "
                         "como ABSORBENTE en esta linea, y aca la causa es aritmetica: con 100 "
                         "valores y 40 %% de abstencion el objetivo MODAL es la abstencion "
                         "(0,40 contra 0,006 de cada valor), asi que el mejor predictor CONSTANTE "
                         "es callarse y el modelo llega a ese optimo antes de que `wo` (que "
                         "arranca en cero) haga util al archivo. Calentando sin abstencion la "
                         "unica forma de bajar la perdida es LEER, y recien despues se ensenia a "
                         "callarse. No se puede aprender a decir «no se» antes de saber que es "
                         "saber.")
    ap.add_argument("--p-dos", type=float, default=0.30)
    ap.add_argument("--p-una", type=float, default=0.30)
    ap.add_argument("--p-nose-rel", type=float, default=0.20)
    ap.add_argument("--cada", type=int, default=25)
    ap.add_argument("--pool", default="pool_tinyllama.json")
    # Para la ESCALERA. El plan pedia cuatro cosas y se agregaron las cuatro de una, asi que si el
    # banco no aprende no se sabe cual la rompio. Con esto se sube un escalon por vez desde la
    # configuracion del 6-sep, que es la unica que esta medida funcionando.
    ap.add_argument("--n-ent", type=int, default=None, help="usar solo las primeras N entidades")
    ap.add_argument("--n-val", type=int, default=None, help="usar solo los primeros N valores")
    ap.add_argument("--n-rel", type=int, default=len(RELACIONES), choices=[1, 2])
    # PREREG_ARCHIVO_CONTRA_PREENTRENAMIENTO.md (9b14aa1f). Cambia las entidades inventadas por
    # HITOS REALES sobre los que TinyLlama YA tiene una creencia, para que el archivo tenga que
    # ganarle al preentrenamiento y no solo llenar un hueco vacio.
    ap.add_argument("--mundo", default=None, choices=["real", "inventado"],
                    help="real = 23 hitos con confianza medida de 0,55 a 0,98 · "
                         "inventado = 15 de la misma forma con confianza ~0,20, el control")
    ap.add_argument("--sondas", default="sondas_mundo.json")
    ap.add_argument("--reps", type=int, default=40,
                    help="repeticiones por entidad en la evaluacion final, para que el acierto "
                         "por entidad tenga n suficiente y H2 se pueda medir")
    ap.add_argument("--salida", default=None)
    ap.add_argument("--sin-ord", action="store_true",
                    help="ABLACION: `ord` en cero y sin gradiente. Sin sello de orden las dos "
                         "versiones son indistinguibles y `vigente` deberia caer a ~0,5.")
    a = ap.parse_args()
    torch.manual_seed(a.semilla)

    mid = VEHICULOS.get(a.modelo, a.modelo)
    pool = json.load(open(a.pool))
    verdad, VAL_, CONF_ = None, {}, {}
    if a.mundo:
        sd = json.load(open(a.sondas))
        RELACIONES = ["is in the city of"]
        a.n_rel = 1
        if a.mundo == "real":
            pool["entidades"] = ["The " + e["entidad"] for e in sd["mundo"]]
            # `verdad` mapea el indice de entidad -> su respuesta verdadera, para EXCLUIRLA al
            # sortear. Asi el archivo contradice SIEMPRE, que es lo que el prereg pide medir.
            verdad = {i: e["verdad"] for i, e in enumerate(sd["mundo"])}
            VAL_ = {i: e["verdad"] for i, e in enumerate(sd["mundo"])}
            CONF_ = {i: e["confianza"] for i, e in enumerate(sd["mundo"])}
            conf = [e["confianza"] for e in sd["mundo"]]
            print(f"MUNDO REAL · {len(pool['entidades'])} hitos · confianza previa del modelo "
                  f"{min(conf):.4f} a {max(conf):.4f} (media {sum(conf)/len(conf):.4f})")
        else:
            pool["entidades"] = ["The " + e["entidad"] for e in sd["control"]]
            VAL_ = {}
            CONF_ = {i: e["confianza"] for i, e in enumerate(sd["control"])}
            conf = [e["confianza"] for e in sd["control"]]
            print(f"MUNDO INVENTADO · {len(pool['entidades'])} entidades de control · "
                  f"confianza previa {sum(conf)/len(conf):.4f}")
    if a.n_ent:
        pool["entidades"] = pool["entidades"][:a.n_ent]
    if a.n_val:
        pool["valores"] = pool["valores"][:a.n_val]
    RELACIONES = RELACIONES[:a.n_rel]
    # Dos guardas de forma, antes de cargar 1,1 B de pesos para morir en el paso 1.
    assert a.distractores <= len(pool["entidades"]) - a.batch, (
        f"con batch {a.batch} quedan {len(pool['entidades']) - a.batch} entidades libres para "
        f"distractores y se piden {a.distractores}. Los distractores NO pueden ser entidades "
        f"preguntadas: si lo fueran, rotar el archivo no le sacaria la respuesta a nadie.")
    assert a.arch <= a.distractores, (
        f"el archivo pide {a.arch} entradas y hay {a.distractores} distractores. Una muestra sin "
        f"hechos propios (nose_aus) necesita {a.arch} de relleno.")
    tok = AutoTokenizer.from_pretrained(mid)
    tok.pad_token = tok.pad_token or tok.eos_token
    tok.padding_side = "right"
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    modelo = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.float32).to(dev)
    modelo.eval()
    for p in modelo.parameters():
        p.requires_grad_(False)
    cfg = getattr(modelo.config, "text_config", modelo.config)
    d = cfg.hidden_size
    capas = modelo.model.layers
    if a.capa_escritura is None:
        a.capa_escritura = len(capas) // 2
    assert 0 <= a.capa_lectura < len(capas) and 0 < a.capa_escritura <= len(capas), \
        f"capas fuera de rango para un modelo de {len(capas)} bloques"
    # En un hibrido importa DONDE cae la inyeccion respecto de la atencion completa, asi que se
    # dice en el log en vez de dejarlo implicito.
    # Qwen3 tambien trae `layer_types`, pero con TODAS en full_attention: es denso. Solo es
    # hibrido si hay al menos una capa lineal, si no el log dice "HIBRIDO · 0 lineales" y confunde.
    lt = getattr(cfg, "layer_types", None)
    full = [i for i, x in enumerate(lt) if "full" in x] if lt else list(range(len(capas)))
    if lt and len(full) < len(lt):
        print(f"HIBRIDO · {len(lt)-len(full)} capas lineales de {len(lt)} · atencion completa en "
              f"{full} · conv corta kernel {getattr(cfg,'linear_conv_kernel_dim','?')}")
        print(f"  la lectura entra en la capa {a.capa_lectura}, que es "
              f"{'de ATENCION COMPLETA' if a.capa_lectura in full else 'LINEAL'}; la primera "
              f"completa esta en la {full[0]}")
    else:
        print(f"DENSO · atencion completa en las {len(capas)} capas · sin conv corta")

    arch = Archivo(d, a.rango, a.turnos).to(dev)
    if a.sin_ord:
        nn.init.zeros_(arch.ord)
        arch.ord.requires_grad_(False)
    n_entr = sum(p.numel() for p in arch.parameters() if p.requires_grad)
    n_cong = sum(p.numel() for p in modelo.parameters())
    print(f"{mid} · {n_cong/1e9:.2f} B CONGELADOS · entrenables {n_entr/1e6:.2f} M "
          f"({100*n_entr/n_cong:.3f} %) · rango {a.rango} · {dev}")
    print(f"escritura capa {a.capa_escritura} · lectura capa {a.capa_lectura} · "
          f"{len(pool['entidades'])} entidades · {len(pool['valores'])} valores · "
          f"{len(RELACIONES)} relaciones · batch {a.batch} · archivo {a.arch} · "
          f"{len(capas)} bloques")
    assert not any(p.requires_grad for p in modelo.parameters()), "el modelo NO esta congelado"

    # Objetivos: palabras de UN token, asi que el objetivo es la palabra entera.
    def objtok(pal):
        ids = tok(f" {pal}", add_special_tokens=False)["input_ids"]
        assert len(ids) == 1, f"{pal!r} no es de un token: {ids}"
        return ids[0]
    TOK_VAL = {v: objtok(v) for v in pool["valores"]}
    TOK_ABS = objtok(pool["abstencion"])
    SET_VAL = set(TOK_VAL.values())
    p_nose = 1 - a.p_dos - a.p_una
    print(f"abstencion {pool['abstencion']!r} -> {TOK_ABS} · piso trivial de responder al azar "
          f"1/{len(pool['valores'])} = {1/len(pool['valores']):.4f}")
    print(f"PISO DEL MUDO: abstenerse SIEMPRE da GLOBAL {p_nose:.4f}. Por debajo de eso el "
          f"modelo es peor que el silencio.")
    if a.calentar:
        print(f"calentando {a.calentar} pasos sin casos de abstencion")

    opt = torch.optim.AdamW([p for p in arch.parameters() if p.requires_grad], lr=a.lr)
    capa = capas[a.capa_lectura]
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
    hist, t0 = [], time.time()

    for paso in range(1, a.pasos + 1):
        calentando = paso <= a.calentar
        frases, turnos_f, dis_idx, propios, objetivos, clases, _ = lote(a, pool, g, calentando, verdad)
        idx, tur, _ = armar(propios, dis_idx, turnos_f, a.arch, g)
        V = escribir(modelo, tok, frases, a.capa_escritura, estado)     # (F, d)
        archivo = V[idx.to(dev)]                                        # (B, N, d)

        preg = [f"{pool['entidades'][e]} {RELACIONES[r]}" for (e, r, _) in propios]
        ids = tok(preg, return_tensors="pt", padding=True).to(dev)
        objetivo = torch.tensor([TOK_VAL.get(o, TOK_ABS) for o in objetivos], device=dev)
        estado["archivo"], estado["turnos"] = archivo, tur.to(dev)
        salida = modelo(**ids)
        ultimo = ids["attention_mask"].sum(1) - 1
        logits = salida.logits[torch.arange(len(preg), device=dev), ultimo]
        perdida = nn.functional.cross_entropy(logits, objetivo)
        opt.zero_grad(); perdida.backward(); opt.step()

        if paso == 1 or paso % a.cada == 0 or paso == a.pasos:
            with torch.no_grad():
                pred = logits.argmax(-1)
            m = medir(pred, objetivo, clases, TOK_ABS, SET_VAL)
            m.update(paso=paso, perdida=round(perdida.item(), 4),
                     s_paso=round((time.time() - t0) / paso, 2))
            hist.append(m)
            print(f"  paso {paso:4d} · perdida {m['perdida']:7.4f} · GLOBAL {m['global']:.4f} · "
                  f"acierto {m['acierto']:.4f} · vigente {m['vigente']:.4f} · "
                  f"nose {m['nose']:.4f} · invento {m['invento']:.4f} · {m['s_paso']:.1f} s/paso"
                  f"{'  (calentando, sin abstencion)' if calentando else ''}", flush=True)

    # ---- Los controles, sobre el modelo ya entrenado y en el MISMO lote ----
    print("\n  CONTROLES sobre el ultimo lote:")
    res = {}
    with torch.no_grad():
        for nombre, arc in (("con archivo", archivo),
                            ("BARAJADO", archivo.roll(1, dims=0)),
                            ("VACIO (ceros)", torch.zeros_like(archivo)),
                            ("lectura APAGADA", None)):
            estado["archivo"] = arc
            lg = modelo(**ids).logits[torch.arange(len(preg), device=dev), ultimo]
            m = medir(lg.argmax(-1), objetivo, clases, TOK_ABS, SET_VAL)
            res[nombre] = m
            print(f"    {nombre:18s} GLOBAL {m['global']:.4f} · acierto {m['acierto']:.4f} · "
                  f"vigente {m['vigente']:.4f} · nose {m['nose']:.4f} · invento {m['invento']:.4f}")
        estado["archivo"] = archivo
    print("\n    BARAJADO alto = no lee el archivo · VACIO alto = contesta desde el "
          "preentrenamiento\n    (VACIO es el control que pedia el plan por vocabulario abierto)")

    # --- EVALUACION FINAL POR ENTIDAD (PREREG_ARCHIVO_CONTRA_PREENTRENAMIENTO, H2) ---
    # Cada entidad se pregunta `--reps` veces con un valor distinto del archivo cada vez, para que
    # el acierto por entidad tenga n suficiente y se pueda cruzar con su confianza previa.
    por_ent = {}
    if a.mundo:
        with torch.no_grad():
            for ei in range(len(pool["entidades"])):
                ok = 0
                for _ in range(a.reps):
                    v = None
                    while v is None or (verdad and VAL_[ei] == pool["valores"][v]):
                        v = int(torch.randint(0, len(pool["valores"]), (1,), generator=g))
                    fr = [f"{pool['entidades'][ei]} {RELACIONES[0]} {pool['valores'][v]}."]
                    V1 = escribir(modelo, tok, fr, a.capa_escritura, estado)
                    arc = V1.unsqueeze(1)                                  # (1, 1, d)
                    tu = torch.zeros(1, 1, dtype=torch.long, device=dev)
                    ids1 = tok([f"{pool['entidades'][ei]} {RELACIONES[0]}"],
                               return_tensors="pt", padding=True).to(dev)
                    estado["archivo"], estado["turnos"] = arc, tu
                    lg = modelo(**ids1).logits[0, ids1["attention_mask"].sum(1)[0] - 1]
                    ok += int(int(lg.argmax()) == TOK_VAL[pool["valores"][v]])
                por_ent[pool["entidades"][ei]] = {"acierto": ok / a.reps, "n": a.reps,
                                                  "confianza": CONF_.get(ei)}
        estado["archivo"] = archivo
        print(f"\n  POR ENTIDAD ({a.reps} repeticiones cada una):")
        for k, v in sorted(por_ent.items(), key=lambda kv: -(kv[1]["confianza"] or 0)):
            print(f"    c={v['confianza'] if v['confianza'] is not None else 0:.4f}  "
                  f"acierto {v['acierto']:.3f}  {k}")

    if a.salida:
        json.dump({"args": vars(a), "hist": hist, "controles": res, "por_entidad": por_ent},
                  open(a.salida, "w"), indent=1)
        print(f"\n  escrito {a.salida}")
    print(f"\nlisto en {(time.time()-t0)/60:.1f} min")


def medir(pred, objetivo, clases, tok_abs, set_val):
    """La vara del proyecto: (acierto + acierto_nose)/n, no `nose` a secas."""
    import numpy as np
    pred = pred.cpu().numpy(); obj = objetivo.cpu().numpy()
    cl = np.array(clases)
    con_resp = np.isin(cl, ["una", "dos"])
    sin_resp = ~con_resp
    ok = pred == obj
    def prom(m):
        return float(ok[m].mean()) if m.sum() else float("nan")
    dos = cl == "dos"
    return {
        "global": float(ok.mean()),                                  # la metrica de Maxi
        "acierto": prom(con_resp),                                   # solo las que tienen respuesta
        "vigente": prom(dos),                                        # version ULTIMA en las de dos
        "nose": prom(sin_resp),                                      # abstiene cuando debe
        "nose_rel": prom(cl == "nose_rel"),                          # la dura: entidad si, relacion no
        "nose_aus": prom(cl == "nose_aus"),
        "invento": float(np.isin(pred[sin_resp], list(set_val)).mean()) if sin_resp.sum() else 0.0,
        "n": {c: int((cl == c).sum()) for c in ["una", "dos", "nose_rel", "nose_aus"]},
    }


if __name__ == "__main__":
    main()
