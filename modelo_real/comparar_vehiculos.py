"""¿Cual conviene como vehiculo? TinyLlama vs Qwen3-0.6B vs Qwen3.5-0.8B · 2026-09-09

No se elige por el paper, se elige por lo que el montaje NECESITA. Cuatro preguntas, y las cuatro
se contestan midiendo, no leyendo:

  1. ¿Se puede montar?    hace falta `model.model.layers[i]` para el hook de lectura y
                          `output_hidden_states` para la escritura. Si no, no hay experimento.
  2. ¿Tiene el mecanismo? cuales capas son lineales y cuales de atencion completa, y si hay una
                          convolucion corta formando la query. Es lo que las leyes describen.
  3. ¿Da objetivos limpios? cuantas piezas del banco tokenizan a UN token.
  4. ¿Cuanto cuesta?      segundos por forward en esta CPU.
"""
import json, time, warnings
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

warnings.filterwarnings("ignore")
MODELOS = ["TinyLlama/TinyLlama-1.1B-Chat-v1.0", "Qwen/Qwen3-0.6B", "Qwen/Qwen3.5-0.8B"]
POOL = json.load(open("pool_tinyllama.json"))
PIEZAS = POOL["entidades"] + POOL["valores"] + [POOL["abstencion"]]

for mid in MODELOS:
    print("=" * 78)
    print(mid)
    print("=" * 78)
    cfg = AutoConfig.from_pretrained(mid)
    t = getattr(cfg, "text_config", cfg)
    lt = getattr(t, "layer_types", None)
    print(f"  capas {t.num_hidden_layers} · hidden {t.hidden_size} · vocab {t.vocab_size}")
    if lt:
        full = [i for i, x in enumerate(lt) if "full" in x]
        print(f"  HIBRIDO · intervalo {getattr(t,'full_attention_interval','?')} · "
              f"conv corta kernel {getattr(t,'linear_conv_kernel_dim','?')}")
        print(f"  capas de atencion COMPLETA: {full}")
        print(f"  capas LINEALES (regla delta): {len(lt)-len(full)} de {len(lt)}")
    else:
        print("  DENSO · atencion completa en todas las capas · sin conv corta")

    tok = AutoTokenizer.from_pretrained(mid)
    uno = sum(1 for w in PIEZAS
              if len(tok(f" {w}", add_special_tokens=False)["input_ids"]) == 1)
    print(f"  piezas del banco que son de UN token: {uno} de {len(PIEZAS)} "
          f"({100*uno/len(PIEZAS):.0f} %)")

    try:
        t0 = time.time()
        m = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.float32)
        m.eval()
        carga = time.time() - t0
    except Exception as e:
        print(f"  ⚠️ NO carga con AutoModelForCausalLM: {type(e).__name__}: {str(e)[:150]}")
        print()
        continue

    # ¿Existe el arbol que el hook necesita?
    ruta, mod = None, m
    for cand in ("model.layers", "model.language_model.layers", "model.model.layers"):
        o = m
        try:
            for p in cand.split("."):
                o = getattr(o, p)
            ruta, mod = cand, o
            break
        except AttributeError:
            continue
    print(f"  ruta de capas para el hook: {ruta or 'NO ENCONTRADA'} "
          f"({len(mod) if ruta else '?'} bloques)")

    ids = tok(["Ana lives in Boston.", "Paul works in Berlin."],
              return_tensors="pt", padding=True)
    tok.pad_token = tok.pad_token or tok.eos_token
    t0 = time.time()
    with torch.no_grad():
        o = m(**ids, output_hidden_states=True)
    dt = time.time() - t0
    print(f"  hidden_states: {len(o.hidden_states)} niveles · forward de 2 frases {dt:.2f} s "
          f"· carga {carga:.0f} s")
    n = sum(p.numel() for p in m.parameters())
    print(f"  parametros {n/1e9:.2f} B · un archivo de rango 256 seria "
          f"{4*256*t.hidden_size*1e-6:.2f} M ({100*4*256*t.hidden_size/n:.3f} %)")
    del m
    print()
