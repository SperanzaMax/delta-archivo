"""LA LEY EN UN MODELO QUE NO ES NUESTRO · Mamba-130M en CPU · 2-sep

Evalua `PREREG_MODELO_REAL.md` (SHA 91684b97), congelado antes de descargar el modelo.

Lo que se pone a prueba NO es que un modelo recurrente ignore tokens lejanos —eso seria falso— sino
la DISOCIACION: el ESTADO ve toda la secuencia y la QUERY con la que ese estado se lee ve una ventana.
En Mamba `conv1d` (kernel 4) se aplica a x ANTES de calcular B, C y Delta, y C es el analogo de la
query de lectura. Si la ley vale, la salida de conv1d en la posicion t es funcion EXACTA de x[t-3..t].

R-1  el movimiento de la salida de conv1d es 0,0 EXACTO para d >= 4 y > 0 para d <= 3
R-2  el movimiento de la salida de la CAPA es > 0 tambien para d >= 4  (la disociacion)
R-3  se sostiene en >= 4 de 5 posiciones y con dos textos
"""
import os
import sys

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODELO = os.environ.get("MODELO", "state-spaces/mamba-130m-hf")
DS = [1, 2, 3, 4, 5, 6, 7, 8]

TEXTOS = [
    ("en", "The capital city of the country that borders Spain to the west is Lisbon , and the "
           "population of that city is about half a million people ."),
    ("es", "la altura del faro que esta en el puerto del norte es de treinta metros , y el precio "
           "de la entrada al museo del centro es de diez pesos ."),
]


def main():
    torch.set_grad_enabled(False)
    tok = AutoTokenizer.from_pretrained(MODELO)
    modelo = AutoModelForCausalLM.from_pretrained(MODELO, dtype=torch.float32)
    modelo.eval()
    capa0 = modelo.backbone.layers[0].mixer
    k = capa0.conv1d.kernel_size[0]
    print("=" * 96)
    print(f"LA LEY EN {MODELO} · prereg SHA 91684b97")
    print("=" * 96)
    print(f"  conv1d de la capa 0: kernel {k}  ->  alcance {k - 1} tokens hacia atras")
    print(f"  parametros: {sum(p.numel() for p in modelo.parameters()):,}\n")

    capt = {}
    capt_h = capa0.conv1d.register_forward_hook(
        lambda m, i, o: capt.__setitem__("conv", o.detach().clone()))
    capa_h = modelo.backbone.layers[0].register_forward_hook(
        lambda m, i, o: capt.__setitem__("capa", (o[0] if isinstance(o, tuple) else o).detach().clone()))

    filas, ok1, ok2, celdas = [], 0, 0, 0
    for nom, texto in TEXTOS:
        ids = tok(texto, return_tensors="pt").input_ids
        T = ids.shape[1]
        # cinco posiciones de lectura repartidas, siempre con margen para d=8 hacia atras
        posiciones = [T - 1 - i * max(1, (T - 12) // 5) for i in range(5)]
        posiciones = [p for p in posiciones if p >= 9]
        print(f"--- texto «{nom}» · {T} tokens · posiciones de lectura {posiciones}")
        for pos in posiciones:
            modelo(ids)
            base_conv = capt["conv"][0, :, pos].clone()
            base_capa = capt["capa"][0, pos].clone()
            fila = []
            for d in DS:
                j = pos - d
                if j < 0:
                    fila.append((d, float("nan"), float("nan")))
                    continue
                otro = ids.clone()
                # se reemplaza por un token cualquiera DISTINTO, del mismo texto, para no meter uno
                # que el modelo nunca vio en ese contexto
                nuevo = int(ids[0, (j + 7) % T])
                if nuevo == int(ids[0, j]):
                    nuevo = int(ids[0, (j + 13) % T])
                otro[0, j] = nuevo
                modelo(otro)
                dc = float((capt["conv"][0, :, pos] - base_conv).abs().max())
                dh = float((capt["capa"][0, pos] - base_capa).abs().max())
                fila.append((d, dc, dh))
                celdas += 1
                dentro = d <= k - 1
                ok1 += int((dc > 0) == dentro)
                ok2 += int(dh > 0)
            filas.append((nom, pos, fila))
            print(f"  pos {pos:3d}  " + "  ".join(
                f"d{d}:{'conv ' + ('0.0     ' if dc == 0 else f'{dc:.2e}')}" for d, dc, dh in fila))
            print(f"           " + "  ".join(
                f"d{d}:{'capa ' + ('0.0     ' if dh == 0 else f'{dh:.2e}')}" for d, dc, dh in fila))
    capt_h.remove(); capa_h.remove()

    print("\n" + "=" * 96)
    print(f"R-1  conv1d se mueve si y solo si d <= {k - 1}:  {ok1} de {celdas}"
          f"   {'CUMPLE' if ok1 == celdas else '** NO CUMPLE **'}")
    print(f"R-2  la salida de la CAPA se mueve SIEMPRE:      {ok2} de {celdas}"
          f"   {'CUMPLE' if ok2 == celdas else '** NO CUMPLE **'}")
    print("=" * 96)
    if ok1 == celdas and ok2 == celdas:
        print("  -> DISOCIACION: el estado ve toda la secuencia y la query ve una ventana de "
              f"{k} tokens.")


if __name__ == "__main__":
    main()
