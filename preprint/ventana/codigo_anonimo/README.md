# The Query Cannot See the Question — code and data

Anonymous supplementary material for a TMLR submission.

Everything in the paper can be reproduced from what is here. There are no API keys, no personal
paths and no orchestration scripts, because none of those are needed to reproduce a result.

## What is where

```
micro_lm/         the 3.5 MB model with a co-trained persistent archive (Results 1-3)
  modelo.py         architecture, including convq and the configurable kernel
  datos.py          the closed synthetic language, 242 tokens
  entrenar.py       training and evaluation
  ablacion_taps.py  Result 3, ablating the tap that reaches the relation
  chequeo_alineamiento_conv.py   measures the real reach of a convolution from its weights

modelo_real/      the measurements on Mamba (Results 4-6)
  tarea_real.py       the same task in English, one-token answers
  entrenar_real.py    fine-tune of mamba-130m, and the sensitivity probe after training
  geometria_formas.py the token distances, counted with the tokenizer and not assumed
  sonda_combinacion.py where conv1d moves when the relation token is replaced
  escalera_v2.py      Result 5, the attenuation curve across 15 forms and 6 distances
  juzgar_distancia.py the pre-registered criteria, with the guards applied
  bench_mambapy.py    equivalence and speed of the parallel scan
  campana_remota.py   runs a list of units back to back
  vocabulario.json    entities, values, relations and question templates

prereg/           the criteria, frozen before running. Each was committed with its SHA256
resultados/       the raw JSON every number in the paper is computed from
```

## Reproducing the numbers

The measurements in Results 4 and 5 need **no training and no GPU**. They download a public
checkpoint and probe it:

```bash
python modelo_real/geometria_formas.py     # the distances, verified against the tokenizer
python modelo_real/sonda_combinacion.py    # where conv1d moves, per layer
MODELO=state-spaces/mamba-130m-hf python modelo_real/escalera_v2.py
MODELO=state-spaces/mamba-370m-hf python modelo_real/escalera_v2.py
```

The last two take a few minutes each on CPU and write `escalera_v2_130m.json` and
`escalera_v2_370m.json`, which are the files already included in `resultados/`.

Result 6 needs a GPU. One unit is

```bash
python modelo_real/entrenar_real.py --condicion cerca --semilla 0 --pasos 800 \
    --batch 8 --acum 1 --largo 64 --n-hechos 4 --cada 100 --n-eval 32 \
    --modelo state-spaces/mamba-130m-hf --salida g0_cerca_s0.json
```

and the verdict over all twelve units is

```bash
python modelo_real/juzgar_distancia.py
```

## One dependency worth naming

`entrenar_real.py` turns on `use_mambapy`, a parallel associative scan in pure PyTorch that
`transformers` supports but leaves off by default. Without it, HuggingFace walks the sequence token
by token in Python and a training step costs about nine times more. We verified it is numerically
equivalent to the sequential path before using it, on CPU and on GPU, and `bench_mambapy.py` is that
check. Install it with `pip install mambapy`.

## A note on the code you are reading

These scripts are research code written while the questions were being answered, so the comments
record what was measured, what was ruled out, and what turned out to be wrong. That includes mistakes
we made and caught, such as a control that could not have come out any other way, and a statistic that
did not survive sweeping its threshold. Both are reported in the paper. The comments are in Spanish
because that is the language they were written in.
