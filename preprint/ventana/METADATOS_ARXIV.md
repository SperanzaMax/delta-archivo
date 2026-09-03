# Envío a arXiv del paper de la ventana · todo lo que pide el formulario

**Archivo a subir:** `ventana_arxiv.tar.gz` (contiene sólo `ventana_en.tex`; el paper no depende de
figuras ni de un `.bib` externo, y compila solo con `pdflatex` en dos pasadas, 8 páginas).

⚠️ arXiv quiere **el fuente LaTeX, no el PDF**. Subir el PDF directamente está permitido pero
desaconsejado, y hace que el paper no se pueda regenerar ni indexar bien.

---

## 1. Title

```
The Query Cannot See the Question: A Short Convolution's Reach Decides Which Part of a Query Conditions Retrieval
```

*(el subtítulo largo del `\title` no entra en el campo de arXiv, que quiere una sola línea)*

## 2. Authors

```
Maximiliano Speranza
```

## 3. Abstract

El del `\begin{abstract}` del `.tex`, **en texto plano**: hay que sacar el LaTeX. arXiv acepta `$...$`
para matemática simple pero no `\textbf`, `\emph` ni `\texttt`. Versión lista para pegar:

> Linear-attention and state-space models form their query with a short causal depthwise convolution.
> Kernel size 4 is the de facto default, so the query at a given layer is a function of the current
> token and three predecessors. We show that this reach is not a detail of local smoothing but a hard
> limit on which part of a question is allowed to condition retrieval, and that what falls outside
> does not degrade gracefully: it disappears exactly, and the model answers confidently from the part
> it can still see.
>
> We measure the effect in a small language model with a co-trained persistent archive queried across
> sequences. The sensitivity of retrieval to a query token is 0.000000 as soon as the token passes the
> convolution's reach, and the cut-off moves with the kernel: sixty cells out of sixty, two
> architectures, three seeds each. Widening the kernel from 3 to 5, which costs 1,280 parameters out
> of 865,395, lifts correct abstention on the hard case from 0.5850-0.7349 to 0.9931-1.0000 with no
> overlap between conditions.
>
> We then check the mechanism outside our own model. In mamba-130m-hf, changing one token five to
> eight positions before the read moves the layer output at every distance and leaves the convolution
> output at exactly zero, 80 cells out of 80. The state sees the whole sequence; the query that reads
> it does not.
>
> Depth changes the law without repealing it, and we measure by how much. Across fifteen question
> forms, six distances and two checkpoints (mamba-130m with 24 layers and mamba-370m with 48), the cut
> in layer 0 is exact in both, while from layer 1 recurrence restores the signal attenuated, at 1.077
> and 1.028 per token of distance (r = 0.978 in both). Three times the parameters and twice the depth
> leave the rate unchanged.
>
> Finally we test whether access predicts behaviour, and report a negative. Fine-tuning mamba-130m
> with the discriminating component outside the window costs +0.167, +0.181 and +0.141 of correct
> abstention at step 100, three seeds out of three above our pre-registered threshold, and nothing at
> all by step 400. Where there are no layers left to pay with, the window sets a ceiling; where there
> are, it sets a toll.

## 4. Categorías

| | |
|---|---|
| **Primaria** | `cs.LG` — Machine Learning |
| **Cross-list** | `cs.CL` — Computation and Language |

Razón: la contribución es sobre **arquitectura** (convolución de la query en modelos de espacio de
estados), que vive en `cs.LG`, y la tarea es de lenguaje, que justifica el cruce a `cs.CL`. No poner
`cs.AI`, que en la práctica es un cajón de sastre y resta.

## 5. Comments

```
8 pages. Pre-registered; criteria frozen with SHA before running. Includes a pre-registered negative
result. Spanish version available from the author.
```

Decir «pre-registered» y «negative result» en Comments **es a favor**, no en contra: es lo que
distingue al trabajo del ruido y lo que TMLR valora.

## 6. Licencia

```
CC BY 4.0
```

La misma que el preprint del sello. Es la más permisiva compatible con enviar después a una revista.

## 7. ACM / MSC class

Dejar en blanco. Son opcionales y para este trabajo no aportan.

## 8. DOI / Journal reference

Dejar en blanco **para este paper**, que es nuevo y no está en ningún lado.

⚠️ Para los que **ya tienen DOI de Research Square** (el de la gemación y el del sello), al subirlos
a arXiv hay que **poner el DOI en el campo DOI**. Research Square permite el depósito paralelo pero
advierte que dos versiones fragmentan las métricas: declarar el DOI es lo que las vuelve a unir.

---

## Lo que hay que saber antes de darle enviar

1. **Hace falta endorsement.** Para el primer envío a `cs.LG` arXiv pide que un autor establecido te
   avale, salvo que tu dominio de correo esté auto-endorsed. Al registrarte con
   `mrsperanza@frba.utn.edu.ar` el sistema te dirá cuál de las dos cosas aplica. Si pide endorsement,
   arXiv te da un **código** para pasarle a quien te avale.
2. **El registro y la contraseña los hacés vos.** No creo cuentas ni cargo credenciales.
3. **arXiv no borra.** Una vez anunciado, el paper queda; se puede reemplazar por una versión nueva
   (v2, v3) pero no retirar. Por eso conviene subir la versión que ya tiene el negativo adentro, que
   es la que está lista ahora, y no una anterior.
4. **Hay un hold de moderación** de hasta unos días en el primer envío de un autor nuevo. Es normal.
