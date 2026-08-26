# Paquete de envío — el trípode de abstención

Mismo formato que el envío de gemación del 11-ago, que salió publicado sin fricción. Un solo
registro, PDF en inglés como principal y el español como suplemento.

---

## Dónde se envía, y por qué

| destino | estado | motivo |
|---|---|---|
| **Research Square** | ✅ **el elegido** | ya hay cuenta con ORCID conectado, es gratis, da DOI, indexa en Scholar, y ya publicó un trabajo de esta línea sin problemas |
| arXiv | ❌ bloqueado | desde el 21-ene-2026 hace falta endorsement de un autor establecido. El mail a Cohen y de Melo salió el 23-ago y no contestaron todavía |
| TMLR / OpenReview | ❌ bloqueado | el perfil sigue sin activar (tres rechazos, hace falta avalista). Y TMLR permite preprints previos, así que esto no lo quema |
| Preprints.org | ❌ descartado | rechazó cinco envíos, dos de ellos por duplicado con Zenodo |
| Zenodo | ❌ **NO** | anclar acá es lo que quemó Preprints.org. La prioridad ya está anclada por los pre-registros congelados |

**Regla que se respeta:** no se sube a Zenodo un paper que se va a enviar a otro lado.

---

## Título

```
Where Abstention Lives: A Pre-Registered Four-Way Comparison of Abstention Interfaces in a Small Language Model with Versioned Memory
```

## Autor

| campo | valor |
|---|---|
| Nombre | Maximiliano Speranza |
| Afiliación | Independent Researcher, Buenos Aires, Argentina |
| Email | maximiliano.speranza@gmail.com |
| ORCID | https://orcid.org/0009-0005-0413-8554 |
| Autor de correspondencia | sí (autor único) |

**La UTN no va.** Firma como investigador independiente; la condición de estudiante va sólo en la
nota de autor de eventos presenciales, nunca en un depósito indexado.

## Abstract (texto plano, para pegar)

```
Teaching a language model to say "I don't know" has at least three published implementations, and they are never compared against each other. The uncertainty can be a token in the vocabulary ([IDK]), a separate binary head (SelectiveNet), or an entry in the memory the model reads from (pointer sentinel, the no-answer score of SQuAD 2.0). Each was introduced in a different architecture, on a different task, against a different baseline. Nobody has asked the obvious question: holding the model, the data, the seeds and the training budget fixed, does it matter where the abstention decision lives?

We answer it in a 863,730-parameter language model trained from scratch, with a co-trained persistent archive and versioned facts, where an answer is a single token and the metric is therefore exact. Four interfaces are compared pairwise across 27 training units, with every prediction hash-frozen before the corresponding data existed.

It matters, and the ordering is stable. A separate binary head - 129 parameters, 0.015% of the model - reduces the false-abstention rate by a factor of 2 to 2.8x against the vocabulary token at matched budget in 3/3 seeds, and moves the training frontier at which abstention becomes learnable down by approximately 0.10 of accuracy margin: with a head, a model that recovers 10 points worse can still be taught to keep quiet. Renormalizing the token's embedding vector - the explanation the [IDK] paper offers for its own failure in small models - does not help, leaving a gap of +0.1465 in false abstention against the head. The mechanism is not the norm of the vector; it is that two decisions of different natures compete for one softmax.

The memory slot fails, and fails informatively. Its attention mass converges to 0.4074, 0.4046 and 0.4020 against an empirical base rate of absent questions of 0.4048: it learned the prior, not membership. A 400-point threshold sweep rescues nothing (Youden's J between +0.038 and +0.078). Read together with an archive matching score that sits at chance (AUC 0.4984), this says something sharper than "the slot did not work": in a co-trained memory read by softmax, absence has no representation, and adding a place for it to live is not sufficient to create one.

We state explicitly what these results do not license. The claim supported here is "separating the head makes abstention learnable in this regime", not "the model knows when it does not know": the residual ceiling is one of calibration, not capacity (AUC 0.777-0.998), and seven pre-registered attempts to obtain the decision without labels all landed in AUC 0.50-0.67.
```

## Keywords

```
selective prediction; abstention; hallucination; language model memory; versioned knowledge; pre-registration; negative results
```

## Área / categoría

Computer Science → Machine Learning (alternativa: Artificial Intelligence)

## Licencia

**CC-BY 4.0**, la misma de todo el cuerpo de trabajo.

## Archivos

| archivo | rol |
|---|---|
| `tripode_en.pdf` | manuscrito principal (10 páginas) |
| `tripode_es.pdf` | material suplementario, versión completa en español |

## Declaraciones

- **Funding:** None.
- **Competing interests:** The author declares no competing interests.
- **Author contributions:** Sole author; responsible for design, implementation, analysis and
  writing.
- **Data availability:** All pre-registrations with their SHA hashes, deviation files, analysis
  scripts, per-seed raw results and machine-generated reports are archived in the project
  repository. Every verdict in the paper is emitted by a script frozen before the corresponding
  data existed.
- **Ethics / human subjects:** Not applicable — no human or animal subjects; all data is synthetic.
- **Preprint posted elsewhere:** No.

## Qué NO contratar

Research Square ofrece edición, formato, traducción y «premium». Ninguno hace falta: publicar el
preprint es gratis. Y cuando salga van a llegar mails de revistas depredadoras citando el título
exacto — ya pasó dos veces con el preprint anterior. **No responder y no tocar «Unsubscribe».**

---

## Trazabilidad de cada número del paper

Por si un revisor pregunta, o por si hay que rehacer una cifra.

| afirmación del paper | de dónde sale |
|---|---|
| 21 unidades, `cabeza` 4/5 vs `token` y `escala` 0/5 | `INFORME_CABEZA_20260819.md` §2-3 |
| brecha `escala` +0,1465 y la sonda de normas | `INFORME_CABEZA_20260819.md` §3-4 |
| presupuesto, `token` cruza 4/5 a 20000 | `INFORME_PRESUPUESTO_TOKEN.md` P-1/P-2 |
| pareado a 20000, 2,0× / 1,7× / 2,8× | `INFORME_PRESUPUESTO_TOKEN.md` P-3 |
| frontera, Spearman −0,8033 / −0,8886 | `INFORME_FRONTERA_20260819.md` §3-4 |
| celda cruzada, 7/7 signos p=0,0078 | `INFORME_CELDA_CRUZADA_20260819.md` |
| slot, masa 0,4074/0,4046/0,4020 vs base 0,4048 | `INFORME_SLOT_20260825.md` §2 |
| barrido de 400 umbrales, J +0,038 a +0,078 | `INFORME_SLOT_20260825.md` §2.bis |
| instrumento roto y su regresión bit a bit | `NOTA_INSTRUMENTO_SLOT_20260825.md` |
| score del archivo AUC 0,4984 / 0,5022 | `INFORME_SCORE_ARCHIVO_20260816.md` |
| foco 0,65 en posiciones intermedias | `INFORME_FOCO_LECTURA_20260816.md` |
| `err_fuera` 0,0000 en los cuatro niveles | `INFORME_SER_20260815.md` |
| monitor de desacuerdo, 0/8 con M-3 8/8 | `INFORME_MONITOR_20260820.md` |
| AUC del logit de la cabeza 0,777-0,998 | `INFORME_CABEZA_20260819.md` §4bis |
| las siete vías sin etiquetas en 0,50-0,67 | `PLAN_FOCO_20260824.md` §3 + `INFORME_TEST_K_20260824.md` |
