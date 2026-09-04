# Envío a Research Square — el paper de la ventana

## ✅ ENVIADO EL 2026-09-04 a las 07:43 (GMT-3) · RSID `rs-10929866`

| campo | valor |
|---|---|
| **RSID** | `rs-10929866` |
| **estado** | **Prescreening** (hasta 72 h hábiles; el del sello tardó día y medio) |
| **DOI** | *Available once preprint is posted* |
| panel | https://www.researchsquare.com/article/rs-10929866/private/article-overview |

Es el **cuarto** preprint de esta línea. Lo que se cargó, exacto:

| sección | qué se puso |
|---|---|
| 1. manuscrito | `ventana_en.pdf`, 10 páginas |
| 2. título | el de abajo, título y subtítulo unidos por dos puntos |
| 3. article type | Research Article |
| 4. area of study | Computer Science → Artificial Intelligence and Machine Learning |
| 5. autor | Maximiliano Rodrigo Speranza · Independent Researcher · **`mrsperanza@frba.utn.edu.ar`** · ORCID · corresponding |
| 6. abstract | el de `abstract_rs.txt`, 4618 de 5000 caracteres |
| 7. keywords | los diez de abajo |
| 8. figuras | salteada, van dentro del PDF |
| 9. suplementario | `ventana_es.pdf`, «Complete Spanish-language version of the manuscript» |
| 10. institución | salteada. La UTN **no** patrocinó ni va como afiliación |
| 11. funding | salteada, no hay |
| 12. ethics | guardada sin marcar: ni sujetos humanos ni vertebrados |
| 13. competing interests | «No, I declare that authors have no interests…» |
| 14. submit | términos + CC BY 4.0, con código de 6 dígitos por mail |

**★ El mail del autor pasó a ser el INSTITUCIONAL, por pedido de Maxi durante la carga.** Los dos
`.tex` se editaron y los dos PDF se **recompilaron antes de subirlos**, así que el registro y el
manuscrito dicen lo mismo. Los tres preprints anteriores llevan el Gmail; desde éste es
`mrsperanza@frba.utn.edu.ar`. ⚠️ **`ventana_arxiv.tar.gz` sigue teniendo el Gmail**: hay que
regenerarlo antes del envío a arXiv.

**El código de 6 dígitos se leyó del Gmail con el conector**, no hizo falta que lo pasara a mano.
Expira en 10 minutos y la sesión del navegador aguantó sin problema, así que el envío entero se puede
hacer de una sentada.

## ⏳ 2026-09-04 · TICKET `#11738664` para corregir ANTES de que postee

La auditoría del mismo día (`AUDITORIA_VENTANA_20260904.md`) encontró que el abstract dice
**«two blocks deep»** y el micro-LM tiene **cuatro** bloques. Corregido en el manuscrito, pero el
envío ya estaba adentro.

**No corresponde una v2**, y se verificó en la plataforma en vez de suponerlo: `/article/rs-10929866/v1`
devuelve **`Resource not found`**, o sea que **no hay versión pública ni DOI todavía**. Lo que
corresponde es corregir el envío para que la versión pública nazca sin el error.

**★ La vía que NO necesita sesión, y es lo reutilizable:** el formulario de soporte
`support.researchsquare.com/support/tickets/new` **no pide login**. Pide correo, categoría, RSID,
título y descripción. Se abrió el ticket con categoría *Preprint posting status*, pidiendo que
devuelvan el envío a draft o reemplacen el archivo, con el link al PDF corregido en GitHub.

Sirvió porque **la sesión de Research Square no sobrevive a un reinicio de Chrome** (la de OpenReview
sí) y el login sólo lo puede hacer Maxi.

**Riesgo de esperar, bajo:** el prescreening tarda de día y medio a 72 h hábiles.


---

## Archivos

| archivo | rol |
|---|---|
| `ventana_en.pdf` | manuscrito principal, 10 páginas |
| `ventana_es.pdf` | material suplementario, versión completa en español, 11 páginas |

**Es el PDF CON nombre**, no el anónimo de `tmlr/`. Research Square no es ciego.

## Título

```
The Query Cannot See the Question: A Short Convolution's Reach Decides Which Part of a Query Conditions Retrieval, and What Falls Outside Becomes a Confident Wrong Answer
```

## Autor

| campo | valor |
|---|---|
| Nombre | Maximiliano Rodrigo Speranza |
| Afiliación | Independent Researcher, Buenos Aires, Argentina |
| Email | mrsperanza@frba.utn.edu.ar |
| ORCID | https://orcid.org/0009-0005-0413-8554 |
| Autor de correspondencia | sí, autor único |

**La UTN no va como afiliación.** El correo institucional sí, que es distinto: identifica dónde
estudia, no bajo qué institución firma.

## Abstract, en texto plano para pegar

```
Linear-attention and state-space models form their query with a short causal depthwise convolution. Kernel size 4 is the de facto default, so the query at a given layer is a function of the current token and three predecessors. We show that this reach is not a detail of local smoothing but a hard limit on which part of a question is allowed to condition retrieval, and that what falls outside does not degrade gracefully: it disappears exactly, and the model answers confidently from the part it can still see. We measure the effect in a small language model with a co-trained persistent archive queried across sequences. The sensitivity of retrieval to a query token, measured as the total-variation movement of the read distribution when that token is replaced, is 0.000000 as soon as the token passes the convolution's reach, and the cut-off moves with the kernel: with kernel 3 (reach 2) the step falls between distances 2 and 3, and with kernel 5 (reach 4) between 4 and 5. Sixty cells out of sixty, two architectures, three seeds each, two query components, five distances. The behavioural consequence is a specific and common failure. In our task, questions have the form what is the <relation> of <entity>?, with the entity at distance 1 from the read position and the relation at distance 3. With kernel 3, the relation is outside the window in 100% of queries, so the model retrieves on the entity alone. When asked about a relation that was never stated for an entity that was, it answers from the wrong stored fact instead of abstaining: correct abstention is 0.5850 to 0.7349 across three seeds. Widening the kernel to 5, which costs 1,280 parameters out of 865,395, lifts it to 0.9931 to 1.0000 with no overlap between conditions, and overall accuracy to 0.988-0.993 against a trivial floor of 0.4065. We then check the mechanism outside our own model. In mamba-130m-hf, 129M parameters, changing one token five to eight positions before the read moves the layer output at every distance and leaves the convolution output at exactly zero, 80 cells out of 80. The state sees the whole sequence; the query that reads it does not. Incidentally, the oldest convolution tap in that checkpoint is exactly zero in all 24 layers, so its effective reach is 2 rather than the nominal 3; we report the measurement and not a cause. Depth changes the law without repealing it, and we measure by how much. In a recurrent model the combination of two query components happens wherever both are available, so the distance that decides is the one between them. Across fifteen question forms, six distances and two checkpoints - mamba-130m with 24 layers and mamba-370m with 48 - the cut in layer 0 is exact and falls at the measured reach in both, while from layer 1 recurrence restores the signal attenuated, at 1.077 and 1.028 per token of distance (r = 0.978 in both, curves correlated at 0.955). Three times the parameters and twice the depth leave the rate unchanged, which makes the attenuation a property of the architecture. A control holding the distance fixed and varying the filler separates distance from lexical content. Finally we test whether access predicts behaviour, and report a negative. Fine-tuning mamba-130m on the same task with the discriminating component outside the window costs

1

+0.167, +0.181 and +0.141 of correct abstention at step 100 - three seeds out of three above our pre-registered threshold - and nothing at all by step 400. Our small model, two blocks deep, never recovers; the 24-layer model recovers quickly. Where there are no layers left to pay with, the window sets a ceiling; where there are, it sets a toll. The strong claim therefore holds for memory consulted from an early layer, and not as a behavioural prediction for a deep model trained to convergence. Finally we ask how often this configuration arises when nobody is arranging it. Across four question-answering corpora - SQuAD, Natural Questions, TriviaQA and HotpotQA, 33,585 questions - between 90% and 100% of real questions have their discriminating parts further apart than the measured reach, and it worsens with question difficulty, reaching 1.0000 on multi-hop questions. The configuration our synthetic task was built to study is the ordinary case, not a corner case. We give the diagnosis as a recipe that requires no training: change one token of the query and see whether retrieval moves. If it does not, that token is invisible to the search, and any confidence the model expresses about it is unearned - though, as our negative shows, invisible to one layer's query is not the same as unusable by the model.

1
```

## Keywords

```
state space models; linear attention; associative recall; abstention; selective prediction; mechanistic interpretability; retrieval; hallucination; pre-registration; negative results
```

## Área / categoría

Computer Science → **Machine Learning** (alternativa: Artificial Intelligence)

## Licencia

**CC-BY 4.0**, la misma de todo el cuerpo de trabajo y la que va a ir en arXiv.

## Declaraciones

- **Funding:** None.
- **Competing interests:** The author declares no competing interests.
- **Author contributions:** Sole author; responsible for design, implementation, analysis and writing.
- **Data availability:** All code, the frozen pre-registrations with their SHA256 hashes, and the raw
  JSON from which every number in the paper is computed are available at
  https://github.com/SperanzaMax/delta-archivo
- **Ethics / human subjects:** Not applicable. No human or animal subjects. The synthetic language is
  generated; the four question corpora used in Result 7 are public benchmarks.
- **Preprint posted elsewhere:** No.

---

## Antes de darle enviar

1. **Pushear el repo.** El paper dice que el código está en GitHub y hoy hay 43 commits sin subir, así
   que quien siga el link no encuentra un solo archivo de los que promete. Esto hay que resolverlo
   antes, no después.
2. **El envío pide un código de 6 dígitos por mail que expira en 10 minutos**, y la sesión del
   navegador caduca. Es 2FA de la plataforma; el envío final lo hacés vos.
3. **Prescreen editorial** de hasta 72 horas hábiles. El del sello tardó día y medio.
4. **Después del DOI**, salen los mails a la facultad con el link. Están en `endorsement/`, en versión
   corta y larga.

## Y lo que viene después del DOI

| destino | cuándo |
|---|---|
| **TMLR** | ya, no depende de esto. `tmlr/ventana_tmlr.pdf` + `suplementario_anonimo.zip` |
| **arXiv** | cuando llegue el endorsement, **declarando el DOI de Research Square** en el campo DOI |
| TechRxiv | si reabre. Hoy no acepta envíos |
