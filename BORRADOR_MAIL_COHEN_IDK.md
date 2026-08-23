# Borrador de contacto con los autores de [IDK] — NO ENVIADO

**Estado: borrador. No se mandó nada.** Maxi decide si sale, y con qué cambios.

**Destinatarios.** Roi Cohen, Konstantin Dobler, Eden Biran, Gerard de Melo — *"I Don't Know:
Explicit Modeling of Uncertainty with an [IDK] Token"*, **NeurIPS 2024** (arXiv 2412.06676).
Gerard de Melo (HPI) es el autor senior y el que casi seguro califica como endorser de arXiv.

---

## La recomendación, primero

**Mandar DOS mensajes separados en el tiempo, no uno.**

El primero es el científico y no pide nada. El segundo, sólo si contestan, pide el endorsement.
Mezclarlos en un mail convierte «encontré algo que contradice tu explicación» en «te corrijo y de
paso hacéme un favor», y eso baja las chances de las dos cosas a la vez. El endorsement no se
pierde por esperar una semana; la buena impresión sí se pierde por apurarla.

Dicho eso, el pedido en sí es legítimo y hasta es el camino que arXiv recomienda: desde el
21-ene-2026, quien no tiene correo institucional va por el segundo camino, que es **endorsement
personal de un autor establecido del mismo dominio**, y la guía sugiere explícitamente contactar a
los autores cuyo trabajo uno cita o sobre el que construye. Que es exactamente este caso.

---

## Mail 1 — el científico (este es el que se manda primero)

> **Subject:** A small-model replication of the [IDK] failure mode, with a different cause
>
> Dear Roi, Konstantin, Eden and Gerard,
>
> I have been building a very small language model (863k parameters, trained from scratch) with an
> explicit persistent memory, to study when such a model can say that an answer is not in its
> memory. Your NeurIPS 2024 paper is the closest prior work to what I am doing, and I ran into the
> failure mode you report.
>
> You note that the [IDK] method fails on the smallest models you tried (pythia-70m and
> pythia-160m) and attribute it to numerical precision in the [IDK] embedding initialization. My
> setup sits entirely in that regime, so I tested that explanation directly.
>
> I ran three paired conditions, identical in every other respect, three seeds each:
>
> 1. **token**, which is your design, with the abstention symbol as one more entry of the
>    vocabulary softmax
> 2. **scale**, which is your hypothesis made as generous as I could, renormalizing the abstention
>    embedding to the mean norm of the value tokens at the start of the phase
> 3. **head**, a separate binary abstention output outside the vocabulary softmax, 129 extra
>    parameters, 0.015 percent of the model
>
> Condition 2 fails just like condition 1. Condition 3 passes the same gate where the other two do
> not. So in my setup the obstacle does not look like initialization or vector norm. It looks like
> the two decisions, whether an answer exists and which answer it is, competing for mass in one
> softmax, which is a structural constraint rather than a numerical one.
>
> I am aware this is a synthetic task at a scale far below yours, and that the result may simply
> not transfer. That is part of why I am writing. If the mechanism is the shared softmax, your Pi
> cap at 0.5 would be treating the symptom, and a separate head might be worth one ablation at your
> scale, where I cannot go.
>
> I would be glad to share the code, the pre-registration and the runs if any of this is useful to
> you. And if I have misread your paper on this point, I would rather hear it from you than keep
> building on a misreading.
>
> Best regards,
> Maximiliano R. Speranza
> Independent researcher
> ORCID 0009-0005-0413-8554

---

## Mail 2 — el endorsement (SÓLO si contestan, y no antes)

> **Subject:** arXiv endorsement for cs.CL
>
> Dear Gerard,
>
> Thank you for the exchange about the [IDK] result.
>
> I would like to post the write-up of these experiments on arXiv, in cs.CL. I am an independent
> researcher without an institutional address, so under the policy that came into effect in January
> 2026 I need a personal endorsement from an established author in the domain.
>
> Would you be willing to endorse me? The arXiv endorsement code is XXXXXX and the request can be
> completed at https://arxiv.org/auth/endorse?x=XXXXXX. There is no obligation attached to it and
> it says nothing about the quality of the work, only that the submission belongs in the category.
>
> If you would rather not, that is completely fine and I am grateful for the discussion either way.
>
> Best regards,
> Maximiliano R. Speranza
> ORCID 0009-0005-0413-8554

---

## Antes de mandar el mail 1, hay que verificar

- Que las tres condiciones estén realmente corridas con **3 semillas cada una** y que los números
  del párrafo sean los del checkpoint, no los de la memoria. Si un revisor los pide y no coinciden,
  el contacto se convierte en un problema.
- Que la afirmación «`escala` es la hipótesis de ellos hecha generosa» sea justa. Ellos hablan de
  **precisión numérica en la inicialización**; nosotros probamos **la norma del vector**. Se parecen
  pero no son idénticas, y conviene decirlo con esa precisión en vez de ponerles en la boca algo que
  no dijeron.
- Que los correos sean los correctos, del paper de NeurIPS y no de una lista vieja.
