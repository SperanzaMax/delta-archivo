# Borradores de contacto — convergencia y (eventual) aval de arXiv

**2026-08-16.** Para revisar antes de enviar. **No se envió nada.**

---

## ⚠ Dos cosas verificadas hoy que cambian la estrategia

**1. El paper que teníamos como mejor candidato CAMBIÓ DE TÍTULO.** Lo que citábamos como
*«Don't Ask the LLM to Track Freshness»* (arXiv 2606.01435) hoy figura como **«Reliable Post-Retrieval
Assembly for Agent Memory: Separating Evidence Extraction from Policy Execution»**, de **Vikas Reddy y
Sumanth Reddy Challaram**. Hay que citarlo por el título actual.

Y la convergencia es **más fuerte** de lo que creíamos. Su tesis: los sistemas de memoria fallan
cuando **varias tareas ocurren en un solo paso** (filtrado semántico, resolución de conflicto,
supresión de lo previo, generación), y separarlas en etapas los arregla — de 54 % a 82-93 % en
single-hop de MemoryAgentBench.

**Eso es exactamente lo que medimos hoy, un nivel más abajo:** nuestro softmax de lectura hace
selección **e** integración en un solo paso y **no tiene estado de vacío** — por eso «siempre encuentra
algo». Su receta a nivel de sistema y nuestro hallazgo a nivel de mecanismo dicen lo mismo. Ese es el
gancho, y es genuino.

**2. Ojo con quién puede avalar.** arXiv exige que el endorser tenga **entre 2 y 5 artículos ya
publicados en arXiv en esa categoría**, hechos públicos entre hace 5 años y hace 3 meses.
**Youwang Deng** (Substrate Asymmetry) figura como **autor único y sin afiliación** — muy posiblemente
otro investigador independiente, lo que lo hace un interlocutor natural pero **quizá no elegible como
endorser**. Lo mismo puede pasar con los autores de 2606.01435.

**Conclusión práctica: separar los dos objetivos.** Estos mails sirven para **construir interlocución
real** (que es lo valioso y lo que puede derivar en co-autoría o citas). Para el **aval**, conviene
alguien con trayectoria acumulada en cs.CL/cs.LG — un autor establecido de predicción selectiva o de
memoria, o un contacto académico local. Mezclar las dos cosas en el primer mail baja la probabilidad
de las dos.

---

## Mail 1 · Vikas Reddy / Sumanth Reddy Challaram (arXiv 2606.01435)

> **Asunto:** Mechanism-level evidence for your single-step failure claim (small co-trained memory)

Dear Vikas Reddy and Sumanth Reddy Challaram,

I read *Reliable Post-Retrieval Assembly for Agent Memory* with interest. Your central claim — that
memory systems fail when semantic filtering, conflict resolution and generation are collapsed into a
single step — matches something I measured this month from a very different angle, and I thought the
convergence might be useful to you.

I train a small language model **from scratch** (863k parameters, closed 242-token vocabulary) with a
**persistent archive co-trained inside the network**, and I evaluate it on multi-session facts with
elliptical corrections, with the recurrent state reset between sessions. Three pre-registered
experiments (hashes committed before looking at the data):

- The **retrieval score against the archived keys does not distinguish whether the queried fact is
  present at all**: AUC 0.4984 and 0.5022 — chance — replicated on two checkpoints. Reconstructing the
  model's logits from the extracted score matches bit-for-bit (0.000e+00), so this is a property of
  the model, not of my instrumentation.
- The model **does focus** (up to 0.65 of the read mass on a single entry), but at intermediate
  positions; measuring at the answering position hides it. **Where it focuses most, the score is still
  at chance.**
- Two probes locate the error: the neighbouring fact is **intact** (0.83), and the older version is
  degraded too — so it is neither corrupted writing nor a lost revision.

The mechanism is the part that speaks to your paper: **a softmax read sums to one, so the best entry
wins with comparable mass whether or not a good candidate exists.** Selection and integration happen
in one step and there is no "empty" state. It always finds something, and it finds with the same
conviction when there is nothing to find. Your staged separation is, in that light, a repair for a
mechanism that structurally cannot abstain.

The negative results are all pre-registered and the code is public:
https://github.com/SperanzaMax/delta-archivo — happy to share the draft if it is of any use. Related
prior work of mine on versioned memory retrieval is at DOI 10.21203/rs.3.rs-10669947.

Best regards,
Maximiliano Speranza — Independent researcher · ORCID 0009-0005-0413-8554

---

## Mail 2 · Youwang Deng (arXiv 2606.11712) — enviar SÓLO después de tener respuesta del primero

> **Asunto:** A third substrate for your factual-absence axis

Dear Youwang Deng,

*Substrate Asymmetry in User-Side Memory* frames memory along three axes and shows that no single
technique wins all of them — with RAG clearly ahead on **factual absence**. I have been measuring a
**third substrate** that your comparison does not cover, and the result fits your framework directly.

Instead of parametric memory or external retrieval, I train a small model **from scratch** with a
**persistent index co-trained inside the network**. On your absence axis it behaves like the
**parametric** side, not like retrieval: the score of the query against the archived keys separates
present from absent facts at **AUC 0.4984 / 0.5022** — chance — across two checkpoints and three
pre-registered controls.

I believe the reason is the one your asymmetry implies but does not state: **an external retriever can
return the empty set, and that is where free abstention comes from.** A co-trained index read by
softmax cannot — the mass sums to one, so something always wins. Putting the index inside the network
buys co-training and **loses the abstention that retrieval had for free**.

If that reading is right, your third axis has a structural explanation rather than an empirical one,
and the repair is a learned null slot (the pointer-sentinel / SQuAD 2.0 no-answer mechanism) applied
at the memory interface.

Code and pre-registrations: https://github.com/SperanzaMax/delta-archivo

Best regards,
Maximiliano Speranza — Independent researcher · ORCID 0009-0005-0413-8554

---

## Notas de envío

- **Uno por vez.** arXiv considera inapropiado escribir a muchos endorsers a la vez; y aunque estos
  mails no piden aval, el criterio de no parecer un envío masivo vale igual.
- **Ningún mail pide endorsement.** Si aparece interés, el pedido sale natural en el segundo
  intercambio, y para entonces ya vieron el trabajo — que es lo que arXiv pide del endorser.
- **Conseguir los correos**: figuran en el PDF de cada paper (primera página) o en la página de arXiv.
- **Revisar antes de enviar** la [política actualizada de enero 2026](https://blog.arxiv.org/2026/01/21/attention-authors-updated-endorsement-policy/).
