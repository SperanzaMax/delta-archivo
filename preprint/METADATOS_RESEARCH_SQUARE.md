# Paquete de envío — Research Square

Todo listo para pegar. Un solo registro: el PDF en inglés como principal, el español como suplemento.

---

## Título

```
Geometric Indexing Does Not Improve Versioned Memory Retrieval: Three Pre-Registered Negative Results and a Silent Total-Failure Mode
```

## Autor

| campo | valor |
|---|---|
| Nombre | Maximiliano Speranza |
| Afiliación | Independent Researcher, Buenos Aires, Argentina |
| Email | maximiliano.speranza@gmail.com |
| ORCID | https://orcid.org/0009-0005-0413-8554 |
| Autor de correspondencia | sí (autor único) |

## Abstract (texto plano)

```
Conversational agents are increasingly expected to remember what a user told them across sessions, including the case where the user corrects something said earlier. A natural proposal is to exploit the geometry of the embedding space: rather than overwriting a memory, deposit the revised version near its predecessor, so that proximity encodes the relation between versions. We test that proposal directly, in a non-parametric index over a frozen encoder, with every prediction registered and hash-frozen before the corresponding data existed.

The result is a consistent negative with an identified mechanism. With a fixed displacement step, the revision cluster performs an unbounded random walk away from its anchor: cosine to the query falls from 0.811 to 0.324 after four revisions and to -0.139 after six, and coverage collapses to 0.000 from K >= 4. Bounding the displacement repairs the drift exactly as intended - the cosine becomes flat at 0.8177, slope +0.0000 per revision - and the mechanism still loses, by -0.0864 (95% CI [-0.1014, -0.0714]) at K = 8. The reason is a constant toll of approximately 0.036 cosine: the embedding of the revision's own text is already optimally placed, because it contains the entity the query mentions, so any artificial displacement can only move it away.

That last clause is a property of the generator, not of corrections. Real conversational corrections are elliptical: "no, it's Beto" does not name the entity. We therefore ran an adversarial third experiment in the regime that should favour geometry, against the honest baseline (coreference hydration, as deployed by production memory systems), sweeping its failure rate tau. Geometry only wins above tau* = 0.45 - a coreference resolver worse than a coin flip - so the negative extends rather than reverses.

The useful positive finding is independent of the mechanism under test: raw elliptical corrections are unrecoverable, scoring 0.0000 recall at k = 5 across all ten seeds - not low, never. In a separate check, their top-1 rank is likewise indistinguishable from chance. A memory system that archives conversational turns without resolving coreference loses 100% of corrections, and loses them silently - the index always returns something, just never the right thing.
```

## Keywords

```
conversational memory; retrieval-augmented generation; knowledge updates; embedding geometry; pre-registration; negative results; coreference resolution
```

## Área / categoría

Computer Science → Machine Learning (alternativa: Artificial Intelligence)

## Licencia

**CC-BY 4.0** — la misma que se viene usando en Zenodo, para que la reutilización sea coherente en
todo el cuerpo de trabajo.

## Archivos

| archivo | rol |
|---|---|
| `gemacion_en.pdf` | manuscrito principal |
| `gemacion_es.pdf` | material suplementario (versión completa en español) |

## Declaraciones (por si el formulario las pide por separado)

- **Funding:** None.
- **Competing interests:** The author declares no competing interests.
- **Author contributions:** Sole author; responsible for design, implementation, analysis and writing.
- **Data availability:** All pre-registrations, hashes, analysis scripts, raw per-seed results and
  machine-generated reports are archived in the project repository. Every verdict in the paper is
  emitted by a script frozen before the corresponding data existed.
- **Ethics / human subjects:** Not applicable — no human or animal subjects; all data is synthetic.
- **Preprint posted elsewhere:** No.

## Qué NO contratar

Research Square ofrece servicios pagos (edición, formato, traducción, «premium»). Ninguno hace falta:
la publicación del preprint es gratuita.
