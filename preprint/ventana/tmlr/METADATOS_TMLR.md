# Envío a TMLR del paper de la ventana

**Archivo a subir:** `ventana_tmlr.pdf` (7 páginas). TMLR sube **el PDF** a OpenReview, no el fuente.

**Anonimato verificado en las tres capas**, no supuesto:
1. el encabezado sale «Under review as submission to TMLR / Anonymous authors / Paper under
   double-blind review»;
2. `pdftotext` no devuelve ninguna coincidencia de nombre, ORCID, mail ni dominio institucional;
3. **los metadatos del PDF están vacíos** (`Author:`, `Title:`, `Subject:`, `Keywords:`). Ésta es la
   que se filtra siempre y la que nadie mira.

⚠️ Antes de cada recompilado hay que repetir el chequeo 3: `pdfinfo` puede volver a poblar `Author`
si alguien agrega `\hypersetup{pdfauthor=...}` o si se compila desde un editor que lo inyecta.

---

## Title

```
The Query Cannot See the Question: A Short Convolution's Reach Decides Which Part of a Query Conditions Retrieval
```

## Abstract

El del PDF, en texto plano. OpenReview acepta `$...$` para matemática simple; sacar `\textbf`,
`\emph` y `\texttt`.

## Keywords

```
state space models, linear attention, associative recall, abstention, selective prediction,
mechanistic interpretability, retrieval, pre-registration
```

## Lo que hay que declarar en el formulario

- **Preprint previo:** decir que sí, con el enlace de arXiv cuando exista. TMLR lo permite
  explícitamente: *«It is acceptable for a submission to overlap with the author's previous work if
  it was shared […] on preprint servers such as arXiv and bioRxiv.»* No declararlo sí sería un
  problema.
- **Conflictos de interés:** la Facultad Regional Buenos Aires de la UTN, que es lo que ya quedó
  cargado en el perfil de OpenReview. El sistema lo toma de ahí.

## Candidatos a Action Editor

Elegir de la lista viva de AEs de TMLR (cambia seguido, así que hay que mirarla al momento de
enviar) a alguien cuyo perfil incluya **modelos de espacio de estados / atención lineal** o
**interpretabilidad mecanicista**. No elegir por prestigio general: el criterio de TMLR es si los
claims están sostenidos, y eso lo juzga mejor quien conoce la arquitectura.

---

## Por qué este paper y no los otros dos, con el número que lo decide

La cuota es la *Generalized Harmonic Quota Rule* con N₁ = 2: firmando solo, **dos envíos al año**. Y
textual de la política: *«Budget is spent for all submissions by an author, including those which are
desk rejected.»* **Un desk reject cuesta medio año**, así que el primer envío tiene que ser el más
difícil de rebotar de entrada.

| candidato | dónde vive la evidencia |
|---|---|
| **ventana** (éste) | micro-LM **+ `mamba-130m` y `mamba-370m`**, modelos que no son nuestros |
| E1 de Ligamento | sólo el micro-LM |
| CENTINELA | sólo simulación propia |

Y este paper trae **un negativo pre-registrado adentro** (Resultado 6: el efecto de acceso no se
traduce en falla conductual en un modelo de 24 capas), con la explicación de por qué el negativo es
coherente con el positivo. Un editor que busque motivos para rebotarlo se encuentra la objeción ya
contestada en el texto, que es exactamente lo contrario de lo que produce un desk reject.
